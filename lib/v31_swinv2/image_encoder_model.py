#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch.nn as nn

from .components.misc_helpers import MLP2Layers
from .components.patch_merge import PatchMerge
from .components.windowed_attention import WindowAttentionWithRelPos


# ---------------------------------------------------------------------------------------------------------------------
#%% Main model

class SwinV2Model4Stages(nn.Module):
    
    '''
    Simplified implementation of the SwinV2 backbone model, from:
        Swin Transformer V2: Scaling Up Capacity and Resolution
        Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, et. al
        @ https://arxiv.org/abs/2111.09883
    
    The code here is derived from the timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer_v2.py
    
    This implementation removes most of the flexibility as well as the training-specific
    elements of the original implementation, for the sake of clarity. It is also modified
    in such a way as to be purpose-built for use in the MiDaS v3.1 SwinV2 model
    (e.g. the forward method explicitly returns the 4 internal tokens needed for DPT)
    '''
    
    # .................................................................................................................
    
    def __init__(self,
                 features_per_patch = 96,
                 heads_per_stage = (3,6,12,24),
                 layers_per_stage = (2,2,6,2),
                 patch_grid_hw = (64,64),
                 window_size_hw = (16,16),
                 pretrained_window_size_per_stage = (16,16,16,8),
                 enable_cache = True,
                 ):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.features_per_patch = features_per_patch
        self.window_size_hw = window_size_hw
        self.patch_grid_hw = patch_grid_hw

        # Figure out the patch sizing at each layer (changes due to match merging)
        num_stages = len(heads_per_stage)
        base_grid_h, base_grid_w = patch_grid_hw
        base_patch_shape_hwc = (base_grid_h, base_grid_w, features_per_patch)
        patch_shape_hwc_per_merge = PatchMerge.compute_patch_shape_per_merge(base_patch_shape_hwc, num_stages)
        
        # Build stages
        stage_iter = zip(layers_per_stage, heads_per_stage, pretrained_window_size_per_stage)
        self.stages = nn.ModuleList()
        for stage_idx, (num_layers, num_heads, pretrained_window_size) in enumerate(stage_iter):
            
            # Determine the input (to patch merge) and output (from merge) patch shape for each stage
            num_prev_merges = max(stage_idx - 1, 0)
            in_shape_hwc = patch_shape_hwc_per_merge[num_prev_merges]
            out_shape_hwc = patch_shape_hwc_per_merge[stage_idx]
            
            # We don't need patch merging on the first-most block
            need_patch_merge = stage_idx > 0
            patch_merge_module = PatchMerge(in_shape_hwc) if need_patch_merge else None
            
            swin_stage = SwinStage(
                patch_merge_module = patch_merge_module,
                num_layers = num_layers,
                num_heads = num_heads,
                patch_shape_hwc = out_shape_hwc,
                window_size_hw = window_size_hw,
                pretrained_window_size = pretrained_window_size,
                enable_cache = enable_cache,
            )
            
            self.stages.append(swin_stage)
        
        pass
    
    # .................................................................................................................
    
    def forward(self, patch_tokens, patch_grid_hw = None):
        
        '''
        Main function of model.
        Takes in image patch tokens and the image patch sizing,
        returns tokens from 4 intermediate stages
        '''
        
        # Perform 4-stage processing, grabbing intermediate results
        stage_1_tokens = self.stages[0](patch_tokens)
        stage_2_tokens = self.stages[1](stage_1_tokens)
        stage_3_tokens = self.stages[2](stage_2_tokens)
        stage_4_tokens = self.stages[3](stage_3_tokens)
        
        return stage_1_tokens, stage_2_tokens, stage_3_tokens, stage_4_tokens
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Model components

class SwinStage(nn.Sequential):
    
    '''
    Simplified implementation of a single stage within the SwinV2 transformer model.
    The code here is derived from the timm library (originally called 'BasicLayer'):
        @ https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer_v2.py
    
    This stage takes in tokens (either from a patch embedding or a previous swin stage),
    and passes them through multiple swin transformer blocks in sequence. The output is another
    set of tokens.
    
    If the stage contains a patch merge (which is any stage except the one following patch embedding),
    then the number of output tokens will be reduced by a factor of 4, while the features per token
    will be increased by a factor of 2!
    '''
    
    # .................................................................................................................
    
    def __init__(self, patch_merge_module, num_layers, num_heads, patch_shape_hwc, window_size_hw,
                 pretrained_window_size=None, enable_cache = True):
        
        # Inclue pre-merge step, if needed
        block_list = []
        if patch_merge_module is not None:
            block_list.append(patch_merge_module)
        
        # Bundle shared block arguments for clarity
        shared_attn_kwargs = {
            "num_heads": num_heads,
            "patch_shape_hwc": patch_shape_hwc,
            "window_size_hw": window_size_hw,
            "pretrained_window_size": pretrained_window_size,
            "enable_cache": enable_cache,
        }
        
        # Build window/shifted-window transformer blocks
        num_block_pairs = num_layers // 2
        for i in range(num_block_pairs):
            
            # Build & add windowed transformer block to stage
            block1 = SwinTransformerBlock(**shared_attn_kwargs, is_shift_block=False)
            block_list.append(block1)
            
            # Build & add second transformer block, which is allowed to shift windows if possible
            block2 = SwinTransformerBlock(**shared_attn_kwargs, is_shift_block=True)
            block_list.append(block2)
        
        # Set up block sequence, from parent class
        super().__init__(*block_list)
    
    # .................................................................................................................


# =====================================================================================================================


class SwinTransformerBlock(nn.Module):
    
    '''
    Simplified implementation of a swinv2 transformer block, from (see figure 1):
        "Swin Transformer V2: Scaling Up Capacity and Resolution"
        Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, et. al
        @ https://arxiv.org/abs/2111.09883
    
    This module can be used to represent a 'windowed' transformer block as well as the
    'shifted window' transformer block, by passing in a shifting amount
    This design was originally described in swin-v1 (see figure 3):
        https://arxiv.org/abs/2103.14030
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_heads, patch_shape_hwc, window_size_hw, pretrained_window_size,
                 is_shift_block = False, enable_cache = True, mlp_ratio = 4):
        
        # Inherit from parent
        super().__init__()
        
        # Setup sub-layers
        _, _, features_per_token = patch_shape_hwc
        self.attn = WindowAttentionWithRelPos(num_heads, patch_shape_hwc, window_size_hw, 
                                              pretrained_window_size, is_shift_block, enable_cache)
        self.norm1 = nn.LayerNorm(features_per_token)
        self.mlp = MLP2Layers(features_per_token, mlp_ratio)
        self.norm2 = nn.LayerNorm(features_per_token)
    
    # .................................................................................................................
    
    def forward(self, tokens):
        
        # Calculate (post-norm) attention with residual connection
        attn_tokens = self.attn(tokens)
        attn_tokens = self.norm1(attn_tokens)
        attn_tokens = tokens + attn_tokens
        
        # Calculate (post-norm) feedforward output with residual connection
        output_tokens = self.mlp(attn_tokens)
        output_tokens = self.norm2(output_tokens)
        output_tokens = attn_tokens + output_tokens
        
        return output_tokens
    
    # .................................................................................................................
