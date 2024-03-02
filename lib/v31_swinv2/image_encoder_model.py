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
                 features_per_stage = (96, 192, 384, 768),
                 heads_per_stage = (3,6,12,24),
                 layers_per_stage = (2,2,6,2),
                 window_size_hw = (16,16),
                 pretrained_window_size_per_stage = (16,16,16,8),
                 enable_cache = True,
                 ):
        
        # Inherit from parent
        super().__init__()
        
        # Build patch merge layers, used in front of stages 2, 3, 4
        features_in_out = zip(features_per_stage[:-1], features_per_stage[1:])
        self.patch_merge_layers = nn.ModuleList(
            PatchMerge(feat_in, feat_out) for feat_in, feat_out in features_in_out
        )
        
        # Build stages
        stage_iter = zip(features_per_stage, layers_per_stage, heads_per_stage, pretrained_window_size_per_stage)
        self.stages = nn.ModuleList()
        for features_per_token, num_layers, num_heads, pretrained_window_size in stage_iter:
            
            swin_stage = SwinStage(
                num_layers = num_layers,
                num_heads = num_heads,
                features_per_token = features_per_token,
                window_size_hw = window_size_hw,
                pretrained_window_size = pretrained_window_size,
                enable_cache = enable_cache,
            )
            
            self.stages.append(swin_stage)
        
        pass
    
    # .................................................................................................................
    
    def forward(self, patch_tokens, patch_grid_hw):
        
        '''
        Main function of model.
        Takes in image patch tokens and the image patch sizing,
        returns tokens from 4 intermediate stages
        
        Each of the later stages (2,3,4) require a patch merging step before the transformer block!
        '''
        
        # Perform 4-stage processing, with patch merging
        stage_1_tokens = self.stages[0](patch_tokens, patch_grid_hw)
        
        stage_2_tokens, stage_2_grid_hw = self.patch_merge_layers[0](stage_1_tokens, patch_grid_hw)
        stage_2_tokens = self.stages[1](stage_2_tokens, stage_2_grid_hw)
        
        stage_3_tokens, stage_3_grid_hw = self.patch_merge_layers[1](stage_2_tokens, stage_2_grid_hw)
        stage_3_tokens = self.stages[2](stage_3_tokens, stage_3_grid_hw)
        
        stage_4_tokens, stage_4_grid_hw = self.patch_merge_layers[2](stage_3_tokens, stage_3_grid_hw)
        stage_4_tokens = self.stages[3](stage_4_tokens, stage_4_grid_hw)
        
        return stage_1_tokens, stage_2_tokens, stage_3_tokens, stage_4_tokens
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Model components

class SwinStage(nn.Module):
    
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
    
    def __init__(self, num_layers, num_heads, features_per_token, window_size_hw,
                 pretrained_window_size=None, enable_cache = True):
        
        # Inherit from parent
        super().__init__()
        
        # Bundle shared block arguments for clarity
        shared_attn_kwargs = {
            "num_heads": num_heads,
            "features_per_token": features_per_token,
            "window_size_hw": window_size_hw,
            "pretrained_window_size": pretrained_window_size,
            "enable_cache": enable_cache,
        }
        
        # Build alternating window/shifted-window transformer blocks
        block_list = []
        num_block_pairs = num_layers // 2
        for i in range(num_block_pairs):
            
            # Build & add windowed transformer block to stage
            block1 = SwinTransformerBlock(**shared_attn_kwargs, is_shift_block=False)
            block_list.append(block1)
            
            # Build & add second transformer block, which is allowed to shift windows if possible
            block2 = SwinTransformerBlock(**shared_attn_kwargs, is_shift_block=True)
            block_list.append(block2)
        
        # Store swin blocks
        self.blocks = nn.ModuleList(block_list)
    
    # .................................................................................................................
    
    def forward(self, tokens, patch_grid_hw):
        
        for block in self.blocks:
            tokens = block(tokens, patch_grid_hw)
        
        return tokens
    
    # .................................................................................................................


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
    
    def __init__(self, num_heads, features_per_token, window_size_hw, pretrained_window_size,
                 is_shift_block = False, enable_cache = True, mlp_ratio = 4):
        
        # Inherit from parent
        super().__init__()
        
        # Setup sub-layers
        self.attn = WindowAttentionWithRelPos(num_heads, features_per_token, window_size_hw, 
                                              pretrained_window_size, is_shift_block, enable_cache)
        self.norm1 = nn.LayerNorm(features_per_token)
        self.mlp = MLP2Layers(features_per_token, mlp_ratio)
        self.norm2 = nn.LayerNorm(features_per_token)
    
    # .................................................................................................................
    
    def forward(self, tokens, patch_grid_hw):
        
        # Calculate (post-norm) attention with residual connection
        attn_tokens = self.attn(tokens, patch_grid_hw)
        attn_tokens = self.norm1(attn_tokens)
        attn_tokens = tokens + attn_tokens
        
        # Calculate (post-norm) feedforward output with residual connection
        output_tokens = self.mlp(attn_tokens)
        output_tokens = self.norm2(output_tokens)
        output_tokens = attn_tokens + output_tokens
        
        return output_tokens
    
    # .................................................................................................................
