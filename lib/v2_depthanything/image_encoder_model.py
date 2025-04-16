#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn

from .components.position_encoder import PositionEncoder
from .components.transformer_block import TransformerBlock
from .components.misc_helpers import LayerNormEPS6


# ---------------------------------------------------------------------------------------------------------------------
#%% Main model

class DinoV2Model4Stages(nn.Module):
    
    '''
    Simplified implementation of the DINOv2 image encoder of Depth-Anything.
    The original model architecture comes from:
        "DINOv2: Learning Robust Visual Features without Supervision"
        By: Maxime Oquab, Timothée Darcet, Théo Moutakanni et al.
        @ https://arxiv.org/abs/2304.07193
    
    The code here is derived from the Depth-Anything V2 copy of the dinov2 implementation:
        @ https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dinov2.py
    
    The original dinov2 code repo can be found here:
        @ https://github.com/facebookresearch/dinov2
        
    This implementation removes most of the flexibility as well as training-specific elements
    of the original model, for the sake of readability. It has also been modified specifically 
    for use in the DPT model architecture for depth estimation, originally proposed in the
    paper "Vision Transformers for Dense Prediction":
        @ https://arxiv.org/abs/2103.13413
    '''
    
    # .................................................................................................................
    
    def __init__(self, features_per_token=768, num_heads=12, num_blocks=12,
                 base_patch_grid_hw=(37,37), enable_cache=False, enable_optimizations=True):
        
        # Inherit from parent
        super().__init__()
        
        # Set up positional encoder for image patches
        self.posenc = PositionEncoder(features_per_token, base_patch_grid_hw, enable_cache=enable_cache)
        
        # Set up model components
        self.cls_token = nn.Parameter(torch.zeros(1, 1, features_per_token))
        self.outnorm = LayerNormEPS6(features_per_token)
        
        # Set up transformer stages (bulk of model processing)
        num_stages = 4
        layers_per_stage = int(round(num_blocks / num_stages))
        stages_list= []
        for _ in range(num_stages):
            one_stage = TransformerStage(layers_per_stage, features_per_token, num_heads, enable_optimizations)
            stages_list.append(one_stage)
        self.stages = nn.ModuleList(stages_list)

    # .................................................................................................................

    def forward(self, patch_tokens, patch_grid_hw):
        
        # Add positional embeddings & prepend cls token to patches
        clstok, patch_tokens = self.posenc(self.cls_token, patch_tokens, patch_grid_hw)
        tokens = torch.cat((clstok.expand(patch_tokens.shape[0], -1, -1), patch_tokens), dim=1)
        
        # Run each stage and store resulting tokens for passing to DPT structure
        stage_results = []
        for stage in self.stages:
            tokens = stage(tokens)
            stage_results.append(tokens)
        
        # Apply (shared) layer norm to all outputs
        stage_1, stage_2, stage_3, stage_4 = [self.outnorm(result) for result in stage_results]
        return stage_1, stage_2, stage_3, stage_4
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Component classes

class TransformerStage(nn.Module):
    
    '''
    Helper wrapper around groups of blocks. Each group of blocks generates a single
    output of the model (of which there are 4, by default) which are then passed on
    to the rest of the DPT structure for further processing.
    
    This could be done as a simple nn.Sequential, but is handled this way mostly for
    nicer readability in model weights.
    For example, this gives us weights like: stages.2.blocks.4.qkv...
    instead of (if using nn.Sequential): stages.2.4.qkv...
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_blocks, features_per_token, num_heads, enable_optimizations):
        
        # Inherit from parent
        super().__init__()
        
        # Create list of transformer blocks (note all blocks share the exact same config!)
        self.blocks = nn.ModuleList(
            TransformerBlock(features_per_token, num_heads, enable_optimizations) for _ in range(num_blocks)
        )
    
    # .................................................................................................................
    
    def forward(self, tokens):
        
        '''
        Process tokens sequentially through all blocks. Note output has the same shape as input: BxNxF
        -> B is batch dimension
        -> N is number of tokens
        -> F is features per token
        '''
        
        for each_block in self.blocks:
            tokens = each_block(tokens)
    
        return tokens
