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
            
    The code here is derived from the Depth-Anything copy of the dinov2 repo:
        @ https://github.com/LiheYoung/Depth-Anything/tree/e7ef4b4b7a0afd8a05ce9564f04c1e5b68268516/torchhub/facebookresearch_dinov2_main
    
    The original dinov2 code repo can be found here:
        @ https://github.com/facebookresearch/dinov2
        
    This implementation removes most of the flexibility as well as training-specific elements
    of the original model, for the sake of readability. It has also been modified specifically 
    for use in the DPT model architecture for depth estimation, originally proposed in the
    paper "Vision Transformers for Dense Prediction":
        @ https://arxiv.org/abs/2103.13413
    
    Note that unlike the original DPT models, this model does not output intermediate tokens
    from the transformers, instead it outputs the last 4 (consecutive) blocks!
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
        self.blocks = nn.ModuleList(
            [TransformerBlock(features_per_token, num_heads, enable_optimizations) for _ in range(num_blocks)]
        )
        self.outnorm = LayerNormEPS6(features_per_token)

    # .................................................................................................................

    def forward(self, patch_tokens, patch_grid_hw):
        
        '''
        Takes in image patch tokens along with the original image 
        patch dimensions and returns the last 4 transformer block outputs.
        
        Note: The input patch tokens are assumed to be made from an image-like 
        representation with shape: BxHxWxC (i.e. batch, height, width, channels),
        which has been reshaped into a 'rows of tokens' format of shape: BxNxC
        
        The output tokens include a cls token, prepended to the original inputs.
        This means that the output of this model is 4 tokens, each of shape:
            Bx(1+N)xC
        '''
        
        # Add positional embeddings & prepend cls token to patches
        clstok, patch_tokens = self.posenc(self.cls_token, patch_tokens, patch_grid_hw)
        tokens = torch.cat((clstok.expand(patch_tokens.shape[0], -1, -1), patch_tokens), dim=1)
        
        # Run all but final 4 transformer blocks of the model
        for block in self.blocks[:-4]:
            tokens = block(tokens)
        
        # Run the last 4 blocks, which create the 4-stage output of the model
        stages_1to4 = []
        for block in self.blocks[-4:]:
            tokens = block(tokens)
            stages_1to4.append(tokens)
        
        # Apply (shared) layer norm to all outputs
        stage_1, stage_2, stage_3, stage_4  = [self.outnorm(result) for result in stages_1to4]
        return stage_1, stage_2, stage_3, stage_4
    
    # .................................................................................................................
