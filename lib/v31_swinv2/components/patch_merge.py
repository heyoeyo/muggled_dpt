#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class PatchMerge(nn.Module):
    
    '''
    Module which performs 'patch merging' as described by the Swin (V1) paper:
        "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        By: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
        @ https://arxiv.org/abs/2103.14030
    
    The code here is derived from the timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/ce4d3485b690837ba4e1cb4e0e6c4ed415e36cea/timm/models/swin_transformer_v2.py#L360
    
    The effect of patch merging is to take an input set of tokens which come from an
    image-like representation of shape: BxHxWxC and halve the spatial sizings, while doubling
    the channel length, so that the output is of shape: Bx(H/2)x(W/2)x(2C), though this
    image-like shape is converted to a 'rows of tokens' format: BxNx(2C) for output.
    '''
    
    # .................................................................................................................
    
    def __init__(self, features_per_patch_in, features_per_patch_out):
        
        # Inherit from parent
        super().__init__()
        
        # Account for features after stacking tl/bl/tr/br samples
        features_after_stacking = 4 * features_per_patch_in
        self.reduction = nn.Linear(features_after_stacking, features_per_patch_out, bias=False)
        self.norm = nn.LayerNorm(features_per_patch_out)

    # .................................................................................................................
    
    def forward(self, tokens, patch_grid_hw):
        
        '''
        Combines tokens together in a way that reduces the number of tokens
        by a factor of 4, while doubling the number of features per token.
        Note that this transformation uses an image-like interpretation of the tokens, where
        every 2x2 set of 'image patch tokens' is merged into a single token
        
        The input is expected to be a 'rows of tokens' with shape: BxNxC,
        but with an image-like representation of shape: BxHxWxC, given by
        the patch_grid_hw
        
        Returns:
            tokens_merged, patch_grid_hw
        
        Note: the merged tokens have a shape: Bx(N/4)x(2C)
        '''
        
        # For convenience
        H, W = patch_grid_hw
        B, N, C = tokens.shape

        # Re-arrange tokens into image-like tensors, so we can perform 'every-other' spatial indexing
        imagelike_tensor = tokens.view(B, H, W, C)

        # Sample input patch grid into 4 smaller copies which sample even/odd indices of rows and columns
        # Each copy 'decimates' the original by picking from top-left/top-right/bot-left/bot-right
        #  TL:     BL:     TR:     BR:
        #  █░█░█░  ░░░░░░  ░█░█░█  ░░░░░░
        #  ░░░░░░  █░█░█░  ░░░░░░  ░█░█░█
        #  █░█░█░  ░░░░░░  ░█░█░█  ░░░░░░
        #  ░░░░░░  █░█░█░  ░░░░░░  ░█░█░█
        # Shape of each samping: B x H/2 x W/2 x C
        tl_samples = imagelike_tensor[:, 0::2, 0::2, :]
        bl_samples = imagelike_tensor[:, 1::2, 0::2, :]
        tr_samples = imagelike_tensor[:, 0::2, 1::2, :]
        br_samples = imagelike_tensor[:, 1::2, 1::2, :]
        
        # Stack 4 decimated grids on top of each other (each has half the original input height & width)
        # Shape: B x H/2 x W/2 x 4C
        imagelike_tensor = torch.cat([tl_samples, bl_samples, tr_samples, br_samples], -1)
        
        # Form output sizing, for future layers
        _, out_h, out_w, _ = imagelike_tensor.shape
        out_patch_grid_hw = (out_h, out_w)
        
        # Revert to 'rows of tokens' tensor shape and use linear layer to halve new feature count
        # -> Note this has the effect of halving both spatial dimensions of the 'patch grid'
        #    while doubling the channels per patch! Overall a halving of total features
        #    (i.e. (H/2)*(W/2)*(2C) = H*W*C/2)
        # Final shape: B x N/4 x 2C
        tokens = imagelike_tensor.view(B, N//4, 4*C)
        tokens = self.reduction(tokens)
        tokens = self.norm(tokens)
        
        return tokens, out_patch_grid_hw
    
    # .................................................................................................................

