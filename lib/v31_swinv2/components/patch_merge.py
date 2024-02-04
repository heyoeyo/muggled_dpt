#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class PatchMerge(nn.Module):
    
    # .................................................................................................................
    
    def __init__(self, patch_shape_hwc):
        
        # Inherit from parent
        super().__init__()
        
        # Store sizing info, needed to interpret transformer tokens as an image patch grid
        _, _, features_per_patch = patch_shape_hwc
        self.patch_shape_hwc = patch_shape_hwc
        
        # Pre-compute input/output feature counts
        features_after_stacking = 4 * features_per_patch
        output_features_per_patch = features_after_stacking // 2
        
        self.reduction = nn.Linear(features_after_stacking, output_features_per_patch, bias=False)
        self.norm = nn.LayerNorm(output_features_per_patch)

    # .................................................................................................................
    
    def forward(self, tokens):
        
        '''
        Combines tokens together in a way that reduces the number of tokens
        by a factor of 4, while doubling the number of features per token.
        Note that this transformation uses an image-like interpretation of the tokens, where
        every 2x2 set of 'image patch tokens' is merged into a single token
        '''
        
        # For convenience
        H, W, _ = self.patch_shape_hwc
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
        
        # Revert to 'rows of tokens' tensor shape and use linear layer to halve new feature count
        # -> Note this has the effect of halving both spatial dimensions of the 'patch grid'
        #    while doubling the channels per patch! Overall a halving of total features
        #    (i.e. (H/2)*(W/2)*(2C) = H*W*C/2)
        # Final shape: B x N/4 x 2C
        tokens = imagelike_tensor.view(B, N//4, 4*C)
        tokens = self.reduction(tokens)
        tokens = self.norm(tokens)
        
        return tokens
    
    # .................................................................................................................
    
    @staticmethod
    def compute_patch_shape_per_merge(base_patch_shape_hwc, num_merges):
        
        '''
        Function used to determine the output patch sizing (height x width x channels)
        after some number of consecutive patch merging, which have the effect of halving
        the spatial dimensions and doubling the channel count
        
        Returns a list of shapes in hwc format
        For example for input shape (8,8,64) and num_merges = 4:
            [
              [8,8,64],
              [4,4,128],
              [2,2,256],
              [1,1,512]
            ]
        '''
        
        # For convenience
        base_h, base_w, base_c = base_patch_shape_hwc
        
        # Computer output patch shape for each (consequtive) patch merge
        patch_shapes_hwc_per_merge = []
        for n in range(num_merges):
            doubling_factor = int(2 ** n)
            
            out_h = base_h // doubling_factor
            out_w = base_w // doubling_factor
            out_c = int(base_c * doubling_factor)
            
            patch_shapes_hwc_per_merge.append((out_h, out_w, out_c))
        
        return patch_shapes_hwc_per_merge
    
    # .................................................................................................................
