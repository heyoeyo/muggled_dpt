#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn

from .components.misc_helpers import Conv3x3Layer


# ---------------------------------------------------------------------------------------------------------------------
#%% Main model

class ReassembleModel(nn.Module):
    
    '''
    Simplified implementation of the 'reassembly' model/component described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    This implementation is intended for use with a SwinV2 DPT model, which has slightly different
    scaling (per stage) compared to the description in the original DPT paper. This is a result
    of the SwinV2 model performing it's own spatial down-scaling, so that the reassembly model
    does not need to do this!
    
    This model includes all 4 (hard-coded) reassembly blocks into a single model
    '''
    
    # .................................................................................................................
    
    def __init__(self, hidden_channels_list, num_output_channels):
        
        # Inherit from parent
        super().__init__()
        
        # Make sure we get exactly 4 hidden channel counts
        ok_hidden_counts = len(hidden_channels_list) == 4
        assert ok_hidden_counts, "Expecting 4 reassembly channel counts, got: {}".format(hidden_channels_list)
        hidden_1, hidden_2, hidden_3, hidden_4 = hidden_channels_list
        
        # Build reassembly blocks for each transformer output stage
        self.spatial_noscale = ReassembleBlock(hidden_1, num_output_channels, downscale_factor=1)
        self.spatial_downx2 = ReassembleBlock(hidden_2, num_output_channels, downscale_factor=2)
        self.spatial_downx4 = ReassembleBlock(hidden_3, num_output_channels, downscale_factor=4)
        self.spatial_downx8 = ReassembleBlock(hidden_4, num_output_channels, downscale_factor=8)
    
    # .................................................................................................................
    
    def forward(self, stage_1_tokens, stage_2_tokens, stage_3_tokens, stage_4_tokens, patch_grid_hw):
        
        # Perform all 4 reassembly stages
        stage_1_imglike = self.spatial_noscale(stage_1_tokens, patch_grid_hw)
        stage_2_imglike = self.spatial_downx2(stage_2_tokens, patch_grid_hw)
        stage_3_imglike = self.spatial_downx4(stage_3_tokens, patch_grid_hw)
        stage_4_imglike = self.spatial_downx8(stage_4_tokens, patch_grid_hw)
        
        return stage_1_imglike, stage_2_imglike, stage_3_imglike, stage_4_imglike
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Model components

class ReassembleBlock(nn.Module):
    
    '''
    According to paper, reassembly consists of 3 steps + 1 undocumented (?) step:
        1. Read (handle readout token)
        2. Concatenate (into image-like tensor)
        3. Project + Resample
        4. Project channels to match fusion input sizing (not documented in paper!)
    
    However, for SwinV2, there is no readout token and the inputs to the reassembly model
    are already spatially scaled at different stages. This means that the reassembly
    block only needs to reshape the tokens into an image-like format, as well as
    performing the final (undocumented) step of projecting the channel count to
    match the fusion model input sizing.
    
    This block represents a single reassembly stage, which takes tokens output from
    a specific stage of a transformer model and 'reassembles' the token back into
    an image-like format, which is eventually given to a fusion model that
    combines all reassembled stages together into a single output (depth) image.
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_hidden_channels, num_output_channels, downscale_factor = 1):
        
        # Inherit from parent
        super().__init__()
        
        self._downscale_factor = downscale_factor
        self.token_to_2d = TokensTo2DLayer()
        self.fuse_proj = Conv3x3Layer(num_hidden_channels, num_output_channels, bias=False)
    
    # .................................................................................................................
    
    def forward(self, tokens_bnc, patch_grid_hw):
        
        # Account for downscaling of the patch grid, based on reassembly stage
        scaled_grid_hw = [size // self._downscale_factor for size in patch_grid_hw]
        
        # Convert to image-like tensor, along with projection (change channel count to match fusion model)
        output = self.token_to_2d(tokens_bnc, scaled_grid_hw)
        output = self.fuse_proj(output)
        
        return output
    
    # .................................................................................................................


class TokensTo2DLayer(nn.Module):
    
    '''
    The purpose of this layer is to convert transformer tokens into an
    image-like representation. That is, re-joining the image patch tokens (vectors)
    output from a transformer back into a 2D representation (with many channels)
    More formally, this layer reshapes inputs from: BxNxC -> BxCxHxW
    (This layer does not compute/modify values, it just reshapes the tensors!)
    '''
    
    # .................................................................................................................
    
    def __init__(self):
        
        # Inherit from parent
        super().__init__()
    
    # .................................................................................................................
    
    def forward(self, tokens_bnc, patch_grid_hw):
        
        '''
        Assume input tokens have shape:
            B x N x C
        -> Where B is batch size
        -> N is number of tokens
        -> C is the feature size ('channels') of the tokens
        
        Returns a single output tensor of shape:
            BxCxHxW
        -> Where H & W correspond to the number of image patches vertically (H) and horizontally (W)
        '''
        
        # Transpose to get tokens in last dimension: BxCxN
        output = torch.transpose(tokens_bnc, 1, 2)
        
        # Expand last (token) dimension into HxW to get image-like (patch-grid) shape: BxDxN -> BxDxHxW
        output = torch.unflatten(output, 2, patch_grid_hw)
        
        return output
    
    # .................................................................................................................
