#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch.nn as nn

from .components.misc_helpers import SpatialUpsampleLayer, Conv3x3Layer, Conv1x1Layer


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class MonocularDepthHead(nn.Module):
    
    '''
    Implementation of the 'Monocular depth estimate head' model, described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    This model is a multi-stage convolution model, whose main purpose is to reduce the number of
    channels of the given input down to 1, representing the final depth-estimate value from
    a DPT (dense prediction transformer) model. It also doubles the width/height of the output,
    compared to the input.
    
    For more info, see the appendix of the paper, section: "Monocular depth estimation head"
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_channels_in):
        
        # Inherit from parent
        super().__init__()
        
        # For clarity, define layer config settings
        spatial_upsample_factor = 2
        channels_in = num_channels_in
        channels_half = num_channels_in // 2
        channels_fixed = 32
        channels_out = 1

        # Create component that spatially upsamples input features (while reducing channels)
        self.spatial_upsampler = nn.Sequential(
            Conv3x3Layer(channels_in, channels_half),
            SpatialUpsampleLayer(spatial_upsample_factor),
        )
        
        # Create component which collapses features to a single channel
        self.proj_1ch = nn.Sequential(
            Conv3x3Layer(channels_half, channels_fixed),
            nn.ReLU(True),
            Conv1x1Layer(channels_fixed, channels_out),
            nn.ReLU(True),
        )
    
    # .................................................................................................................
    
    def forward(self, fusion_feature_map):
        
        # Halve channels while upsampling spatially
        output = self.spatial_upsampler(fusion_feature_map)
        
        # Collapse down to 1 channel for depth output
        output = self.proj_1ch(output)
        
        # Remove unitary channel for output (shape goes from: Bx1xHxW -> BxHxW)
        return output.squeeze(dim=1)
    
    # .................................................................................................................
