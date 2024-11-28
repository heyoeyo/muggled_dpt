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
    a DPT (dense prediction transformer) model and restore the original input resolution.
    
    For more info, see the appendix of the paper, section: "Monocular depth estimation head"
    
    This implementation has modifications specific to the Depth-Anything model,
    which does not simply scale the output by a factor of 2 like the original DPT models.
    The original Depth-Anything implementation scales the result to a target size
    (which is taken as an input), though the implementation here does something different...
    
    In this implementation, the scaling is fixed at a factor of P/8, where P is the
    patch size used by the patch embedding (14 by default).
    To see why this works, consider that the DPT model takes an input image of 
    size N and divides it into N/P patches. The fusion component of the model
    ends up scaling this patch sizing by a factor of 2*2*2, and then the
    head model (this module) is meant to restore the sizing back to the original
    input size. So we need a factor H which multiplies the sizing from the fusion
    step back to the original size N:
        
        H * (2*2*2) * (N/P) = N
        H * 8 / P = 1
        
        So: H = P / 8
    
    This approach is more fragile than the original Depth-Anything version, but allows
    the model to share an implementation with existing DPT models, so is preferred here.
    
    Also note that the V2 metric models use a slightly modified head model, which
    is supported by this implementation (though this isn't part of the original DPT structure).
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_channels_in, patch_size_px = 14, is_metric = False):
        
        # Inherit from parent
        super().__init__()
        
        # For clarity, define layer config settings
        spatial_upsample_factor = patch_size_px / 8
        channels_in = num_channels_in
        channels_half = num_channels_in // 2
        channels_fixed = 32
        channels_out = 1

        # Create component that spatially upsamples input features (while reducing channels)
        self.spatial_upsampler = nn.Sequential(
            Conv3x3Layer(channels_in, channels_half),
            SpatialUpsampleLayer(scale_factor = spatial_upsample_factor),
        )
        
        # Create component which collapses features to a single channel
        self.proj_1ch = nn.Sequential(
            Conv3x3Layer(channels_half, channels_fixed),
            nn.ReLU(True),
            Conv1x1Layer(channels_fixed, channels_out),
            nn.ReLU(True) if not is_metric else nn.Sigmoid(),
        )
    
    # .................................................................................................................
    
    def forward(self, imagelike_bchw):
        
        '''
        Spatially upscales the input image-like tokens while projecting down
        to a single channel.
        
        The input is expected to have shape: BxCxHxW
        The output has shape: Bx1xH'xW'
        -> Where H', W' are the upscaled height & width (factor of 1.75x by default)
        '''
        
        # Halve channels while upsampling spatially
        output = self.spatial_upsampler(imagelike_bchw)
        
        # Collapse down to 1 channel for depth output
        output = self.proj_1ch(output)
        
        # Remove unitary channel for output (shape goes from: Bx1xHxW -> BxHxW)
        return output.squeeze(dim=1)
    
    # .................................................................................................................
