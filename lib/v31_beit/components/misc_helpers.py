#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class SpatialUpsampleLayer(nn.Module):
    
    '''
    Simpler upscaling layer. Just a wrapper around the interpolation function.
    Note, there are no learnable parameters!
    (purpose is to allow for defining upsample layers within sequential models)
    '''
    
    # .................................................................................................................
    
    def __init__(self, scale_factor = 2, interpolation_mode = "bilinear"):
        
        # Inherit from parent
        super().__init__()
        
        # Store layer config
        self._scale_factor = scale_factor
        self._mode = interpolation_mode
    
    # .................................................................................................................
    
    def forward(self, feature_map_2d):
        return nn.functional.interpolate(feature_map_2d,
                                         scale_factor=self._scale_factor,
                                         mode=self._mode,
                                         align_corners=True)
    
    # .................................................................................................................
    
    def extra_repr(self):
        return "scale_factor={}, interpolation_mode={}".format(self._scale_factor, self._mode)
    
    # .................................................................................................................


class Conv3x3Layer(nn.Conv2d):
    
    '''
    Helper class used to create 3x3 2D convolution layers.
    Configured so that the width & height of the output match the input.
    If an output channel count isn't specified, will default to matching input channel count.
    '''
    
    # .................................................................................................................
    
    def __init__(self, in_channels, out_channels = None, bias = True):
        
        # Instantiate parent with some fixed arguments for 3x3 operation
        out_channels = in_channels if (out_channels is None) else out_channels
        super().__init__(in_channels, out_channels, bias = bias,
                         kernel_size = 3, stride = 1, padding = 1)
    
    # .................................................................................................................


class Conv1x1Layer(nn.Conv2d):
    
    '''
    Helper class used to create 1x1 2D convolution layers (i.e. depthwise convolution).
    Configured so that the width & height of the output match the input.
    If an output channel count isn't specified, will default to matching input channel count.
    '''
    
    # .................................................................................................................
    
    def __init__(self, in_channels, out_channels = None, bias = True):
        
        # Instantiate parent with some fixed arguments for 1x1 operation
        out_channels = in_channels if (out_channels is None) else out_channels
        super().__init__(in_channels, out_channels, bias=bias,
                         kernel_size = 1, stride = 1, padding = 0)
    
    # .................................................................................................................
