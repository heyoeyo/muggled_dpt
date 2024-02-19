#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch.nn as nn

from .components.misc_helpers import SpatialUpsampleLayer, Conv3x3Layer, Conv1x1Layer


# ---------------------------------------------------------------------------------------------------------------------
#%% Main model

class FusionModel(nn.Module):
    
    '''
    Simplified implementation of the 'fusion' model/component described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    This model includes logic for handling the entire fusion process
    (including) merging all layers together.
    
    The purpose of this model is to process the multi-stage outputs
    taken from a transformer model into 1-channel depth 'image'
    
    This model seems to be based off of an earlier idea (RefineNet) from the paper:
        "RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation"
        By: Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid
        @ https://arxiv.org/abs/1611.06612
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_fusion_channels, num_blocks = 4):
        
        # Inherit from parent
        super().__init__()
        
        # Make lower-layer fusion blocks
        num_lower_blocks = num_blocks - 1
        self.blocks = nn.ModuleList(FusionBlock(num_fusion_channels) for _ in range(num_lower_blocks))
        
        # Add top-most fusion block, which works differently (no prior fusion input)
        self.blocks.append(TopMostFusionBlock(num_fusion_channels))
    
    # .................................................................................................................
    
    def forward(self, upx4_featuremap, upx2_featuremap, noscale_featuremap, downx2_featuremap):
        
        # Fuse all results together
        fuse_3 = self.blocks[3](downx2_featuremap)
        fuse_2 = self.blocks[2](noscale_featuremap, fuse_3)
        fuse_1 = self.blocks[1](upx2_featuremap, fuse_2)
        fuse_0 = self.blocks[0](upx4_featuremap, fuse_1)
        
        return fuse_0
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Model Components

class TopMostFusionBlock(nn.Module):
    
    '''
    Helper class, used  to distinguish the final fusion block from 'regular' fusion blocks
    Unlike other fusion blocks, the final block does not take in a 'prior fusion' feature map
    as part of it's forward pass. As a result, there is no residual addition step between the
    reassembly feature map & prior fusion. The original implementation also chooses to remove
    the convolution applied to the reassembly feature map, presumably, since the output
    convolution/projection handles this.
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_channels):
        
        # Inherit from parent
        super().__init__()
        
        # Include only the final project block for top-most layer
        # -> Compared to 'regular' fusion blocks: there is no conv_assembly or residual addtition step!
        # -> The lack of conv_assembly is somewhat surprising, given this layer still takes in assembly data...
        self.proj_seq = UpsampleProjectionBlock(num_channels)
    
    # .................................................................................................................
    
    def forward(self, imagelike_feature_map):
        return self.proj_seq(imagelike_feature_map)
    
    # .................................................................................................................


class FusionBlock(nn.Module):
    
    '''
    Simplified implementation of a single 'fusion block', described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    Specifically, this block implements a single 'fusion' step as shown in
    Figure 1 of the paper (right-most diagram), with consists of two
    'residual convolution units', an upsample step and a projection layer.
    
    The purpose of this block is to process & combine the multi-stage
    outputs taken from a transformer model into a single image-like
    output (with many feature channels).
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_channels):
        
        # Inherit from parent
        super().__init__()
        
        # Define models for the layers referenced in paper (see right-most figure 1)
        self.conv_reassembly = ResidualConv2D(num_channels)
        self.proj_seq = UpsampleProjectionBlock(num_channels)
    
    # .................................................................................................................
    
    def forward(self, reassembly_feature_map, previous_fusion_feature_map):
        
        # Calculate residual addition of 'reassembled & previous fusion' inputs
        output = self.conv_reassembly(reassembly_feature_map) + previous_fusion_feature_map
        
        # Perform further processing & projection
        output = self.proj_seq(output)
        
        return output
    
    # .................................................................................................................


class UpsampleProjectionBlock(nn.Sequential):
    
    '''
    Helper block which implements part of the fusion block, described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    Specifically, this block implements the 3 steps following the 'addition' step
    within the 'fusion' block: residual convolution -> upsample -> projection
    
    Expects inputs of shape: BxCxHxW
    Returns output of shape: BxCx(2H)x(2W)
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_channels):
        
        # Inherit from parent
        super().__init__(
            ResidualConv2D(num_channels),
            SpatialUpsampleLayer(scale_factor=2),
            Conv1x1Layer(num_channels),
        )
    
    # .................................................................................................................


class ResidualConv2D(nn.Module):
    
    '''
    Simplified implementation of the 'Residual Conv Unit', described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    This model is really just a two-stage 3x3 convolution model (with relu activations), along with
    a resnet-like skip connection (i.e. adding the input to the convolution result). Note that the
    convolutions do NOT change the size of the data, that is, the output has the same number of
    channels along with the same width & height as the input!
    
    For more info, see the appendix of the paper, section: "Residual convolutional units"
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_channels):
        
        # Inherit from parent
        super().__init__()
        
        # Define 2-layer convolution network
        self.conv_seq = nn.Sequential(
            nn.ReLU(False),         # Don't do in-place, since input data may be used elsewhere!
            Conv3x3Layer(num_channels),
            nn.ReLU(True),
            Conv3x3Layer(num_channels),
        )
    
    # .................................................................................................................
    
    def forward(self, feature_map):
        return self.conv_seq(feature_map) + feature_map
    
    # .................................................................................................................
