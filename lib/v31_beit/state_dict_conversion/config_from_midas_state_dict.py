#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch

from .key_regex import get_nth_integer


# ---------------------------------------------------------------------------------------------------------------------
#%% Main function

def get_model_config_from_midas_state_dict(state_dict, enable_cache, enable_optimizations):
    
    # Get model config from state dict
    config_dict = {
        "features_per_token": get_transformer_features_per_token(state_dict),
        "num_blocks": get_num_transformer_blocks(state_dict),
        "num_heads": get_num_transformer_heads(state_dict),
        "reassembly_features_list": get_reassembly_channel_sizes(state_dict),
        "fusion_channels": get_num_fusion_channels(state_dict),
        "patch_size_px": get_patch_size_px(state_dict),
        "base_patch_grid_hw": get_base_patch_grid_size(state_dict),
        "enable_cache": enable_cache,
        "enable_optimizations": enable_optimizations,
    }
    
    return config_dict

# ---------------------------------------------------------------------------------------------------------------------
#%% Component functions

def get_num_transformer_blocks(state_dict):
    
    '''
    State dict contains keys like:
        'pretrained.model.blocks.0.gamma_1'
        'pretrained.model.blocks.3.attn.q_bias',
        'pretrained.model.blocks.17.norm1.weight',
        ... etc
    This function tries to find the largest number from the '...blocks.#...' part of these keys,
    since this determines how many layers (aka depth) are in the transformer.
    '''
    
    # Search for all model blocks, looking for the highest index
    max_block_idx = -1
    for key in state_dict.keys():
        if "pretrained.model.blocks" in key:
            block_idx = get_nth_integer(key, 0)
            max_block_idx = max(max_block_idx, block_idx)
    
    # Blocks start counting at 0, so we need to add 1 to get total number of layers
    assert max_block_idx > 0, "Error determining number of transformer blocks! Could not find any blocks"
    num_transformer_blocks = 1 + max_block_idx
    
    return int(num_transformer_blocks)

# .....................................................................................................................

def get_num_transformer_heads(state_dict):
    
    # Make sure there is a relative positional encoding bias table key in the given state dict
    relpos_bias_table_key = "pretrained.model.blocks.0.attn.relative_position_bias_table"
    assert relpos_bias_table_key in state_dict.keys(), \
        "Error determining number of heads in transformer! Couldn't find {} key".format(relpos_bias_table_key)
    
    # Expecting table shape: LxH
    # -> L is number of look-up table entries in table, approx ~ grid_h*grid_w*4 (for base patch grid size)
    # -> H is number of heads in transformer
    _, num_heads = state_dict[relpos_bias_table_key].shape
    
    return int(num_heads)

# .....................................................................................................................

def get_num_fusion_channels(state_dict):
    
    '''
    The state dict contains 'layer#_rn' keys, which are used to project reassembly
    feature maps down to a consistent channel count used by all fusion blocks. Since
    all fusion blocks use the same channel count, it's enough to grab a single one of these
    layers and read off the channel count from there.
    '''
    
    # Make sure the desired layer key is in the given state dict
    layer_rn_key = "scratch.layer1_rn.weight"
    assert layer_rn_key in state_dict.keys(), \
        "Error determining fusion channel count! Couldn't find {} key".format(layer_rn_key)
    
    # Expecting weights with shape: CxRx3x3
    # -> C is fusion channel count
    # -> R is reassembly layer channel count
    # -> 3x3 is for 3x3 convolution
    num_fusion_channels, _, _, _ = state_dict[layer_rn_key].shape
    
    return int(num_fusion_channels)

# .....................................................................................................................

def get_transformer_features_per_token(state_dict):
    
    '''
    The state dict is expected to contain weights for the patch embedding layer.
    This is a convolutional layer responsible for 'chopping up the image' into image patch tokens.
    The number of output channels from this convolution layer determines the number of features per token.
    '''
    
    # Make sure there is a patch embedding key in the given state dict
    patch_embed_key = "pretrained.model.patch_embed.proj.weight"
    assert patch_embed_key in state_dict.keys(), \
        "Error determining transformer features per token! Couldn't find {} key".format(patch_embed_key)
    
    # Expecting weights with shape: Fx3xPxP
    # -> F is num features per token in transformer
    # -> 3 is expected channel count of input RGB images
    # -> P is patch size in pixels
    features_per_token, _, _, _ = state_dict[patch_embed_key].shape
    
    return int(features_per_token)

# .....................................................................................................................

def get_reassembly_channel_sizes(state_dict):
    
    '''
    The state dict is expected to contain 'layer#_rn' entries, which are used to project
    reassembly feature maps into the same output channel dimension (the number of fusion channels).
    The input to each of these layers represents the reassembly feature sizes, which
    are generally different for each of the 4 reassembly stages, and the actual number we
    want to get out from this function.
    '''
    
    # For clarity, these are the layers we'll check to get the 4 reassembly feature sizes
    target_keys = {
        "scratch.layer1_rn.weight",
        "scratch.layer2_rn.weight",
        "scratch.layer3_rn.weight",
        "scratch.layer4_rn.weight",
    }
    
    # Loop over each target layer and read the weight shape to get reassembly feature sizes
    size_per_target_key = {}
    for each_target_key in target_keys:
        
        # Bail if we can't find the key we're after
        assert each_target_key in state_dict.keys(), \
            "Error determining reassembly features! Couldn't find {} key".format(each_target_key)
        
        # Expecting weights with shape: CxRx3x3
        # -> C is fusion channel count
        # -> R is reassembly layer channel count
        # -> 3x3 is for 3x3 convolution
        _, num_reassembly_channels, _, _ = state_dict[each_target_key].shape
        size_per_target_key[each_target_key] = int(num_reassembly_channels)
    
    # Make sure we get correct ordering, in case dict was indexed out-of-order
    reassembly_features_list = []
    for key in sorted(size_per_target_key.keys()):
        reassembly_features_list.append(size_per_target_key[key])
    
    return reassembly_features_list

# .....................................................................................................................

def get_patch_size_px(state_dict):
    
    '''
    The state dict is expected to contain weights for the patch embedding layer.
    This is a convolutional layer responsible for 'chopping up the image' into image patch tokens.
    The kernel size (and stride) of this convolution layer corresponds to the patch sizing (in pixels)
    that we're after. We assume the kernel is square, so patch width & height are the same.
    '''
    
    # Make sure there is a patch embedding key in the given state dict
    patch_embed_key = "pretrained.model.patch_embed.proj.weight"
    assert patch_embed_key in state_dict.keys(), \
        "Error determining patch size! Couldn't find {} key".format(patch_embed_key)

    # Expecting weights with shape: Fx3xPxP
    # -> F is num features per token in transformer
    # -> 3 is expected channel count of input RGB images
    # -> P is patch size in pixels
    _, _, _, patch_size_px = state_dict[patch_embed_key].shape
    
    return int(patch_size_px)

# .....................................................................................................................


def get_base_patch_grid_size(state_dict):
    
    '''
    The state dict is expected to contain a relative position bias 'table', which is a learned
    set of positional encodings that are added to the attention matrix calculated by each
    transformer block. This table is a 'reference' table, that gets scaled to match the inference input
    dimensions (to allow for images that are sized differently from the model's trained image size).
    The reference table has a shape of: LxH
    -> H is the number of transformer heads
    -> L is the table length. It is calculated assuming a base patch grid size as follows:
        
        L = (2*gh - 1) * (2*gw - 1) + 3
        -> Where gh, gw are the base patch grid height & width respectively
        
    In this function we try to estimate gh & gw, but assume they are square (i.e. gh = gw),
    by finding the value of L from the stored bias table and reversing the above equation to get gh, gw.
    '''
    
    # Make sure there is a relative positional encoding bias table key in the given state dict
    relpos_bias_table_key = "pretrained.model.blocks.0.attn.relative_position_bias_table"
    assert relpos_bias_table_key in state_dict.keys(), \
        "Error determining base patch grid size. Couldn't find {} key".format(relpos_bias_table_key)
    
    # Expecting table shape: LxH
    # -> L is number of look-up table entries in table
    # -> H is number of heads in transformer
    num_lut_entries, _ = state_dict[relpos_bias_table_key].shape
    
    # Calculate (square) grid size for lut count
    num_cls_lut_entries = 3
    num_token_lut_entries = num_lut_entries - num_cls_lut_entries
    num_relative_entries = torch.sqrt(torch.tensor(num_token_lut_entries))
    grid_side_length = (num_relative_entries + 1)/2
    
    # Sanity checks
    is_square_num = torch.equal(torch.round(num_relative_entries), num_relative_entries)
    is_int_length = torch.equal(torch.round(grid_side_length), grid_side_length)
    if not (is_square_num and is_int_length):
        raise ValueError("Error calculating base grid size. Got non-integer results, base grid is not square?")
    
    grid_side_length = int(grid_side_length)
    return (grid_side_length, grid_side_length)

# .....................................................................................................................
