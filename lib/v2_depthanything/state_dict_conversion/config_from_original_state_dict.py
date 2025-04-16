#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch

from .key_regex import get_nth_integer


# ---------------------------------------------------------------------------------------------------------------------
#%% Main function

def get_model_config_from_state_dict(state_dict, enable_cache, enable_optimizations):
    
    # Special check for a key that WE add during loading if the model name suggests it's metric
    # -> This is needed because metric model weights are identical to normal relative model weights
    # -> This hack allows us to at least generate properly formatted metric outputs, instead of errors
    is_metric = "is_metric" in state_dict.keys()
    
    # Get feature count separate, since we need it to determine number of heads
    features_per_token = get_transformer_features_per_token(state_dict)
    num_heads = get_num_transformer_heads(features_per_token)
    
    # Get model config from state dict
    config_dict = {
        "features_per_token": features_per_token,
        "num_blocks": get_num_transformer_blocks(state_dict),
        "num_heads": num_heads,
        "reassembly_features_list": get_reassembly_channel_sizes(state_dict),
        "fusion_channels": get_num_fusion_channels(state_dict),
        "patch_size_px": get_patch_size_px(state_dict),
        "base_patch_grid_hw": get_base_patch_grid_size(state_dict),
        "is_metric": is_metric,
        "enable_cache": enable_cache,
        "enable_optimizations": enable_optimizations,
    }
    
    return config_dict


# ---------------------------------------------------------------------------------------------------------------------
#%% Component functions

def get_num_transformer_blocks(state_dict):
    
    '''
    State dict contains keys like:
        'pretrained.blocks.0.ls1.gamma'
        'pretrained.blocks.3.attn.qkv.bias',
        'pretrained.blocks.7.norm1.weight',
        ... etc
    This function tries to find the largest number from the '...blocks.#...' part of these keys,
    since this determines how many layers (aka depth) are in the transformer.
    '''
    
    # Search for all model blocks, looking for the highest index
    max_block_idx = -1
    for key in state_dict.keys():
        if "pretrained.blocks" in key:
            block_idx = get_nth_integer(key, 0)
            max_block_idx = max(max_block_idx, block_idx)
    
    # Blocks start counting at 0, so we need to add 1 to get total number of layers
    assert max_block_idx > 0, "Error determining number of transformer blocks! Could not find any blocks"
    num_transformer_blocks = 1 + max_block_idx
    
    return int(num_transformer_blocks)

# .....................................................................................................................

def get_num_transformer_heads(features_per_token):
    
    '''
    It does not seem possible to infer the number of heads directly
    from the model weights, however for all known cases, the number of
    heads is exactly 64 times smaller than the number of features per token
    For example:
        vit-small: features_per_token = 384,  num_heads = 384/64  =  6
         vit-base: features_per_token = 768,  num_heads = 768/64  = 12
        vit-large: features_per_token = 1024, num_heads = 1024/64 = 16
        vit-giant: features_per_token = 1536, num_heads = 1536/64 = 24
    '''
    
    return features_per_token // 64

# .....................................................................................................................

def get_num_fusion_channels(state_dict):
    
    '''
    The state dict contains 'layer#_rn' keys, which are used to project reassembly
    feature maps down to a consistent channel count used by all fusion blocks. Since
    all fusion blocks use the same channel count, it's enough to grab a single one of these
    layers and read off the channel count from there.
    '''
    
    # Make sure the desired layer key is in the given state dict
    layer_rn_key = "depth_head.scratch.layer1_rn.weight"
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
    patch_embed_key = "pretrained.patch_embed.proj.weight"
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
        "depth_head.scratch.layer1_rn.weight",
        "depth_head.scratch.layer2_rn.weight",
        "depth_head.scratch.layer3_rn.weight",
        "depth_head.scratch.layer4_rn.weight",
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
    patch_embed_key = "pretrained.patch_embed.proj.weight"
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
    The state dict is expected to contain a positional embedding which has a shape
    of: 1xNxF
    -> Where N is the number of patch grid tokens + 1 (to account for class token)
    -> F is the features per token
    
    The original patch grid is assumed to be square, so that we can say:
        
        N = 1 + H*W = 1 + S^2,
        -> where S is the base grid size, assuming H = W
    '''
    
    # Make sure we find a positional embedding key
    position_embed_key = "pretrained.pos_embed"
    assert position_embed_key in state_dict.keys(), \
        "Error determining base patch grid size. Couldn't find {} key".format(position_embed_key)
    
    # Expecting embedding shape: 1 x N x F
    _, num_tokens_total, _ = state_dict[position_embed_key].shape
    
    # Extract the base grid sizing
    num_cls_token = 1
    num_patch_grid_tokens = num_tokens_total - num_cls_token
    base_grid_size = int(torch.sqrt(torch.tensor(num_patch_grid_tokens)))
    
    return (base_grid_size, base_grid_size)

# .....................................................................................................................
