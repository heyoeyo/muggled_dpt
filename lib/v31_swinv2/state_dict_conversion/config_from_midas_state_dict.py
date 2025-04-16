#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from math import sqrt

from .key_regex import get_nth_integer, has_prefix

from collections import defaultdict


# ---------------------------------------------------------------------------------------------------------------------
#%% Main function

def get_model_config_from_midas_state_dict(state_dict, enable_cache, enable_optimizations):
    
    # Figure out window sizing separate, since pretrained size depends on it
    window_size_wh = get_transformer_window_size_hw(state_dict)
    pretrained_window_sizes_per_stage = get_pretrained_window_sizes(window_size_wh)
    
    # Get model config from state dict
    config_dict = {
        "features_per_stage": get_features_per_stage(state_dict),
        "heads_per_stage": get_transformer_heads_per_stage(state_dict),
        "layers_per_stage": get_transformer_layers_per_stage(state_dict),
        "base_patch_grid_hw": get_base_patch_grid_size(state_dict),
        "window_size_hw": window_size_wh,
        "pretrained_window_sizes_per_stage": pretrained_window_sizes_per_stage,
        "fusion_channels": get_num_fusion_channels(state_dict),
        "patch_size_px": get_patch_size_px(state_dict),
        "enable_cache": enable_cache,
        "enable_optimizations": enable_optimizations,
    }
    
    return config_dict

# ---------------------------------------------------------------------------------------------------------------------
#%% Component functions

# .....................................................................................................................

def get_transformer_heads_per_stage(state_dict):
    
    '''
    State dict contains keys like:
        pretrained.model.layers.0.blocks.0.attn.logit_scale
        pretrained.model.layers.0.blocks.1.attn.logit_scale
        pretrained.model.layers.2.blocks.3.attn.logit_scale
        ... etc
    
    These entries are sized according to the number of heads per stage
    (here stage is indicated by the layer.#). More specifically,
    they have a size of: [h,1,1], where h is the number of heads.
    There are multiple 'logit_scale' parameters per stage, though they 
    should all be the same size.
    This function pulls out the single unique size per stage (expect to have 4!)
    '''
    
    # Record target tensor size separately for every 'layer.#' (overwriting if needed, since target key repeats)
    target_suffix = "logit_scale"
    heads_per_stage_dict = defaultdict(int)
    for key in state_dict.keys():
        if key.endswith(target_suffix):
            stage_idx = get_nth_integer(key, 0)
            num_heads = state_dict[key].shape[0]
            heads_per_stage_dict[stage_idx] = num_heads
    
    # Bundle the num_heads in order of stage indices
    stage_idx_list = sorted(heads_per_stage_dict.keys())
    heads_per_stage = [heads_per_stage_dict[stage_idx] for stage_idx in stage_idx_list]
    assert len(heads_per_stage) == 4, "Expecting 4 stages in swinv2 dpt, got: {}".format(len(heads_per_stage))
    
    return heads_per_stage

# .....................................................................................................................

def get_transformer_layers_per_stage(state_dict):
    
    '''
    State dict contains keys like:
        pretrained.model.layers.0.blocks.0.mlp.fc1.weight
        pretrained.model.layers.0.blocks.1.attn_mask
        pretrained.model.layers.2.blocks.4.attn.logit_scale
        ... etc
    What the original model refers to as 'layers' are referred to as 'stages' here,
    while in the original 'blocks' are referred to as 'layers' now.
    
    This function tries to determine how many layers there are per stage. This corresponds
    to finding the largest '...block.#' for each of the different '...layers.#' prefixes.
    '''
    
    # Record highest 'block.#' for every 'layer.#'
    target_prefix = "pretrained.model.layers.#.blocks.#"
    layers_per_stage_dict = defaultdict(int)
    for key in state_dict.keys():
        if has_prefix(key, target_prefix):
            stage_idx = get_nth_integer(key, 0)
            layer_idx = get_nth_integer(key, 1)
            layers_per_stage_dict[stage_idx] = max(layers_per_stage_dict[stage_idx], 1 + layer_idx)
    
    # Bundle the layer counts in order of stage indices
    stage_idx_list = sorted(layers_per_stage_dict.keys())
    layers_per_stage = [layers_per_stage_dict[stage_idx] for stage_idx in stage_idx_list]
    assert len(layers_per_stage) == 4, "Expecting 4 stages in swinv2 dpt, got: {}".format(len(layers_per_stage))
    
    return layers_per_stage

# .....................................................................................................................

def get_transformer_window_size_hw(state_dict):
    
    '''
    The window size in the original swinv2 models is square for all models,
    for example: 16, 24, 24 for tiny/base/large respectively.
    The size is encoded in the first-most attention mask used by the shifted-window
    attention to mask out shifted components. Specifically, the mask has a size
    of: [n, A, A] where 'n' is the number of windows and 'A' is the window area (= w*w),
    so the window size can be obtained as the square root of A.
    
    This value is shared across the entire model, so we only need to find it once,
    it is stored in keys that look like:
        pretrained.model.layers.0.blocks.1.attn_mask
    '''
    
    # Search for attention mask to get window area, in order to calculate window size
    target_suffix = "attn_mask"
    window_area = None
    for key in sorted(state_dict.keys()):
        if key.endswith(target_suffix):
            window_area = state_dict[key].shape[1]
            break
    
    # Convert area to size and tuple format, used by model
    assert window_area is not None, "Error, couldn't find attn_mask key, can't determine window size!"
    window_size = int(sqrt(window_area))
    window_size_hw = (window_size, window_size)
    
    return window_size_hw

# .....................................................................................................................

def get_base_patch_grid_size(state_dict):
    
    '''
    The base patch grid size (the size of the image after patch embedding) is encoded
    in a complex way, an can be determined from the first-most attention mask.
    The shape of the attention mask is [n, A, A], where 'n' is the number of windows
    and 'A' is the window area. The number of windows is given by the patch grid area divided
    by the window area. Or if we re-arrange for the patch grid area:
    
        patch grid area = H * W = n * A
        grid H & W = sqrt(n * A)
    
    The patch grids and window sizes are all square in the original swinv2 models, so we
    can assume H=W in these calculations.
    '''
    
    # Search for attention mask to get number of windows and window area, in order to calculate grid sizing
    target_suffix = "attn_mask"
    num_windows = None
    window_area = None
    for key in sorted(state_dict.keys()):
        if key.endswith(target_suffix):
            num_windows, window_area = state_dict[key].shape[0:2]
            break
    
    # Convert area to size and tuple format, used by model
    assert num_windows is not None, "Error, couldn't find attn_mask key, can't determine number of windows!"
    assert window_area is not None, "Error, couldn't find attn_mask key, can't determine window area!"
    patch_grid_area = num_windows * window_area
    patch_grid_side_length = int(sqrt(patch_grid_area))
    patch_grid_hw = (patch_grid_side_length, patch_grid_side_length)
    
    return patch_grid_hw

# .....................................................................................................................

def get_pretrained_window_sizes(window_size_hw):
    
    '''
    The original model has hard-coded config values referred to as
    'pretrained_window_sizes' which end up altering the scaling of the
    values of the relative positional encoding bias table. The values
    can vary across the swin stages. However, these are only used to generate
    the relative position bias table on start-up and are not part of the model weights!
    
    Therefore, this function simply tries to match the model window size
    to the pretrained sizes from the existing saved model weights. If a match
    isn't found, then the pretrained window sizes are assume to match the given
    window height sizing for all stages
    '''
    
    win_h, _ = window_size_hw
    
    pretrained_size_per_window_size_lut = {
        16: [16, 16, 16, 8],
        24: [12, 12, 12, 6],
    }
    
    # Try to look up the pretrained window sizes, based on existing model definitions
    pretrained_window_sizes = pretrained_size_per_window_size_lut.get(win_h, [None] * 4)
        
    return pretrained_window_sizes

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

def get_features_per_stage(state_dict):
    
    '''
    The state dict is expected to contain weights for the patch embedding layer.
    This is a convolutional layer responsible for 'chopping up the image' into image patch tokens.
    The number of output channels from this convolution layer determines the number of features per patch,
    which is then doubled for every stage (in the reference swinv2 model files).
    '''
    
    # Make sure there is a patch embedding key in the given state dict
    patch_embed_key = "pretrained.model.patch_embed.proj.weight"
    assert patch_embed_key in state_dict.keys(), \
        "Error determining transformer features per token! Couldn't find {} key".format(patch_embed_key)
    
    # Expecting weights with shape: Fx3xPxP
    # -> F is num features per token in transformer
    # -> 3 is expected channel count of input RGB images
    # -> P is patch size in pixels
    features_per_patch, _, _, _ = state_dict[patch_embed_key].shape
    
    # Original models double the number of features per stage after the patch embedding
    # (we could look for this in state dict, but will just hard-code it for simplicity here)
    features_per_stage = [int(features_per_patch) * (2**i) for i in range(4)]
    
    return features_per_stage

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
