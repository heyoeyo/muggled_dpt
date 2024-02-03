#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch

from .make_swinv2_dpt import (
    make_swinv2_dpt_from_midas_v31_state_dict as make_dpt_swinv2,
    make_opencv_image_prepost_processor as make_imgprep_swinv2,
)

from .make_beit_dpt import (
    make_beit_dpt_from_midas_v31_state_dict as make_dpt_beit,
    make_opencv_image_prepost_processor as make_imgprep_beit,
)


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def make_dpt_from_midas_v31(path_to_midas_v31_weights, enable_relpos_cache = False, strict_load = True):
    
    # Load model weights with fail check in case weights are in cuda format and user doesn't have cuda
    try:
        midas_state_dict = torch.load(path_to_midas_v31_weights)
    except RuntimeError:
        midas_state_dict = torch.load(path_to_midas_v31_weights, map_location="cpu")
    
    # For clarity, define all the known models (and the appropriate creation function mappings)
    model_type_to_funcs_lut = {
        "swinv2": (make_dpt_swinv2, make_imgprep_swinv2),
        "beit": (make_dpt_beit, make_imgprep_beit),
    }
    
    # Try to figure out which type of model we're creating from state dict keys (e.g. beit vs swinv2)
    guessed_model_type = determine_model_type_from_state_dict(midas_state_dict)
    if guessed_model_type not in model_type_to_funcs_lut.keys():
        raise NotImplementedError("Bad model type: {}, no support for this yet!".format(guessed_model_type))
    
    # Build the model & supporting data
    make_func, imgprep_func = model_type_to_funcs_lut[guessed_model_type]
    config_dict, dpt_model = make_func(midas_state_dict, enable_relpos_cache, strict_load)
    imgprep = imgprep_func(config_dict)
    
    return config_dict, dpt_model, imgprep

# .....................................................................................................................

def determine_model_type_from_state_dict(state_dict):
    
    '''
    Helper used to figure out which model type (e.g. swinv2 vs. beit) we're working with,
    given a state dict (e.g. model weights). This works by looking for (hard-coded) keys
    that are expected to be unique among each of the model's state dicts
    '''
    
    # Initialize output
    guessed_model_type = None
    sd_keys = state_dict.keys()
    
    swinv2_target_key = "pretrained.model.layers.0.blocks.0.attn.logit_scale"
    if swinv2_target_key in sd_keys:
        return "swinv2"
    
    beit_target_key = "pretrained.model.blocks.0.attn.relative_position_bias_table"
    if beit_target_key in sd_keys:
        return "beit"
    
    depth_anything_target_key = "pretrained.blocks.0.ls1.gamma"
    if depth_anything_target_key in sd_keys:
        return "depth_anything"
    
    return guessed_model_type