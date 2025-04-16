#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os.path as osp
import torch

from time import sleep


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def make_dpt_from_state_dict(
        path_to_state_dict, enable_cache=False, enable_optimizations=True, strict_load=True, model_type=None
    ):
    
    # Load model weights with fail check in case weights are in cuda format and user doesn't have cuda
    try:
        state_dict = torch.load(path_to_state_dict)
    except RuntimeError:
        state_dict = torch.load(path_to_state_dict, map_location="cpu")
    
    # Try to figure out which type of model we're creating from state dict keys (e.g. beit vs swinv2)
    if model_type is None:
        model_type = determine_model_type_from_state_dict(path_to_state_dict, state_dict)
    
    # Error out if we don't understand the model type
    known_model_types = ["swinv2", "beit", "depthanythingv1", "depthanythingv2"]
    if model_type not in known_model_types:
        print("Accepted model types:", *known_model_types, sep = "\n")
        raise NotImplementedError(f"Bad model type: {model_type}, no support for this yet!")
    
    # Special hack! If we get a depth-anything v2 model with the word 'metric' in it's name
    # add an extra key to the state dict to indicate it's a metric model
    # (the metric model weights are otherwise indistinguishable from the relative models!)
    if model_type == "depthanythingv2" and "metric" in path_to_state_dict:
        state_dict["is_metric"] = torch.tensor((1), dtype=torch.float32)
        print(
            "",
            "Warning: Metric Depth-Anything V2 model detected!",
            "  These models are not officially supported,",
            "  model outputs may be incorrect...",
            sep="\n",
            flush=True,
        )
        sleep(1.5)
    
    # Build the model & supporting data
    make_dpt_func, make_imgprep_func = import_model_functions(model_type)
    config_dict, dpt_model = make_dpt_func(state_dict, enable_cache, enable_optimizations, strict_load)
    imgprep = make_imgprep_func(config_dict)
    
    return config_dict, dpt_model, imgprep

# .....................................................................................................................

def determine_model_type_from_state_dict(model_path, state_dict):
    
    '''
    Helper used to figure out which model type (e.g. swinv2 vs. beit) we're working with,
    given a state dict (e.g. model weights). This works by looking for (hard-coded) keys
    that are expected to be unique among each of the model's state dicts
    '''
    
    sd_keys = state_dict.keys()
    
    swinv2_target_key = "pretrained.model.layers.0.blocks.0.attn.logit_scale"
    if swinv2_target_key in sd_keys:
        return "swinv2"
    
    beit_target_key = "pretrained.model.blocks.0.attn.relative_position_bias_table"
    if beit_target_key in sd_keys:
        return "beit"
    
    depth_anything_target_key = "pretrained.blocks.0.ls1.gamma"
    if depth_anything_target_key in sd_keys:
        
        # Guess at depth-anything versino from file name
        model_name = osp.basename(model_path).lower()
        is_v2 = "v2" in model_name
        is_v1 = (not is_v2) and (("anything_vit" in model_name) or ("v1" in model_name))
        
        # Assume v2 by default, but switch to v1 if needed
        depthanything_type = "depthanythingv1" if is_v1 else "depthanythingv2"
        if (not is_v1) and (not is_v2):
            print("",
                  "WARNING: Unable to determine DepthAnything model version!",
                  "-> Will assume v2",
                  "-> Will use v1 if the file name contains 'v1'",
                  sep="\n")
        
        return depthanything_type
    
    return "unknown"

# .....................................................................................................................

def import_model_functions(model_type):
    
    '''
    Function used to import the 'make dpt' and 'make image prep' functions for
    all known model types. This is a hacky-ish thing to do, but helps avoid
    importing all model code even though we're only loading one model.
    '''
    
    if model_type == "swinv2":
        from .make_swinv2_dpt import (
            make_swinv2_dpt_from_midas_v31_state_dict as make_dpt_func,
            make_opencv_image_prepost_processor as make_imgprep_func,
        )
        
    elif model_type == "beit":
        from .make_beit_dpt import (
            make_beit_dpt_from_midas_v31_state_dict as make_dpt_func,
            make_opencv_image_prepost_processor as make_imgprep_func,
        )
        
    elif model_type == "depthanythingv1":
        from .make_depthanythingv1_dpt import (
            make_depthanythingv1_dpt_from_original_state_dict as make_dpt_func,
            make_opencv_image_prepost_processor as make_imgprep_func,
        )
        
    elif model_type == "depthanythingv2":
        from .make_depthanythingv2_dpt import (
            make_depthanythingv2_dpt_from_original_state_dict as make_dpt_func,
            make_opencv_image_prepost_processor as make_imgprep_func,
        )
        
    else:
        raise TypeError(f"Cannot import model functions, Unknown model type: {model_type}")
    
    return make_dpt_func, make_imgprep_func
