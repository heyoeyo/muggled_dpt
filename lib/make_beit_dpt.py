#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch

from .dpt_model import MuggledDPT

from .v31_beit.image_prepost_processor import DPTImageProcessor
from .v31_beit.image_encoder_model import BEiTModel4Stage
from .v31_beit.reassembly_model import ReassembleModel
from .v31_beit.fusion_model import FusionModel
from .v31_beit.head_model import MonocularDepthHead

from .v31_beit.state_dict_conversion.config_from_midas_state_dict import get_model_config_from_midas_state_dict
from .v31_beit.state_dict_conversion.convert_midas_state_dict_keys import convert_midas_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def make_beit_dpt_from_midas_v31(path_to_midas_v31_weights, enable_relpos_cache = False):
    
    # Get model config from weights (i.e. beit_large_512 vs beit_base_384) & convert to new keys/state dict
    midas_state_dict = torch.load(path_to_midas_v31_weights)
    config_dict = get_model_config_from_midas_state_dict(midas_state_dict)
    new_state_dict = convert_midas_state_dict_keys(config_dict, midas_state_dict)
    
    # Load model & set model weights
    dpt_model = make_beit_dpt(**config_dict, enable_relpos_cache = enable_relpos_cache)
    dpt_model.imgencoder.load_state_dict(new_state_dict["imgencoder"])
    dpt_model.reassemble.load_state_dict(new_state_dict["reassemble"])
    dpt_model.fusion.load_state_dict(new_state_dict["fusion"])
    dpt_model.head.load_state_dict(new_state_dict["head"])
    
    return config_dict, dpt_model

# .....................................................................................................................

def make_opencv_image_prepost_processor(model_config_dict):
    
    # Calculate the model 'base image size' using the config info
    base_grid_h, _ = model_config_dict["base_patch_grid_hw"]
    patch_size_px = model_config_dict["patch_size_px"]
    base_image_size = int(base_grid_h * patch_size_px)
    
    return DPTImageProcessor(base_image_size, 2*patch_size_px)

# .....................................................................................................................

def make_beit_dpt(features_per_token, num_heads, num_blocks, reassembly_features_list, base_patch_grid_hw,
                  patch_size_px = 16, fusion_channels = 256, enable_relpos_cache = False):
    
    # Construct model components
    imgenc_model = \
        BEiTModel4Stage(features_per_token, num_heads, num_blocks, patch_size_px, base_patch_grid_hw, enable_relpos_cache)
    reassembly_model = ReassembleModel(features_per_token, reassembly_features_list, fusion_channels)
    fusion_model = FusionModel(fusion_channels)
    head_model = MonocularDepthHead(fusion_channels)
    
    # Build combined DPT model!
    dpt_model = MuggledDPT(imgenc_model, reassembly_model, fusion_model, head_model)
    
    return dpt_model

# .....................................................................................................................

# def make_beit_large_512(enable_relpos_cache = False):
    
#     # Hard-coded configuration of model (from timm library)
#     base_image_size = 512
#     features_per_token = 1024
#     num_heads = 16
#     num_layers = 24
    
#     # DPT-specific configuration
#     reassembly_features_list = [256, 512, 1024, 1024]
    
#     return _make_beit(base_image_size, features_per_token, num_heads, num_layers,
#                       reassembly_features_list, enable_relpos_cache)

# .....................................................................................................................

# def make_beit_large_384(enable_relpos_cache = False):
    
#     # Hard-coded configuration of model (from timm library)
#     base_image_size = 384
#     features_per_token = 1024
#     num_heads = 16
#     num_layers = 24
    
#     # DPT-specific configuration
#     reassembly_features_list = [256, 512, 1024, 1024]
    
#     return _make_beit(base_image_size, features_per_token, num_heads, num_layers,
#                       reassembly_features_list, enable_relpos_cache)

# .....................................................................................................................

# def make_beit_base_384(enable_relpos_cache = False):
    
#     # Hard-coded configuration of model (from timm library)
#     base_image_size = 384
#     features_per_token = 768
#     num_heads = 12
#     num_layers = 12
    
#     # DPT-specific configuration
#     reassembly_features_list = [96, 192, 384, 768]
    
#     return _make_beit(base_image_size, features_per_token, num_heads, num_layers,
#                       reassembly_features_list, enable_relpos_cache)

# ---------------------------------------------------------------------------------------------------------------------
#%%

