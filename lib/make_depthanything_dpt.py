#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from .dpt_model import DPTModel

from .v1_depthanything.components.image_prep import DPTImagePrep
from .v1_depthanything.patch_embed import PatchEmbed
from .v1_depthanything.image_encoder_model import DinoV2Model4Stages
from .v1_depthanything.reassembly_model import ReassembleModel
from .v1_depthanything.fusion_model import FusionModel
from .v1_depthanything.head_model import MonocularDepthHead

from .v1_depthanything.state_dict_conversion.config_from_original_state_dict import get_model_config_from_state_dict
from .v1_depthanything.state_dict_conversion.convert_original_state_dict_keys import convert_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def make_depthanything_dpt_from_original_state_dict(state_dict, enable_cache = False, strict_load = True):
    
    # Feedback on using non-strict loading
    if not strict_load:
        print("",
              "WARNING:",
              "  Loading model weights without 'strict' mode enabled!",
              "  Some weights may be missing or unused!", sep = "\n", flush = True)
    
    # Get model config from weights (i.e. vit-small vs vit-large) & convert to new keys/state dict
    config_dict = get_model_config_from_state_dict(state_dict)
    new_state_dict = convert_state_dict_keys(config_dict, state_dict)
    
    # Load model & set model weights
    dpt_model = make_depthanything_dpt(**config_dict, enable_cache = enable_cache)
    dpt_model.patch_embed.load_state_dict(new_state_dict["patch_embed"], strict_load)
    dpt_model.imgencoder.load_state_dict(new_state_dict["imgencoder"], strict_load)
    dpt_model.reassemble.load_state_dict(new_state_dict["reassemble"], strict_load)
    dpt_model.fusion.load_state_dict(new_state_dict["fusion"], strict_load)
    dpt_model.head.load_state_dict(new_state_dict["head"], strict_load)
    
    return config_dict, dpt_model

# .....................................................................................................................

def make_opencv_image_prepost_processor(model_config_dict):
    
    # Calculate the model 'base image size' using the config info
    base_grid_h, _ = model_config_dict["base_patch_grid_hw"]
    patch_size_px = model_config_dict["patch_size_px"]
    base_image_size = int(base_grid_h * patch_size_px)
    
    return DPTImagePrep(base_image_size, patch_size_px)

# .....................................................................................................................

def make_depthanything_dpt(features_per_token, num_heads, num_blocks, reassembly_features_list, base_patch_grid_hw,
                           patch_size_px = 14, fusion_channels = 256, enable_cache = False):
    
    # Construct model components
    patch_embed_model = PatchEmbed(features_per_token, patch_size_px)
    imgenc_model = DinoV2Model4Stages(features_per_token, num_heads, num_blocks, base_patch_grid_hw, enable_cache)
    reassembly_model = ReassembleModel(features_per_token, reassembly_features_list, fusion_channels)
    fusion_model = FusionModel(fusion_channels)
    head_model = MonocularDepthHead(fusion_channels, patch_size_px)
    
    # Build combined DPT model!
    dpt_model = DPTModel(patch_embed_model, imgenc_model, reassembly_model, fusion_model, head_model)
    
    return dpt_model

# .....................................................................................................................

