#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from .dpt_model import DPTModel

from .v31_swinv2.components.image_prep import DPTImagePrep
from .v31_swinv2.patch_embed import PatchEmbed
from .v31_swinv2.image_encoder_model import SwinV2Model4Stages
from .v31_swinv2.reassembly_model import ReassembleModel
from .v31_swinv2.fusion_model import FusionModel
from .v31_swinv2.head_model import MonocularDepthHead

from .v31_swinv2.state_dict_conversion.config_from_midas_state_dict import get_model_config_from_midas_state_dict
from .v31_swinv2.state_dict_conversion.convert_midas_state_dict_keys import convert_midas_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def make_swinv2_dpt_from_midas_v31_state_dict(midas_v31_state_dict, enable_relpos_cache = False, strict_load = True):
    
    # Feedback on using non-strict loading
    if not strict_load:
        print("",
              "WARNING:",
              "  Loading model weights without 'strict' mode enabled!",
              "  Some weights may be missing or unused!", sep = "\n", flush = True)
    
    # Get model config from weights (i.e. swinv2_large_512 vs swinv2_base_384) & convert to new keys/state dict
    config_dict = get_model_config_from_midas_state_dict(midas_v31_state_dict)
    new_state_dict = convert_midas_state_dict_keys(config_dict, midas_v31_state_dict)
    
    # Load model & set model weights
    dpt_model = make_swinv2_dpt(**config_dict, enable_relpos_cache = enable_relpos_cache)
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

def make_swinv2_dpt(features_per_patch, heads_per_stage, layers_per_stage,
                    base_patch_grid_hw, window_size_hw, pretrained_window_sizes_per_stage,
                    reassembly_features_list, patch_size_px = 4, fusion_channels = 256, enable_relpos_cache = False):
    
    num_stages = len(heads_per_stage)
    features_per_stage = [features_per_patch // (2**stage_idx) for stage_idx in range(num_stages)]
    
    # Construct model components
    patch_embed_model = PatchEmbed(features_per_patch, patch_size_px)
    imgenc_model = SwinV2Model4Stages(features_per_patch, heads_per_stage, layers_per_stage, base_patch_grid_hw, window_size_hw, pretrained_window_sizes_per_stage, enable_relpos_cache)
    reassembly_model = ReassembleModel(features_per_stage, reassembly_features_list, fusion_channels)
    fusion_model = FusionModel(fusion_channels)
    head_model = MonocularDepthHead(fusion_channels)
    
    # Build combined DPT model!
    dpt_model = DPTModel(patch_embed_model, imgenc_model, reassembly_model, fusion_model, head_model)
    
    return dpt_model

# .....................................................................................................................

