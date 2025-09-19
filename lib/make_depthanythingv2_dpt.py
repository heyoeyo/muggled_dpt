#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from .dpt_model import DPTModel, DPTImagePrep

from .v2_depthanything.patch_embed import PatchEmbed
from .v2_depthanything.image_encoder_model import DinoV2Model4Stages
from .v2_depthanything.reassembly_model import ReassembleModel
from .v2_depthanything.fusion_model import FusionModel
from .v2_depthanything.head_model import MonocularDepthHead

from .v2_depthanything.state_dict_conversion.config_from_original_state_dict import get_model_config_from_state_dict
from .v2_depthanything.state_dict_conversion.convert_original_state_dict_keys import convert_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def make_depthanythingv2_dpt_from_original_state_dict(
        state_dict, enable_cache=False, enable_optimizations=True, strict_load=True
    ):
    
    '''
    Function used to initialize a Depth-Anything V2 DPT model from a state dictionary (i.e. model weights) file.
    This function will automatically figure out the model sizing parameters from the state dict,
    assuming it comes from the original Depth-AnythingV2 repo.
    Returns:
        model_config_dict, dpt_model
    '''
    
    # Feedback on using non-strict loading
    if not strict_load:
        print("",
              "WARNING:",
              "  Loading model weights without 'strict' mode enabled!",
              "  Some weights may be missing or unused!", sep = "\n", flush = True)
    
    # Get model config from weights (i.e. vit-small vs vit-large) & convert to new keys/state dict
    config_dict = get_model_config_from_state_dict(state_dict, enable_cache, enable_optimizations)
    new_state_dict = convert_state_dict_keys(config_dict, state_dict)
    
    # Load model & set model weights
    dpt_model = make_depthanythingv2_dpt(**config_dict)
    dpt_model.patch_embed.load_state_dict(new_state_dict["patch_embed"], strict_load)
    dpt_model.imgencoder.load_state_dict(new_state_dict["imgencoder"], strict_load)
    dpt_model.reassemble.load_state_dict(new_state_dict["reassemble"], strict_load)
    dpt_model.fusion.load_state_dict(new_state_dict["fusion"], strict_load)
    dpt_model.head.load_state_dict(new_state_dict["head"], strict_load)
    
    return config_dict, dpt_model

# .....................................................................................................................

def make_opencv_image_prepost_processor(model_config_dict):
    
    '''
    Helper used to set up an image pre-processor for the DPT model.
    The preprocessor is used to make sure input images are sized correctly.
    
    The Depth-Anything DPT model (in this repo implementation) requires
    that the width & height of the input image be a multiple of 2 times
    the patch sizing. This ensures that the input patch grid size can be
    spatially downscaled by a factor of 2 (due to reassembly model) and
    then re-upscaled by a factor of 2 (due to fusion model).
    '''
    
    # For convenience
    base_grid_h, _ = model_config_dict["base_patch_grid_hw"]
    patch_size_px = model_config_dict["patch_size_px"]
    
    # Figure out input image sizing requirements
    base_image_size = round(base_grid_h * patch_size_px)
    to_multiples = 2 * patch_size_px
    
    # Set hard-coded mean/std normalization
    rgb_mean = (0.485, 0.456, 0.406)
    rgb_std = (0.229, 0.224, 0.225)
    
    return DPTImagePrep(base_image_size, patch_size_px, to_multiples, rgb_mean, rgb_std)

# .....................................................................................................................

def make_depthanythingv2_dpt(
        features_per_token,
        num_heads,
        num_blocks,
        reassembly_features_list,
        base_patch_grid_hw,
        fusion_channels=256,
        patch_size_px=14,
        is_giant=False,
        is_metric=False,
        enable_cache=False,
        enable_optimizations=True,
    ):
    
    '''
    Helper used to build all Depth-Anything DPT components. The arguments for this function are
    expected to come from the 'make_depthanything_dpt_from_original_state_dict' function, which
    will use arguments based on a loaded state dictionary.
    
    However, if you want to make a model without pretrained weights
    here are the following standard configs (from Depth-Anything/DinoV2):
    
    # vit-large:
        features_per_token = 1024
        num_heads = 16
        num_blocks = 24
        reassembly_features_list = [256, 512, 1024, 1024]
        base_patch_grid_hw = (37, 37)
        fusion_channels = 256
        patch_size_px = 14
    
    # vit-base
        features_per_token = 768
        num_heads = 12
        num_blocks = 12
        reassembly_features_list = [96, 192, 384, 768]
        base_patch_grid_hw = (37, 37)
        fusion_channels = 128
        patch_size_px = 14
    
    # vit-small
        features_per_token = 384
        num_heads = 6
        num_blocks = 12
        reassembly_features_list = [48, 96, 192, 384]
        base_patch_grid_hw = (37, 37)
        fusion_channels = 64
        patch_size_px = 14
    '''
    
    # Construct model components
    patch_embed_model = PatchEmbed(features_per_token, patch_size_px)
    imgenc_model = DinoV2Model4Stages(
        features_per_token, num_heads, num_blocks, base_patch_grid_hw, is_giant, enable_cache, enable_optimizations
    )
    reassembly_model = ReassembleModel(features_per_token, reassembly_features_list, fusion_channels)
    fusion_model = FusionModel(fusion_channels)
    head_model = MonocularDepthHead(fusion_channels, patch_size_px, is_metric)
    
    # Build combined DPT model!
    dpt_model = DPTModel(patch_embed_model, imgenc_model, reassembly_model, fusion_model, head_model)
    
    return dpt_model

# .....................................................................................................................
