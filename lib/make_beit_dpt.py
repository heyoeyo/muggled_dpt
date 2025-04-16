#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from .dpt_model import DPTModel, DPTImagePrep

from .v31_beit.patch_embed import PatchEmbed
from .v31_beit.image_encoder_model import BEiTModel4Stage
from .v31_beit.reassembly_model import ReassembleModel
from .v31_beit.fusion_model import FusionModel
from .v31_beit.head_model import MonocularDepthHead

from .v31_beit.state_dict_conversion.config_from_midas_state_dict import get_model_config_from_midas_state_dict
from .v31_beit.state_dict_conversion.convert_midas_state_dict_keys import convert_midas_state_dict_keys


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def make_beit_dpt_from_midas_v31_state_dict(
        midas_v31_state_dict, enable_cache=False, enable_optimizations=True, strict_load=True
    ):
    
    '''
    Function used to initialize a BEiT DPT model from a state dictionary (i.e. model weights) file.
    This function will automatically figure out the model sizing parameters from the state dict,
    assuming it comes from the original MiDaS repo (v3.1).
    Returns:
        model_config_dict, dpt_model
    '''
    
    # Feedback on using non-strict loading
    if not strict_load:
        print("",
              "WARNING:",
              "  Loading model weights without 'strict' mode enabled!",
              "  Some weights may be missing or unused!", sep = "\n", flush = True)
    
    # Get model config from weights (i.e. beit_large_512 vs beit_base_384) & convert to new keys/state dict
    config_dict = get_model_config_from_midas_state_dict(midas_v31_state_dict, enable_cache, enable_optimizations)
    new_state_dict = convert_midas_state_dict_keys(config_dict, midas_v31_state_dict)
    
    # Load model & set model weights
    dpt_model = make_beit_dpt(**config_dict)
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
    
    The BEiT DPT model requires that the width & height of the input image
    be a multiple of 2 times the patch sizing.
    This ensures that the input patch grid size can be spatially downscaled
    by a factor of 2 (due to reassembly model) and then upscaled by a factor
    of 2 to match the input sizing (due to fusion model).
    '''
    
    # For convenience
    base_grid_h, _ = model_config_dict["base_patch_grid_hw"]
    patch_size_px = model_config_dict["patch_size_px"]
    
    # Figure out input image sizing requirements
    base_image_size = round(base_grid_h * patch_size_px)
    to_multiples = 2 * patch_size_px
    
    # Set hard-coded mean/std normalization
    rgb_mean = (0.5, 0.5, 0.5)
    rgb_std = (0.5, 0.5, 0.5)
    
    return DPTImagePrep(base_image_size, patch_size_px, to_multiples, rgb_mean, rgb_std)

# .....................................................................................................................

def make_beit_dpt(features_per_token, num_heads, num_blocks, reassembly_features_list, base_patch_grid_hw,
                  fusion_channels = 256, patch_size_px = 16, enable_cache = False, **unused_kwargs):
    
    '''
    Helper used to build all BEiT DPT components. The arguments for this function are
    expected to come from the 'make_beit_dpt_from_midas_v31_state_dict' function, which
    will use arguments based on a loaded state dictionary.
    
    However, if you want to make a model without pretrained weights
    here are the following standard configs (from timm library):
    
    # beit_large_512:
        features_per_token = 1024
        num_heads = 16
        num_blocks = 24
        reassembly_features_list = [256, 512, 1024, 1024]
        base_patch_grid_hw = (32, 32)
        fusion_channels = 256
        patch_size_px = 16
    
    # beit_large_384
        features_per_token = 1024
        num_heads = 16
        num_blocks = 24
        reassembly_features_list = [256, 512, 1024, 1024]
        base_patch_grid_hw = (24, 24)
        fusion_channels = 256
        patch_size_px = 16
    
    # beit_base_384
        features_per_token = 768
        num_heads = 12
        num_blocks = 12
        reassembly_features_list = [96, 192, 384, 768]
        base_patch_grid_hw = (24, 24)
        fusion_channels = 256
        patch_size_px = 16
    '''
    
    # Construct model components
    patch_embed_model = PatchEmbed(features_per_token, patch_size_px)
    imgenc_model = BEiTModel4Stage(features_per_token, num_heads, num_blocks, base_patch_grid_hw, enable_cache)
    reassembly_model = ReassembleModel(features_per_token, reassembly_features_list, fusion_channels)
    fusion_model = FusionModel(fusion_channels)
    head_model = MonocularDepthHead(fusion_channels)
    
    # Build combined DPT model!
    dpt_model = DPTModel(patch_embed_model, imgenc_model, reassembly_model, fusion_model, head_model)
    
    return dpt_model

# .....................................................................................................................
