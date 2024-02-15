#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from .key_regex import get_nth_integer, has_prefix, find_match_by_lut, get_suffix_terms


# ---------------------------------------------------------------------------------------------------------------------
#%% Main function

def convert_state_dict_keys(config_dict, midas_state_dict):
    
    '''
    Function which converts the given Depth-Anything state dict into the new format
    needed by the model implementation in this repo
    (most layer names are renamed to make the model easier to understand, some are deleted)
    
    Returns:
        new_state_dict
    
    Note: The new state dict has keys corresponding the the model components:
        "patch_embed", "imgencoder", "reassemble", "fusion", "head"
    '''
    
    # Allocate storage for state dict of each (new) model component
    patch_embed_sd = {}
    imgenc_sd = {}
    reassemble_sd = {}
    fusion_sd = {}
    head_sd = {}
    
    # Key conversion functions output None when no matching key was found
    found_key = lambda key: key is not None
    
    # Loop over all midas state dict keys and convert them to new formatting
    for orig_key, orig_data in midas_state_dict.items():
        
        # For convenience, get the last term in the layer name (usually 'weight' or 'bias')
        orig_key = str(orig_key)
        weight_or_bias = get_suffix_terms(orig_key, 1)
        
        new_key = _convert_patch_embed_keys(orig_key, weight_or_bias)
        if found_key(new_key):
            patch_embed_sd[new_key] = orig_data
            continue
        
        new_key = _convert_imgenc_keys(orig_key)
        if found_key(new_key):
            imgenc_sd[new_key] = orig_data
            continue
        
        new_key = _convert_reassembly_keys(orig_key, weight_or_bias)
        if found_key(new_key):
            reassemble_sd[new_key] = orig_data
            continue
        
        new_key = _convert_fusion_keys(orig_key, weight_or_bias)
        if found_key(new_key):
            fusion_sd[new_key] = orig_data
            continue
        
        new_key = _convert_head_keys(orig_key, weight_or_bias)
        if found_key(new_key):
            head_sd[new_key] = orig_data
            continue
    
    # Special adjustment, used to split position encoding tensor into cls token embedding + patch embeddings
    imgenc_sd = _split_posenc_tensor(imgenc_sd)
    
    # Bundle new state dict model components together for easier handling
    new_state_dict = {
        "patch_embed": patch_embed_sd,
        "imgencoder": imgenc_sd,
        "reassemble": reassemble_sd,
        "fusion": fusion_sd,
        "head": head_sd,
    }
    
    return new_state_dict


# ---------------------------------------------------------------------------------------------------------------------
#%% Component functions

# .....................................................................................................................

def _convert_patch_embed_keys(key, weight_or_bias):
    
    '''
    Handles conversion of layers that belong to the patch embedding component of the model
    In the original model weights, these layers all begin with: "pretrained.patch_embed"
    '''
    
    # Convert patch_embed
    patch_embed_prefix = "pretrained.patch_embed"
    if key.startswith(patch_embed_prefix):
        new_key = "proj.{}".format(weight_or_bias)
        return new_key
    
    return None

# .....................................................................................................................

def _convert_imgenc_keys(key):
    
    '''
    Converts keys associated with the image encoder component of the model
    Takes care of:
        - cls token
        - position embeddings (gets flagged for removal/splitting)
        - transformer block components
    '''
    
    # Convert cls token
    if key == "pretrained.cls_token":
        return "cls_token"
    
    # Pass this along, since we'll deal with it elsewhere
    if key == "pretrained.pos_embed":
        return key
    
    # Skip mask token (not used)
    if key == "pretrained.mask_token":
        return None
    
    # Convert output norm layer ('pretrained.norm.weight' -> 'outnorm.weight')
    norm_str = "pretrained.norm"
    if key.startswith(norm_str):
        return key.replace(norm_str, "outnorm")
    
    # Group pretrained.blocks (this is the bulk of the ViT model)
    block_prefix = "pretrained.blocks.#"
    if has_prefix(key, block_prefix):
        
        # Remove pretrained prefix
        new_key = key.replace("pretrained.", "")
        
        # Update mlp layer naming ('mlp.fc1.weight' -> 'mlp.layers.0.weight')
        mlp_str = "mlp.fc"
        if mlp_str in new_key:
            new_key = new_key.replace("fc1", "layers.0")
            new_key = new_key.replace("fc2", "layers.2")
        
        # Update layer scale naming ('ls1.gamma' -> 'scale_attn')
        gamma_str = "gamma"
        if gamma_str in new_key:
            new_key = new_key.replace("ls1.gamma", "scale_attn")
            new_key = new_key.replace("ls2.gamma", "scale_mlp")
        
        return new_key
    
    return None

# .....................................................................................................................

def _convert_reassembly_keys(key, weight_or_bias):
    
    '''
    Handles conversion of layers that belong to the reassembly component of the model
    In the original weights, these layers are split among
    entries starting with either "depth_head.projects" or "depth_head.resize_layers"
    
    There is an addition set of layers, not documented in the paper that are
    also handled by this function, which begin with: "depth_head.scratch.layer#_rn"
    '''
    
    stage_to_name_list = ["spatial_upx4", "spatial_upx2", "spatial_noscale", "spatial_downx2"]
    
    projection_prefix = "depth_head.projects"
    if key.startswith(projection_prefix):
        stage_idx = get_nth_integer(key, 0)
        new_prefix = stage_to_name_list[stage_idx]
        new_key = "{}.resample.0.{}".format(new_prefix, weight_or_bias)
        return new_key
    
    resize_prefix = "depth_head.resize_layers"
    if key.startswith(resize_prefix):
        stage_idx = get_nth_integer(key, 0)
        new_prefix = stage_to_name_list[stage_idx]
        new_key = "{}.resample.1.{}".format(new_prefix, weight_or_bias)
        return new_key
    
    # Group scratch.layer_rn (this is a pre-fusion step of the model, not part of documentation?)
    layer_rn_prefix = "depth_head.scratch.layer#_rn"
    if has_prefix(key, layer_rn_prefix):
        stage_idx = get_nth_integer(key, 0) - 1
        new_prefix = stage_to_name_list[stage_idx]
        new_key = "{}.fuse_proj.{}".format(new_prefix, weight_or_bias)
        return new_key
    
    return None

# .....................................................................................................................

def _convert_fusion_keys(key, weight_or_bias):
    
    '''
    Handles conversion of layers that belong to the fusion component of the model
    In the original model weights, these layers all begin with: "depth_head.scratch.refinenet"
    '''
    
    # Group scratch.refinenet (this is the fusion section of the model)
    refinenet_prefix = "depth_head.scratch.refinenet"
    if key.startswith(refinenet_prefix):
        
        # Throw out this data, since it's not actually used in model!
        # -> This is the convolution of reassembly data for the last fusion layer
        # -> The last fusion layer does not take in a prior fusion, and so doesn't end up using these weights!
        if key.startswith("depth_head.scratch.refinenet4.resConfUnit1"):
            return None
        
        # Get block indexing (e.g. 'refinenet1' -> 'blocks.0')
        refinenet_idx = get_nth_integer(key)
        block_idx = refinenet_idx - 1
        block_prefix = "blocks.{}".format(block_idx)
        
        # Convert old sequence naming to newer layer naming
        assembly_component_lut = {
            "resConfUnit1": "conv_reassembly",
            "resConfUnit2": "scale_proj_seq.0",
            "out_conv": "scale_proj_seq.2",
        }
        new_seq_str = find_match_by_lut(key, assembly_component_lut)
        
        # Handle original 'out_conv' cases, which are structured differently from resConfUnits
        is_out_conv = (".out_conv." in key)
        if is_out_conv:
            new_key = "{}.{}.{}".format(block_prefix, new_seq_str, weight_or_bias)
            return new_key
        
        # Convert old indexing to new sequence indexing for remaining resConfUnit layers
        resconv_seq_idx_lut = {
            "conv1": "resconv_seq.1",
            "conv2": "resconv_seq.3",
        }
        new_conv_str = find_match_by_lut(key, resconv_seq_idx_lut)
        new_key = "{}.{}.{}.{}".format(block_prefix, new_seq_str, new_conv_str, weight_or_bias)
        return new_key
    
    return None

# .....................................................................................................................

def _convert_head_keys(key, weight_or_bias):
    
    '''
    Handles conversion of layers that belong to the final head component of the model
    In the original model weights, these layers all begin with: "depth_head.scratch.output_conv"
    '''
    
    # Convert scratch.output_conv
    output_head_prefix = "depth_head.scratch.output_conv"
    if key.startswith(output_head_prefix):
        
        # Convert old layer naming to new (more descriptive) naming
        out_conv_lut = {
            "output_conv1": "spatial_upsampler.0",
            "output_conv2.0": "proj_1ch.0",
            "output_conv2.2": "proj_1ch.2",
        }
        new_prefix = find_match_by_lut(key, out_conv_lut)
        new_key = "{}.{}".format(new_prefix, weight_or_bias)
        return new_key
    
    return None

# .....................................................................................................................

def _split_posenc_tensor(state_dict):
    
    '''
    Special handler used to split the original (single) position embedding into
    two tensors, one for the cls token embedding and the other for image patch embeddings.
    
    It is useful to distinguish between the two embeddings because the image patch embeddings
    can be resized at runtime and need to be handled separately from the cls embedding.
    It also helps for readability.
    '''
    
    # Remove target tensor data from state dict
    target_key = "pretrained.pos_embed"
    orig_tensor = state_dict.pop(target_key)
    
    # Split tensor into component parts
    cls_embedding = orig_tensor[:, :1, :]
    base_patch_embedding = orig_tensor[:, 1:, :]
    
    # Re-add split data back into state dict!
    state_dict["posenc.cls_embedding"] = cls_embedding
    state_dict["posenc.base_patch_embedding"] = base_patch_embedding
    
    return state_dict

# .....................................................................................................................