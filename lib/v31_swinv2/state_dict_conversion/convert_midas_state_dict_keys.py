#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch

from .key_regex import get_nth_integer, has_prefix, replace_prefix, find_match_by_lut, get_suffix_terms


# ---------------------------------------------------------------------------------------------------------------------
#%% Main function

def convert_midas_state_dict_keys(config_dict, midas_state_dict):
    
    '''
    Function which converts the given midas state dict into the new format
    needed by beit models in this repo
    (most layer names are renamed to make the model easier to understand, some are deleted)
    
    Returns:
        new_state_dict
    
    Note: The new state dict has keys corresponding the the model components:
        "imgencoder", "reassemble", "fusion", "head"
    '''
    
    # Get sizing info needed to properly format transformer stages
    heads_per_stage = config_dict["heads_per_stage"]
    layers_per_stage = config_dict["layers_per_stage"]
    # pretrained_window_sizes_per_stage = config_dict["pretrained_window_sizes_per_stage"]
    
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
        
        new_key = _convert_imgenc_keys(orig_key, layers_per_stage)
        if found_key(new_key):
            new_data = _convert_logit_scaling_tensors(new_key, orig_data)
            new_data = _convert_qv_bias_tensors(new_key, new_data, heads_per_stage)
            imgenc_sd[new_key] = new_data
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
    In the original model weights, these layers all begin with: "pretrained.model.patch_embed"
    '''
    
    # Convert patch_embed
    patch_embed_prefix = "pretrained.model.patch_embed."
    if key.startswith(patch_embed_prefix):
        new_key = key.replace(patch_embed_prefix, "")
        return new_key
    
    return None

# .....................................................................................................................

def _convert_logit_scaling_tensors(key, key_tensor):
    
    '''
    Performs clamping & exponentiation of logit scaling factors on load.
    (these weights are the 'scaling' part of 'scaled cosine attention')
    
    The original model computes these values within every attention block
    during inference! Doing this here means we can compute once, store the
    result, and re-use it to avoid the runtime computation!
    '''
    
    is_logit_scale_key = "logit_scale" in key
    if is_logit_scale_key:
        max_logit_value = torch.log(torch.tensor(1.0 / 0.01)).to(key_tensor.device)
        new_tensor = torch.clamp(key_tensor, max=max_logit_value).exp()
        return new_tensor
    
    return key_tensor

# .....................................................................................................................

def _convert_qv_bias_tensors(key, key_tensor, heads_per_stage):
    
    '''
    Reshapes the q/v-bias tensors so that they can be added after the qkv linear layer
    The original model creates a qkv bias tensor and inserts it into the linear layer
    computation on every model run. This isn't especially slow or inefficient, but
    it's a very messy implementation.
    
    The reshaping follows the self-attention block reshaping, which requires the
    bias values to have a shape of: BxHxNxf, where f is F/H (features per head),
    however, we can have a unitary dimension in place of B & N, since the
    bias values are shared across these dimensions!
    The stored values (that we're changing) are F-length vectors.
    
    Note that all the weirdness with the qkv bias is due to the k-bias
    being all zeros! (and is presumably meant to be untrainable)
    '''
    
    is_bias_key = any((target in key) for target in ["q_bias", "v_bias"])
    if is_bias_key:
        stage_idx = get_nth_integer(key, 0)
        num_heads = heads_per_stage[stage_idx]
        new_tensor = key_tensor.reshape(1, num_heads, 1, -1)
        return new_tensor
    
    return key_tensor

# .....................................................................................................................

def _convert_imgenc_keys(key, layers_per_stage):
    
    '''
    Converts keys associated with the image encoder component of the model
    Takes care of:
        - transformer blocks (including relative positional encodings)
    '''
    
    # Group pretrained.blocks (this is the bulk of the ViT model)
    block_prefix = "pretrained.model.layers.#.blocks.#"
    if has_prefix(key, block_prefix):
        
        # Dump attention masks (they are generated at run time, not a learned parameter)
        is_unneeded_layer = ("attn_mask" in key)
        if is_unneeded_layer:
            return None
        
        # Replace layer/block indexing with stage & sequence indexing
        # ex: 'pretrained.model.layers.0.blocks.0.attn.logit_scale' -> 'stages.0.0.attn.attn.logit_scale'
        stage_idx = get_nth_integer(key, 0)
        block_idx = get_nth_integer(key, 1)
        
        # Replace prefix to match new model format
        # ex: 'pretrained.model.layers.0.blocks.0' -> 'stages.0.0'
        new_prefix = "stages.{}.blocks.{}".format(stage_idx, block_idx)
        new_key = replace_prefix(key, block_prefix, new_prefix)
        
        # Update cpb_mlp layer naming ('attn.cpb_mlp.0.weight' -> 'attn.relpos_enc.bias_mlp.0.weight')
        cpb_str = "cpb_mlp"
        if cpb_str in new_key:
            new_key = new_key.replace(cpb_str, "relpos_enc.bias_mlp")
        
        # Update mlp layer naming ('mlp.fc1.weight' -> 'mlp.layers.0.weight')
        mlp_str = "mlp.fc"
        if mlp_str in new_key:
            new_key = new_key.replace("fc1", "layers.0")
            new_key = new_key.replace("fc2", "layers.2")
        
        return new_key
    
    # Handle patch merging layers
    # ex: 'pretrained.model.layers.0.downsample.reduction.weight' -> 'patch_merge_layers.0.reduction.weight'
    patch_merge_prefix = "pretrained.model.layers.#.downsample"
    if has_prefix(key, patch_merge_prefix):
        merge_layer_idx = get_nth_integer(key, 0)
        last_2_suffix = get_suffix_terms(key, 2)
        new_key = "patch_merge_layers.{}.{}".format(merge_layer_idx, last_2_suffix)
        return new_key
    
    return None

# .....................................................................................................................

def _convert_reassembly_keys(key, weight_or_bias):
    
    '''
    Handles conversion of layers that belong to the reassembly component of the model
    In the original model weights, most of these layers all begin with: "pretrained.act_postprocess"
    There is an addition set of layers, not documented in the paper that are
    also handled by this function, which begin with: "scratch.layer#_rn"
    '''
    
    # Group pretrained.act_postprocess (this is the bulk of the reassembly section of the model)
    postproc_prefix = "pretrained.act_postprocess"
    if key.startswith(postproc_prefix):
        
        # Set up mapping between original 'act_postprocess#' to new reassembly layer names
        act_postprocess_lut = {
            "act_postprocess1": "spatial_noscale",
            "act_postprocess2": "spatial_downx2",
            "act_postprocess3": "spatial_downx4",
            "act_postprocess4": "spatial_downx8",
        }
        new_prefix = find_match_by_lut(key, act_postprocess_lut)
        
        # Set up mapping between act_postprocess sequence layers to new layer naming
        act_postprocess_seqidx_lut = {
            "0.project": "readout_proj.1",
            ".3.": "resample.0",
            ".4.": "resample.1",
        }
        new_seq_str = find_match_by_lut(key, act_postprocess_seqidx_lut)
        
        # Build final new key from prefix/sequence name and .weight/.bias components
        new_key = "{}.{}.{}".format(new_prefix, new_seq_str, weight_or_bias)
        return new_key
    
    # Group scratch.layer_rn (this is a pre-fusion step of the model, not part of documentation?)
    layer_rn_prefix = "scratch.layer#_rn"
    if has_prefix(key, layer_rn_prefix):
        
        # Set up mapping between original 'layer#_rn' to new reassembly layer names
        layer_rn_lut = {
            "layer1_rn": "spatial_noscale",
            "layer2_rn": "spatial_downx2",
            "layer3_rn": "spatial_downx4",
            "layer4_rn": "spatial_downx8",
        }
        new_prefix = find_match_by_lut(key, layer_rn_lut)
        new_key = "{}.fuse_proj.{}".format(new_prefix, weight_or_bias)
        return new_key
    
    return None

# .....................................................................................................................

def _convert_fusion_keys(key, weight_or_bias):
    
    '''
    Handles conversion of layers that belong to the fusion component of the model
    In the original model weights, these layers all begin with: "scratch.refinenet"
    '''
    
    # Group scratch.refinenet (this is the fusion section of the model)
    refinenet_prefix = "scratch.refinenet"
    if key.startswith(refinenet_prefix):
        
        # Throw out this data, since it's not actually used in model!
        # -> This is the convolution of reassembly data for the last fusion layer
        # -> The last fusion layer does not take in a prior fusion, and so doesn't end up using these weights!
        if key.startswith("scratch.refinenet4.resConfUnit1"):
            return None
        
        # Get block indexing
        block_lut = {
            "refinenet1": "blocks.0",
            "refinenet2": "blocks.1",
            "refinenet3": "blocks.2",
            "refinenet4": "blocks.3",
        }
        block_prefix = find_match_by_lut(key, block_lut)
        
        # Convert old sequence naming to newer layer naming
        assembly_component_lut = {
            "resConfUnit1": "conv_reassembly",
            "resConfUnit2": "proj_seq.0",
            "out_conv": "proj_seq.2",
        }
        new_seq_str = find_match_by_lut(key, assembly_component_lut)
        
        # Handle original 'out_conv' cases, which are structured differently from resConfUnits
        is_out_conv = (".out_conv." in key)
        if is_out_conv:
            new_key = "{}.{}.{}".format(block_prefix, new_seq_str, weight_or_bias)
            return new_key
        
        # Convert old indexing to new sequence indexing for remaining resConfUnit layers
        conv_seq_idx_lut = {
            "conv1": "conv_seq.1",
            "conv2": "conv_seq.3",
        }
        new_conv_str = find_match_by_lut(key, conv_seq_idx_lut)
        new_key = "{}.{}.{}.{}".format(block_prefix, new_seq_str, new_conv_str, weight_or_bias)
        return new_key
    
    return None

# .....................................................................................................................

def _convert_head_keys(key, weight_or_bias):
    
    '''
    Handles conversion of layers that belong to the final head component of the model
    In the original model weights, these layers all begin with: "scratch.output_conv"
    '''
    
    # Convert scratch.output_conv
    output_head_prefix = "scratch.output_conv"
    if key.startswith(output_head_prefix):
        
        # Convert old layer naming to new (more descriptive) naming
        out_conv_lut = {
            "output_conv.0": "spatial_upsampler.0",
            "output_conv.2": "proj_1ch.0",
            "output_conv.4": "proj_1ch.2",
        }
        new_prefix = find_match_by_lut(key, out_conv_lut)
        new_key = "{}.{}".format(new_prefix, weight_or_bias)
        return new_key
    
    return None

# .....................................................................................................................
