#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

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
    num_stages = len(config_dict["reassembly_features_list"])
    blocks_per_stage = config_dict["num_blocks"] // num_stages
    num_heads = config_dict["num_heads"]
    
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
        
        new_key = _convert_imgenc_keys(orig_key, blocks_per_stage)
        if found_key(new_key):
            imgenc_sd[new_key] = _convert_qv_bias_tensors(new_key, orig_data, num_heads)
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
    patch_embed_prefix = "pretrained.model.patch_embed"
    if key.startswith(patch_embed_prefix):
        new_key = "proj.{}".format(weight_or_bias)
        return new_key
    
    return None

# .....................................................................................................................

def _convert_qv_bias_tensors(key, key_tensor, num_heads):
    
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
        new_tensor = key_tensor.reshape(1, num_heads, 1, -1)
        return new_tensor
    
    return key_tensor

# .....................................................................................................................

def _convert_imgenc_keys(key, blocks_per_stage):
    
    '''
    Converts keys associated with the image encoder component of the model
    Takes care of:
        - cls token
        - transformer blocks (including relative positional encodings)
    '''
    
    # Convert cls token
    if key == "pretrained.model.cls_token":
        new_key = "cls_token"
        return new_key
    
    # Group pretrained.blocks (this is the bulk of the ViT model)
    block_prefix = "pretrained.model.blocks.#"
    if has_prefix(key, block_prefix):
        
        # Dump relative position index table (it is not a learned parameter and is shared for all blocks)
        is_unneeded_layer = ("relative_position_index" in key)
        if is_unneeded_layer:
            return None
        
        # Replace the block prefix, which is shared for all sub-layers of every block
        # ex: "pretrained.model.blocks.1..." -> "stages.0.blocks.1...
        block_idx = get_nth_integer(key, 0)
        stage_idx = block_idx // blocks_per_stage
        new_block_idx = int(block_idx % blocks_per_stage)
        new_prefix = "stages.{}.blocks.{}".format(stage_idx, new_block_idx)
        new_key = replace_prefix(key, block_prefix, new_prefix)
        
        # Update relative position bias table naming
        bias_table_str = "relative_position_bias_table"
        if bias_table_str in new_key:
            new_key = new_key.replace(bias_table_str, "relpos_enc.ref_bias_lut")
        
        # Update 'gamma' naming
        gamme_str = "gamma_"
        if gamme_str in new_key:
            new_key = new_key.replace("gamma_1", "scale_attn")
            new_key = new_key.replace("gamma_2", "scale_mlp")
        
        # Update mlp layer naming ('mlp.fc1.weight' -> 'mlp.layers.0.weight')
        mlp_str = "mlp.fc"
        if mlp_str in new_key:
            new_key = new_key.replace("fc1", "layers.0")
            new_key = new_key.replace("fc2", "layers.2")
        
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
            "act_postprocess1": "spatial_upx4",
            "act_postprocess2": "spatial_upx2",
            "act_postprocess3": "spatial_noscale",
            "act_postprocess4": "spatial_downx2",
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
            "layer1_rn": "spatial_upx4",
            "layer2_rn": "spatial_upx2",
            "layer3_rn": "spatial_noscale",
            "layer4_rn": "spatial_downx2",
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

