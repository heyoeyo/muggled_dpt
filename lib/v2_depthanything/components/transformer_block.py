#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn

from .misc_helpers import MLP2Layers, LayerNormEPS6

# Check if XFormers library is installed
# -> This gives improved float16 performance compared to normal 'pytorch' code
# -> The improvement is significant when running larger image sizes (2x speed up)
# -> It also dramatically reduces memory usage, so input image sizes can be used!
try:
    from xformers.ops import memory_efficient_attention, unbind
    XFORMERS_AVAILABLE = torch.cuda.is_available()
except ImportError:
    XFORMERS_AVAILABLE = False

# Allow disabling of XFormers by setting a 'NO_XFORMERS' env variable
# ex: $ export NO_XFORMERS=1
if XFORMERS_AVAILABLE:
    import os
    _xformers_disable_key = "NO_XFORMERS"
    XFORMERS_AVAILABLE = _xformers_disable_key not in os.environ
    if XFORMERS_AVAILABLE:
        print("",
              "XFormers Detected! (enabled)",
              f"  To disable, set environment variable: '{_xformers_disable_key}'",
              sep = "\n")
    else:
        print("",
              "XFormers Detected! (disabled)",
              f"  To enable, un-set environment variable: '{_xformers_disable_key}'",
              sep = "\n")


# ---------------------------------------------------------------------------------------------------------------------
#%% Main model

class TransformerBlock(nn.Module):
    
    # .................................................................................................................
    
    def __init__(self, features_per_token, num_heads, mlp_ratio = 4.0):
        
        # Inherit from parent
        super().__init__()
        
        # Special check to switch to high-performance attention, if possible
        attn_args = (features_per_token, num_heads)
        self.attn = AttentionXFormers(*attn_args) if XFORMERS_AVAILABLE else Attention(*attn_args)
        
        # Define components for self-attention
        self.norm1 = LayerNormEPS6(features_per_token)
        self.scale_attn = nn.Parameter(torch.ones(features_per_token))
        
        # Define components for feed-forward transformation of tokens after attention
        self.norm2 = LayerNormEPS6(features_per_token)
        self.mlp = MLP2Layers(features_per_token, mlp_ratio)
        self.scale_mlp = nn.Parameter(torch.ones(features_per_token))

    # .................................................................................................................
    
    def forward(self, tokens, patch_grid_hw = None):
        
        # Calculate (pre-norm) attention with residual connection
        attn_tokens = self.norm1(tokens)
        attn_tokens = self.attn(attn_tokens)
        attn_tokens = tokens + (self.scale_attn * attn_tokens)
        
        # Calculate (pre-norm) feedforward output with residual connection
        output_tokens = self.norm2(attn_tokens)
        output_tokens = self.mlp(output_tokens)
        output_tokens = attn_tokens + (self.scale_mlp * output_tokens)
        
        return output_tokens
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Model components


class Attention(nn.Module):
    
    '''
    Attention block for the DINOv2 model.
    
    This code is lightly modified from the Depth-Anything/facebookresearch code:
        @ https://github.com/LiheYoung/Depth-Anything/blob/main/torchhub/facebookresearch_dinov2_main/dinov2/layers/attention.py
    '''
    
    # .................................................................................................................
    
    def __init__(self, features_per_token, num_heads):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.num_heads = num_heads
        
        # Set up scale factor from 'scaled dot-product attention'
        self.features_per_head = features_per_token // num_heads
        self.qk_scale_factor = self.features_per_head ** -0.5

        # Set up query/key/value weights & output feedforward network
        self.qkv = nn.Linear(features_per_token, features_per_token * 3, bias=True)
        self.proj = nn.Linear(features_per_token, features_per_token, bias=True)
        
        # Set up softmax as dedicated 'module' so that we can hook into it for debugging/analysis!
        self.softmax = nn.Softmax(dim=-1)

    # .................................................................................................................
    
    def forward(self, tokens):
        
        '''
        Input & output have same shape: BxNxC
        -> B is batch dimension
        -> N is number of tokens (= 1 + H*W, where H & W are patch grid size, +1 comes from readout token)
        -> C is features per token (or Channels)
        
        Internally has shape variables:
        -> H is number of heads
        -> c is number of features per head (= C//H)
        '''
        
        # For convenience
        B, N, C = tokens.shape
        
        # Convert shape: BxNx(3C) -> BxNx3xHxc -> 3xBxHxNxc, so that we can split the q/k/v components
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.features_per_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Perform 'scaled dot-product' attention (see: 'Attention is all you need' paper)
        # BxHxNxc @ BxHxcxN -> BxHxNxN
        q = q * self.qk_scale_factor
        attn = q @ k.transpose(-2, -1)

        # Generate new tokens from weighted value tokens & reshape to match original input token shape
        # Shape: BxHxNxN @ BxHxNxc -> BxHxNxc -> BxNxHxc -> BxNxC
        value_weighting = self.softmax(attn)
        tokens = (value_weighting @ v).transpose(1, 2).reshape(B, N, C)
        tokens = self.proj(tokens)
        
        return tokens
    
    # .................................................................................................................


class AttentionXFormers(Attention):
    
    '''
    This is a faster executing version of the attention module, which
    makes use of functions from the XFormers library.
    
    Due to my own uncertainty about the compatibility/stability of XFormers,
    this is implemented as a 'special upgrade', rather than being the
    default implementation. Note this also breaks when using cpu-only!
    '''
    
    # .................................................................................................................
    
    def forward(self, tokens):
        
        ''' Faster self-attention implementation due to XFormers '''

        # Create query/key/value as usual
        B, N, C = tokens.shape
        qkv = self.qkv(tokens).reshape(B, N, 3, self.num_heads, self.features_per_head)
        
        # XFormers magic!
        q, k, v = unbind(qkv, 2)
        tokens = memory_efficient_attention(q, k, v)
        
        # Reshape to match input
        tokens = tokens.reshape([B, N, C])
        tokens = self.proj(tokens)
        
        return tokens
    
    # .................................................................................................................
