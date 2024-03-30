#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn

from .components.relative_positional_encoder import RelativePositionEncoding


# ---------------------------------------------------------------------------------------------------------------------
#%% Main model

class BEiTModel4Stage(nn.Module):
    
    '''
    Simplified implementation of the BEiT backbone model, used in:
        BEiT: BERT Pre-Training of Image Transformers
        Hangbo Bao, Li Dong, Songhao Piao, Furu Wei
        @ https://arxiv.org/abs/2106.08254
    
    The code here is derived from the timm library:
        @ https://github.com/huggingface/pytorch-image-models
    
    This implementation removes most of the flexibility as well as the training-specific
    elements of the original implementation, for the sake of clarity. It is also modified
    in such a way as to be purpose-built for use in the MiDas v3.1 BEiT model
    (e.g. the forward method explicitly returns the 4 internal tokens needed for DPT)
    '''
    
    # .................................................................................................................
    
    def __init__(self, features_per_token=1024, num_heads=16, num_layers=24,
                 base_patch_grid_hw = (32,32), enable_relpos_cache = False):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.features_per_token = features_per_token
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.enable_cache = enable_relpos_cache
        
        # Set up classifier/global token (called 'readout' token in original paper)
        self.cls_token = nn.Parameter(torch.empty(1, 1, features_per_token))

        # Generate multiple transformer stages (layers of transformer blocks)
        num_stages = 4
        self.num_layers_per_stage = int(round(num_layers / num_stages))
        stage_args = (self.num_layers_per_stage, features_per_token, num_heads, base_patch_grid_hw)
        self.stages = nn.ModuleList(TransformerStage(stage_idx, *stage_args) for stage_idx in range(num_stages))
    
    # .................................................................................................................
    
    def forward(self, patch_tokens, patch_grid_hw):
        
        '''
        Main function of model.
        Takes in image patch tokens and the image patch sizing (needed for positional encoding).
        Returns tokens from 4 intermediate stages
        '''
        
        # Append the readout ('cls') token to image patch tokens, for all batches
        # -> Important, readout is assumed to be 0-th token for reassembly stages later on!
        num_batch = patch_tokens.shape[0]
        readout_token_per_batch = self.cls_token.expand(num_batch, -1, -1)
        tokens = torch.cat((readout_token_per_batch, patch_tokens), dim=1)
        
        # Cache relative position encoding, if needed
        if self.enable_cache:
            self.precompute_relpos_and_cache(patch_grid_hw)
        
        # Perform 4-stage processing, grabbing intermediate results
        stage_1_tokens = self.stages[0](tokens, patch_grid_hw)
        stage_2_tokens = self.stages[1](stage_1_tokens, patch_grid_hw)
        stage_3_tokens = self.stages[2](stage_2_tokens, patch_grid_hw)
        stage_4_tokens = self.stages[3](stage_3_tokens, patch_grid_hw)
        
        return stage_1_tokens, stage_2_tokens, stage_3_tokens, stage_4_tokens
    
    # .................................................................................................................
    
    def precompute_relpos_and_cache(self, patch_grid_hw):
        
        '''
        Function used to 'drill' down into attention blocks to compute & cache
        relative positional encodings for the given patch grid size
        '''
        
        try:
            for stage in self.stages:
                stage.update_relpos_cache(patch_grid_hw)
            
        except torch.cuda.OutOfMemoryError as err:
            self.enable_cache = False
            print("",
                  "*** WARNING ***",
                  "Not enough memory for caching! Caching will be disabled...",
                  "Error message:",
                  str(err), "", sep = "\n", flush = True)
            self.clear_relpos_cache()
        
        return
    
    # .................................................................................................................
    
    def clear_relpos_cache(self):
        ''' Convenience function which wipes out cached data inside attention blocks '''
        self.precompute_relpos_and_cache(None)
        torch.cuda.empty_cache()
        return
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Component classes

class TransformerStage(nn.Module):
    
    '''
    Helper module which consists of a sequence of transformer blocks which are executed
    sequentially. This module helps handle the need for a shared 'patch_grid_hw' arguments
    that is passed to all blocks as an input (otherwise this module is just a 'sequential' module!)
    '''
    
    # .................................................................................................................
    
    def __init__(self, stage_index, num_layers, features_per_token, num_heads, base_patch_grid_hw):
        
        # Inherit from parent
        super().__init__()
        
        # Create list of transformer blocks (note all blocks share the exact same config!)
        block_args = (features_per_token, num_heads, base_patch_grid_hw)
        self.blocks = nn.ModuleList(TransformerBlock(*block_args) for _ in range(num_layers))
    
    # .................................................................................................................
    
    def forward(self, tokens, patch_grid_hw):
        
        '''
        Process tokens sequentially through all blocks. Note output has the same shape as input: BxNxF
        -> B is batch dimension
        -> N is number of tokens
        -> F is features per token
        '''
        
        for each_block in self.blocks:
            tokens = each_block(tokens, patch_grid_hw)
    
        return tokens
    
    # .................................................................................................................
    
    def update_relpos_cache(self, patch_grid_hw):
        
        '''
        Function used to 'drill' down into attention blocks to compute & cache
        relative positional encodings for the given patch grid size
        '''
        
        for block_idx, block in enumerate(self.blocks):
            block.update_relpos_cache(patch_grid_hw)
        
        return
    
    # .................................................................................................................


class TransformerBlock(nn.Module):
    
    '''
    Simplified implementation of a single transformer block layer from:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    Based on code from timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/beit.py
    
    Purpose is to act as a single 'layer' of a multi-layer transformer model, including
    the application of self-attention along with normalization & a fully-connected output layer
    The main simplification is to remove most of the original flexibility of the model, so
    that it is easier to follow for use in the DPT model.
    '''
    
    # .................................................................................................................
    
    def __init__(self, features_per_token, num_heads, base_patch_grid_hw, mlp_ratio=4):
        
        # Inherit from parent
        super().__init__()
        
        # Define components for self-attention (with pre-norm & residual connection)
        self.norm1 = nn.LayerNorm(features_per_token, eps=1e-6)
        self.attn = SelfAttentionRelPos(features_per_token, num_heads, base_patch_grid_hw)
        self.scale_attn = nn.Parameter(torch.empty(features_per_token))
        
        # Define components for feed-forward transformation of tokens after attention
        self.norm2 = nn.LayerNorm(features_per_token, eps=1e-6)
        self.mlp = MLP2Layers(features_per_token, hidden_feature_ratio = mlp_ratio)
        self.scale_mlp = nn.Parameter(torch.empty(features_per_token))
    
    # .................................................................................................................
    
    def forward(self, tokens, patch_grid_hw):
        
        '''
        Single transformer block, input & output have the same shape: BxNxF
        -> B is batch dimension
        -> N is number of tokens
        -> F is features per token
        '''
        
        # Calculate (pre-norm!) attention with residual connection
        attn_tokens = self.norm1(tokens)
        attn_tokens = self.attn(attn_tokens, patch_grid_hw)
        attn_tokens = tokens + (self.scale_attn * attn_tokens)
        
        # Calculate (pre-norm!) feedforward output with residual connection
        output_tokens = self.norm2(attn_tokens)
        output_tokens = self.mlp(output_tokens)
        output_tokens = attn_tokens + (self.scale_mlp * output_tokens)
        
        return output_tokens
    
    # .................................................................................................................
    
    def update_relpos_cache(self, patch_grid_hw):
        return self.attn.update_relpos_cache(patch_grid_hw)
    
    # .................................................................................................................


class SelfAttentionRelPos(nn.Module):
    
    '''
    Simplified implementation of the self-attention block from:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    Based on code from timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/beit.py
    
    Includes (modified implementation of) relative positional encoding, originally from:
        "Self-Attention with Relative Position Representations"
        By: Peter Shaw, Jakob Uszkoreit, Ashish Vaswani
        @ https://arxiv.org/abs/1803.02155
    '''
    
    # .................................................................................................................
    
    def __init__(self, features_per_token, num_heads, base_patch_grid_hw):
        
        # Inherit from parent
        super().__init__()
        
        # Store model config
        self.features_per_token = features_per_token
        self.num_heads = num_heads
        
        # Set up scale factor from 'scaled dot-product attention'
        features_per_head = features_per_token // num_heads
        internal_feature_count = features_per_head * self.num_heads
        self.qk_scale_factor = features_per_head ** -0.5
        
        # Set up query/key/value weights & bias parameters
        # Note: bias is set up separate from weights, because there is no k-bias!
        bias_shape = (1, num_heads, 1, features_per_head)
        self.q_bias = nn.Parameter(torch.empty(bias_shape))
        self.v_bias = nn.Parameter(torch.empty(bias_shape))
        self.qkv = nn.Linear(features_per_token, internal_feature_count * 3, bias=False)
        
        # Set up position encoding
        self.relpos_enc = RelativePositionEncoding(num_heads, base_patch_grid_hw)
        
        # Set up output projection layer to get back our input feature count (if it was changed inside attention)
        self.proj = nn.Linear(internal_feature_count, features_per_token)
        
        # Set up softmax as dedicated 'module' so that we can hook into it for debugging/analysis!
        self.softmax = nn.Softmax(dim=-1)
    
    # .................................................................................................................
    
    def forward(self, tokens, patch_grid_hw):
        
        '''
        Input & output have same shape: BxNxF
        -> B is batch dimension
        -> N is number of tokens (= 1 + H*W, where H & W are patch grid size, +1 comes from readout token)
        -> F is features per token
        
        Internally has shape variables:
        -> H is number of heads
        -> f is number of features per head (= F//H)
        
        Basic idea:
            1. Use input tokens to make new set of tokens: Q, K, V (there are H copies of Q, K & V)
            2. Calculate (attention) A = dotproduct(Q, K) * qk_scale
            3. Add relative positional encoding to attention, A = A + relpos_encoding
            4. Calculate (weights) W = softmax(A), calculated on the N 'columns' of A
            5. Calculate output = W * V
            6. Apply final linear/feed-forward layer on output token feature values (output has shape: NxF)
        '''
        
        # Get input shape, which we'll need to proeprly reshape intermediate results
        B, N, F = tokens.shape
        
        # Convert shape: BxNx(3F) -> BxNx3xHxf -> 3xBxHxNxf, so that we can split the q/k/v components
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Add bias terms (we don't do this as part of the linear layer, since the k-tokens don't have a bias!)
        # -> bias has shape: BxHxNxf
        q = q + self.q_bias
        v = v + self.v_bias
        
        # Perform 'scaled dot-product' attention (see: 'Attention is all you need' paper)
        # BxHxNxf @ BxHxfxN -> BxHxNxN
        q = q * self.qk_scale_factor
        attn = (q @ k.transpose(-2, -1))
        attn = self.relpos_enc(attn, patch_grid_hw)
        
        # Generate new tokens from weighted value tokens & reshape to match original input token shape
        # Shape: BxHxNxN @ BxHxNxf -> BxHxNxf -> BxNxHxf -> BxNxF
        value_weighting = self.softmax(attn)
        output = (value_weighting @ v).transpose(1, 2).reshape(B, N, -1)
        output = self.proj(output)
        
        return output
    
    # .................................................................................................................
    
    def update_relpos_cache(self, patch_grid_hw):
        return self.relpos_enc.update_cache(patch_grid_hw)
    
    # .................................................................................................................


class MLP2Layers(nn.Module):
    
    '''
    Simplified implementation of the MLP model from the timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/23e7f177242e7516e8e3fc02ea1071b8cbc41ca8/timm/layers/mlp.py#L13
    
    This implementation removes most of the flexibility options, so that only the functionality used
    by the BEiT implementation remains. Also removes training-related (i.e. dropout) components.
    
    This model is a simple feed-forward network, which is intended to be used at the end of each
    transformer block. Note that it defaults to including an 'expansion' type of hidden layer
    (i.e hidden layer has more features than input/output), based on feature_ratio input.
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_features, hidden_feature_ratio = 4, bias = True):
        
        # Inherit from parent
        super().__init__()
        
        # Calculate number of hidden features
        num_hidden_features = int(round(hidden_feature_ratio * num_features))
        
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden_features, bias=bias),
            nn.GELU(),
            nn.Linear(num_hidden_features, num_features, bias=bias)
        )
    
    # .................................................................................................................
    
    def forward(self, x):
        return self.layers(x)
    
    # .................................................................................................................

