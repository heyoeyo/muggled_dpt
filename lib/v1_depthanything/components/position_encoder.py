#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class PositionEncoder(nn.Module):
    
    '''
    This is a simplified/re-organized version of the original DinoV2/Depth-Anything implementation:
        https://github.com/LiheYoung/Depth-Anything/blob/e7ef4b4b7a0afd8a05ce9564f04c1e5b68268516/torchhub/facebookresearch_dinov2_main/vision_transformer.py#L110
    
    This model is used to handle the position encoding of the Depth-Anything
    DPT model. The model uses a learned embedding and only runs once at the
    start of the image encoder, just after patch embedding.
    
    Note this implementation gives very slightly different results compared to the
    original, due to changes in how the positional encodings are interpolated when resizing!
    '''
    
    # .................................................................................................................
    
    def __init__(self, features_per_token, base_patch_grid_hw, enable_cache = True):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.features_per_token = features_per_token
        self.base_grid_hw = base_patch_grid_hw
        
        # Set up base embedding, which can be resized for different image sizes
        base_grid_h, base_grid_w = base_patch_grid_hw
        num_base_patches = base_grid_h * base_grid_w
        self.base_patch_embedding = nn.Parameter(torch.zeros(1, num_base_patches, features_per_token))
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, features_per_token))
        
        # Set up cache for holding resized position encodings & cls token embedding (which doesn't change)
        self.cls_embed_cache = GridCache(enable_cache)
        self.pos_embed_cache = GridCache(enable_cache)
    
    # .................................................................................................................
    
    def forward(self, cls_token, image_patch_tokens, patch_grid_hw):
        
        '''
        Takes in the model class token, image patch tokens (in 'rows of tokens' format)
        and the original image patch sizing.
        Returns:
            cls_token, image_patch_tokens
            
        Note that both outputs have had positional encoding added and are ready
        for processing by a transformer model. This model will cache the positional
        encodings (if enabled) for faster future execution!
        '''
        
        # Handle cls token and image patches separately, since image may need resizing!
        cls_token = self._apply_cls_embedding(cls_token)
        image_patch_tokens = image_patch_tokens + self._get_position_embedding(patch_grid_hw)
        
        return cls_token, image_patch_tokens
    
    # .................................................................................................................
    
    def _apply_cls_embedding(self, cls_token):
        
        '''
        The cls token + bias doesn't change from one input to the next,
        so we can cache the result after the first time we generate it
        '''
        
        no_cache_key = (0,0)
        is_in_cache, cls_token_with_embedding = self.cls_embed_cache.retrieve_tensor(no_cache_key)
        if not is_in_cache:
            cls_token_with_embedding = cls_token + self.cls_embedding
            self.cls_embed_cache.store_tensor(cls_token_with_embedding, no_cache_key)
        
        return cls_token_with_embedding
    
    # .................................................................................................................
    
    def _get_position_embedding(self, patch_grid_hw):
        
        ''' Helper used to handle scaling/caching the image-patch positional embeddings '''
        
        is_in_cache, pos_embed = self.pos_embed_cache.retrieve_tensor(patch_grid_hw)
        if not is_in_cache:
            pos_embed = self._scale_to_patch_grid(patch_grid_hw)
            self.pos_embed_cache.store_tensor(pos_embed, patch_grid_hw)
        
        return pos_embed

    # .................................................................................................................
    
    def _scale_to_patch_grid(self, patch_grid_hw):
        
        '''
        Helper used to make sure the position embeddings are scaled
        to match the input patch sizing, by linear interpolation.
        We don't bother checking for a match to the original sizing,
        since changes to the implementation mean we'll never see the
        original 37x37 grid size!
        
        Note that this implementation differs slightly from the original,
        in that the output size is defined directly and there is no offset
        added to the scaling factor (no scaling factor used even).
        This results in very slight differences in model output compared to
        the original model!
        
        For comparison, the original implementation can be found here:
            https://github.com/LiheYoung/Depth-Anything/blob/e7ef4b4b7a0afd8a05ce9564f04c1e5b68268516/torchhub/facebookresearch_dinov2_main/vision_transformer.py#L179
        '''
        
        # Get original shape/data type, so we can restore this on the output
        base_h, base_w = self.base_grid_hw
        _, N, C = self.base_patch_embedding.shape
        orig_dtype = self.base_patch_embedding.dtype
        
        # Force embedding to float32 for computing interpolation
        # -> If we don't do this, we could get bad results/errors on lower precision dtypes
        pos_embed_f32 = self.base_patch_embedding.float()
        
        # Convert tokens to image-like shape and resize to match given patch grid
        pos_embed_imagelike_bchw = pos_embed_f32.reshape(1, base_h, base_w, C).permute(0,3,1,2)
        pos_embed_imagelike_bchw = nn.functional.interpolate(
            pos_embed_imagelike_bchw,
            size = patch_grid_hw,
            mode = "bicubic",
            antialias = False
        )
        
        # Convert interpolated encodings back into 'rows of tokens' shape
        resized_patch_embedding = pos_embed_imagelike_bchw.permute(0, 2, 3, 1).view(1, -1, C)
        return resized_patch_embedding.to(orig_dtype)
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Model components


class GridCache(nn.Module):
    
    '''
    Helper used to maintain cached data that depends on a grid size (i.e. height & width)
    This can be helpful for re-using heavy-to-compute tensors that are deterministic
    for a given grid sizing (i.e. positional embeddings)
    '''
    
    # .................................................................................................................
    
    def __init__(self, enable_cache = True):
        
        # Inherit from parent
        super().__init__()
        
        # Set up cache storage, which should not be part of the model state dict!
        self._is_enabled = enable_cache
        self._cache_key = None
        self.register_buffer("cache_tensor", None, persistent = False)
    
    # .................................................................................................................
    
    def check_is_cached(self, grid_hw):
        
        cache_key = self._make_cache_key(grid_hw)
        bias_is_cached = (cache_key == self._cache_key)
        
        return bias_is_cached
    
    # .................................................................................................................
    
    def store_tensor(self, tensor_data, grid_hw):
        if self._is_enabled:
            self.cache_tensor = tensor_data
            self._cache_key = self._make_cache_key(grid_hw)
        return
    
    # .................................................................................................................
    
    def retrieve_tensor(self, grid_hw):
        
        # Get cached bias data, and indicator of whether we're storing it
        cache_key = self._make_cache_key(grid_hw)
        is_in_cache = (cache_key == self._cache_key)
        relpos_bias = self.cache_tensor
        
        return is_in_cache, relpos_bias
    
    # .................................................................................................................
    
    def clear(self):
        
        self.cache_tensor = None
        self._cache_key = None
        
        return
    
    # .................................................................................................................
    
    @staticmethod
    def _make_cache_key(grid_hw):
        return "h{}w{}".format(grid_hw[0], grid_hw[1])
    
    # .................................................................................................................
    
    def extra_repr(self):
        
        '''
        For debugging: This prints a string inside the model repr, used to indicate args
        For example, repr prints out: Classname(*** extra_repr string goes here ***)
        '''
        
        bias_str = "no bias cache"
        if self.cache_tensor is not None:
            bias_shape_str = "x".join(str(num) for num in self.cache_tensor.shape)
            bias_str = "bias={}".format(bias_shape_str)
        
        return bias_str
    
    # .................................................................................................................
