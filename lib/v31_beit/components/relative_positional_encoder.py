#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class RelativePositionEncoding(nn.Module):
    
    '''
    Modified implementation of relative position encoding, originally from timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/beit.py
    
    Includes modifications to support dynamic 'window' sizing (i.e. different patch grid sizes),
    based on code originally from MiDaS model:
        @ https://github.com/isl-org/MiDaS/blob/master/midas/backbones/beit.py
    
    The original concept seems to be based on a modified interpretation
    of the encoding introduced in the paper:
        "Self-Attention with Relative Position Representations"
        By: Peter Shaw, Jakob Uszkoreit, Ashish Vaswani
        @ https://arxiv.org/abs/1803.02155
    
    This implementation includes support for caching of relative position bias terms
    (helpful if running multiple images of the same size through the model), since
    (re-)computing the bias term is extremely slow!
    '''
    
    # .................................................................................................................
    
    def __init__(self, num_heads, reference_patch_grid_hw = (32,32)):
        
        # Inherit from parent
        super().__init__()
        
        # Store sizing info
        self.num_heads = num_heads
        self.reference_patch_grid_hw = reference_patch_grid_hw
        
        # Calculate size of bias table. Note there are 3 cls entries (cls-to-cls, token-to-cls, cls-to-token)
        ref_grid_h, ref_grid_w = reference_patch_grid_hw
        self.ref_num_relative_h = int((2 * ref_grid_h) - 1)
        self.ref_num_relative_w = int((2 * ref_grid_w) - 1)
        self.ref_num_token_lut_entries = (self.ref_num_relative_h * self.ref_num_relative_w)
        self.num_cls_lut_entries = 3

        # Set up reference bias table (this table is interpolated to handle other patch grid sizes)
        full_bias_lut_length = self.ref_num_token_lut_entries + self.num_cls_lut_entries
        self.ref_bias_lut = nn.Parameter(torch.zeros(full_bias_lut_length, num_heads))
        
        # Set up cache for index & bias values, so we can re-use them if we get repeated input image sizes
        self.cache = GridCache()
    
    # .................................................................................................................
    
    def forward(self, attention_tensor, patch_grid_hw):
        
        '''
        Relative position encoding simply 'adds' to an existing attention tensor.
        Note that the added term is a learnable parameter!
        '''
        
        return attention_tensor + self._get_position_bias(patch_grid_hw)
    
    # .................................................................................................................
    
    def update_cache(self, patch_grid_hw = None):
        
        '''
        Helper used to pre-compute relative position bias and store in cache, if needed
        Expected to be called by the image encoder model, prior to running transformer layers
        Returns:
            is_already_cached (True if we already have data for the given patch_grid_hw)
        '''
        
        # Interpret missing patch grid size as a 'clear cache' command
        if patch_grid_hw is None:
            self.cache.clear()
            return False
        
        # Generate & store bias data, if needed
        bias_already_cached = self.cache.check_is_cached(patch_grid_hw)        
        if not bias_already_cached:
            relpos_bias = self._generate_position_bias_lut(patch_grid_hw)
            self.cache.store_bias(relpos_bias, patch_grid_hw)
        
        return bias_already_cached
    
    # .................................................................................................................
    
    def _get_position_bias(self, patch_grid_hw):
        
        '''
        Helper used to retrieve cached relative position bias values (if stored)
        or otherwise generate and store them for re-use (for a given grid size).
        Returns data with shape: HxNxN
        -> H is number of heads
        -> N is number of tokens (= grid_h*grid_w + 1)
        '''
        
        # Use cached bias table, if we have it
        is_in_cache, relpos_bias = self.cache.retrieve_bias(patch_grid_hw)
        if not is_in_cache:
            relpos_bias = self._generate_position_bias_lut(patch_grid_hw)
        
        return relpos_bias
    
    # .................................................................................................................
    
    @staticmethod
    def _generate_relative_position_index(patch_grid_hw, device=None):
        
        '''
        Function which generates a patterned matrix of integers which are used as indices into
        a (separate) relative positional bias lookup table, in order to create the relative positional
        encoding bias which is added to the transformer attention result.
        For the original implementation, see the timm library:
            @ https://github.com/huggingface/pytorch-image-models/blob/21647c0a0c3dde0176d427dcf3e3ab48f3a1d5c2/timm/models/beit.py#L61
        
        The matrix generated by this function is NxN, where N is the number of tokens (based on patch grid size).
        Each element in the matrix is an integer representing the relative 2D offset between a pair of tokens.
        For example, row 5, column 9 of the matrix represents the offset between token 5 and token 9.
        The offsets can be thought of as (x,y) pairs representing the distance between pairs of tokens.
        For example, given the offset for tokenA-to-tokenB: (-5, +3), this implies tokenA and tokenB are
        separated by 5 columns (-5) and 3 rows (+3).
        This function figures out the correct token-to-token offsets and then represents each
        unique (x,y) pair with an integer value, to be used to index into a (learned) relative positional
        bias lookup table. The function also handles the encodings for the readout/cls token, which is given
        a special index for cls-to-cls, cls-to-token and token-to-cls 'offsets'.
        
        Note that the matrix of integers generated here is deterministic for a given window size
        (i.e. will get the same result given the same input size, there is no learnable component here)
        
        Returns a matrix of integers of shape: NxN
        -> N is the number of tokens for the given grid size
           where N = grid_h*grid_w + 1 (this +1 is for the cls/readout token!)
        '''
        
        # For clarity
        int_dtype = torch.int32
        grid_h, grid_w = patch_grid_hw
        num_tokens = grid_h * grid_w + 1
        output_hw = (num_tokens, num_tokens)
        
        # Generate counting patterns used to create indexing table
        all_y_idxs = torch.arange(grid_h, dtype=int_dtype, device=device)
        all_x_idxs = torch.arange(grid_w, dtype=int_dtype, device=device)
        coords = torch.stack(torch.meshgrid([all_y_idxs, all_x_idxs], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        # coords_flatten has shape: 2xA, A = grid area = grid_h*grid_w
        # Example: on the left below is a 2x3 grid showing each (x,y) coordinate,
        # to the right is the coords_flattened result
        #  ┌                     ┐    [
        #  │ (0,0), (1,0), (2,0) │ ->   [0, 0, 0, 1, 1, 1],
        #  │ (0,1), (1,1), (2,1) │      [0, 1, 2, 0, 1, 2],
        #  └                     ┘    ]
        # Notice that the first row of the flattened result is a flattened listing of the y-index values
        # from the grid on the left, while the second row is a flattened listing of the x-index values
        
        # Some undocumented mathemagic! This step creates all of the relative offset (x,y) pairs
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # relative_coords has shape: AxAx2
        # Example: on the left below is the 0th channel for a 2x3 grid size, showing y-offsets,
        # on the right is the 1st channel showing x-offsets.
        #     [[ 0,  0,  0, -1, -1, -1],       [[ 0, -1, -2,  0, -1, -2],
        #      [ 0,  0,  0, -1, -1, -1],        [ 1,  0, -1,  1,  0, -1],
        #      [ 0,  0,  0, -1, -1, -1],        [ 2,  1,  0,  2,  1,  0],
        #      [ 1,  1,  1,  0,  0,  0],        [ 0, -1, -2,  0, -1, -2],
        #      [ 1,  1,  1,  0,  0,  0],        [ 1,  0, -1,  1,  0, -1],
        #      [ 1,  1,  1,  0,  0,  0]]        [ 2,  1,  0,  2,  1,  0]]
        # Note that there is an odd looking tiling to the x/y-offsets, to account for flattening.
        # For example, the first row of y-offsets shows: 0,0,0,-1,-1,-1. This is because the first row
        # corresponds to the offset between token 0 (row 0) and tokens 0,1,2,3,4,5 (each of the columns).
        # For a 2x3 grid, and the flattening used, the token indexing is as follows:
        #  ┌         ┐
        #  │ 0, 1, 2 │
        #  │ 3, 4, 5 │
        #  └         ┘
        # Therefore, token 0 has no y-offset between tokens 0, 1, 2 and then a y-offset of -1 between
        # tokens 3, 4, 5, hence the 0,0,0,-1,-1,-1 pattern in the first row of the 0th channel of
        # the relative_coords results. A similar pattern holds for the x-offsets. The first row
        # pattern is 0,-1,-2,0,-1,-2, since it indicates the x-offset between token 0 and tokens: 0,1,2,3,4,5,
        # for example, notice that the x-offset between token 0 and token 3 is 0, since they are in the same column.
        
        # Normalize coords to positive integer values
        relative_coords[:, :, 0] += grid_h - 1
        relative_coords[:, :, 0] *= (2 * grid_w) - 1
        relative_coords[:, :, 1] += grid_w - 1
        # This step shifts the x/y-offsets so that they become positive integers. Additionally,
        # after shifting the y-offsets to start counting at zero (to avoid negative numbers), the values are
        # scaled so that the smallest non-zero value (e.g. 1) is greater than the largest x-offset after
        # similar shifting. This is done so that the sum of (x,y) values generates a unique integer
        # representation for every unique (x,y) pair
        # Example: on the left below is the 0th channel for a 2x3 grid size, showing the shifted & scaled
        # y-offsets, while the right shows the 1st channel with shifted x-offsets
        #     [[ 5,  5,  5,  0,  0,  0],       [[ 2,  1,  0,  2,  1,  0],
        #      [ 5,  5,  5,  0,  0,  0],        [ 3,  2,  1,  3,  2,  1],
        #      [ 5,  5,  5,  0,  0,  0],        [ 4,  3,  2,  4,  3,  2],
        #      [10, 10, 10,  5,  5,  5],        [ 2,  1,  0,  2,  1,  0],
        #      [10, 10, 10,  5,  5,  5],        [ 3,  2,  1,  3,  2,  1],
        #      [10, 10, 10,  5,  5,  5]]        [ 4,  3,  2,  4,  3,  2]]
        # Note that the smallest non-zero y-offset (5) is larger than the largest x-offset (4), so that
        # y-offsets can be added to x-offsets to get unique (single) integer representations.
        
        # Set up special index values for cls (readout) to token/cls entries
        max_token_index = (2 * grid_h - 1) * (2 * grid_w - 1) - 1
        cls_to_token_index = max_token_index + 1
        token_to_cls_index = max_token_index + 2
        cls_to_cls_index = max_token_index + 3
        
        # Build final result, which uses the x/y coords sum to uniquely index all token-to-token entries,
        # and prepends the special cls entries to the first row & column
        relative_position_index = torch.zeros(size=output_hw, dtype=int_dtype, device=device)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = cls_to_token_index
        relative_position_index[0:, 0] = token_to_cls_index
        relative_position_index[0, 0] = cls_to_cls_index
        # Relative position index has shape: NxN
        # Example for grid_wh = (2,3):
        #   [[17, 15, 15, 15, 15, 15, 15],
        #    [16,  7,  6,  5,  2,  1,  0],
        #    [16,  8,  7,  6,  3,  2,  1],
        #    [16,  9,  8,  7,  4,  3,  2],
        #    [16, 12, 11, 10,  7,  6,  5],
        #    [16, 13, 12, 11,  8,  7,  6],
        #    [16, 14, 13, 12,  9,  8,  7]]
        
        return relative_position_index
    
    # .................................................................................................................
    
    def _generate_position_bias_lut(self, patch_grid_hw):
        
        '''
        Function which generates relative position bias values for a given patch grid size.
        This function takes a 'reference' table of bias values and interpolates them to get a
        new table with dimensions matching the target patch grid size. The reference table is
        a learned parameter of the model (it should be included in the model weights on loading).
        
        This function also does some work to manage the last 3 bias values of the reference table,
        which are special values used for cls-to-cls, cls-to-token and token-to-cls positional encodings
        (these are excluded from the interpolation step).
        
        Based on implementation from MiDaS:
            @ https://github.com/isl-org/MiDaS/blob/bdc4ed64c095e026dc0a2f17cabb14d58263decb/midas/backbones/beit.py#L29C21-L29C21
        
        Which itself is a modification of code from the timm library:
            @ https://github.com/huggingface/pytorch-image-models/blob/21647c0a0c3dde0176d427dcf3e3ab48f3a1d5c2/timm/models/beit.py#L130C9-L130C26
        
        Returns a tensor of shape: HxNxN
        -> H is number of heads of transformer model
        -> N is number of tokens
        
        Note: This process adds considerable time to the model execution, and the results are the same
        for a given patch_grid_hw, so can be cached to (substantially) speed up inference speed.
        Surprisingly, the slow down is not due to the computations (e.g. interpolation step), but
        instead simply due to the indexing step (index positions into the bias luts), likely due to the
        very random memory access pattern...?
        '''
        
        # For convenience
        grid_h, grid_w = patch_grid_hw
        num_tokens = (grid_h * grid_w) + 1
        heads = self.num_heads
        
        # Get 'relative width/height' sizes, which determine number of unique entries in bias table
        ref_rel_h, ref_rel_w = self.ref_num_relative_h, self.ref_num_relative_w
        new_rel_h, new_rel_w = 2*patch_grid_hw[0] - 1, 2*patch_grid_hw[1] - 1 # Written weirdly, for onnx support!
        
        # Get the reference token so we can spatially interpolate the values
        # -> Ref token LUT shape: (refH * refW) x Heads
        # -> Ref cls LUT shape: 3 x Heads
        ref_num_token_lut_entries = self.ref_num_token_lut_entries
        ref_token_lut = self.ref_bias_lut[:ref_num_token_lut_entries, :]
        ref_cls_lut = self.ref_bias_lut[ref_num_token_lut_entries:, :]
        
        # Scale/interpolate the original reference entries to the new patch grid size
        # -> Ref shape 2D: 1 x Heads x refH x refW
        # -> New shape 2D: 1 x Heads x newH x newW
        ref_token_2d = ref_token_lut.reshape(1, ref_rel_h, ref_rel_w, heads).permute(0, 3, 1, 2)
        new_token_2d = nn.functional.interpolate(ref_token_2d, size=(new_rel_h, new_rel_w), mode="bilinear")
        
        # Reshape interpolated (2D) values back into 1D (per head) LUTs
        # -> New token LUT shape: (newH * newW) x Heads
        new_num_token_lut_entries = (new_rel_h * new_rel_w)
        new_token_lut = new_token_2d.permute(0, 2, 3, 1).reshape(new_num_token_lut_entries, heads)
        
        # Create full table by combining upscaled table with cls entries of the original table
        new_relpos_bias_lut = torch.cat([new_token_lut, ref_cls_lut])
        # Make final bias by indexing into bias table, using proper indexing input
        # -> idx shape: N x N
        # -> bias shape: (N * N) x Heads, N is number of tokens
        relpos_idx = self._generate_relative_position_index(patch_grid_hw, new_relpos_bias_lut.device)
        relpos_bias = new_relpos_bias_lut[relpos_idx.view(-1)]     # <- This step is extremely slow!
        
        # Reshape bias values to match attention tensor shape: Heads x N x N
        relpos_bias = relpos_bias.view(num_tokens, num_tokens, heads)
        relpos_bias = relpos_bias.permute(2, 0, 1).contiguous()
        relpos_bias = relpos_bias.unsqueeze(0)
        return relpos_bias
    
    # .................................................................................................................
    
    @staticmethod
    def calculate_bytes_per_layer(num_heads, patch_grid_hw, bytes_per_bias_element = 4):
        
        '''
        Helper used to determine how many bytes of RAM are needed to store a single
        relative position bias matrix (i.e. for a single attention layer)
        Returns:
            bytes_per_layer
        '''
        
        # Calculate the number of elements (i.e. numbers) in a single bias lookup table (i.e. per transformer layer)
        grid_h, grid_w = patch_grid_hw
        num_tokens = 1 + (grid_h * grid_w)
        num_bias_lut_elements = (num_tokens * num_tokens * num_heads)
        bytes_per_layer = int(num_bias_lut_elements * bytes_per_bias_element)
        
        return bytes_per_layer
    
    # .................................................................................................................
    
    def extra_repr(self):
        
        '''
        For debugging: This prints a string inside the model repr, used to indicate args
        For example, repr prints out: Classname(*** extra_repr string goes here ***)
        '''
        
        return "num_heads={}, reference_patch_grid_hw={}".format(self.num_heads, self.reference_patch_grid_hw)
        
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Model components

class GridCache(nn.Module):
    
    '''
    Helper used to maintain a cache of results from relative positional encoding.
    Caching the final bias result improves the execution speed of the model.
    However, this does come at the cost of a significant increase in VRAM usage!
    
    Bytes per attention layer = H x N x N x Q
    -> H is number of heads
    -> N is number of tokens
    -> Q is number of bytes per tensor entry (i.e. Q = 4 for float32, Q = 2 for float16)
    
    Example: For BeiT large-512 with a 512x512 image input:
        - the patch grid size is 32x32 (32 patches in height & width from: 512/16)
        - the number of tokens is 1025 (= 32*32 + 1, number of patches +1 for cls/readout token)
        - the number of heads, H, is 16
        - the number of attention layers is 24
        
        So for float32 (Q = 4):
            bytes per layer = 16 x 1025 x 1025 x 4 = 67240000
            total bytes for all layers = 24 * 67240000 = 1613760000
                                       = 1.6GB (!!!)
    '''
    
    # .................................................................................................................
    
    def __init__(self):
        
        # Inherit from parent
        super().__init__()
        
        # Set up cache storage, which should not be part of the model state dict!
        self._cache_key = None
        self.register_buffer("bias_cache", None, persistent = False)
    
    # .................................................................................................................
    
    def check_is_cached(self, patch_grid_hw):
        
        cache_key = self._make_cache_key(patch_grid_hw)
        bias_is_cached = (cache_key == self._cache_key)
        
        return bias_is_cached
    
    # .................................................................................................................
    
    def store_bias(self, relative_position_bias, patch_grid_hw):
        self.bias_cache = relative_position_bias
        self._cache_key = self._make_cache_key(patch_grid_hw)
        return
    
    # .................................................................................................................
    
    def retrieve_bias(self, patch_grid_hw):
        
        # Get cached bias data, and indicator of whether we're storing it
        cache_key = self._make_cache_key(patch_grid_hw)
        is_in_cache = (cache_key == self._cache_key)
        relpos_bias = self.bias_cache
        
        return is_in_cache, relpos_bias
    
    # .................................................................................................................
    
    def clear(self):
        
        self.bias_cache = None
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
        if self.bias_cache is not None:
            bias_shape_str = "x".join(str(num) for num in self.bias_cache.shape)
            bias_str = "bias={}".format(bias_shape_str)
        
        return bias_str
    
    # .................................................................................................................

