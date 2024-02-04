#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class RelativePositionEncoding(nn.Module):
    
    # .................................................................................................................
    
    def __init__(self, num_heads, pretrained_window_size = None, enable_cache = True, num_mlp_hidden = 512):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size
        self.enable_cache = enable_cache
        
        # mlp to generate continuous relative position bias
        num_xy = 2
        self.bias_mlp = nn.Sequential(
            nn.Linear(num_xy, num_mlp_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(num_mlp_hidden, num_heads, bias=False)
        )
        
        # Set up cache for index & bias values, so we can re-use them if we get repeated input image sizes
        self.register_buffer("device_tensor", torch.tensor((0.0)), persistent=False)
        self.lut_cache = GridCache(enable_cache)
        self.index_cache = GridCache(enable_cache)
        self.bias_cache = GridCache(enable_cache)
    
    # .................................................................................................................
    
    def forward(self, attention_tensor, window_size_hw):
        return attention_tensor + self._get_position_bias(window_size_hw)
    
    # .................................................................................................................
    
    def _get_position_bias(self, window_size_hw):
        
        '''
        Helper used to retrieve cached relative position bias values (if stored)
        or otherwise generate and store them for re-use (for a given grid size).
        Returns data with shape: HxNxN
        -> H is number of heads
        -> N is number of tokens
        '''
        
        # Use cached bias table, if we have it
        is_in_cache, relpos_bias = self.bias_cache.retrieve_tensor(window_size_hw)
        if not is_in_cache:
            
            # For convenience
            win_h, win_w = window_size_hw
            window_area = win_h * win_w
            
            # Get (deterministic) base table and indexing matrix
            relative_coords_table = self._get_coords_table(window_size_hw)
            relative_position_index = self._get_position_index(window_size_hw)
            
            # Apply mlp to deterministic coords to get 'learned' mapping
            relpos_bias_table = self.bias_mlp(relative_coords_table).view(-1, self.num_heads)
            relpos_bias = relpos_bias_table[relative_position_index.view(-1)]
            relpos_bias = 16 * torch.sigmoid(relpos_bias)
            
            # Reshape the bias result to be added to attention tensor (1 x H x N x N)
            relpos_bias = relpos_bias.view(window_area, window_area, -1)  # Wh*Ww,Wh*Ww,nH
            relpos_bias = relpos_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relpos_bias = relpos_bias.unsqueeze(0)
            
            self.bias_cache.store_tensor(relpos_bias, window_size_hw)
        
        return relpos_bias
    
    # .................................................................................................................
    
    def _get_coords_table(self, window_size_hw):
        
        is_in_cache, relative_coords_table = self.lut_cache.retrieve_tensor(window_size_hw)
        if not is_in_cache:
            device, dtype = self.device_tensor.device, self.device_tensor.dtype
            relative_coords_table = self._make_relpos_lut(window_size_hw, self.pretrained_window_size, device, dtype)
            self.lut_cache.store_tensor(relative_coords_table, window_size_hw)
        
        return relative_coords_table
    
    # .................................................................................................................
    
    def _get_position_index(self, window_size_hw):
        
        is_in_cache, relative_position_index = self.index_cache.retrieve_tensor(window_size_hw)
        if not is_in_cache:
            device = self.device_tensor.device
            relative_position_index = self._make_relpos_index(window_size_hw, device)
            self.index_cache.store_tensor(relative_position_index, window_size_hw)
        
        return relative_position_index
    
    # .................................................................................................................
    
    @staticmethod
    def _make_relpos_lut(window_size_hw, pretrained_window_size = None, device = None, dtype = None):
        
        '''
        Simplified implementation of the (base) relative positional bias table, described in:
            Swin Transformer V2: Scaling Up Capacity and Resolution
            Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, et. al
            @ https://arxiv.org/abs/2111.09883
        
        The code here is derived from the timm library:
            @ https://github.com/huggingface/pytorch-image-models/blob/ce4d3485b690837ba4e1cb4e0e6c4ed415e36cea/timm/models/swin_transformer_v2.py#L160
        
        The code includes several strange scaling tricks, mostly to improve the positional encoding
        results when using a model with different window sizes compared to sizes used during pretraining,
        at least according to the swinv2 paper (see page 5 of the paper for more details).
        
        Returns a tensor of shape: 1 x (2H - 1) x (2W - 1) x 2
          -> Where H & W come from the given window size
        '''
        
        # For convenience
        win_h, win_w = window_size_hw
        dtype = dtype if dtype is not None else torch.float32
        
        # Generate 'table' of all possible (y-offset, x-offset) pairs for the given window size
        # This table has shape: 1 x nY x nX x 2
        #     -> Where nY, nX are the number of possible y & x offsets, respectively
        #        (e.g. for y offsets, this is all indices from [-win_h - 1] to [+win_h - 1], so nY = 2*win_h - 1)
        #     -> The first factor of 1 is for compatibility with batches
        #     -> The final factor of 2 are the (y, x) offset values
        all_y_idxs = torch.arange(-(win_h - 1), win_h, dtype=dtype, device=device)
        all_x_idxs = torch.arange(-(win_w - 1), win_w, dtype=dtype, device=device)
        yx_offset_pairs = torch.stack(torch.meshgrid([all_y_idxs, all_x_idxs], indexing="ij"))
        relative_coords_table = yx_offset_pairs.permute(1, 2, 0).contiguous().unsqueeze(0)
        # This table can be thought of as a 2D array holding all possible (y, x) positional offsets,
        # where the center-most entry is the (0,0) offset.
        # For a window size of (2,3) the table looks like:
        #  ┌                                              ┐
        #  │ (-1,-2), (-1, -1), (-1, 0), (-1, 1), (-1, 2) │
        #  │ ( 0,-2), ( 0, -1), ( 0, 0), ( 0, 1), ( 0, 2) │
        #  │ ( 1,-2), ( 1, -1), ( 1, 0), ( 1, 1), ( 1, 2) │
        #  └                                              ┘
        # Note: the entries are in (y,x) format!
        # -> Also, the depiction as (y,x) pairs is for visualization only. The table holds these
        #    two values as the 0th and 1st index (respectively) of the final dimension of the table
        
        # Normalize the (y,x) offsets to a target range
        # - If the given window size matches the pretraining size, this normalizes to +/- 1
        # - This can normalize to a different range, which is part of the
        #   'transferring across window sizes' idea introduced by the swinv2 paper
        divider_h = (win_h if pretrained_window_size is None else pretrained_window_size)
        divider_w = (win_w if pretrained_window_size is None else pretrained_window_size)
        relative_coords_table[:, :, :, 0] /= max(divider_h - 1, 1)
        relative_coords_table[:, :, :, 1] /= max(divider_w - 1, 1)
        
        # Perform log scaling as described in swinv2 paper (see page 5)
        # - There is an undocumented scale factor of 8 in the original code, reproduced here for consistency
        scale_factor = 8
        log2_scale_factor = torch.log2(torch.tensor(scale_factor))
        table_sign = torch.sign(relative_coords_table)
        table_log2 = torch.log2(torch.abs(relative_coords_table * scale_factor) + 1.0) / log2_scale_factor
        relative_coords_table = table_sign * table_log2
        
        return relative_coords_table
    
    # .................................................................................................................
    
    @staticmethod
    def _make_relpos_index(window_size_hw, device = None):
        
        '''
        Function which generates a patterned matrix of integers which are used as indices into
        a (separate) relative positional bias lookup table, in order to create the relative positional
        encoding bias which is added to the transformer attention result.
        For the original implementation, see the timm library:
            @ https://github.com/huggingface/pytorch-image-models/blob/ce4d3485b690837ba4e1cb4e0e6c4ed415e36cea/timm/models/swin_transformer_v2.py#L178
        
        The matrix generated by this function is NxN, where N is the number of tokens (based on the window size).
        Each element in the matrix is an integer representing the relative 2D offset between a pair of tokens.
        For example, row 5, column 9 of the matrix represents the offset between token 5 and token 9.
        The offsets can be thought of as (y,x) pairs representing the distance between pairs of tokens.
        For example, given the offset for tokenA-to-tokenB: (+3, -5), this implies tokenA and tokenB are
        separated by 3 rows (+3) and 5 columns (-5) apart using (y,x) ordering (to match hw ordering).
        
        Returns a matrix of integers of shape: NxN
        -> N = (window_h * window_w) is the number of tokens for the given window size
        '''
        
        # For clarity
        int_dtype = torch.int64
        win_h, win_w = window_size_hw
        max_abs_h_idx = win_h - 1
        max_abs_w_idx = win_w - 1
        
        # Generate counting patterns used to create indexing table
        all_y_idxs = torch.arange(win_h, dtype=int_dtype, device=device)
        all_x_idxs = torch.arange(win_w, dtype=int_dtype, device=device)
        coords = torch.stack(torch.meshgrid([all_y_idxs, all_x_idxs], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        # coords_flatten has shape: 2xA, A = grid area = win_h*win_w
        # Example: on the left below is a 2x3 grid showing each (y,x) coordinate
        # (i.e. the coords result), to the right is the coords_flattened result
        #  ┌                     ┐    [
        #  │ (0,0), (0,1), (0,2) │ ->   [0, 0, 0, 1, 1, 1],
        #  │ (1,0), (1,1), (1,2) │      [0, 1, 2, 0, 1, 2],
        #  └                     ┘    ]
        # Notice that the first row of the flattened result is a flattened listing of the y-index values
        # from the (y,x) grid on the left, while the second row is a flattened listing of the x-index values
        
        # Some undocumented mathemagic! This step creates all of the relative offset (y,x) pairs
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
        # Note that there is an odd looking tiling to the offsets to account for flattening.
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
        relative_coords[:, :, 0] += max_abs_h_idx
        relative_coords[:, :, 0] *= (2 * max_abs_w_idx) + 1
        relative_coords[:, :, 1] += max_abs_w_idx
        # This step shifts the offsets so that they become positive integers. Additionally,
        # after shifting the y-offsets to start counting at zero (to avoid negative numbers), the values are
        # scaled so that the smallest non-zero value (e.g. 1) is greater than the largest x-offset after
        # similar shifting. This is done so that the sum of y+x values generates a unique integer
        # representation for every unique (y,x) pair
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
        
        # Add the y & x entries together to get final indices
        # -> the prior scaling steps ensure that all unique (y,x) pairs sum to unique integers!
        relative_position_index = relative_coords.sum(-1)
        
        return relative_position_index
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Model components

class GridCache(nn.Module):
    
    '''
    Helper used to maintain cached data that depends on a grid size (i.e. height & width)
    This can be helpful for re-using heavy-to-compute tensors that are deterministic
    for a given grid sizing (i.e. positional bias values)
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

