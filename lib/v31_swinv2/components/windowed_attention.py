#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

from .relative_positional_encoder import RelativePositionEncoding


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class WindowAttentionWithRelPos(nn.Module):
    
    # .................................................................................................................
    
    def __init__(self, num_heads, features_per_token, window_size_hw,
                 pretrained_window_size=None, is_shift_block = False, enable_cache = True):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.num_heads = num_heads
        
        # Set up pre-/post-windowing functionality
        self.windowing = Windowing(window_size_hw, is_shift_block)
        
        # Set up query/key/value weights & bias parameters
        # Note: bias is set up separate from weights, because there is no k-bias!
        features_per_head = features_per_token // num_heads
        bias_shape = (1, num_heads, 1, features_per_head)
        self.q_bias = nn.Parameter(torch.empty(bias_shape))
        self.v_bias = nn.Parameter(torch.empty(bias_shape))
        self.qkv = nn.Linear(features_per_token, features_per_token * 3, bias=False)
        
        # Set up scaling factors for cosine attention & position encoding bias
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.relpos_enc = RelativePositionEncoding(num_heads, pretrained_window_size, enable_cache)
        
        # Set up output project layer
        self.proj = nn.Linear(features_per_token, features_per_token)
        
        # Set up softmax as dedicated 'module' so that we can hook into it for debugging/analysis!
        self.softmax = nn.Softmax(dim=-1)
    
    # .................................................................................................................
    
    def forward(self, tokens, patch_grid_hw):
        
        # For convenience
        B, N, C = tokens.shape
        H, W, = patch_grid_hw
        
        # Convert to image-like representation
        img_tokens = tokens.view(B, H, W, C)
        orig_img_shape_bhwc = img_tokens.shape
        
        window_size_hw = self.windowing.resize(patch_grid_hw)
        
        # Apply windowing on image tokens before processing
        window_tokens, num_win_xy = self.windowing.partition(img_tokens, window_size_hw)
        window_tokens = self.attention_on_windows(window_tokens, B, window_size_hw)
        img_tokens = self.windowing.reverse(window_tokens, window_size_hw, num_win_xy, orig_img_shape_bhwc)
        
        # Convert back to token-like representation (from image-like)
        tokens = img_tokens.view(B, N, C)
        
        return tokens
    
    # .................................................................................................................
    
    def attention_on_windows(self, window_tokens, num_batches, window_size_hw):
        
        # For convenience
        # (N is number of tokens, C is features per token and P = batch_size * num_window_partitions)
        P, N, C = window_tokens.shape
        
        # Convert shape: PxNx(3F) -> PxNx3xHxf -> 3xPxHxNxf, so that we can split the q/k/v components
        qkv = self.qkv(window_tokens)
        qkv = qkv.reshape(P, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Add bias terms (we don't do this as part of the linear layer, since the k-tokens don't have a bias!)
        # -> bias has shape: PxHxNxf
        q = q + self.q_bias
        v = v + self.v_bias
        
        # Cosine attention with relative positional encodings
        # -> Shape after attention: PxHxNxN
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        attn = attn * self.logit_scale
        attn = self.relpos_enc(attn, window_size_hw)
        
        # Apply window mask to account for shifting, if needed. Doesn't affect shape (PxHxNxN)
        attn = self.windowing.add_mask(attn, num_batches)
        
        # Generate new tokens from weighted value tokens & reshape to match original input token shape
        # Shape: PxHxNxN @ PxHxNxf = PxHxNxf. Final output is shaped: PxNxC (same as original input)
        value_weighting = self.softmax(attn)
        output = (value_weighting @ v).transpose(1, 2).reshape(P, N, C)
        output = self.proj(output)
        
        return output
    
    # .................................................................................................................


class Windowing(nn.Module):
    
    '''
    Class used to handle windowing functionality for SwinV2
    More specifically, this class is responsible for taking image-like tensors
    and splitting them into window 'tiles', which are smaller groupings of 'pixels'
    that are processed together using self attention. By splitting into these smaller
    windows, the attention calculation can run faster (due to having fewer tokens!).
    
    This module is also responsible for the 'shifting' used by Swin models before windowing,
    as well as generating & applying a shift-mask to attention results to prevent attention
    between window segments that are not adjacent in the original input, but which end
    up side-by-side due to the use of cyclic shifts.
    
    Note, this 'module' does not include any learnable parameters! It is a module purely
    to make sure that masking makes use of the correct device at runtime.
    '''
    
    # .................................................................................................................
    
    def __init__(self, window_size_hw, is_shift_block = False):
        
        # Inherit from parent
        super().__init__()
        
        # Store config
        self.target_window_size_hw = window_size_hw
        self.is_shift_block = is_shift_block
        
        # Set up buffers for holding mask & device/dtype information
        self.register_buffer("attn_mask", None, persistent=False)
        self.register_buffer("device_tensor", torch.tensor(0.0), persistent=False)
        
        # Allocate storage for windowing/shifting settings
        # -> These depend on patch sizing, which we don't assume in advance!
        self._need_shift = False
        self._grid_h, self._grid_w = (0, 0)
        self._actual_window_size_hw = (0,0)
        self._window_area = 0
        self.shift_hw = (0, 0)
        self.reverse_shift_hw = (0, 0)

    # .................................................................................................................
    
    def add_mask(self, attention_tensor, num_batches):
        
        ''' Applies shift mask (if needed) to account for shifted-windowing '''
        
        # Apply masking to account for shifted-windows, if needed
        if self._need_shift:
            attention_tensor += self.attn_mask.repeat(num_batches, 1, 1, 1)
        
        return attention_tensor
    
    # .................................................................................................................
    
    def partition(self, image_tokens_bhwc, window_size_hw):
        
        '''
        Function resposible for taking image-like tokens and splitting them
        into smaller 'windows' (tiles of the original image-like shape)
        '''

        # For convenience
        C = image_tokens_bhwc.shape[-1]
        window_area = window_size_hw[0] * window_size_hw[1]
        
        # Apply shifting for swin, if needed
        if self._need_shift:
            image_tokens_bhwc = torch.roll(image_tokens_bhwc, shifts=self.shift_hw, dims=(1, 2))
        
        # Convert image-like tokens into window token (tiles) of the image
        window_tokens, num_win_xy = image_to_windows(image_tokens_bhwc, window_size_hw)
        window_tokens = window_tokens.view(-1, window_area, C)
        
        return window_tokens, num_win_xy
    
    # .................................................................................................................
    
    def reverse(self, window_tokens, window_size_hw, num_windows_xy, orig_image_shape_bhwc):
        
        '''
        Function responsible for taking window tokens and re-arranging them
        into an image-like format, to reverse the effects of window partitioning.
        '''
        
        # For convenience
        B, H, W, C = orig_image_shape_bhwc
        win_h, win_w = window_size_hw
        
        # Merge windows back into image-like tokens (i.e. undo partitioning step)
        window_tokens = window_tokens.view(-1, win_h, win_w, C)
        image_tokens_bhwc = windows_to_image(window_tokens, window_size_hw, num_windows_xy, orig_image_shape_bhwc)
        
        # Undo prior shifting, if needed
        if self._need_shift:
            image_tokens_bhwc = torch.roll(image_tokens_bhwc, shifts=self.reverse_shift_hw, dims=(1, 2))
        
        return image_tokens_bhwc
    
    # .................................................................................................................
    
    def resize(self, patch_grid_hw):
        
        grid_h, grid_w = patch_grid_hw
        need_sizing_update = (grid_h != self._grid_h) or (grid_w != self._grid_w)
        if need_sizing_update:
        
            # Get appropriate window/shift sizing for given patch grid size
            window_size_hw, shift_hw = adjust_window_and_shift_sizes(patch_grid_hw, self.target_window_size_hw)
            
            # Update shifting
            (shift_h, shift_w) = shift_hw
            is_shifting = (shift_h > 0) or (shift_w > 0)
            self.shift_hw = (-shift_h, -shift_w)
            self.reverse_shift_hw = (shift_h, shift_w)
            self._need_shift = is_shifting and self.is_shift_block
            
            # Re-build mask, if we have new shifting/patch sizing
            if self._need_shift:
                device = self.device_tensor.device
                dtype = self.device_tensor.dtype
                self.attn_mask = make_shift_mask(patch_grid_hw, window_size_hw, shift_hw, device, dtype)
            
            # Record new window & grid sizing for future checks
            self._actual_window_size_hw = window_size_hw
            self._window_area = window_size_hw[0] * window_size_hw[1]
            self._grid_h = grid_h
            self._grid_w = grid_w
        
        return self._actual_window_size_hw
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def image_to_windows(image_like_bhwc, window_size_hw):
    
    """
    Function which reshapes a given image-like tensor, of shape: B x H x W x C
    into smaller image-like tensors (windows) which tile the original input
    and have shape: NB x win_h x win_w x C
    where NB is the number of windows times the batch size.
    
    Note that the windows must be sized to perfectly tile the original image!
    This means that the given window height & width must divide into the 
    input image size height & width an integer number of times.
    
    Original implementation from timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/swin_transformer_v2.py
    
    Returns:
        window_tiles (shape: NB x win_h x win_w x C)
        -> Where NB is number of windows times batch size
    """
    
    B, H, W, C = image_like_bhwc.shape
    win_h, win_w = window_size_hw
    
    # Figure out how many window 'tiles' fit in the image along x/y directions
    # (Note: win_w/_h correspond to how many 'pixels' are in the windows, not the number of windows)
    num_win_y = H // win_h
    num_win_x = W // win_w
    NB = num_win_y * num_win_x * B
    num_win_xy = (num_win_x, num_win_y)
    
    image_like_bhwc = image_like_bhwc.view(B, num_win_y, win_h, num_win_x, win_w, C)
    windows = image_like_bhwc.permute(0, 1, 3, 2, 4, 5).contiguous().view(NB, win_h, win_w, C)
    
    return windows, num_win_xy

# .....................................................................................................................

def windows_to_image(window_tokens, window_size_hw, num_windows_yx, orig_image_shape_bhwc):
    
    """
    Function which reverses the effect of window partitioning. Takes in window tokens,
    which are small image-like tensors and converts them back into a larger image-like
    tensor from which they originated (the windows are expected to be 'tiles' of the original image).
    
    Original implementation from timm library:
        @ https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/models/swin_transformer_v2.py

    Returns:
        image_like_bhwc (shape: B x H x W x C)
        -> Where B is number of original image-like batches and
           HxWxC is original image shape
    """
    
    B, H, W, C = orig_image_shape_bhwc
    win_h, win_w = window_size_hw
    num_win_x, num_win_y = num_windows_yx
    
    # Figure out how many window tiles fit in the image along x/y directions
    # (Note: win_w/_h correspond to how many 'pixels' are in the windows, not the number of windows)
    # num_win_y = H // win_h
    # num_win_x = W // win_w
    
    image_like_bhwc = window_tokens.view(B, num_win_y, num_win_x, win_h, win_w, -1)
    image_like_bhwc = image_like_bhwc.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    
    return image_like_bhwc

# .....................................................................................................................

def adjust_window_and_shift_sizes(patch_grid_hw, target_window_size_hw, needs_shift = True):
    
    '''
    Function which determines the correct window and shift sizing (if any) to use
    for a given target window size & patch grid sizing. For example, if the window sizing
    is larger than the patch grid size itself, then the window size is adjusted (shrinks) to match
    the patch grid size.
    
    Returns:
        is_shifting (bool), shift_hw, win_hw
    '''
    
    # For convenience
    patch_h, patch_w = patch_grid_hw
    targ_h, targ_w = target_window_size_hw
    
    # Downsize the window sizing if it's bigger than the patch grid
    win_h = min(targ_h, patch_h)
    win_w = min(targ_w, patch_w)
    
    # Force window size to be an integer multiple of the patch grid sizing
    # -> This is needed to make sure we can 'tile' the patch grid with windows of size win_h x win_w
    h_doesnt_tile = (patch_h % win_h) != 0
    if h_doesnt_tile:
        h_divisors = [div for div in range(win_h // 2, 2*win_h) if (patch_h % div) == 0]
        win_h = min(h_divisors, key=lambda div: abs(patch_h - div))
    w_doesnt_tile = (patch_w % win_w) != 0
    if w_doesnt_tile:
        w_divisors = [div for div in range(win_w // 2, 2*win_w) if (patch_w % div) == 0]
        win_w = min(w_divisors, key=lambda div: abs(patch_w - div))
    
    # Shift half the window size if we have a big enough patch grid, otherwise don't shift at all
    targ_shift_h, targ_shift_w = win_h // 2, win_w // 2
    shift_h = 0 if patch_h <= win_h else targ_shift_h
    shift_w = 0 if patch_w <= win_w else targ_shift_w
    
    # Bundle outputs
    shift_hw = (shift_h, shift_w)
    win_hw = (win_h, win_w)
    
    return win_hw, shift_hw

# .....................................................................................................................

def make_shift_mask(patch_grid_hw, window_size_hw, shift_hw, device = None, dtype = None):
    
    '''
    Function used to generate a mask applied to shifted-window attention results.
    This is done is order to prevent attention between sections of the input that
    are on opposite sides, but become adjacent (and end up in the same window)
    due to the cyclic shifting that is used.
    
    This mask consists of 0 and -100 entries only. When added to a computed 
    attention matrix and followed by a softmax, this has the effect of
    'knocking out' entries that have -100 in them, so that they do not affect
    the weighting on value tokens.
    '''
    
    # For convenience
    img_h, img_w = patch_grid_hw
    win_h, win_w = window_size_hw
    shift_h, shift_w = shift_hw
    window_area = win_h * win_w
    
    # Don't create a shift mask if we're not shifting!
    need_shift = (shift_h > 0) or (shift_w > 0)
    if not need_shift:
        return None
    
    # calculate attention mask for SW-MSA
    img_mask = torch.zeros((1, img_h, img_w, 1), device = device, dtype = dtype)
    cnt = 0
    h_slices = [slice(0, -win_h), slice(-win_h, -shift_h), slice(-shift_h, None)]
    w_slices = [slice(0, -win_w), slice(-win_w, -shift_w), slice(-shift_w, None)]
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1
    
    mask_windows, _ = image_to_windows(img_mask, window_size_hw)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_area)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask.unsqueeze(1)
