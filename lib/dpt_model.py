#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn
from torch.nn.functional import interpolate as tensor_resize

# Only Needed for image pre-/post-processing
import cv2
import numpy as np

# For basic type hints
from torch import Tensor
from numpy.typing import NDArray


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DPTModel(nn.Module):
    
    '''
    Simplified implementation of a 'Dense Prediction Transformer' model, described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    Original implementation details come from the MiDaS project repo:
        https://github.com/isl-org/MiDaS
    '''
    
    # .................................................................................................................
    
    def __init__(self,
                 patch_embed_model,
                 image_encoder_4stage_model,
                 reassemble_4stage_model,
                 fusion_4stage_model,
                 head_model):
        
        # Inherit from parent
        super().__init__()
        
        # Store models for use in forward pass
        self.patch_embed = patch_embed_model
        self.imgencoder = image_encoder_4stage_model
        self.reassemble = reassemble_4stage_model
        self.fusion = fusion_4stage_model
        self.head = head_model
        
        # Default to eval mode, expecting to use inference only
        self.eval()
    
    # .................................................................................................................
    
    def forward(self, image_rgb_normalized_bchw: Tensor) -> Tensor:
        
        '''
        Depth prediction function. Expects an image tensor of shape BxCxHxW, with RGB ordering.
        Pixel values should have a mean near 0.0 and a standard-deviation near 1.0
        The height & width of the image need to be compatible with the patch sizing of the model.
        
        Use the 'verify_input(...)' function to test inputs if needed.
        
        Returns single channel inverse-depth 'image' of shape: BxHxW
        '''
        
        # Convert image (shape: BxCxHxW) to patch tokens (shape: BxNpxF)
        patch_tokens, patch_grid_hw = self.patch_embed(image_rgb_normalized_bchw)
        
        # Process patch tokens back into (multi-scale) image-like tensors
        stage_1, stage_2, stage_3, stage_4 = self.imgencoder(patch_tokens, patch_grid_hw)
        reasm_1, reasm_2, reasm_3, reasm_4 = self.reassemble(stage_1, stage_2, stage_3, stage_4, patch_grid_hw)
        
        # Generate a single (fused) feature map from multi-stage input & project into (1ch) depth image output
        fused_feature_map = self.fusion(reasm_1, reasm_2, reasm_3, reasm_4)
        inverse_depth_tensor = self.head(fused_feature_map)
        
        return inverse_depth_tensor
    
    # .................................................................................................................
    
    def inference(self,
                  image_bgr: NDArray,
                  max_side_length: int | None = None,
                  use_square_sizing: bool = True) -> Tensor:
        
        '''
        Helper function used to run the model with built-in image preprocessing and without recording gradients.
        Expects images loaded from opencv:
            image_bgr = cv2.imread(image_path)
        
        To process an image tensor directly (e.g. for batching), use the forward method of this model.
        This is accessible by 'calling' the model directly, for example: model(image_tensor)
        
        Returns:
            inverse_depth_tensor (shape 1xHxW)
        '''
        
        with torch.inference_mode():
            img_tensor_bchw = self.patch_embed.prepare_image(image_bgr, max_side_length, use_square_sizing)
            inverse_depth_tensor = self(img_tensor_bchw)
        
        return inverse_depth_tensor
    
    # .................................................................................................................
    
    def prepare_image_bgr(self,
                          image_bgr: NDArray,
                          max_side_length: int | None = None,
                          use_square_sizing: bool = True,
                          interpolation_mode="bilinear") -> Tensor:
        
        '''
        Helper function which performs image pre-processing needed to convert an
        image loaded from opencv (in bgr order) into a tensor that is properly
        sized and normalized for use with the DPT model.
        Note that this is automatically called when using the 'inference(...)' function!
        
        Returns:
            image_tensor_bchw
        '''
        return self.patch_embed.prepare_image(image_bgr, max_side_length, use_square_sizing, interpolation_mode)
    
    # .................................................................................................................
    
    def verify_input(self, image_rgb_normalized_bchw: Tensor) -> bool:
        
        '''
        Helper function used to verify that the input image into the model is correctly
        formatted for inference. Note this function is meant as a debugging tool, it should
        not be used before every inference, as it does a lot of unnecessary work (assuming input is ok)
        
        - Raises an AssertionError if something is wrong with the input
        - Returns True if there are no problems
        '''
        
        # Check image is in tensor format
        ok_tensor = isinstance(image_rgb_normalized_bchw, torch.Tensor)
        assert ok_tensor, "Image must be provided as a tensor!"
        
        # Check matching device & data types
        img_device = image_rgb_normalized_bchw.device
        model_device = next(self.parameters()).device
        ok_device = (img_device == model_device)
        assert ok_device, "Device mismatch! Image: {}, model: {}".format(img_device, model_device)
        img_dtype = image_rgb_normalized_bchw.dtype
        model_dtype = next(self.parameters()).dtype
        ok_dtype = (img_dtype == model_dtype)
        assert ok_dtype, "Data type mismatch! Image: {}, model: {}".format(img_dtype, model_dtype)
        
        # Check that the shape has 4 terms (meant to be BxCxHxW)
        img_shape = image_rgb_normalized_bchw.shape
        ok_shape = len(img_shape) == 4
        shape_str = "x".join(str(item) for item in img_shape)
        assert ok_shape, "Bad image shape! Image ({}) should have a shape of BxCXHxW".format(shape_str)
        
        # Also run verification from first stage of processing, which varies by model implementation
        ok_patch_verify = self.patch_embed.verify_input(image_rgb_normalized_bchw)
        
        return ok_patch_verify
    
    # .................................................................................................................
