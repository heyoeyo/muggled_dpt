#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import torch.nn as nn


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
    
    def forward(self, image_rgb_normalized_bchw):
        
        '''
        Depth prediction function. Expects an image tensor of shape BxCxHxW, with RGB ordering.
        Pixel values should have a mean near 0.0 and a standard-deviation near 1.0
        The height & width of the image need to be compatible with the patch sizing of the model.
        
        Use the 'verify_input(...)' function to test inputs if needed.
        
        Returns single channel inverse-depth 'image' of shape: BxHxW
        '''
        
        # Convert image (shape: BxCxHxW) to patch tokens (shape: BxNpxF)
        patch_tokens, patch_grid_hw = self.patch_embed(image_rgb_normalized_bchw)
        
        # Process tokens with transformer and retrieve results from multiple stages (shape: BxNxF)
        stage_1, stage_2, stage_3, stage_4 = self.imgencoder(patch_tokens, patch_grid_hw)
        
        # Convert transformer tokens into image-like tensors (shape: BxFxH'xW')
        reasm_1, reasm_2, reasm_3, reasm_4 = self.reassemble(stage_1, stage_2, stage_3, stage_4, patch_grid_hw)
        
        # Generate a single (fused) feature map from multi-stage input & project into (1ch) depth image output
        fused_feature_map = self.fusion(reasm_1, reasm_2, reasm_3, reasm_4)
        depth_image = self.head(fused_feature_map)
        
        return depth_image
    
    # .................................................................................................................
    
    @torch.inference_mode()
    def inference(self, image_rgb_normalized_bchw):
        return self(image_rgb_normalized_bchw)
    
    # .................................................................................................................
    
    def verify_input(self, image_rgb_normalized_bchw):
        
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
        
        # Check batch dimension
        img_shape = image_rgb_normalized_bchw.shape
        ok_shape = len(img_shape) == 4
        shape_str = "x".join(str(item) for item in img_shape)
        assert ok_shape, "Bad image shape! Image ({}) should have a shape of BxCXHxW".format(shape_str)
        
        # Also run verification from first stage of processing, which varies by model implementation
        ok_patch_verify = self.patch_embed.verify_input(image_rgb_normalized_bchw)
        
        return ok_patch_verify
    
    # .................................................................................................................
