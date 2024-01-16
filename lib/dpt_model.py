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
        Depth prediction function. Expects an image of shape BxCxHxW, with RGB ordering.
        Pixel values should be between 0.0 and 1.0. The image dimensions (H & W) should be divisible
        by twice the patch size (e.g. divisible by 32 for default settings).
        Use the 'verify_input(...)' function to test inputs if needed.
        
        Returns single channel inverse-depth 'image' of shape: BxHxW
        '''
        
        # Convert image (shape: BxCxHxW) to patch tokens (shape: BxN'xF)
        # (the number of tokens is 1 less than transformer tokens, since a cls/readout token gets included!)
        patch_tokens, patch_grid_hw = self.patch_embed(image_rgb_normalized_bchw)
        
        # Process tokens with transformer and retrieve results from multiple stages
        # (Each stage has the same shape: BxNxF -> B batches, N tokens, F features per token)
        stage_1, stage_2, stage_3, stage_4 = self.imgencoder.forward(patch_tokens, patch_grid_hw)
        
        # Re-assemble transformer tokens (shape: BxNxF) into image-like tensors (shape: BxFxHxW)
        # (Note each of the stages has a different width & height due to up-/down-scaling)
        upx4, upx2, noscale, downx2 = self.reassemble(stage_1, stage_2, stage_3, stage_4, patch_grid_hw)
        
        # Generate a single (fused) feature map from multi-stage input & project into (1ch) depth image output
        fused_feature_map = self.fusion.forward(upx4, upx2, noscale, downx2)
        depth_image = self.head.forward(fused_feature_map)
        
        # Remove unitary channel dimension (shape goes from: Bx1xHxW -> BxHxW)
        return depth_image.squeeze(dim = 1)
    
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
        
        - Returns True if there are no problems
        - Raises an AssertionError if something is wrong with the input
        '''
        
        # Check image is tensor
        ok_tensor = isinstance(image_rgb_normalized_bchw, torch.Tensor)
        assert ok_tensor, "Image must be a tensor!"
        
        # Check batch dimension
        img_shape = image_rgb_normalized_bchw.shape
        ok_shape = len(img_shape) == 4
        shape_str = "x".join(str(item) for item in img_shape)
        assert ok_shape, "Bad image shape! Image ({}) should have a shape of Bx3XHxW".format(shape_str)
        
        # Check image channels
        B, C, H, W = img_shape
        ok_channels = C == 3
        assert ok_channels, "Bad image channels! Image should have 3 channels: RGB (got {})".format(C)
        
        # Check input image shape
        # -> Needs to be divisble by 16 for patch embedding
        # -> Patch grid size itself needs to be divisible by 2 for downscaling
        ok_height = (H % 32 == 0)
        ok_width = (W % 32 == 0)
        assert (ok_height and ok_width), \
            "Bad width/height! Image must have height ({}) & width ({}) both divisible by 32".format(H, W)
        
        # Check matching device
        img_device = image_rgb_normalized_bchw.device
        model_device = next(self.parameters()).device
        ok_device = img_device == model_device
        assert ok_device, "Device mismatch! Image: {}, model: {}".format(img_device, model_device)
        
        # Check matching data types
        img_dtype = image_rgb_normalized_bchw.dtype
        model_dtype = next(self.parameters()).dtype
        ok_dtype = (img_dtype == model_dtype)
        assert ok_dtype, "Data type mismatch! Image: {}, model: {}".format(img_dtype, model_dtype)
        
        # Check image value normalization
        img_min = image_rgb_normalized_bchw.min()
        img_max = image_rgb_normalized_bchw.max()
        ok_min = torch.abs(img_min + 1) < 0.01
        ok_max = torch.abs(img_max - 1) < 0.01
        if not (ok_min and ok_max):
            print("WARNING:",
                  "Image values are not in expected normalization range!",
                  "Model expects image values to be between 0.0 and 1.0",
                  "Instead got: {:.1f} - {:.1f}".format(img_min, img_max),
                  sep = "\n", flush = True)
        
        return True
    
    # .................................................................................................................
