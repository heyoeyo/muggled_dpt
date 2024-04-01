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
    
    def inference(self, image_rgb_normalized_bchw: Tensor) -> Tensor:
        
        ''' Helper function used to call model without recording gradients '''
        
        with torch.inference_mode():
            inverse_depth_tensor = self(image_rgb_normalized_bchw)
        
        return inverse_depth_tensor
    
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


class DPTImagePrep:
    
    '''
    Image pre-/post-processor for DPT models.
    
    This class has many helper function to deal with manipulating image data.
    However, the most basic use is simply to prepare images for input into the DPT model.
    As an example:
        
        # Set up image prep for model base & patch sizing (this can vary by model)
        imgprep = DPTImagePrep(base_size_px = 512, patch_size_px = 16)
        
        # Load image and prepare for input into DPT model
        image = cv2.imread("path/to/image.jpg")
        image_tensor = imgprep.prepare_image_bgr(image)
        
        # Run DPT model (assuming it has been loaded already!)
        depth_prediction = dpt_model(image_tensor)
    '''
    
    # .................................................................................................................
    
    def __init__(self, base_size_px, patch_size_px, to_multiples,
                 rgb_mean = (0.5, 0.5, 0.5), rgb_std = (0.5, 0.5, 0.5)):
        
        # Store image sizing info
        self._to_multiples = int(round(to_multiples))
        self.base_size_px = self.set_base_size(base_size_px)
        
        # Store mean & standard deviation normalization values
        self._image_rgb_mean = np.float32(rgb_mean)
        self._image_rgb_std = np.float32(rgb_std)
    
    # .................................................................................................................
    
    def set_base_size(self, override_size_px):
        
        '''
        Function used to alter the preprocessor base sizing. This causes the model
        to run at a lower/higher resolution
        '''
        
        self.base_size_px = round(override_size_px / self._to_multiples) * self._to_multiples
        return self.base_size_px
    
    # .................................................................................................................
    
    def prepare_image_bgr(self, image_bgr: NDArray, force_square = False) -> Tensor:
        
        '''
        Function used to pre-process input (BGR, the opencv default) images for use in DPT model.
        Assumes input images of shape: HxWxC, with BGR ordering (the default when using opencv).
        
        This function performs the necessary scaling/normalizing/dimension ordering needed
        to convert an input (opencv) image into the format needed by the DPT model.
        The output of this function has the following properties:
            - is a torch tensor
            - has dimension ordering: BxCxHxW
            - has mean near 0, standard deviation near 1
            - height & width are scaled to match model requirements
        '''
        
        # Scale image to size acceptable by the dpt model
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if force_square:
            scaled_img = self.scale_to_square_side_length(image_rgb, self.base_size_px)
        else:
            scaled_img = self.scale_to_max_side_length(image_rgb, self.base_size_px, self._to_multiples)
        
        # Normalize image values, between -1 and +1 and switch dimension ordering: HxWxC -> CxHxW
        img_norm = (np.float32(scaled_img / 255.0) - self._image_rgb_mean) / self._image_rgb_std
        img_norm = np.transpose(img_norm, (2, 0, 1))
        
        # Convert to tensor with unit batch dimension. Shape goes from: CxHxW -> 1xCxHxW
        return torch.from_numpy(img_norm).unsqueeze(0)

    # .................................................................................................................
    
    @staticmethod
    def scale_prediction(prediction_tensor: Tensor, target_wh, interpolation="bilinear") -> Tensor:
        
        ''' Helper used to scale raw depth prediction. Assumes input is of shape: BxHxW '''
        
        target_hw = (int(target_wh[1]), int(target_wh[0]))
        return tensor_resize(prediction_tensor.unsqueeze(1), size=target_hw, mode=interpolation).squeeze(1)

    # .................................................................................................................
    
    @staticmethod
    def scale_to_square_side_length(image_bgr: NDArray, side_length_px, to_multiples = None) -> NDArray:
        
        if to_multiples is not None:
            side_length_px = int(side_length_px // to_multiples) * to_multiples
        
        return cv2.resize(image_bgr, dsize = (int(side_length_px), int(side_length_px)))
    
    # .................................................................................................................
    
    @staticmethod
    def scale_to_max_side_length(image_bgr: NDArray, max_side_length_px = 800, to_multiples = None) -> NDArray:
        
        '''
        Helper used to scale an image to a target maximum side length. The other side of the image
        is scaled to preserve the image aspect ratio (within rounding error).
        
        Optionally, the image dimensions can be forced to be integer multiples of a 'to_multiples' value.
        
        Expects opencv (numpy array) image with dimension ordering of HxWxC
        '''
        
        # For convenience
        in_h, in_w = image_bgr.shape[0:2]
        
        # Figure out how to scale image to get target max side length and maintain aspect ratio
        input_max_side = max(in_h, in_w)
        scale_factor = max_side_length_px / input_max_side
        scaled_w = int(round(in_w * scale_factor))
        scaled_h = int(round(in_h * scale_factor))
        
        # Force sizes to be multiples of a given number, if needed
        if to_multiples is not None:
            scaled_w = int(round(scaled_w // to_multiples) * to_multiples)
            scaled_h = int(round(scaled_h // to_multiples) * to_multiples)
        
        return cv2.resize(image_bgr, dsize = (scaled_w, scaled_h))
    
    # .................................................................................................................
    
    @staticmethod
    def remove_infinities(data: Tensor | NDArray, inf_replacement_value = 0.0, in_place = True) -> Tensor | NDArray:
        
        '''
        Helper used to remove +/- inf values, which can sometimes be output
        by the DPT model, especially when using reduced precision dtypes!
        Works on pytorch tensors and numpy arrays
        
        Returns:
            data_with_inf_removed
        '''
        
        try:
            # Works with pytorch tensors
            inf_mask = data.isinf()
            data = data if in_place else data.clone()
            
        except AttributeError:
            # Works with numpy arrays
            inf_mask = np.isinf(data)
            data = data if in_place else data.copy()
        
        # Replace infinity values
        data[inf_mask] = inf_replacement_value
        return data
    
    # .................................................................................................................
    
    @staticmethod
    def normalize_01(data: Tensor | NDArray) -> Tensor | NDArray:
        
        '''
        Helper used to normalize depth prediction, to 0-to-1 range.
        Works on pytorch tensors and numpy arrays
        
        Returns:
            depth_normalized_0_to_1
        '''
        
        pred_min = data.min()
        pred_max = data.max()
        return (data - pred_min) / (pred_max - pred_min)
    
    # .................................................................................................................
    
    @classmethod
    def convert_to_uint8(cls, depth_prediction_tensor: Tensor) -> Tensor:
        
        '''
        Helper used to convert depth prediction into 0-255 uint8 range,
        used when displaying result as an image.
        
        Note: The result will also still be on the device as a tensor
        Returns:
            depth_as_uint8_tensor
        '''
        
        return (255.0 * cls.normalize_01(depth_prediction_tensor)).byte()
    
    # .................................................................................................................
