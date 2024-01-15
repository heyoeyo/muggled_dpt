#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np

import torch
from torch.nn.functional import interpolate as tensor_resize


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DPTImagePrep:
    
    '''
    Image pre-/post-processor for MiDaS v3.1 BeiT DPT model.
    These models (BeiT-large-512, BeiT-base-384, etc.) support input images
    of varying sizes, as long as the width & height are both divisible by the model patch size times 2
    -> Factor of 2 is needed to allow downsampling (by factor of 2) of image patch grid, which is done
       in the reassembly part of the model!
    '''
    
    # Hard-coded mean & standard deviation normalization values
    _image_rgb_mean = np.float32((0.5,0.5,0.5))
    _image_rgb_std = np.float32((0.5,0.5,0.5))
    
    # .................................................................................................................
    
    def __init__(self, base_size_px, patch_size_px = 16):
        self.base_size_px = base_size_px
        self._to_multiples = int(2 * patch_size_px)
    
    # .................................................................................................................
    
    def __call__(self, image_bgr):
        return self.prepare_image_bgr(image_bgr)
    
    # .................................................................................................................
    
    def prepare_image_bgr(self, image_bgr, force_square = False):
        
        '''
        Function used to pre-process input (BGR, the opencv default) images for use in DPT model.
        Assumes input images of shape: HxWxC, with BGR ordering (the default when using opencv).
        
        This function performs the necessary scaling/normalizing/dimension ordering needed
        to convert an input (opencv) image into the format needed by the DPT model.
        The output of this function has the following properties:
            - is a torch tensor
            - has values normalizing between -1 and +1
            - has dimension ordering: BxCxHxW
            - height & width are scaled to match model requirements
        '''
        
        # Scale image to size acceptable by the dpt model
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if force_square:
            scaled_img = self.scale_to_square_side_length(image_rgb, self.base_size_px)
        else:
            scaled_img = self.scale_to_min_side_length(image_rgb, self.base_size_px, self._to_multiples)
        
        # Normalize image values, between -1 and +1 and switch dimension ordering: HxWxC -> CxHxW
        img_norm = (np.float32(scaled_img / 255.0) - self._image_rgb_mean) / self._image_rgb_std
        img_norm = np.transpose(img_norm, (2, 0, 1))
        
        # Convert to tensor with unit batch dimension. Shape goes from: CxHxW -> 1xCxHxW
        return torch.from_numpy(img_norm).unsqueeze(0)
    
    # .................................................................................................................
    
    def override_base_size(self, override_size_px):
        
        '''
        Function used to alter the preprocessor base sizing. This causes the model
        to run at a lower/higher resolution (by scaling up position encodings).
        Mostly for experimentation, usually makes things worse
        '''
        
        self.base_size_px = round(override_size_px / self._to_multiples) * self._to_multiples
        return self.base_size_px

    # .................................................................................................................
    
    @staticmethod
    def scale_prediction(prediction_tensor, target_wh, interpolation="bilinear"):
        
        ''' Helper used to scale raw depth prediction. Assumes input is of shape: BxHxW '''
        
        target_hw = (int(target_wh[1]), int(target_wh[0]))
        return tensor_resize(prediction_tensor.unsqueeze(1), size=target_hw, mode=interpolation).squeeze(1)

    # .................................................................................................................
    
    @staticmethod
    def scale_to_square_side_length(image_bgr, side_length_px):
        return cv2.resize(image_bgr, dsize = (int(side_length_px), int(side_length_px)))
    
    # .................................................................................................................
    
    @staticmethod
    def scale_to_min_side_length(image_bgr, min_side_length_px, to_multiples = None):
        
        '''
        Helper used to scale an image to a target minimum side length. The other side of the image
        is scaled to preserve the image aspect ratio (within rounding error).
        Optionally, the image dimensions can be forced to be integer multiples of a 'to_multiples' value.
        Expects opencv (numpy array) image with dimension ordering of HxWxC
        '''
        
        # For convenience, grab input image dimensions
        in_h, in_w = image_bgr.shape[0:2]
        
        # Figure out how to scale width/height to get the target minimum side length
        input_min_side = min(in_h, in_w)
        scale_factor = min_side_length_px / input_min_side
        scaled_w = (in_w * scale_factor)
        scaled_h = (in_h * scale_factor)
        
        # Force sizes to be multiples of a given number, if needed
        if to_multiples is not None:
            scaled_w = int(scaled_w // to_multiples) * to_multiples
            scaled_h = int(scaled_h // to_multiples) * to_multiples
        
        scaled_wh = (int(scaled_w), int(scaled_h))
        return cv2.resize(image_bgr, dsize = scaled_wh)
    
    # .................................................................................................................
    
    @staticmethod
    def scale_to_max_side_length(image_bgr, max_side_length_px = 800):
        
        '''
        Helper used to scale an image to a target maximum side length. The other side of the image
        is scaled to preserve the image aspect ratio (within rounding error).
        Expects opencv (numpy array) image with dimension ordering of HxWxC
        '''
        
        # For convenience
        in_h, in_w = image_bgr.shape[0:2]
        
        # Figure out how to scale image to get target max side length and maintain aspect ratio
        max_side = max(in_h, in_w)
        scale_factor = max_side_length_px / max_side
        scaled_w = int(round(in_w * scale_factor))
        scaled_h = int(round(in_h * scale_factor))
        
        return cv2.resize(image_bgr, dsize = (scaled_w, scaled_h))
    
    # .................................................................................................................
    
    @classmethod
    def normalize_01(cls, tensor_or_array):
        
        '''
        Helper used to normalize depth prediction, to 0-to-1 range.
        Also handles infinity values (can result when using lower float precision)
        Returns:
            depth_normalized_0_to_1
        '''
        
        # Handle negative infinities (for pytorch or numpy input)
        pred_min = tensor_or_array.min()
        try:                   is_neginf = pred_min.isinf()
        except AttributeError: is_neginf = np.isinf(pred_min)
        if is_neginf:
            tensor_or_array[tensor_or_array == pred_min] = 0
            return cls.normalize_01(tensor_or_array)
        
        # Handle positive infinities
        pred_max = tensor_or_array.max()
        try:                   is_posinf = pred_max.isinf()
        except AttributeError: is_posinf = np.isinf(pred_max)
        if is_posinf:
            tensor_or_array[tensor_or_array == pred_max] = 0
            return cls.normalize_01(tensor_or_array)
        
        return (tensor_or_array - pred_min) / (pred_max - pred_min)
    
    # .................................................................................................................
    
    @classmethod
    def convert_to_uint8(cls, depth_prediction_tensor):
        
        '''
        Helper used to convert depth prediction into 0-255 uint8 range,
        used when displaying result as an image.
        
        Note: The result will also still be on the device as a tensor
        Returns:
            depth_as_uint8_tensor
        '''
        
        return (255.0 * cls.normalize_01(depth_prediction_tensor)).byte()
    
    # .................................................................................................................
    
    @staticmethod
    def apply_colormap(image_uint8_1ch, opencv_colormap_code = None):
        
        '''
        Converts a uint8 image (numpy array) into a bgr color image using opencv colormaps
        Expects an image of shape: HxWxC (with 1 or no channels, i.e. HxW only)
        Colormap code should be from opencv, which are accessed with: cv2.COLORMAP_{name}
        If the colormap code is None, then a grayscale (3ch) image is returned
        '''
        
        # Special case, if no colormap code is given, return 3ch grayscale image
        if opencv_colormap_code is None:
            return cv2.cvtColor(image_uint8_1ch, cv2.COLOR_GRAY2BGR)
        
        return cv2.applyColorMap(image_uint8_1ch, opencv_colormap_code)
    
    # .................................................................................................................
