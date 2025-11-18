#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

import torch
import torch.nn as nn

# For type hints
from numpy import ndarray
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class PatchEmbed(nn.Module):
    """
    Simplified implementation of the patch embedding step for:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413

    Based on code from Depth-Anything/facebookresearch:
        @ https://github.com/LiheYoung/Depth-Anything/blob/e7ef4b4b7a0afd8a05ce9564f04c1e5b68268516/torchhub/facebookresearch_dinov2_main/dinov2/layers/patch_embed.py#L26

    Purpose is to take input images and convert them to 'lists' of (1D) tokens,
    one token for each (14x14 default) image patch.
    """

    # Set hard-coded mean/std normalization for input images
    rgb_offset = (0.485, 0.456, 0.406)
    rgb_stdev = (0.229, 0.224, 0.225)

    # .................................................................................................................

    def __init__(
        self,
        features_per_token: int,
        patch_size_px: int = 14,
        default_image_size: int = 518,
        num_input_channels: int = 3,
        bias: bool = True,
    ):

        # Inherit from parent
        super().__init__()

        # Both grouping + linear transformation is handled with a single strided convolution step!
        self.proj = nn.Conv2d(
            num_input_channels,
            features_per_token,
            kernel_size=patch_size_px,
            stride=patch_size_px,
            bias=bias,
        )

        # Store expected size of input images (e.g. size to scale to when handling images)
        # -> Tiling size is a constraint on the image input sizing, based on the need for
        #    having a whole number of patches (for patch embedding) while also having
        #    the number of patches be divisble by 2, due to downscaling used within the model
        self._default_size_px = round(default_image_size)
        self._tiling_size = round(2 * patch_size_px)

        # Store rgb scaling factors, for preparing input images
        self.register_buffer("mean_rgb", torch.tensor(self.rgb_offset).view(-1, 1, 1), persistent=False)
        self.register_buffer("stdev_scale_rgb", 1.0 / torch.tensor(self.rgb_stdev).view(-1, 1, 1), persistent=False)

    # .................................................................................................................

    def forward(self, image_tensor_bchw: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """
        Projects & reshapes image tensor: BxCxHxW -> BxNxF
            -> Where B is batch size
            -> C is image channels (i.e. 3 for RGB image)
            -> H, W are the height & width of the image
            -> N is the number of tokens (equal to number of image patches)
            -> F is the number of features per token

        Returns:
            patch_tokens_bnf, patch_grid_hw
        """

        # Convert image width/height to patch grid width/height, and image channels to feature count
        # -> result has shape: BxFxhxw, where F is features per token, h & w are the patch grid height & width
        patch_tokens_bfhw = self.proj(image_tensor_bchw)

        # Convert from image-like shape to 'rows of tokens' shape
        # -> result has shape: BxNxF
        patch_grid_hw = patch_tokens_bfhw.shape[2:]
        patch_tokens_bnf = patch_tokens_bfhw.flatten(2).transpose(1, 2)

        return patch_tokens_bnf, patch_grid_hw

    # .................................................................................................................

    def prepare_image(
        self,
        image_bgr: ndarray,
        max_side_length: int | None = None,
        use_square_sizing: bool = True,
        interpolation_mode: str = "bilinear",
    ) -> Tensor:
        """
        Helper used to convert opencv-formatted images (e.g. from loading: cv2.imread(path_to_image))
        into the format needed by the patch embedding model (includes scaling and RGB normalization steps)
        Returns:
            image_as_tensor_bchw
        """

        # Fill in missing max side length
        if max_side_length is None:
            max_side_length = self._default_size_px

        # Figure out scaling factor to get target side length
        img_h, img_w = image_bgr.shape[0:2]
        largest_side = max(img_h, img_w)
        scale_factor = max_side_length / largest_side

        # Force sizing to multiples of a specific tiling size
        # -> For example, if image size is 256x256, but we have a 30px tile sizing,
        #    we need to scale to the nearest multiple of 30: 270x270
        targ_hw = (largest_side, largest_side) if use_square_sizing else (img_h, img_w)
        scaled_hw = [max(1, round(side * scale_factor / self._tiling_size)) * self._tiling_size for side in targ_hw]

        # Scale RGB image to correct size and re-order from HWC to BCHW (with batch of 1)
        device, dtype = self.mean_rgb.device, self.mean_rgb.dtype
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor_chw = torch.tensor(np.transpose(image_rgb, (2, 0, 1)), device=device, dtype=dtype)
        image_tensor_bchw = nn.functional.interpolate(
            image_tensor_chw.unsqueeze(0),
            size=scaled_hw,
            align_corners=False,
            antialias=True,
            mode=interpolation_mode,
        )

        # Perform mean/scale normalization
        return ((image_tensor_bchw / 255.0) - self.mean_rgb) * self.stdev_scale_rgb

    # .................................................................................................................

    def verify_input(self, image_tensor_bchw: Tensor) -> bool:

        # Assume input is tensor with bchw shape
        b, c, h, w = image_tensor_bchw.shape

        # Check that the input channel count matches our convolution
        targ_c = self.proj.in_channels
        assert c == targ_c, f"Bad channel count! Expected {targ_c} got {c}"

        # Check input image shape
        # -> Needs to be divisble by patch sizing (14)
        # -> Patch grid size itself needs to be divisible by 2 for downscaling
        h_stride, w_stride = self.proj.stride
        assert h % h_stride == 0, f"Bad height! Image must have height ({h}) divisble by {h_stride}"
        assert w % w_stride == 0, f"Bad width! Image must have width ({w}) divisble by {w_stride}"

        return True

    # .................................................................................................................
