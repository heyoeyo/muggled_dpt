#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class CheckerPattern:
    """
    Class used to draw a checker pattern as the background of images
    that are either masked or have transparency (i.e. alpha channels)
    """

    # .................................................................................................................

    def __init__(self, checker_size_px=64, brightness_pct=75, contrast_pct=35, flipped=False):

        # Force percent values to be 0-to-100
        brightness_pct = min(abs(brightness_pct), 100)
        contrast_pct = min(abs(contrast_pct), 100)

        # Figure out the tile brightness (uint8) values
        mid_color_uint8 = 255 * brightness_pct / 100
        max_diff_uint8 = min(255 - mid_color_uint8, mid_color_uint8)
        real_diff_uint8 = max_diff_uint8 * contrast_pct / 100
        color_a = round(max(min(mid_color_uint8 - real_diff_uint8, 255), 0))
        color_b = round(max(min(mid_color_uint8 + real_diff_uint8, 255), 0))
        if flipped:
            color_a, color_b = color_b, color_a

        # Create the base pattern
        base_wh = (checker_size_px, checker_size_px)
        base_pattern = np.uint8(((color_a, color_b), (color_b, color_a)))
        base_pattern = cv2.resize(base_pattern, dsize=base_wh, interpolation=cv2.INTER_NEAREST_EXACT)
        self._base: ndarray = base_pattern
        self._full_pattern: ndarray = cv2.cvtColor(self._base.copy(), cv2.COLOR_GRAY2BGR)

    # .................................................................................................................

    def __repr__(self):
        name = self.__class__.__name__
        color_a = self._base[0, 0]
        color_b = self._base[0, -1]
        return f"{name} ({color_a} | {color_b})"

    # .................................................................................................................

    def draw_like(self, image_uint8: ndarray) -> ndarray:
        """Draw a full checker pattern matching the shape of the given image"""
        img_h, img_w = image_uint8.shape[0:2]
        return self.draw(img_h, img_w)

    # .................................................................................................................

    def draw(self, image_h, image_w) -> ndarray:
        """
        Draw a full checker pattern of the given size
        - Resulting image is cached, so repeat calls at the same size are fast
        """

        # Re-draw the full pattern if the render size doesn't match
        curr_h, curr_w = self._full_pattern.shape[0:2]
        if curr_h != image_h or curr_w != image_w:

            # Figure out how much to pad to fit target shape
            base_h, base_w = self._base.shape[0:2]
            x_pad = max(image_w - base_w, 0)
            y_pad = max(image_h - base_h, 0)

            # Make fully sized pattern but duplicating the base pattern
            l, t = x_pad // 2, y_pad // 2
            r, b = x_pad - l, y_pad - t
            pattern = cv2.copyMakeBorder(self._base, t, b, l, r, cv2.BORDER_WRAP)

            # Funky sanity check, in case the given frame sizing is smaller than our base pattern!
            pattern = pattern[0:image_h, 0:image_w]
            self._full_pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

        return self._full_pattern.copy()

    # .................................................................................................................

    def render_from_mask(self, image_bgr_uint8: ndarray, mask: ndarray, invert_mask=False) -> ndarray:
        """Draw the given image with a checker pattern based on the provided mask"""

        # Create checker pattern matched to other frame
        checker_pattern = self.draw_like(image_bgr_uint8)

        # Make sure mask is uint8
        if mask.dtype == np.bool:
            mask = np.uint8(mask) * 255

        # Make sure the mask is matched to the other frame size
        img_h, img_w = image_bgr_uint8.shape[0:2]
        mask_h, mask_w = mask.shape[0:2]
        is_same_size = (mask_h == img_h) and (mask_w == img_w)
        if not is_same_size:
            mask = cv2.resize(mask, dsize=(img_w, img_h), interpolation=cv2.INTER_NEAREST_EXACT)

        # Flip mask if needed
        if invert_mask:
            mask = np.bitwise_not(mask)

        # Force mask to be 3-channels, if it isn't already
        if not mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Combine masked frame with (inverted) masked checker pattern to form output
        inv_mask = cv2.bitwise_not(mask)
        checker_masked = cv2.bitwise_and(checker_pattern, inv_mask)
        other_masked = cv2.bitwise_and(image_bgr_uint8, mask)
        return cv2.bitwise_or(checker_masked, other_masked)

    # .................................................................................................................

    def render_from_alpha(self, image_bgra_uint8: ndarray, keep_alpha_channel: bool = False) -> ndarray:
        """
        Given an image with an alpha channel (e.g. HxWx4 image shape)
        This function will draw the image blended with a checker pattern,
        based on the alpha value. Where alpha = 255, only the image will
        be visible, in alpha = 0 areas, only the checker pattern is visible.
        """

        # Extract alpha channel from image & convert to 0-to-1 range
        alpha_norm_f32 = np.float32(image_bgra_uint8[:, :, 3]) / 255.0
        alpha_norm_f32 = np.expand_dims(alpha_norm_f32, axis=2)

        # Result is alpha-weighted linear interpolation between input & checker pattern
        checker_f32 = np.float32(self.draw_like(image_bgra_uint8))
        result_f32 = np.float32(image_bgra_uint8[:, :, 0:3]) * alpha_norm_f32 + checker_f32 * (1.0 - alpha_norm_f32)
        result_bgr_uint8 = np.round(result_f32).astype(np.uint8)

        # Result doesn't need an alpha channel, but user may want to keep it for shape consistency
        if keep_alpha_channel:
            return cv2.cvtColor(result_bgr_uint8, cv2.COLOR_BGR2BGRA)
        return result_bgr_uint8

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions
