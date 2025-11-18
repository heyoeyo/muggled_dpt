#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

import torch

# For type hints
from numpy import ndarray
from torch import Tensor


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def scale_prediction(prediction_tensor: Tensor, target_wh: tuple[int, int], interpolation: str = "bilinear") -> Tensor:
    """Helper used to scale raw depth prediction. Assumes input is of shape: BxHxW"""

    target_hw = (int(target_wh[1]), int(target_wh[0]))
    scaled_prediction = torch.nn.functional.interpolate(
        prediction_tensor.unsqueeze(1), size=target_hw, mode=interpolation
    )
    return scaled_prediction.squeeze(1)


# .....................................................................................................................


def scale_to_max_side_length(
    image_bgr: ndarray,
    max_side_length_px: float = 800,
    interpolation: int | None = None,
) -> ndarray:
    """
    Helper used to scale an image to a target maximum side length. The other side of the image
    is scaled to preserve the image aspect ratio (within rounding error).
    Expects opencv (numpy array) image with dimension ordering of HxWxC
    """

    in_h, in_w = image_bgr.shape[0:2]
    scale_factor = max_side_length_px / max(in_h, in_w)
    scaled_wh = [round(side * scale_factor) for side in (in_w, in_h)]
    return cv2.resize(image_bgr, dsize=scaled_wh, interpolation=interpolation)


# .....................................................................................................................


def remove_inf_tensor(data: Tensor, inf_replacement_value: float = 0.0, in_place: bool = True) -> Tensor:
    """Helper used to remove +/- inf values from tensor (pytorch) data"""
    inf_mask = data.isinf()
    data = data if in_place else data.clone()
    data[inf_mask] = inf_replacement_value
    return data


def remove_inf_ndarray(data: ndarray, inf_replacement_value: float = 0.0, in_place: bool = True) -> ndarray:
    """Helper used to remove +/- inf values from ndarray (numpy) data"""
    inf_mask = np.isinf(data)
    data = data if in_place else data.copy()
    data[inf_mask] = inf_replacement_value
    return data


# .....................................................................................................................


def normalize_01(data: Tensor | ndarray) -> Tensor | ndarray:
    """
    Helper used to normalize depth prediction, to 0-to-1 range.
    Works on pytorch tensors and numpy arrays

    Returns:
        depth_normalized_0_to_1
    """

    pred_min = data.min()
    pred_max = data.max()
    return (data - pred_min) / (pred_max - pred_min)


# .....................................................................................................................


def convert_to_uint8(depth_prediction_tensor: Tensor) -> Tensor:
    """
    Helper used to convert depth prediction into 0-255 uint8 range,
    used when displaying result as an image.

    Note: The result will still be on the device as a tensor!
    To move to cpu/numpy use: convert_to_uint8(...).cpu().numpy()
    Returns:
        depth_as_uint8_tensor
    """

    return (255.0 * normalize_01(depth_prediction_tensor)).byte()


# .....................................................................................................................

def histogram_equalization(depth_uint8: np.ndarray, min_pct: float = 0.0, max_pct: float = 1.0) -> np.ndarray:
    """
    Function used to perform histogram equalization on a depth image.
    This function uses the built-in opencv function: cv2.equalizeHist(...)
    When the min/max thresholds are not set (since it works faster),
    however this implementation also supports truncating the low/high
    end of the input.
    This means that equalization can be performed over a subset of
    the input value range, which makes better use of the value range
    when using thresholded inputs.

    Returns:
        depth_uint8_equalized
    """

    # Make sure min/max are properly ordered & separated
    min_value, max_value = [int(round(255 * value)) for value in sorted((min_pct, max_pct))]
    max_value = max(max_value, min_value + 1)
    if min_value == 0 and max_value == 255:
        return cv2.equalizeHist(depth_uint8)

    # Compute histogram of input
    num_bins = 1 + max_value - min_value
    bin_counts, _ = np.histogram(depth_uint8, num_bins, range=(min_value, max_value))

    # Compute cdf of histogram counts
    cdf = bin_counts.cumsum()
    cdf_min, cdf_max = cdf.min(), cdf.max()
    cdf_norm = (cdf - cdf_min) / float(max(cdf_max - cdf_min, 1))
    cdf_uint8 = np.uint8(255 * cdf_norm)

    # Extend cdf to match 256 lut sizing, in case we skipped min/max value ranges
    low_end = np.zeros(min_value, dtype=np.uint8)
    high_end = np.full(255 - max_value, 255, dtype=np.uint8)
    equalization_lut = np.concatenate((low_end, cdf_uint8, high_end))

    # Apply LUT mapping to input
    return equalization_lut[depth_uint8]
