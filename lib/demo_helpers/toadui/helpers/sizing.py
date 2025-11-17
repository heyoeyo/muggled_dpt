#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2

# For type hints
from numpy import ndarray
from .types import IMGSHAPE_HW, HWPX


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def resize_hw(frame: ndarray, hw: HWPX, interpolation=None) -> ndarray:
    """Alternate version of cv2.resize(...) which takes (height, width) instead of (width, height) for sizing"""
    return cv2.resize(frame, dsize=(hw[1], hw[0]), interpolation=interpolation)


def get_image_hw_to_fit_by_ar(aspect_ratio: float, region_hw: IMGSHAPE_HW, fit_within=True) -> HWPX:
    """
    Helper used to find the sizing (height & width) to fit into/or
    around a given target height & width, based on an aspect ratio.
    For example, for an aspect ratio of 1.5 (e.g. width = 1.5 * height):
        get_image_hw_to_fit_by_ar(1.5, (600,400), fit_within=True) -> (267, 400)
        get_image_hw_to_fit_by_ar(1.5, (600,400), fit_within=False) -> (600, 900)

    -> So 267x400 fits 'within' the 600x400 region
    -> While 600x900 fits 'around' the region

    Returns:
        output_height, output_width
    """
    reg_h, reg_w = region_hw[0:2]

    min_or_max = min if fit_within else max
    out_h = min_or_max(reg_h, reg_w / aspect_ratio)
    out_w = min_or_max(reg_w, reg_h * aspect_ratio)

    return round(out_h), round(out_w)


def get_image_hw_to_fit_region(image_shape: IMGSHAPE_HW, region_hw: IMGSHAPE_HW, fit_within=True) -> HWPX:
    """
    Helper used to find the sizing (height & width) of a given image
    if it is scaled to fit into/around the given region height & width,
    assuming the aspect ratio of the image is preserved.
    For example, to fit a 100x200 image into a 600x600 square space,
    while preserving aspect ratio, the image would be scaled to 300x600

    Returns:
        output_height, output_width
    """

    img_h, img_w = image_shape[0:2]
    reg_h, reg_w = region_hw[0:2]

    min_or_max = min if fit_within else max
    scale = min_or_max(reg_h / img_h, reg_w / img_w)
    out_h = round(scale * img_h)
    out_w = round(scale * img_w)

    return out_h, out_w


def get_image_hw_for_max_height(image_shape: IMGSHAPE_HW, max_height_px: int = 800) -> HWPX:
    """
    Helper used to find the height & width of a given image if it
    is scaled to fit to a given target height, assuming the aspect
    ratio is preserved.
    For example, to fit a (HxW) 100x200 image to a max height of
    500, the image would be scaled to 500x1000

    Returns:
        output_height, output_width
    """

    img_h, img_w = image_shape[0:2]
    scale = max_height_px / img_h
    out_h = round(scale * img_h)
    out_w = round(scale * img_w)

    return out_h, out_w


def get_image_hw_for_max_width(image_shape: IMGSHAPE_HW, max_width_px: int = 800) -> HWPX:
    """
    Helper used to find the height & width of a given image if it
    is scaled to fit to a given target width, assuming the aspect
    ratio is preserved.
    For example, to fit a (HxW) 100x200 image to a max width of
    500, the image would be scaled to 250x500

    Returns:
        output_height, output_width
    """

    img_h, img_w = image_shape[0:2]
    scale = max_width_px / img_w
    out_h = round(scale * img_h)
    out_w = round(scale * img_w)

    return out_h, out_w


def get_image_hw_for_max_side_length(image_shape: IMGSHAPE_HW, max_side_length: int = 800) -> HWPX:
    """
    Helper used to find the height & width of a given image if it
    is scaled to a target max side length, assuming the aspect
    ratio is preserved.
    For example, to fit a (HxW) 100x200 image to a max side length
    of 500, the image would be scaled to 250x500

    Returns:
        output_height, output_width
    """

    img_h, img_w = image_shape[0:2]
    scale = min(max_side_length / img_w, max_side_length / img_h)
    out_h = round(scale * img_h)
    out_w = round(scale * img_w)

    return out_h, out_w
