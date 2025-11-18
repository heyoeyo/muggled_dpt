#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

# For type hints
from numpy import ndarray
from .types import COLORU8, XYNORM, XY1XY2PX, XYPX
from .ocv_types import LineTypeCode, OCVLineType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def draw_box_outline(image_uint8: ndarray, color: COLORU8 | None = (0, 0, 0), thickness=1) -> ndarray:
    """
    Helper used to draw a box outline around the outside of a given image.
    If the color given is None, then no box will be drawn
    (can be used to conditionally disable outline)
    Returns:
        image_with_outline_uint8
    """

    # Bail if no color or thickness
    if color is None or thickness <= 0:
        return image_uint8

    img_h, img_w = image_uint8.shape[0:2]
    x1, y1 = thickness - 1, thickness - 1
    x2, y2 = img_w - thickness, img_h - thickness

    # Technique for rendering border depends on the thickness, due to oddities
    # in the way opencv line drawing works
    if thickness < 4:
        # The width (in pixels) of lines drawn by opencv for thickness values
        # less than 4, follows the formula: line width (px) = 2*thickness - 1
        # The sizes are always odd, so we can get the target border thickness
        # by simply drawing lines with corners matching the image itself.
        x1, y1 = 0, 0
        x2, y2 = img_w - 1, img_h - 1
        cv2.rectangle(image_uint8, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_4)
    elif thickness < 8:
        # For thickness values 4 or greater, opencv follows a different
        # pattern of line widths, given by the formula:
        #   line width (px) = (thickness + 2 if odd else 1)
        # So for example, a thickness of 4 is also 5 pixels wide. A thickness
        # of 5 is drawn 7 pixels wide, thickness 9 is drawn 11 pixels wide etc.
        # So we need an extra offset to maintain correct border sizing.
        extra_offset = thickness // 2
        x1, y1 = extra_offset - 1, extra_offset - 1
        x2, y2 = img_w - extra_offset, img_h - extra_offset
        cv2.rectangle(image_uint8, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_4)
    else:
        # Opencv rounds the corners of rectangles, though this is only
        # noticable at higher thickness values. Here we switch to drawing
        # the border using filled rectangles to maintain consistent border
        # sizing (we don't default to doing this because it's slower).
        xl, xr = thickness - 1, img_w - thickness
        yt, yb = thickness - 1, img_h - thickness
        cv2.rectangle(image_uint8, (0, 0), (img_w, yt), color, -1, cv2.LINE_4)
        cv2.rectangle(image_uint8, (0, yb), (img_w, img_h), color, -1, cv2.LINE_4)
        cv2.rectangle(image_uint8, (0, 0), (xl, img_h), color, -1, cv2.LINE_4)
        cv2.rectangle(image_uint8, (xr, 0), (img_w, img_h), color, -1, cv2.LINE_4)

    return image_uint8


def draw_drop_shadow(
    image_uint8: ndarray,
    left=2,
    top=4,
    right=2,
    bottom=0,
    color: COLORU8 = (0, 0, 0),
    blur_strength: float = 3,
    blur_sharpness: float = 1,
) -> ndarray:
    """
    Helper used to draw a shadow (e.g. blurred outline) around the edges of an image.
    The shadow 'size' can be set per edge (left, top, right, bottom)
    Returns:
        image_with_drop_shadow
    """

    # Bail if we're not shadowing
    if all(val <= 0 for val in [left, top, right, bottom, blur_strength]):
        return image_uint8

    # For convenience
    img_h, img_w = image_uint8.shape[0:2]
    x1, y1 = 0, 0
    x2, y2 = img_w - 1, img_h - 1
    xl, xr = x1 + left, x2 - right
    yt, yb = y1 + top, y2 - bottom

    # Draw lines around edges & blur to create drop-shadow effect
    shadow_img = np.full((img_h, img_w), 255, dtype=np.uint8)
    if top > 0:
        cv2.rectangle(shadow_img, (x1, y1), (x2, yt), 0, -1)
    if left > 0:
        cv2.rectangle(shadow_img, (x1, y1), (xl, y2), 0, -1)
    if right > 0:
        cv2.rectangle(shadow_img, (xr, y1), (x2, y2), 0, -1)
    if bottom > 0:
        cv2.rectangle(shadow_img, (x1, yb), (x2, y2), 0, -1)
    shadow_img = np.float32(cv2.GaussianBlur(shadow_img, None, blur_strength)) * (1.0 / 255.0)

    # Scale input by shadow amount. Equivalent to a black (0,0,0) drop-shadow
    if blur_sharpness != 1:
        shadow_img = np.pow(shadow_img, blur_sharpness)
    shadow_img = np.expand_dims(shadow_img, axis=2)
    result_f32 = shadow_img * np.float32(image_uint8)

    # If we have a non-black shadow, then we treat the step above as a weighting
    # -> So add an inversely weighted copy of the color we want to get the final result
    is_black_shadow = all(col == 0 for col in color)
    if not is_black_shadow:
        full_color_img = np.full((img_h, img_w, 3), color, dtype=np.float32)
        result_f32 += (1.0 - shadow_img) * full_color_img
        result_f32 = np.clip(result_f32, 0, 255)

    return np.round(result_f32).astype(np.uint8)


def draw_normalized_polygon(
    image_uint8: ndarray,
    polygon_xy_norm_list: list[tuple[float, float]] | ndarray,
    color: COLORU8 = (0, 255, 255),
    thickness: int = 1,
    bg_color: COLORU8 | None = None,
    line_type: int = cv2.LINE_AA,
    is_closed: bool = True,
) -> ndarray:
    """
    Helper used to draw polygons from 0-to-1 normalized xy coordinates.
    Expects coordinates in the form:
        xy_norm = [(0,0), (0.5, 0), (0.5, 0.75), (1, 1), (0, 1), etc.]
    """

    # Force input to be an array to make normalization easier
    xy_norm_array = polygon_xy_norm_list
    if not isinstance(polygon_xy_norm_list, ndarray):
        xy_norm_array = np.float32(polygon_xy_norm_list)

    # Convert normalized xy into pixel units
    img_h, img_w = image_uint8.shape[0:2]
    norm_to_px_scale = np.float32((img_w - 1, img_h - 1))
    xy_px_array = np.int32(np.round(xy_norm_array * norm_to_px_scale))

    # Draw polygon with background if needed
    if bg_color is not None:
        bg_thick = max(0, thickness) + 1
        cv2.polylines(image_uint8, [xy_px_array], is_closed, bg_color, bg_thick, line_type)

    # Draw polygon outline, or filled in shape if using negative thickness value
    if thickness < 0:
        return cv2.fillPoly(image_uint8, [xy_px_array], color, line_type)
    return cv2.polylines(image_uint8, [xy_px_array], is_closed, color, thickness, line_type)


def draw_rectangle_norm(
    image_uint8: ndarray,
    xy1_norm: XYNORM,
    xy2_norm: XYNORM,
    color: COLORU8 = (0, 0, 0),
    thickness: int = -1,
    pad_xy1xy2_px: XY1XY2PX | XYPX = (0, 0),
    inset_outline: bool = True,
) -> ndarray:
    """
    Helper used to draw a rectangle onto an image, using normalized coordinates
    """

    pad_xy1, pad_xy2 = pad_xy1xy2_px
    img_h, img_w = image_uint8.shape[0:2]
    norm_to_px_scale = np.float32((img_w - 1, img_h - 1))
    x1_px, y1_px = np.int32(np.round(np.float32(xy1_norm) * norm_to_px_scale + np.float32(pad_xy1)))
    x2_px, y2_px = np.int32(np.round(np.float32(xy2_norm) * norm_to_px_scale - np.float32(pad_xy2)))

    if inset_outline and thickness > 1:
        inset_amt = thickness - 1
        x1_px, y1_px = x1_px + inset_amt, y1_px + inset_amt
        x2_px, y2_px = x2_px - inset_amt, y2_px - inset_amt

    pt1, pt2 = (x1_px, y1_px), (x2_px, y2_px)
    return cv2.rectangle(image_uint8, pt1, pt2, color, thickness)


def draw_circle_norm(
    image_uint8: ndarray,
    xy_center_norm: XYNORM,
    radius_px: int = 5,
    color: COLORU8 = (0, 255, 255),
    thickness: int = -1,
    line_type: OCVLineType = LineTypeCode.antialiased,
) -> ndarray:
    """Helper used to draw a circle onto an image, using normalized coordinates"""
    img_h, img_w = image_uint8.shape[0:2]
    xy_px = (round(xy_center_norm[0] * img_w), round(xy_center_norm[1] * img_h))
    return cv2.circle(image_uint8, xy_px, radius_px, color, thickness, line_type)
