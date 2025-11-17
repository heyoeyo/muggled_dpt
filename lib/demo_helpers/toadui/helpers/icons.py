#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .images import blank_image
from .colors import pick_contrasting_gray_color
from .sampling import (
    make_xy_coordinate_mesh,
    sdf_line_segment,
    sdf_rectangle,
    sdf_isoceles_triangle,
    sdf_circle_segment,
    smoothstep,
    pointwise_minimum_of_many,
)

# For type hints
from numpy import ndarray
from .types import COLORU8


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def draw_lock_icons(
    locked_color: COLORU8 = (0, 0, 0),
    locked_fg_color: COLORU8 | None = None,
    unlocked_color: COLORU8 | None = None,
    unlocked_fg_color: COLORU8 | None = None,
    scale_norm: float = 0.75,
    side_length_px: int = 40,
    antialias_px: float = 4,
) -> tuple[ndarray, ndarray]:
    """
    Function used to draw a 'lock' and corresponding 'unlocked' icon
    The icons are drawn using sdfs, so that they should scale nicely to
    different sizes. The unlocked colors can be different from the locked
    colors, but if not provided (e.g. using color of 'None') they will
    match the locked colors.
    Returns:
        locked_icon_image, unlocked_icon_image
    """

    # Build xy coordinate mesh for sdfs
    icon_scale = 1.0 / scale_norm
    xy1, xy2 = (-icon_scale, icon_scale), (icon_scale, -icon_scale)
    mesh_xy = make_xy_coordinate_mesh((side_length_px, side_length_px), xy1, xy2)

    imgs_list = []
    for is_locked in (True, False):

        # Offset mesh when locked vs. unlocked, to help indicate state change
        lock_offset = 0.15
        xy_offset = (0, lock_offset * 0.5) if is_locked else (0, 0)
        mxy = mesh_xy - np.float32(xy_offset) if is_locked else mesh_xy

        # Configure main block/rectangle component of icon
        bar_y_top = 0.2 + (0 if is_locked else lock_offset)
        rect_radius = 0.05
        rect_wh = (1.3 - rect_radius, 0.85 - rect_radius)
        rect_xy = (0, -0.5 * rect_wh[1])

        # Configure bar component of lock icon
        bar_thickness = 0.1
        bar_width_norm = 0.75
        bar_xy1 = (0, bar_y_top)
        bar_xy2 = (0, -0.1)
        bar_radius = ((rect_wh[0] - bar_thickness) * bar_width_norm * 0.5) * 0.5
        bar_diam = 2 * bar_radius

        # Draw bar component
        sdf_bar = sdf_line_segment(mxy, bar_xy1, bar_xy2)
        sdf_bar = np.abs(sdf_bar - bar_diam) - bar_thickness
        if not is_locked:
            neg_box = sdf_rectangle(mxy, (-bar_diam, 0), (bar_diam, bar_y_top * 1.5))
            sdf_bar = np.maximum(sdf_bar, -neg_box)

        # Draw box component
        sdf_box = sdf_rectangle(mxy, rect_xy, rect_wh)
        sdf_box = sdf_box - rect_radius

        # Combine bar & box components into a single 'image'
        sdf_final = pointwise_minimum_of_many(sdf_box, sdf_bar)
        antialias_norm = 0.5 * antialias_px / side_length_px
        img_f32 = smoothstep(antialias_norm, -antialias_norm, sdf_final)
        imgs_list.append(img_f32)

    # Convert colors to [1,1,3] shape, for multiplying into 0-to-1 images
    if locked_fg_color is None:
        locked_fg_color = pick_contrasting_gray_color(locked_color)
    if unlocked_color is None:
        unlocked_color = locked_color
    if unlocked_fg_color is None:
        unlocked_fg_color = locked_fg_color
    colors_iter = (locked_color, unlocked_color, locked_fg_color, unlocked_fg_color)
    to_113_shape = lambda color: np.expand_dims(np.float32(color), axis=(0, 1))
    bgcol_lock, bgcol_unlock, fgcol_lock, fgcol_unlock = [to_113_shape(color) for color in colors_iter]

    # Color images as weighted sum of foreground/background colors
    locked_f32, unlocked_f32 = [np.expand_dims(img, axis=2) for img in imgs_list]
    img_locked_u8 = np.uint8(np.round(fgcol_lock * locked_f32 + (1 - locked_f32) * locked_color))
    img_unlocked_u8 = np.uint8(np.round(fgcol_unlock * unlocked_f32 + (1 - unlocked_f32) * unlocked_color))

    return img_locked_u8, img_unlocked_u8


def draw_play_pause_icons(
    triangle_bg_color: COLORU8 = (60, 60, 225),
    triangle_symbol_symbol: COLORU8 | None = None,
    bars_bg_color: COLORU8 | None = None,
    bars_symbol_color: COLORU8 | None = None,
    side_length_px: int = 40,
) -> tuple[ndarray, ndarray]:
    """
    Helper used to draw the conventional triangle (▶) and bar (⏸︎) icons
    for play/pause controls

    Returns:
        triangle_icon, bar_icon
    """

    # Fill in default color if missing
    if triangle_symbol_symbol is None:
        triangle_symbol_symbol = pick_contrasting_gray_color(triangle_symbol_symbol)
    if bars_bg_color is None:
        bars_bg_color = triangle_bg_color
    if bars_symbol_color is None:
        bars_symbol_color = triangle_symbol_symbol

    # Figure out available drawing space
    pad = max(10, side_length_px // 2)
    avail_side = side_length_px - pad
    half_avail = avail_side // 2

    # Figure out shape boundaries (done a bit strangely to force centering)
    mid_point = (side_length_px - 1) * 0.5
    x1, y1 = [round(val - 0.25 - half_avail) for val in (mid_point, mid_point)]
    x2, y2 = [round(val + 0.25 + half_avail) for val in (mid_point, mid_point)]

    # Draw right-pointing triangle for play state
    triangle_img = blank_image(side_length_px, side_length_px, triangle_bg_color)
    poly_px = [(x1, y1), (x2, round(mid_point)), (x1, y2)]
    cv2.fillConvexPoly(triangle_img, np.int32(poly_px), triangle_symbol_symbol, cv2.LINE_AA)

    # Draw 'double bars' for pause state
    bar_img = blank_image(side_length_px, side_length_px, bars_bg_color)
    barw = round(avail_side / 3)
    pt1, pt2 = (x1, y1), (x1 + barw, y2)
    pt3, pt4 = (x2 - barw, y1), (x2, y2)
    cv2.rectangle(bar_img, pt1, pt2, bars_symbol_color, -1)
    cv2.rectangle(bar_img, pt3, pt4, bars_symbol_color, -1)

    return triangle_img, bar_img


def draw_rotating_arrow_icons(
    color_fg: COLORU8 = (255, 255, 255),
    color_bg: COLORU8 = (0, 0, 0),
    side_length_px: int = 40,
    scale_norm: float = 0.95,
    thickness_norm: float = 0.25,
    arrow_scale_norm: float = 1,
    antialias_px=4,
) -> tuple[ndarray, ndarray]:
    """
    Helper used to draw rotating arrow icons, which
    are conventionally used to represent 'rotation',
    'going back' or 'reloading'.

    Returns:
        left_facing_rot_arrow, right_facing_rot_arrow
    """
    # Build xy coordinate mesh for sdfs
    icon_scale = 1.0 / scale_norm
    xy1, xy2 = (-icon_scale, icon_scale), (icon_scale, -icon_scale)
    mesh_xy = make_xy_coordinate_mesh((side_length_px, side_length_px), xy1, xy2)

    # Figure out sizing
    arrow_base = 0.225 * arrow_scale_norm
    arrow_height = 0.45 * arrow_scale_norm
    radius = scale_norm - (thickness_norm + arrow_base) * 0.5

    # Figure out positioning of circle & arrow head
    circ_xy = (0, 0)
    sdf_circ = sdf_circle_segment(mesh_xy, circ_xy, radius, thickness_norm, start_angle_rad=np.pi)
    arrow_xy = (circ_xy[0], circ_xy[1] + radius)
    sdf_arrow = sdf_isoceles_triangle(mesh_xy, arrow_xy, arrow_base, arrow_height, np.pi) - 0.1

    # Apply antialiasing for nicer result
    sdf_left_rotarrow = pointwise_minimum_of_many(sdf_circ, sdf_arrow)
    antialias_norm = 0.5 * antialias_px / side_length_px
    l_img_f32 = smoothstep(antialias_norm, -antialias_norm, sdf_left_rotarrow)
    l_img_f32 = np.expand_dims(l_img_f32, -1)

    # Color image as weighted sum of foreground/background colors & create right-pointing version
    l_img_u8 = np.uint8(np.round(color_fg * l_img_f32 + (1 - l_img_f32) * color_bg))
    r_img_u8 = np.fliplr(l_img_u8)

    return l_img_u8, r_img_u8
