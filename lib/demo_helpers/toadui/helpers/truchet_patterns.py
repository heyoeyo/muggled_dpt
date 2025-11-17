#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from itertools import product

import cv2
import numpy as np

from .colors import make_colormap_from_keypoints
from .sampling import make_xy_coordinate_mesh, sdf_circle, pointwise_minimum_of_many, smoothstep

# For type hints
from numpy import ndarray
from .types import COLORU8, IMGSHAPE_HW


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def draw_truchet(
    image_hw: IMGSHAPE_HW | int | float,
    tiles_list: list[ndarray],
    fit_by_crop: bool = False,
    rng_seed: int | None = None,
    draw_debug_grid: bool = False,
) -> ndarray:

    # For convenience, allow single-value pattern sizes which are interpreted as square
    if isinstance(image_hw, (int, float)):
        image_hw = (round(image_hw), round(image_hw))

    # Figure out out many tiles are needed
    pat_h, pat_w = image_hw[0:2]
    tile_h, tile_w = tiles_list[0].shape[0:2]
    num_h_tiles = int(np.ceil(pat_h / tile_h))
    num_w_tiles = int(np.ceil(pat_w / tile_w))

    # Initialize blank output which will be filled with tiles
    out_h, out_w = round(num_h_tiles * tile_h), round(num_w_tiles * tile_w)
    out_shape = (out_h, out_w, 3)
    output = np.empty(out_shape, dtype=np.uint8)

    # Place tiles randomly
    num_unique_tiles = len(tiles_list)
    noise_gen = np.random.default_rng(rng_seed)
    for h_idx, w_idx in product(range(num_h_tiles), range(num_w_tiles)):
        rand_idx = noise_gen.integers(0, num_unique_tiles)
        x1, y1 = w_idx * tile_w, h_idx * tile_h
        x2, y2 = x1 + tile_w, y1 + tile_h
        output[y1:y2, x1:x2] = tiles_list[rand_idx]

    # For debugging, show tile boundaries
    if draw_debug_grid:
        draw_color = (0, 0, 0)
        for h_idx in range(num_h_tiles + 1):
            h_px = h_idx * tile_h
            cv2.line(output, (-10, h_px), (out_w + 10, h_px), draw_color, 1)
        for w_idx in range(num_w_tiles + 1):
            w_px = w_idx * tile_w
            cv2.line(output, (w_px, -10), (w_px, out_h + 10), draw_color, 1)

    # Shrink output to target shape, if we over-tiled
    needs_downsize = (out_h > pat_h) or (out_w > pat_w)
    if needs_downsize:
        if fit_by_crop:
            y1 = (out_h - pat_h) // 2
            x1 = (out_w - pat_w) // 2
            y2 = y1 + pat_h
            x2 = x1 + pat_w
            output = output[y1:y2, x1:x2]
        else:
            output = cv2.resize(output, dsize=(pat_w, pat_h))

    return output


# ---------------------------------------------------------------------------------------------------------------------
# %% Tile functions


def make_truchet_tiles_diagonal(
    side_length_px: int,
    thickness_pct: float = 0.5,
    color_fg: COLORU8 = (255, 255, 255),
    color_bg: COLORU8 = (0, 0, 0),
) -> tuple[ndarray, ndarray]:
    """Make simple truchet pattern using diagonal lines"""
    s_px, s_norm = side_length_px, 1.0 / side_length_px

    # Generate sdf of tiling pattern (sdf allows support for any tile size)
    xy_mesh = make_xy_coordinate_mesh(s_px, xy1=(-1, -1), use_wraparound_sampling=True)
    line1 = np.abs(np.sum(xy_mesh, axis=2) - 1)
    line2 = np.abs(np.sum(xy_mesh, axis=2) + 1)
    tile_sdf = pointwise_minimum_of_many(line1, line2)

    # Convert SDF to grayscale image, with anti-aliased edges
    tile_sdf = tile_sdf - 0.5 * thickness_pct
    tile_sdf = smoothstep(-s_norm, s_norm, tile_sdf)
    tile_uint8 = np.uint8(tile_sdf * 255)

    # Map tile into bg/fg colors
    cmap = make_colormap_from_keypoints([color_fg, color_bg])
    tile_uint8 = cv2.LUT(cv2.cvtColor(tile_uint8, cv2.COLOR_GRAY2BGR), cmap)

    return (tile_uint8, np.flipud(tile_uint8))


def make_truchet_tiles_smith(
    side_length_px: int,
    thickness_pct: float = 0.5,
    color_fg: COLORU8 = (255, 255, 255),
    color_bg: COLORU8 = (0, 0, 0),
) -> tuple[ndarray, ndarray]:
    """Make truchet pattern using curved lines aka. Smith tiles"""

    # For clarity
    s_px, s_norm = side_length_px, 1.0 / side_length_px
    min_thickness = 3
    max_antialias = 1.5
    radius_norm = 0.5

    # Make SDF containing 2 circles at tl/br corners
    xy_mesh = make_xy_coordinate_mesh(s_px, use_wraparound_sampling=True)
    circ1 = sdf_circle(xy_mesh, (0, 0), radius_norm)
    circ2 = sdf_circle(xy_mesh, (1, 1), radius_norm)
    tile_sdf = pointwise_minimum_of_many(circ1, circ2)
    tile_sdf = np.abs(tile_sdf)

    # Convert SDF to grayscale image, with anti-aliased edges
    mid_dist_norm = 1.0 / np.sqrt(2)
    max_thickness_norm = mid_dist_norm - radius_norm
    aa_norm = min(max_antialias, s_px * s_px / 2000) * s_norm
    thick_norm = max_thickness_norm * max(thickness_pct, min_thickness * s_norm)
    tile_sdf = smoothstep(thick_norm - aa_norm, thick_norm + aa_norm, tile_sdf)
    tile_uint8 = np.uint8(tile_sdf * 255)

    # Map tile into bg/fg colors
    cmap = make_colormap_from_keypoints([color_fg, color_bg])
    tile_uint8 = cv2.LUT(cv2.cvtColor(tile_uint8, cv2.COLOR_GRAY2BGR), cmap)

    return (tile_uint8, np.flipud(tile_uint8))


def make_dot_tiles(
    side_length_px: int,
    radius_norm: float = 0.5,
    color_fg: COLORU8 = (255, 255, 255),
    color_bg: COLORU8 = (0, 0, 0),
    antialias_px: float = 5,
    is_offset: bool = True,
):

    # Make pattern with for circles (at corners or offset to midpoints)
    xy_mesh = make_xy_coordinate_mesh((side_length_px), use_wraparound_sampling=True)
    if is_offset:
        c_radius = np.sqrt(radius_norm / 8)
        circ1 = sdf_circle(xy_mesh, (0.5, 0), c_radius)
        circ3 = sdf_circle(xy_mesh, (1, 0.5), c_radius)
        circ4 = sdf_circle(xy_mesh, (0.5, 1), c_radius)
        circ2 = sdf_circle(xy_mesh, (0, 0.5), c_radius)
    else:
        c_radius = np.sqrt(radius_norm / 4)
        circ1 = sdf_circle(xy_mesh, (0, 0), c_radius)
        circ2 = sdf_circle(xy_mesh, (1, 0), c_radius)
        circ3 = sdf_circle(xy_mesh, (1, 1), c_radius)
        circ4 = sdf_circle(xy_mesh, (0, 1), c_radius)

    tile_sdf = pointwise_minimum_of_many(circ1, circ2, circ3, circ4)

    # Apply antialiasing & convert to grayscale uint8
    smoothness_norm = antialias_px / side_length_px
    smooth_min = -smoothness_norm * c_radius
    smooth_max = smoothness_norm * (0.5 - c_radius)
    smoothsdf = smoothstep(smooth_min, smooth_max, tile_sdf)
    tile_uint8 = np.uint8(smoothsdf * 255)

    # Apply color mapping
    cmap = make_colormap_from_keypoints([color_fg, color_bg])
    tile_uint8 = cv2.LUT(cv2.cvtColor(tile_uint8, cv2.COLOR_GRAY2BGR), cmap)

    return [tile_uint8]
