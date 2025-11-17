#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

# For type hints
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def make_xy_coordinate_mesh(
    mesh_hw: int | tuple[int, int],
    xy1: tuple[float, float] = (0, 0),
    xy2: tuple[float, float] = (1, 1),
    use_wraparound_sampling=False,
    dtype=np.float32,
) -> ndarray:
    """
    Returns a grid of xy coordinate values (between the given xy1/xy2 values).
    The shape of the result is: HxWx2
    -> Where H, W are from the given mesh_hw
    -> The '2' at the end corresponds to the (x,y) coordinate values at each 'pixel'
    """

    # Support providing height/width as a single value (interpret as square sizing)
    if isinstance(mesh_hw, (int, float)):
        mesh_hw = round(mesh_hw)
        mesh_hw = (mesh_hw, mesh_hw)
    if isinstance(use_wraparound_sampling, (int, bool)):
        use_wraparound_sampling = (bool(use_wraparound_sampling), bool(use_wraparound_sampling))

    # For convenience
    mesh_h, mesh_w = np.int32(mesh_hw[0:2])
    x1, y1 = xy1
    x2, y2 = xy2
    wraparound_x, wrap_around_y = use_wraparound_sampling

    # With wrap-around sampling endpoints are chosen so that first/last samples are 'one step' apart
    # -> For example, no-wrap indexing: [0, 0.5, 1] vs. wrap indexing: [0.167, 0.5, 0.833]
    if wraparound_x:
        half_x_step = 0.5 * (x2 - x1) / mesh_w
        x1 = x1 + half_x_step
        x2 = x2 - half_x_step
    if wrap_around_y:
        half_y_step = 0.5 * (y2 - y1) / mesh_h
        y1 = y1 + half_y_step
        y2 = y2 - half_y_step

    # Form xy mesh
    x_idx = np.linspace(x1, x2, mesh_w, dtype=dtype)
    y_idx = np.linspace(y1, y2, mesh_h, dtype=dtype)
    return np.dstack(np.meshgrid(x_idx, y_idx, indexing="xy"))


def make_xy_complex_mesh(
    mesh_hw: int | tuple[int, int],
    xy1: tuple[float, float] = (0, 0),
    xy2: tuple[float, float] = (1, 1),
    use_wraparound_sampling=False,
    dtype=np.complex64,
) -> ndarray:
    """
    Variation of creating an xy mesh. In this case, the xy coordinates are
    encoded as a single '2D' matrix, but each entry is a complex number
    of the form: x + iy
    -> Where x,y are the xy mesh coordinates
    """
    x1, y1 = xy1
    x2, y2 = xy2
    xy_mesh = make_xy_coordinate_mesh(mesh_hw, (x1, y1 * 1j), (x2, y2 * 1j), use_wraparound_sampling, dtype)
    return np.sum(xy_mesh, axis=2)


def resample_with_xy_mesh(
    image_uint8: ndarray,
    xy_mesh: ndarray,
    xy1=(0, 0),
    xy2=(1, 1),
    border=cv2.BORDER_REFLECT,
    interpolation=cv2.INTER_LINEAR,
) -> ndarray:

    img_h, img_w = image_uint8.shape[0:2]
    x1, y1 = xy1
    x2, y2 = xy2

    xmap = (img_w - 1) * (xy_mesh[:, :, 0] - x1) / (x2 - x1)
    ymap = (img_h - 1) * (xy_mesh[:, :, 1] - y1) / (y2 - y1)
    return cv2.remap(image_uint8, xmap, ymap, interpolation, borderMode=border)


def resample_with_complex_mesh(
    image_uint8: ndarray,
    complex_mesh: ndarray,
    xy1=(0, 0),
    xy2=(1, 1),
    border=cv2.BORDER_REFLECT,
    interpolation=cv2.INTER_LINEAR,
) -> ndarray:
    """
    Re-samples an image based on a grid of xy coordinates,
    represented as a single complex-valued 2D matrix.
    Each value in the matrix is assumed to be an xy
    coordinate as a complex number: z = x + iy

    For each entry in the grid, the complex value is used to
    'look up' the RGB color value of the given image, which is
    then painted into the corresponding grid position.

    Returns: resampled_image_uint8
    -> The output will have the same size as the complex mesh
    """
    img_h, img_w = image_uint8.shape[0:2]
    x1, y1 = xy1
    x2, y2 = xy2

    xmap = (img_w - 1) * (complex_mesh.real - x1) / (x2 - x1)
    ymap = (img_h - 1) * (complex_mesh.imag - y1) / (y2 - y1)
    return cv2.remap(image_uint8, xmap, ymap, interpolation, borderMode=border)


def rotate_mesh_xy(mesh_xy: ndarray, rotation_angle_rad: float, pivot_xy: tuple[float, float] | None = None) -> ndarray:
    """
    Function used to rotate xy coordinates of a mesh.
    Returns:
        rotated_mesh_xy

    - The rotation angle is given in radians
    - A pivot_xy value can be given to rotate around a target point
    """

    # Avoid doing any work if we're not rotating
    if rotation_angle_rad == 0:
        return mesh_xy

    # Build the rotation matrix
    rot_cos, rot_sin = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
    rotmat = np.array([[rot_cos, rot_sin], [-rot_sin, rot_cos]])

    # Rotate directly or with set pivot point
    if pivot_xy is None:
        return np.matmul(mesh_xy, rotmat)
    pivot_array = np.array(pivot_xy, dtype=mesh_xy.dtype)
    return np.matmul(mesh_xy - pivot_array, rotmat) + pivot_array


# ---------------------------------------------------------------------------------------------------------------------
# %% SDFs


def pointwise_minimum_of_many(*sdfs: ndarray) -> ndarray:
    """
    Helper used to compute the minimum of many arrays.
    Note that this preserves the shape of the inputs!

    This function behaves as if the np.minimum(...)
    function accepted an arbitrary number of inputs:
        np.minimum(arr1, arr2, arr3, arr4, ... etc)

    This can be used to 'union' together multiple sdfs

    Returns:
        minimum_of_inputs
    """

    if len(sdfs) == 1:
        return sdfs[0]

    min_result = np.minimum(sdfs[0], sdfs[1])
    if len(sdfs) == 2:
        return min_result

    for remaining_sdf in sdfs[2:]:
        min_result = np.minimum(min_result, remaining_sdf)
    return min_result


def sdf_circle(mesh_xy: ndarray, xy_center: tuple[float, float], radius: float) -> ndarray:
    """
    Helper used to calculate the signed-distance-function (SDF) for a circle.
    This can be used to help draw 'perfect' circles of arbitrary radius,
    unlike the built-in opencv drawing function.

    Notes:
    - The mesh_xy is expected to be a HxWx2 array, containing xy coordinate values
    - Units for the center/radius are based on the provided xy mesh
    - This does not 'draw' a circle into an image, it only computes the sdf
    -> A simple way to 'draw' the result is as follows:
        image_of_circle = np.uint8(sdf_circle(...) < 0) * 255

    Returns:
        sdf_of_circle
    """

    return np.linalg.norm(mesh_xy - np.float32(xy_center), ord=2, axis=2) - radius


def sdf_rectangle(mesh_xy: ndarray, xy_center: tuple[float, float], wh: tuple[float, float]) -> ndarray:
    """
    Helper used to calculate the signed-distance-function (SDF) for a rectangle.
    This can be used to draw 'perfect' rectangular of arbitrary units,
    (most notably, they do not need to be perfectly pixel-aligned), and
    can be used to draw boxes with rounded corners, for example.
    Note that the xy_center and width/height (wh) values should be
    given in units matching the mesh_xy input.

    For an explanation, see:
    https://youtu.be/62-pRVZuS5c?si=9f3Iz6DOk2xoVpfH

    Returns:
        sdf_of_rectangle
    """

    xy_delta = np.abs(mesh_xy - np.float32(xy_center)) - np.float32(wh) * 0.5
    return np.linalg.norm(np.maximum(xy_delta, 0), ord=2, axis=2) + np.minimum(xy_delta.max(axis=2), 0)


def sdf_isoceles_triangle(
    mesh_xy: ndarray,
    xy_position: tuple[float, float],
    base: float,
    height: float,
    rotation_angle_rad: float = 0,
    position_by_base: bool = True,
) -> ndarray:
    """
    Helper used to calculate the signed-distance-function (SDF) for an
    isoceles triangle. Defined by a base & height sizing.

    The xy_position will either correspond to the position of the 'tip' of
    the triangle or (if position_by_base=True) the middle of the 'base' of
    the triangle. The triangle can also be rotated about this point.

    Based on code from shadertoy:
    https://www.shadertoy.com/view/MldcD7
    """

    # For sanity
    base, height = [abs(val) for val in (base, height)]

    # Offset & rotate xy coords to position arrow as desired
    xy_offset = np.array(xy_position, dtype=mesh_xy.dtype)
    if position_by_base:
        xy_offset[0] += np.cos(rotation_angle_rad) * height
        xy_offset[1] += np.sin(rotation_angle_rad) * height
    mxy = mesh_xy - np.array(xy_offset, dtype=mesh_xy.dtype)
    angle_with_offset = (np.pi * 0.5) + rotation_angle_rad
    mxy = rotate_mesh_xy(mxy, -angle_with_offset)
    mxy[:, :, 0] = np.abs(mxy[:, :, 0])

    # Mimicking original shadertoy code verbatim (at least, within python), hard to follow!
    dot_numer = mxy[:, :, 0] * base + mxy[:, :, 1] * height
    dot_denom = base**2 + height**2
    dot_result = np.clip(dot_numer / dot_denom, 0.0, 1.0)

    ax = mxy[:, :, 0] - (base * dot_result)
    ay = mxy[:, :, 1] - (height * dot_result)
    a = np.dstack((ax, ay))
    bx = mxy[:, :, 0] - base * np.clip(mxy[:, :, 0] / base, 0.0, 1.0)
    by = mxy[:, :, 1] - height
    b = np.dstack((bx, by))

    channel_dotprod_2d = "xyc,xyc->xy"
    d = np.minimum(np.einsum(channel_dotprod_2d, a, a), np.einsum(channel_dotprod_2d, b, b))
    s = np.maximum((mxy[:, :, 0] * height - mxy[:, :, 1] * base), (mxy[:, :, 1] - height))

    sdf = np.sqrt(d) * np.sign(s)
    return sdf


def sdf_line_segment(mesh_xy: ndarray, point_a_xy: tuple[float, float], point_b_xy: tuple[float, float]) -> ndarray:
    """
    Helper used to calculate the signed-distance-function for a line segment.
    Note that the point_a and point_b xy coordinates should be given in
    units matching the mesh_xy input.

    For an explanation of this, see:
    https://youtu.be/PMltMdi1Wzg?si=d5PR-7pBYkxZYcIU

    Returns:
        sdf_of_line_segment
    """
    delta_a = mesh_xy - np.float32(point_a_xy)
    delta_ba = np.float32(point_b_xy) - np.float32(point_a_xy)
    dist_from_a = np.clip(np.dot(delta_a, delta_ba) / np.dot(delta_ba, delta_ba), 0.0, 1.0)
    return np.linalg.norm(delta_a - (delta_ba * np.expand_dims(dist_from_a, 2)), ord=2, axis=2)


def sdf_circle_segment(
    mesh_xy: ndarray,
    xy_center: tuple[float, float],
    radius: float,
    thickness: float,
    extent_norm: float = 0.75,
    start_angle_rad: float = 0,
) -> ndarray:
    """
    Helper used to calculate the signed-distance-function (SDF) for a circular
    segment. If the 'extent' is less than 1, then the circle will be incomplete.
    This can be useful for drawing rotation-like icons.

    Based on code from shadertoy:
    https://www.shadertoy.com/view/wl23RK

    Returns:
        sdf_of_circular_segment
    """

    # Center and rotate mesh, to get desired starting point
    mxy = mesh_xy - np.array(xy_center, dtype=mesh_xy.dtype)
    angle_offset = np.pi * (1 - 2 * extent_norm) * 0.5
    rotation_angle = angle_offset - start_angle_rad
    mxy = rotate_mesh_xy(mxy, rotation_angle)
    mxy[:, :, 0] = np.abs(mxy[:, :, 0])

    # Calculate extent terms (these determine the 'cut-out' of the circle)
    extent_rad = np.pi * extent_norm
    sin_ext = np.sin(extent_rad)
    cos_ext = np.cos(extent_rad)
    sin_cos_ext = np.array((sin_ext, cos_ext), dtype=mesh_xy.dtype)

    # Compute cut-out mask
    cutout_mask = mxy[:, :, 0] * cos_ext > mxy[:, :, 1] * sin_ext
    circle_mask = np.bitwise_not(cutout_mask)

    # Compute sdf for circular & cut-out regions
    cutout_sdf = np.linalg.norm(mxy - sin_cos_ext * radius, ord=2, axis=2)
    circle_sdf = np.abs(np.linalg.norm(mxy, ord=2, axis=2) - radius)

    # Combine regions with mask to finalize sdf
    sdf = np.empty(mesh_xy.shape[0:2], dtype=mesh_xy.dtype)
    sdf[cutout_mask] = cutout_sdf[cutout_mask]
    sdf[circle_mask] = circle_sdf[circle_mask]
    sdf = sdf - (thickness * 0.5)

    return sdf


# ---------------------------------------------------------------------------------------------------------------------
# %% Interpolation


def smoothstep(edge0: float, edge1: float, x: ndarray | float) -> ndarray | float:
    """CPU implementation of common shader function"""
    x = (x - edge0) / (edge1 - edge0)
    x = np.clip(x, 0, 1)
    return x * x * (3.0 - 2.0 * x)


def lerp(value_t0, value_t1, t):
    """Linear interpolation from value_t0 to value_t1"""
    return value_t0 * (1.0 - t) + value_t1 * t


def cosine_interp(value_t0, value_t1, t):
    """Interpolation from value_t0 to value_t1 using a (non-linear) cosine curve"""
    weight = 0.5 - 0.5 * np.cos(t * np.pi)
    return value_t0 * (1.0 - weight) + value_t1 * weight
