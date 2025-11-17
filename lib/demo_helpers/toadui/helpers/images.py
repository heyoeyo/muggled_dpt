#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from pathlib import Path
from dataclasses import dataclass
from base64 import b64decode, b64encode

import cv2
import numpy as np

# For type hints
from typing import Iterable
from numpy import ndarray
from .types import COLORU8, IMGSHAPE_HW, XYNORM, XYPX, XY1XY2PX
from .ocv_types import OCVInterp


# ---------------------------------------------------------------------------------------------------------------------
# %% Types


@dataclass
class CropData:
    """
    Container data used to hold information used to crop an image,
    along with some convenience functions.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def crop(self, frame: ndarray) -> ndarray:
        return frame[self.y1 : self.y2, self.x1 : self.x2]

    @property
    def xy1(self) -> tuple[int, int]:
        return (self.x1, self.y1)

    @property
    def xy2(self) -> tuple[int, int]:
        return (self.x2, self.y2)

    @property
    def xy1xy2(self) -> XY1XY2PX:
        return ((self.x1, self.y1), (self.x2, self.y2))

    def get_yx_slices(self) -> tuple[slice, slice]:
        """
        Get x/y indexing slices for cropping
        For example:
            y_slice, x_slice = crop_data.get_xy_slices()
            crop_img = image[y_slice, x_slice]
        """
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    @classmethod
    def from_xy1_xy2_norm(cls, image_shape: IMGSHAPE_HW, xy1_norm: XYNORM, xy2_norm: XYNORM):
        """
        Create crop data from normalized xy1 & xy2 coordinates (and a frame shape)
        Returns:
            new CropData
        """

        # For convenience
        img_h, img_w = image_shape[0:2]
        x_scale = img_w - 1
        y_scale = img_h - 1

        # Compute crop coords
        xy1_px = (round(xy1_norm[0] * x_scale), round(xy1_norm[1] * y_scale))
        xy2_px = (round(xy2_norm[0] * x_scale), round(xy2_norm[1] * y_scale))

        return cls(*xy1_px, *xy2_px)

    def is_valid(self) -> bool:
        """Check if crop data would produce a valid (e.g. non-empty) crop result"""
        return (self.x2 > self.x1) and (self.y2 > self.y1) and (self.x1 >= 0) and (self.y1 >= 0)


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def load_valid_image(image_path: str | Path) -> tuple[bool, ndarray | None]:
    """
    Helper used to provide a True/False flag indicating if a loaded image is valid.
    Returns:
        is_valid, image_data
    """
    img_data = cv2.imread(image_path)
    is_valid = img_data is not None
    return is_valid, img_data


def blank_image(height: int, width: int, bgr_color: None | int | COLORU8 = None) -> ndarray:
    """Helper used to create a blank image of a given size (and optionally provide a fill color)"""

    # If no color is given, default to zeros
    if bgr_color is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # If only 1 number is given for the color, duplicate it to form a gray value
    if isinstance(bgr_color, int):
        bgr_color = (bgr_color, bgr_color, bgr_color)

    return np.full((height, width, 3), bgr_color, dtype=np.uint8)


def blank_image_1ch(height: int, width: int, gray_value: int = 0) -> ndarray:
    """Helper used to create a blank mask (i.e. grayscale/no channels) of a given size"""
    return np.full((height, width), gray_value, dtype=np.uint8)


def pad_image_to_hw(
    image: ndarray,
    height: int | None,
    width: int | None,
    border_color: COLORU8 = (0, 0, 0),
    border_type: int = cv2.BORDER_CONSTANT,
) -> ndarray:
    """
    Helper used to (center) pad an image to a target height & width.
    Either of the target height or width values can be given as None,
    in which case the size will be taken from the image itself,
    this can be useful for only padding along 1 axis.
    If the image exceeds the padding width or height, nothing will
    be done to it (i.e. the image won't be cropped to the target size).

    Returns:
        padded_image
    """

    # Fill in missing values
    img_h, img_w = image.shape[0:2]
    if height is None:
        height = img_h
    if width is None:
        width = img_w

    # Only apply padding if needed
    if height > img_h or width > img_w:
        pad_h, pad_w = max(height - img_h, 0), max(width - img_w, 0)
        pad_t, pad_l = pad_h // 2, pad_w // 2
        pad_b, pad_r = pad_h - pad_t, pad_w - pad_l
        return cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, border_type, border_color)

    return image


def adjust_image_gamma(image_uint8: ndarray, gamma: float | Iterable[float] = 1.0) -> ndarray:
    """
    Helper used to apply gamma correction to an entire image.
    If multiple gamma values are provided, they will be applied
    separately, per-channel, to the input image.
    Returns:
        gamma_corrected_image
    """

    # If we get multiple gamma values, assume we need to apply them per-channel
    if isinstance(gamma, Iterable):
        for ch_idx, ch_gamma in enumerate(gamma):
            image_uint8[:, :, ch_idx] = adjust_image_gamma(image_uint8[:, :, ch_idx], ch_gamma)
        return image_uint8

    # In case we get gamma of 1, do nothing (helpful for iterable case)
    if gamma == 1:
        return image_uint8
    return np.round(255 * np.pow(image_uint8.astype(np.float32) * (1.0 / 255.0), gamma)).astype(np.uint8)


def make_alpha_image_from_mask(
    image_bgr: ndarray,
    mask_image: ndarray,
    mask_bgr_components: bool = True,
    invert_mask: bool = False,
    interpolation: OCVInterp = None,
) -> ndarray:
    """
    Helper used to create an image with an alpha channel (e.g. transparency) from
    a normal image + mask. The provided mask should either be boolean, float or uint8.
    Notes:
    - If a boolean mask is given, it will be converted to uint8 using: uint8(mask) * 255
    - If a non-boolean/uint8 mask (e.g. float) is given is will be converted using: uint8(mask * 255)
    - If a mask with more than 1 channel is given, only the first (e.g. 'red') channel will be used
    - The input image can be given as grayscale, bgr or even bgra
    - If the given image and mask are different sizes, the mask will be scaled to match
      the size of the image using the provided 'interpolation'
    - If 'mask_bgr_components' is True, then the bgr channels will have the mask applied,
      this can reduce the resulting filesize considerably when saving as .png
    - If 'invert_mask' is True, then bright areas of the mask will be made transparent

    Returns:
        image_bgra
    """

    # Make sure we have a uint8 mask
    mask_uint8 = mask_image
    if mask_image.dtype != np.uint8:
        mask_uint8 = np.uint8(mask_image) * 255 if mask_image.dtype == np.bool else np.uint8(mask_image * 255)

    # Make sure mask is single-channeled
    mask_uint8 = mask_uint8[:, :, 0] if mask_uint8.ndim == 3 else mask_uint8
    if mask_uint8.ndim != 2:
        raise TypeError("Unable to handle mask input, expecting 2 or 3 dimensions: HxW or HxWxC")

    # Make sure we're dealing with an image that has color channels
    img_bgra = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR) if image_bgr.ndim == 2 else image_bgr.copy()
    if img_bgra.ndim != 3:
        raise TypeError("Unable to handle image input, expecting 2 or 3 dimensions: HxW or HxWxC")

    # Add an alpha channel to the image if it doesn't have one
    if img_bgra.shape[2] == 3:
        img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2BGRA)
    if img_bgra.shape[2] != 4:
        raise TypeError("Unable to handle image input, expecting BGR or BGRA image shape: HxWx3 or HxWx4")

    # Scale mask to match image size
    mask_h, mask_w = mask_uint8.shape[0:2]
    img_h, img_w = image_bgr.shape[0:2]
    if mask_h != img_h or mask_w != img_w:
        mask_uint8 = cv2.resize(mask_uint8, (img_w, img_h), interpolation=interpolation)

    # Flip mask bits to invert
    if invert_mask:
        mask_uint8 = cv2.bitwise_not(mask_uint8)

    # Store mask as alpha channel
    if mask_bgr_components:
        img_bgra[:, :, 0] = cv2.bitwise_and(img_bgra[:, :, 0], mask_uint8)
        img_bgra[:, :, 1] = cv2.bitwise_and(img_bgra[:, :, 1], mask_uint8)
        img_bgra[:, :, 2] = cv2.bitwise_and(img_bgra[:, :, 2], mask_uint8)
    img_bgra[:, :, 3] = mask_uint8

    return img_bgra


def make_horizontal_gradient_image(
    image_hw: IMGSHAPE_HW,
    left_color: COLORU8 = (0, 0, 0),
    right_color: COLORU8 = (255, 255, 255),
) -> ndarray:
    h, w = image_hw[0:2]
    weight = np.linspace(0, 1, w, dtype=np.float32)
    weight = np.expand_dims(weight, axis=(0, 2))
    col_1px = (1.0 - weight) * np.float32(left_color) + weight * np.float32(right_color)
    return np.tile(col_1px.astype(np.uint8), (h, 1, 1))


def make_vertical_gradient_image(
    image_hw: IMGSHAPE_HW,
    top_color: COLORU8 = (0, 0, 0),
    bottom_color: COLORU8 = (255, 255, 255),
) -> ndarray:
    h, w = image_hw[0:2]
    weight = np.linspace(0, 1, h, dtype=np.float32)
    weight = np.expand_dims(weight, axis=(1, 2))
    col_1px = (1.0 - weight) * np.float32(top_color) + weight * np.float32(bottom_color)
    return np.tile(col_1px.astype(np.uint8), (1, w, 1))


def dirty_blur(
    image_uint8: ndarray,
    blur_strength: float = 2,
    aspect_adjust: float = 0,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT,
    random_seed: int | None = None,
) -> ndarray:
    """
    Function used to apply a blurring effect based on re-sampling.
    The aspect_adjust input should be given as a value between -1 and +1.
    A value of -1 will re-sample entirely along the x-axis, while
    a value of +1 will re-sample entirely along the y-axis. A value
    of 0 will re-sample along both equally (in a circular pattern).

    Note, this function is very slow, as the sampling is re-computed
    every time it's called! This is only meant for infrequent use. For
    cases requiring repeat/frequent 'blurring', consider re-implementing
    with cached sampling meshes for speed-up!

    Returns:
        resampled_image_uint8
    """

    # Use seeded/non-seeded random number generation
    if random_seed is not None:
        noise_gen = np.random.default_rng(random_seed)
        noise_func = lambda h, w: noise_gen.standard_normal((h, w))
    else:
        noise_func = lambda h, w: np.random.randn(h, w)

    # Build base sampling mesh (if we use this mesh only, we would get back the original image)
    img_hw = image_uint8.shape[0:2]
    yxsample = [np.linspace(0, s - 1, s, dtype=np.float32) for s in img_hw]
    ygrid, xgrid = np.meshgrid(*yxsample, indexing="ij")

    # Perturb base mesh using random (circular) jitter to get re-sampling blur effect
    rad_grid = noise_func(*img_hw) * blur_strength
    ang_grid = noise_func(*img_hw) * (np.pi * 2.0)
    x_ar, y_ar = np.clip((1 - aspect_adjust, 1 + aspect_adjust), 0, 1)
    xgrid += np.cos(ang_grid) * rad_grid * x_ar
    ygrid += np.sin(ang_grid) * rad_grid * y_ar

    return cv2.remap(image_uint8, xgrid, ygrid, interpolation, borderMode=border_mode)


def kuwahara_filter(
    image_uint8: ndarray,
    quadrant_size_xy: int | tuple[int, int] = 3,
    internal_dtype=np.float32,
) -> ndarray:
    """
    Function which applies Kuwahara filtering to an image.
    Often produces a painterly looking result.

    Each pixel in the image can be thought of as being contained
    within 4 quadrants, where the pixel is a corner of each of the
    quadrants (e.g. top-left, top-right, bottom-left, bottom-right).
    The Kuwahara filter works by replacing each pixel with the average
    value of the quadrant with the lowest (grayscale/brightness) variance.

    By taking the average value, the filter has a smoothing/blurring effect.
    However, by using the quadrant with the lowest variance, quadrants
    with sharp edges generally won't be used/averaged.
    This helps avoid blurring edges.

    For more information, see:
        https://en.wikipedia.org/wiki/Kuwahara_filter

    Returns:
        filtered_image_uint8

    Notes:
    - the input image is assumed to be in BGR format or grayscale
    - assumes image is uint8, other formats may still work (not tested)
    - the internal_dtype should be np.float32 or np.int32 (others not tested),
      this only effects internal usage, the output will have the same type
      as the input image (e.g. uint8)
    - quadrant_size_xy can be given as a single integer, in which
      case the value will be shared for both x & y
    """

    # Force sizing to be an (x, y) tuple
    if isinstance(quadrant_size_xy, int):
        quadrant_size_xy = (quadrant_size_xy, quadrant_size_xy)

    # Do nothing if we get zero quadrant_size sizing
    quad_x, quad_y = [max(0, size) for size in quadrant_size_xy]
    if quad_x == 0 and quad_y == 0:
        return image_uint8
    q_xy = (quad_x, quad_y)

    # For clarity
    input_has_channels = image_uint8.ndim == 3
    input_dtype = image_uint8.dtype
    kernel_xy = [(1 + size) for size in q_xy]
    border_type = cv2.BORDER_REFLECT101
    anchor = (0, 0)
    # ^^^ Use anchor of 0 for better predictability, otherwise changes on even window sizes!

    # Pad input, so that we can use shifted-indexing for kuwahara quadrants
    padded_img_u8 = cv2.copyMakeBorder(image_uint8, quad_y, quad_y, quad_x, quad_x, borderType=border_type)
    if not input_has_channels:
        padded_img_u8 = cv2.cvtColor(padded_img_u8, cv2.COLOR_GRAY2BGR)

    # Convert input to format with a single 'gray' channel (e.g. YUV, though others work as well)
    # - filter only operates on gray values, non-gray channels are needed to re-build output at the end
    cvt_img_u8 = cv2.cvtColor(padded_img_u8, cv2.COLOR_BGR2YUV)
    gray_img = cvt_img_u8[:, :, 0].astype(internal_dtype)
    u_img_u8 = cvt_img_u8[:, :, 1]
    v_img_u8 = cvt_img_u8[:, :, 2]

    # Compute mean/variance in window around every pixel
    mean_img = cv2.blur(gray_img, kernel_xy, anchor=anchor, borderType=border_type)
    sqmean_img = cv2.blur(np.square(gray_img), kernel_xy, anchor=anchor, borderType=border_type)
    var_img = sqmean_img - np.square(mean_img)

    # Create mean/variance 'quadrants' per-pixel, for computing kuwahara
    slice_ax, slice_ay = [slice(size, -size) if size > 0 else slice(None) for size in q_xy]
    slice_bx, slice_by = [slice(0, -2 * size) if size > 0 else slice(None) for size in q_xy]
    slice_iter = ((slice_ay, slice_ax), (slice_ay, slice_bx), (slice_by, slice_ax), (slice_by, slice_bx))
    var_stack = np.dstack([var_img[sy, sx] for sy, sx in slice_iter])
    mean_stack = np.dstack([mean_img[sy, sx] for sy, sx in slice_iter])
    u_stack = np.dstack([u_img_u8[sy, sx] for sy, sx in slice_iter])
    v_stack = np.dstack([v_img_u8[sy, sx] for sy, sx in slice_iter])

    # Get index of quadrant with the lowest gray variance (main trick of kuwahara filtering!)
    quadrant_idx = np.argmin(var_stack, axis=-1)

    # Re-construct image using kuwahara index to grab per-pixel 'best' quadrants from gray + other channels
    row_idx = np.arange(quadrant_idx.shape[0])[:, None]
    col_idx = np.arange(quadrant_idx.shape[1])[None, :]
    cvt_out = np.empty_like(image_uint8)
    cvt_out[:, :, 0] = mean_stack[row_idx, col_idx, quadrant_idx].astype(input_dtype)
    cvt_out[:, :, 1] = u_stack[row_idx, col_idx, quadrant_idx]
    cvt_out[:, :, 2] = v_stack[row_idx, col_idx, quadrant_idx]
    output = cv2.cvtColor(cvt_out, cv2.COLOR_YUV2BGR)
    if not input_has_channels:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    return output


def convert_xy_norm_to_px(image_shape: IMGSHAPE_HW, *xy_norm_coords: XYNORM) -> tuple[XYPX] | XYPX:
    """
    Helper used to convert normalized xy coordinates to pixel coordinates,
    based on the provided image shape.

    Note that any number of input xy coordinates can be given, and the
    same number of coordinates will be returned by the function as a tuple.
    However, if a single xy coord is provided, then the function will output
    the converted coordinate directly (as opposed to a tuple of 1 xy coord).

    Returns:
        *xy_px_coords
    """

    # For convenience
    img_h, img_w = image_shape[0:2]
    x_scale = img_w - 1
    y_scale = img_h - 1

    # Compute crop coords
    xy_px_list = tuple((round(x * x_scale), round(y * y_scale)) for x, y in xy_norm_coords)

    return xy_px_list if len(xy_px_list) > 1 else xy_px_list[0]


def histogram_equalization(
    image_uint8: ndarray,
    min_pct: float = 0.0,
    max_pct: float = 1.0,
    channel_index: int | None = None,
) -> ndarray:
    """
    Function used to perform histogram equalization on a uint8 image.
    This function uses the built-in opencv function: cv2.equalizeHist(...)
    when using a (0, 1) min/max range.

    When a range other than (0,1) is given, the input
    will be equalized between this range.

    If a multi-channel image is provided, then each channel will
    be independently equalized!

    Returns:
        image_uint8_equalized
    """

    # If a channel index is given, equalize only that channel
    if channel_index is not None:
        result = image_uint8.copy()
        result[:, :, channel_index] = histogram_equalization(image_uint8[:, :, channel_index], min_pct, max_pct, None)
        return result

    # Make sure min/max are properly ordered & separated
    min_uint8, max_uint8 = [int(round(255 * value)) for value in sorted((min_pct, max_pct))]
    min_uint8, max_uint8 = (254, 255) if min_uint8 >= 254 else (min_uint8, max_uint8)
    min_uint8, max_uint8 = (0, 1) if max_uint8 <= 1 else (min_uint8, max_uint8)
    if min_uint8 <= 0 and max_uint8 >= 255:
        if image_uint8.ndim == 1:
            return cv2.equalizeHist(image_uint8).squeeze()
        elif image_uint8.ndim == 2:
            return cv2.equalizeHist(image_uint8)
        elif image_uint8.ndim == 3:
            num_channels = image_uint8.shape[2]
            return np.dstack([histogram_equalization(image_uint8[:, :, c]) for c in range(num_channels)])

    # Compute histogram of input
    h_bins = np.arange(257, dtype=np.int32)
    bin_counts, _ = np.histogram(image_uint8, h_bins)

    # Compute cumulative density function from histogram counts
    # -> Normalize cdf between min_pct & max_pct, so equalization applies to this range only!
    cdf = bin_counts.cumsum()
    cdf_min, cdf_max = cdf[0], cdf[-1]
    cdf_norm = min_pct + (max_pct - min_pct) * (cdf - cdf_min) / (cdf_max - cdf_min)

    # Apply LUT mapping to input
    equalization_lut = np.round(cdf_norm * 255).astype(np.uint8)
    return equalization_lut[image_uint8]


def encode_image_b64str(image: ndarray, file_extention: str = ".png") -> str:
    """Helper used to encode images in a string (base-64) format"""
    ok_encode, encoded_array = cv2.imencode(file_extention, image)
    assert ok_encode, f"Error while base-64 encoding ({file_extention}) image!"
    return b64encode(encoded_array).decode("utf-8")


def decode_b64str_image(b64_str: str) -> ndarray:
    """Helper used to decode images from a base-64 encoding"""
    decoded_bytes = b64decode(b64_str)
    decoded_array = np.frombuffer(decoded_bytes, np.uint8)
    return cv2.imdecode(decoded_array, cv2.IMREAD_UNCHANGED)
