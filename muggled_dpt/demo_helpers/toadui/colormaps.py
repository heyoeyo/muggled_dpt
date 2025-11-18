#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import BaseCallback
from .helpers.styling import UIStyle, get_background_thickness
from .helpers.drawing import draw_box_outline
from .helpers.colors import make_colormap_from_keypoints

# For type hints
from numpy import ndarray
from .helpers.types import SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class ColormapsBar(BaseCallback):
    """
    UI element used to render a horizontal arrangement of colormap 'buttons'
    Similar to using:
        HStack(ToggleImage(colormap_image_1), ToggleImage(...), ... etc)

    However this implementation is more efficient and is better behaved when
    scaling display sizes.
    """

    # .................................................................................................................

    def __init__(self, *colormap_codes_or_luts: int | ndarray | None, height=40, minimum_width=128):
        """
        Colormap inputs should be provided as either:
            1) integer opencv colormap codes, e.g. cv2.COLORMAP_VIRIDIS
            2) LUTs, which are uint8 numpy arrays of shape: 1x256x3, holding BGR mappings
            3) None, which is interpretted to mean a grayscale colormap
        """

        # Store basic state
        self._is_changed = True
        self._cmap_idx = 0

        # Store sizing settings
        self._cached_image = np.zeros((1, 1, 3), dtype=np.uint8)
        self._need_rerender = True
        self._bboxes_xy1xy2_px = np.int32()

        # Check & store each provided colormap and interpret 'None' as grayscale colormap
        self._cmap_luts_list = []
        self._num_cmaps = 0
        for cmap in colormap_codes_or_luts:
            self.add_colormap(cmap)

        # Sanity check. Make sure we have at least 1 colormap
        if self._num_cmaps == 0:
            self._cmap_luts_list = [make_gray_colormap()]
            self._num_cmaps = 1

        # Set up element styling
        self.style = UIStyle(
            outline_color=(0, 0, 0),
            highlight_fg_color=(255, 255, 255),
            highlight_bg_color=(0, 0, 0),
            highlight_thickness=1,
        )

        # Inherit from parent
        super().__init__(height, minimum_width, is_flexible_h=False, is_flexible_w=True)

    # .................................................................................................................

    def read(self) -> tuple[bool, int, ndarray]:
        """
        Returns:
            is_changed, selected_colormap_index, selected_colormap_lut

        -> This function is not needed, unless needing to detect the change
           state or if there is a need for the selected index.
        -> If only colormapping is needed, use the .apply_colormap(image_1ch) function!
        """

        is_changed = self._is_changed
        self._is_changed = False

        cmap_idx = self._cmap_idx
        cmap_lut = self._cmap_luts_list[cmap_idx]

        return is_changed, cmap_idx, cmap_lut

    # .................................................................................................................

    def _on_left_click(self, cbxy, cbflags) -> None:

        # Map normalized x-coord to button boundaries
        x_norm = cbxy.xy_norm[0]
        new_cmap_idx = int(x_norm * self._num_cmaps)
        new_cmap_idx = int(np.clip(new_cmap_idx, 0, self._num_cmaps - 1))

        # Store updated selection
        self.set_colormap(new_cmap_idx)

        return

    # .................................................................................................................

    def apply_colormap(self, image_uint8_1ch) -> ndarray:
        """Apply the currently selected colormap to the provided (1-channel) image"""
        return apply_colormap(image_uint8_1ch, self._cmap_luts_list[self._cmap_idx])

    # .................................................................................................................

    def set_colormap(self, colormap_index: int) -> SelfType:
        self._is_changed |= colormap_index != self._cmap_idx
        self._cmap_idx = colormap_index
        self._need_rerender = self._is_changed
        return self

    # .................................................................................................................

    def next(self, increment: int = 1) -> SelfType:
        """Function used to select the next colormap (with wrap-around)"""
        new_idx = (self._cmap_idx + increment) % self._num_cmaps
        self.set_colormap(new_idx)
        return self

    # .................................................................................................................

    def prev(self, decrement: int = 1) -> SelfType:
        """Function used to select the previous colormap (with wrap-around)"""
        return self.next(-decrement)

    # .................................................................................................................

    def add_colormap(self, colormap_code_or_lut: int | ndarray | None, insert_index: int | None = None) -> SelfType:

        # Handle input. Interpret 'None' as grayscale colormap
        if colormap_code_or_lut is None:
            colormap_code_or_lut = make_gray_colormap()
        elif isinstance(colormap_code_or_lut, int):
            gray_img = make_gray_colormap()
            colormap_code_or_lut = cv2.applyColorMap(gray_img, colormap_code_or_lut)

        if not isinstance(colormap_code_or_lut, ndarray):
            raise TypeError("Unrecognized colormap type! Must be a cv2 colormap code or a 1x256x3 array")

        assert colormap_code_or_lut.shape == (1, 256, 3), "Bad colormap shape, must be: 1x256x3"

        insert_index = insert_index if insert_index is not None else len(self._cmap_luts_list)
        self._cmap_luts_list.insert(insert_index, colormap_code_or_lut)
        self._num_cmaps = len(self._cmap_luts_list)
        self._need_rerender = True

        return self

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Only render if sizing changes or we force a re-render
        # (we don't expect frequent changes to the colormap display!)
        img_h, img_w = self._cached_image.shape[0:2]
        if img_h != h or img_w != w or self._need_rerender:
            self._need_rerender = False

            # Figure out how wide each button should be (without gaps)
            w_per_btn = np.diff(np.int32(np.round(np.linspace(0, 1, self._num_cmaps + 1) * w)))

            imgs_list = []
            for btn_idx, lut in enumerate(self._cmap_luts_list):

                btn_w = w_per_btn[btn_idx]
                img = cv2.resize(lut, dsize=(btn_w, h), interpolation=cv2.INTER_NEAREST_EXACT)
                img = draw_box_outline(img, self.style.outline_color, 1)
                if btn_idx == self._cmap_idx:
                    bg_thick = get_background_thickness(self.style.highlight_thickness)
                    img = draw_box_outline(img, self.style.highlight_bg_color, bg_thick)
                    img = draw_box_outline(img, self.style.highlight_fg_color, self.style.highlight_thickness)

                imgs_list.append(img)

            # Cache final image and clear re-render flag
            self._cached_image = np.hstack(imgs_list)

        return self._cached_image

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def load_colormap(path_to_image: str, interpolation: int | None = None) -> tuple[bool, ndarray]:
    """
    Helper used to load an image and resize to the 1x256x3 sizing
    needed by opencv (for LUTs). The loaded image will be rotated
    if needed (i.e. vertical colormaps can be loaded).

    If the loaded colormap needs to be resized, it will use the
    provided interpolation type, which should be an opencv INTER code,
    for example: cv2.INTER_NEAREST

    Returns:
        ok_load, colormap_array (shape: 1x256x3)
    """

    loaded_cmap = cv2.imread(path_to_image)
    ok_load = loaded_cmap is not None
    if ok_load:

        # Make sure colormap is wider than it is tall (expecting HW: 1x256)
        h, w = loaded_cmap.shape[0:2]
        if h > w:
            loaded_cmap = np.rot90(loaded_cmap, 1)
            h, w = loaded_cmap.shape[0:2]

        # Force colormap to 1x256 sizing
        if h != 1 or w != 256:
            loaded_cmap = cv2.resize(loaded_cmap, dsize=(256, 1), interpolation=interpolation)

        # Make sure colormap has 3 channels
        if loaded_cmap.ndim < 3:
            loaded_cmap = cv2.cvtColor(loaded_cmap, cv2.COLOR_GRAY2BGR)

    return ok_load, loaded_cmap


# .....................................................................................................................


def apply_colormap(image_uint8_1ch: ndarray, colormap_code_or_lut: int | None | ndarray) -> ndarray:
    """
    Converts a uint8 image (numpy array) into a bgr color image using opencv colormaps
    or using LUTs (numpy arrays of shape 1x256x3).
    Colormap code should be from opencv, which are accessed with: cv2.COLORMAP_{name}
    LUTs should be numpy arrays of shape 1x256x3, where each of the 256 entries
    encodes a bgr value which maps on to a 0-255 range.

    Expects an input image of shape: HxW or HxWx1 (e.g. grayscale only)
    """

    if isinstance(colormap_code_or_lut, int):
        # Handle maps provided as opencv colormap codes (e.g. cv2.COLORMAP_VIRIDIS)
        return cv2.applyColorMap(image_uint8_1ch, colormap_code_or_lut)

    elif isinstance(colormap_code_or_lut, ndarray):
        # Handle maps provided as LUTs (e.g. 1x256x3 numpy arrays)
        image_ch3 = cv2.cvtColor(image_uint8_1ch, cv2.COLOR_GRAY2BGR)
        return cv2.LUT(image_ch3, colormap_code_or_lut)

    elif colormap_code_or_lut is None:
        # Return grayscale image if no mapping is provided
        return cv2.cvtColor(image_uint8_1ch, cv2.COLOR_GRAY2BGR)

    # Error if we didn't deal with the colormap above
    raise TypeError(f"Error applying colormap, unrecognized colormap type: {type(colormap_code_or_lut)}")


# .....................................................................................................................


def make_gray_colormap(num_samples=256) -> ndarray:
    """Makes a colormap in opencv LUT format, for grayscale output using cv2.LUT function"""
    return make_colormap_from_keypoints(np.float32([(0, 0, 0), (1, 1, 1)]), num_samples=num_samples)


# .....................................................................................................................


def make_spectral_colormap(num_samples=256) -> ndarray:
    """
    Creates a colormap which is a variation on the built-in opencv 'TURBO' colormap,
    but with muted colors and overall less harsh contrast. The colormap originates from
    the matplotlib library, where it is the reversed version of a colormap called
    'Spectral'. It is being generated this way to avoid requiring the full matplotlib dependency!

    The original colormap definition can be found here:
    https://github.com/matplotlib/matplotlib/blob/30f803b2e9b5e237c5c31df57f657ae69bec240d/lib/matplotlib/_cm.py#L793
    -> The version here uses a slightly truncated copy of the values
    -> This version is also pre-reversed compared to the original
    -> Color keypoints are in bgr order (the original uses rgb ordering, opencv needs bgr)

    Returns a colormap which can be used with opencv, for example:

        spectral_colormap = make_spectral_colormap()
        gray_image_3ch = cv2.cvtColor(gray_image_1ch, cv2.COLOR_GRAY2BGR)
        colormapped_image = cv2.LUT(gray_image_3ch, spectral_colormap)

    The result has a shape of: 1xNx3, where N is number of samples (256 by default and required for cv2.LUT usage)
    """

    # Colormap keypoints from matplotlib. The colormap is produced by linear-interpolation of these points
    spectral_rev_bgr = np.float32(
        (
            (0.635, 0.310, 0.369),
            (0.741, 0.533, 0.196),
            (0.647, 0.761, 0.400),
            (0.643, 0.867, 0.671),
            (0.596, 0.961, 0.902),
            (0.749, 1.000, 1.000),
            (0.545, 0.878, 0.996),
            (0.380, 0.682, 0.992),
            (0.263, 0.427, 0.957),
            (0.310, 0.243, 0.835),
            (0.259, 0.004, 0.620),
        )
    )

    return make_colormap_from_keypoints(spectral_rev_bgr, num_samples=num_samples)


# .....................................................................................................................


def make_tree_colormap(num_samples=256) -> ndarray:
    """
    Creates a custom colormap based on 'tree-like' colors

    Returns a colormap which can be used with opencv, for example:

        tree_colormap = make_tree_colormap()
        gray_image_3ch = cv2.cvtColor(gray_image_1ch, cv2.COLOR_GRAY2BGR)
        colormapped_image = cv2.LUT(gray_image_3ch, tree_colormap)

    Returns a uint8 numpy array with shape: 1xNx3, where N is the number of samples (256 by default)
    """

    tree_bgr = (
        [18, 9, 0],
        [37, 47, 0],
        [56, 81, 0],
        [69, 110, 11],
        [72, 146, 28],
        [66, 176, 58],
        [57, 200, 97],
        [50, 226, 137],
        [66, 243, 180],
        [125, 253, 224],
    )

    # Make sure brightness is evenly distributed
    base_cmap = make_colormap_from_keypoints(np.float32(tree_bgr) / 255.0, num_samples=num_samples)
    yrb = cv2.cvtColor(base_cmap, cv2.COLOR_BGR2YCrCb)
    yrb[:, :, 0] = make_gray_colormap(num_samples)[:, :, 0]
    cmap = cv2.cvtColor(yrb, cv2.COLOR_YCrCb2BGR)

    # Force black/white end-points
    cmap[:, 0, :] = 0
    cmap[:, -1, :] = 255
    return cmap


# .....................................................................................................................


def make_wa_rainbow_colormap(num_samples=256, phase_norm=0) -> ndarray:
    """
    Wrap-around rainbow colormap with pastel-like coloring.
    This is meant as an alternative to the built-in
    opencv rainbow colormap, which does not wrap-around.
    (wrap-around can be useful for plotting cyclical values)

    Returns:
        colormap_array (1xNx3, where N is number of samples)

    Note that this mapping starts/ends with a blue color!
    The starting color (e.g. 'phase') of the colormap can be
    altered using the phase_norm input.
    For example, using a phase_norm value of 1/3 will give a
    colormap starting/ending at red.

    Based on an article by Inigo Quilez:
    https://iquilezles.org/articles/palettes/
    """

    # For clarity
    twopi = 2.0 * np.pi
    brightness = 0.6
    saturation = 0.5

    angle = np.linspace(0, 1, num_samples, endpoint=False, dtype=np.float32) + phase_norm
    rval = brightness + saturation * np.cos(twopi * (angle + 2 / 3))
    gval = brightness + saturation * np.cos(twopi * (angle + 1 / 3))
    bval = brightness + saturation * np.cos(twopi * (angle + 0 / 3))

    # Bundle colors together
    bgr_cmap = np.clip(np.dstack((bval, gval, rval)), 0, 1)
    bgr_cmap = np.round(255 * bgr_cmap).astype(np.uint8)
    return adjust_colormap_gamma(bgr_cmap, 0.75)


def make_hsv_rainbow_colormap(num_samples=256) -> ndarray:
    """
    Function which creates a highly saturated rainbow
    colormap using the HSV colorspace.
    This is very nearly a wrap-around mapping, except
    for duplicated values at the end points.

    Returns:
        colormap_array (1xNx3, where N is number of samples)
    """

    hsv_array = np.full((1, num_samples, 3), 255, dtype=np.uint8)
    hsv_array[:, :, 0] = np.round(np.linspace(0, 255, num_samples)).astype(np.uint8)
    bgr_cmap = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR_FULL)

    return bgr_cmap


# .....................................................................................................................


def adjust_colormap_gamma(colormap: int | None | ndarray, gamma=1.0) -> ndarray:
    """
    Helper used to apply 'gamma correction' to a colormap
    Gamma values below 1 will brighten the colormap,
    while values above 1 will darken it.
    Returns gamma-corrected colormap
    """

    # Assume we're working with regular 1x256x3 colormaps, unless we get something different
    num_samples = 256
    if isinstance(colormap, ndarray):
        num_samples = max(colormap.shape)

    # Create uint8 BGR image from colormap and apply gamma
    gray_img = np.expand_dims(np.arange(num_samples, dtype=np.uint8), 0)
    bgr_img = apply_colormap(gray_img, colormap)
    bgr_img = np.float32(bgr_img / 255.0)
    bgr_img = np.pow(bgr_img, gamma)
    return np.uint8(np.round(255 * bgr_img))
