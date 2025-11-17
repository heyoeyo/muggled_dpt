#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import BaseCallback, CBEventXY, CBEventFlags, CBRenderSizing
from .helpers.images import blank_image
from .helpers.sizing import get_image_hw_to_fit_region, get_image_hw_to_fit_by_ar
from .helpers.styling import UIStyle

# For type hints
from numpy import ndarray
from .helpers.types import HWPX, SelfType, XYNORM, XYPX, IsLMR
from .helpers.ocv_types import OCVInterp


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class DynamicImage(BaseCallback):
    """Element used to hold images that may be updated (using the .set_image(...) function)"""

    # .................................................................................................................

    def __init__(
        self,
        image: ndarray | None = None,
        min_side_length: int = 128,
        resize_interpolation: OCVInterp = None,
        is_flexible_h: bool = True,
        is_flexible_w: bool = True,
    ):

        # Inherit from parent
        super().__init__(min_side_length, min_side_length, is_flexible_h, is_flexible_w)

        # Store sizing info
        self._min_side_length = min_side_length
        self._targ_h = -1
        self._targ_w = -1

        # Store state for mouse interaction
        self._is_lmr_clicked = [False, False, False]
        self._mouse_xy = CBEventXY.default()

        # Default to blank square image if given 'None' image input
        init_image = blank_image(min_side_length, min_side_length) if image is None else image
        self._full_image = None
        self._render_image = blank_image(1, 1)
        self.set_image(init_image)

        # Set up element styling
        self.style = UIStyle(interpolation=resize_interpolation)

    # .................................................................................................................

    def get_render_hw(self) -> HWPX:
        """
        Report the most recent render resolution of the image. This can be
        used to scale new images to match previous render sizes, which can
        help reduce jittering when giving images that repeatedly change size.
        Returns:
            render_height, render_width
        """
        return self._render_image.shape[0:2]

    def set_image(self, image: ndarray) -> SelfType:
        """Set the internally held image data"""
        self._full_image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._targ_h = -1
        self._targ_w = -1
        return self

    def read_mouse_xy(self) -> tuple[IsLMR, CBEventXY]:
        """
        Read most recent mouse interaction, including whether the mouse was clicked.
        Note that 'is_lmr_clicked' means 'is left/middle/right clicked'
        Returns:
            is_lmr_clicked, mouse_xy_event
        """
        is_lmr_clicked = IsLMR(*self._is_lmr_clicked)
        self._is_lmr_clicked = [False, False, False]
        return is_lmr_clicked, self._mouse_xy

    def save(self, save_path: str) -> None:
        """Save current image data to the file system"""
        cv2.imwrite(save_path, self._full_image)

    # .................................................................................................................

    def _on_left_click(self, cbxy, cbflags) -> None:
        self._is_lmr_clicked[0] = True
        self._mouse_xy = cbxy
        return

    def _on_middle_click(self, cbxy, cbflags) -> None:
        self._is_lmr_clicked[1] = True
        self._mouse_xy = cbxy
        return

    def _on_right_click(self, cbxy, cbflags) -> None:
        self._is_lmr_clicked[2] = True
        self._mouse_xy = cbxy
        return

    def _on_move(self, cbxy, cbflags) -> None:
        self._mouse_xy = cbxy
        return

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        img_h, img_w = self._full_image.shape[0:2]
        scaled_w = max(self._cb_rdr.min_w, round(img_w * h / img_h))
        return scaled_w

    def _get_height_given_width(self, w: int) -> int:
        img_h, img_w = self._full_image.shape[0:2]
        scaled_h = max(self._cb_rdr.min_h, round(img_h * w / img_w))
        return scaled_h

    def _get_height_and_width_without_hint(self) -> HWPX:
        return self._full_image.shape[0:2]

    def _get_dynamic_aspect_ratio(self):
        is_flexible = self._cb_rdr.is_flexible_h and self._cb_rdr.is_flexible_h
        return self._full_image.shape[1] / self._full_image.shape[0] if is_flexible else None

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render to target size if needed
        if self._targ_h != h or self._targ_w != w:
            self._targ_h = h
            self._targ_w = w

            # Scale image to fit within given sizing
            img_h, img_w = self._full_image.shape[0:2]
            scale = min(h / img_h, w / img_w)
            fill_wh = (round(scale * img_w), round(scale * img_h))
            scaled_image = cv2.resize(self._full_image, dsize=fill_wh, interpolation=self.style.interpolation)

            # Store rendered result for re-use
            self._render_image = scaled_image

        return self._render_image

    # .................................................................................................................


class StretchImage(DynamicImage):
    """
    Element used to hold images that can be updated and are meant to
    'stretch to fill' the space that they have available. By default,
    this class targets a specific aspect ratio for rendering, for example,
    stretching an image to fill a square (aspect ratio of 1). If the target
    aspect ratio is set to None, then the image aspect ratio will be used
    if there is space to do so, otherwise the image will stretch to
    fill whatever space is available.
    """

    # .................................................................................................................

    def __init__(
        self,
        image: ndarray,
        aspect_ratio: float | None = 1,
        min_h: int = 128,
        min_w: int = 128,
        resize_interpolation: OCVInterp = None,
        is_flexible_h=True,
        is_flexible_w=True,
    ):

        # Precompute aspect ratio settings for scaling
        self._has_ar = aspect_ratio is not None
        self._w_over_h = max(aspect_ratio, 0.001) if self._has_ar else -1
        self._h_over_w = 1.0 / self._w_over_h

        # Override parent sizing & styling
        super().__init__(image, resize_interpolation=resize_interpolation)
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flexible_h, is_flexible_w)

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        ar = self._w_over_h if self._has_ar else self._full_image.shape[1] / self._full_image.shape[0]
        return max(self._cb_rdr.min_w, round(h * ar))

    def _get_height_given_width(self, w: int) -> int:
        ar_inv = self._h_over_w if self._has_ar else self._full_image.shape[0] / self._full_image.shape[1]
        return max(self._cb_rdr.min_h, round(w * ar_inv))

    def _get_height_and_width_without_hint(self) -> HWPX:
        return self._full_image.shape[0:2]

    def _get_dynamic_aspect_ratio(self):
        return self._w_over_h if self._has_ar else None

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render to target size if needed
        if self._targ_h != h or self._targ_w != w:
            self._targ_h = h
            self._targ_w = w
            self._render_image = cv2.resize(self._full_image, dsize=(w, h), interpolation=self.style.interpolation)

        return self._render_image

    # .................................................................................................................


class FixedARImage(DynamicImage):
    """Element used to hold images that render to a fixed aspect ratio"""

    # .................................................................................................................

    def __init__(
        self,
        image: ndarray | None = None,
        aspect_ratio: float | None = None,
        min_side_length: int = 128,
        resize_interpolation: OCVInterp = None,
        is_flexible=True,
    ):

        # Sanity check. We need a target aspect ratio or an image (which will determine the aspect ratio)
        missing_img = image is None
        missing_ar = aspect_ratio is None
        assert not (missing_img and missing_ar), "Must provide one of inputs: image or aspect_ratio"
        if missing_ar:
            aspect_ratio = image.shape[1] / image.shape[0]

        # Precompute aspect ratio settings for scaling
        self._w_over_h = aspect_ratio
        self._h_over_w = 1.0 / self._w_over_h

        # Figure out minimum sizing, based on aspect ratio
        is_tall = self._w_over_h < 1
        min_h = round(min_side_length * self._h_over_w) if is_tall else min_side_length
        min_w = min_side_length if is_tall else round(min_side_length * self._w_over_h)
        if missing_img:
            image = blank_image(min_h, min_w)

        # Override parent sizing & styling
        super().__init__(image, resize_interpolation=resize_interpolation)
        self._cb_rdr = CBRenderSizing(min_h, min_w, is_flexible, is_flexible)

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        return max(self._cb_rdr.min_w, round(h * self._w_over_h))

    def _get_height_given_width(self, w: int) -> int:
        return max(self._cb_rdr.min_h, round(w * self._h_over_w))

    def _get_height_and_width_without_hint(self) -> HWPX:
        img_h, img_w = self._full_image.shape[0:2]
        out_h, out_w = get_image_hw_to_fit_by_ar(self._w_over_h, (img_h, img_w), fit_within=False)
        return out_h, out_w

    def _get_dynamic_aspect_ratio(self):
        return self._w_over_h

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render to target size if needed
        if self._targ_h != h or self._targ_w != w:
            self._targ_h = h
            self._targ_w = w
            scale_h, scale_w = get_image_hw_to_fit_region(self._full_image.shape, (h, w))
            scale_wh = (max(1, scale_w), max(1, scale_h))
            self._render_image = cv2.resize(self._full_image, dsize=scale_wh, interpolation=self.style.interpolation)

        return self._render_image

    # .................................................................................................................


class ZoomImage(FixedARImage):
    """
    Display element used to show a 'zoomed-in' version of an image.
    Handles zooming internally (e.g. user only have to provide full image),
    as well as providing support for panning the zoomed image.
    """

    # .................................................................................................................

    def __init__(
        self,
        image: ndarray | None = None,
        initial_zoom_factor=0.5,
        allow_panning=True,
        min_side_length: int = 128,
        resize_interpolation: OCVInterp = cv2.INTER_NEAREST,
        is_flexible=True,
    ):

        # Allocate storage for re-render/cacheing checks, specific to zooming
        self._prev_full_h, self._prev_full_w = -1, -1
        self._need_zoom_rerender = True
        self._is_zoom_changed = True

        # Store zoom range parameters
        self._zoom_min_px: int = 32
        self._zoom_min_norm: float = 0.0
        self._zoom_delta_norm: float = 0.8

        # Store zoom boundary parameters
        self._zoom_factor: float = initial_zoom_factor
        self._zoom_boundary_px: int = 2
        self._zoom_xy_center_min_px: ndarray = np.int32((0, 0))
        self._zoom_xy_center_max_px: ndarray = np.int32((10, 10))

        # Storage for zooming crop coords (needed for computing relative positioning)
        self._zoom_xy_center_norm = (0.5, 0.5)
        self._zoom_xy1xy2_px = np.int32(((0, 0), (100, 100)))
        self._zoom_xy1xy2_norm = np.float32(((0, 0), (1, 1)))
        self._zoom_hw = (100, 100)

        # Storage for panning state
        self._allow_panning = allow_panning
        self._is_pressed = False
        self._pan_pixel_xy = np.int32((0, 0))
        self._pan_zoom_xy_center_norm = np.float32((0.5, 0.5))
        self._pan_zoom_shape = (200, 200)
        self._pan_zoom_xy1_px = np.int32((0, 0))

        # Inherit from parent
        aspect_ratio = 1
        super().__init__(image, aspect_ratio, min_side_length, resize_interpolation, is_flexible)
        self._cached_zoom_img = self._full_image.copy()

    # .................................................................................................................

    def is_changed(self):
        """Check if zoom has changed (either directly or from panning)"""
        is_changed, self._is_zoom_changed = self._is_zoom_changed, False
        return is_changed

    def map_full_to_zoom_coordinates(
        self, *xy_coords: tuple[float], input_is_normalized=False, normalize_output=True
    ) -> tuple[XYPX | XYNORM] | XYPX | XYNORM:
        """
        Function used to map coordinates on the original (unzoomed) image
        into coordinates on the zoomed image. This can be useful, for example,
        if trying to draw indicator (e.g. a circle) on both the full image
        and a zoomed copy, while maintaining location consistency.

        This function can take input in pixel units or 0-to-1 normalized units,
        and can return coordinate in pixel/normalized format, depending on the
        input_is_normalized & normalize_output arguments.

        Note that this function take any number of coords as input, and will
        return a matching sized tuple as output. However, if only a single
        xy coord. is given as input, the output will be a single xy coord.
        (as opposed to a tuple of 1 xy coord)

        Returns:
            *zoomed_xy_coords
        """

        # Offset xy coordinates with zoom crop top-left
        zx1, zy1 = (self._zoom_xy1xy2_norm[0] if input_is_normalized else self._zoom_xy1xy2_px[0]).tolist()
        offset_xy = (((x - zx1), (y - zy1)) for x, y in xy_coords)

        # Convert from pixels-to-normalized units or vice versa, depending on input/output units
        if input_is_normalized and not normalize_output:
            zoom_h, zoom_w = self._zoom_hw
            x_scale, y_scale = (zoom_w - 1), (zoom_h - 1)
            offset_xy = ((round(x_norm * x_scale), round(y_norm * y_scale)) for x_norm, y_norm in offset_xy)
        elif not input_is_normalized and normalize_output:
            zoom_h, zoom_w = self._zoom_hw
            x_scale, y_scale = 1.0 / (zoom_w - 1), 1.0 / (zoom_h - 1)
            offset_xy = ((x_px * x_scale, y_px * y_scale) for x_px, y_px in offset_xy)

        return tuple(offset_xy) if len(xy_coords) > 1 else offset_xy[0]

    def set_image(self, image: ndarray) -> SelfType:
        """Update image used for zooming"""
        self._full_image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._need_zoom_rerender = True

        # Check if we need to re-compute zoom region
        new_h, new_w = self._full_image.shape[0:2]
        image_hw_changed = new_h != self._prev_full_h or new_w != self._prev_full_w
        if image_hw_changed:
            self._update_zoom_boundary()

        self._prev_full_h = new_h
        self._prev_full_w = new_w

        return self

    def set_zoom_center(self, center_xy: XYPX | XYNORM, is_normalized=True) -> SelfType:
        """
        Update zoom center location. Can be given in pixel units or normalize 0 to 1.
        If pixel units are given, be sure to set is_normalized=False.

        Note that the zoom center cannot be changed if the user is dragging
        the zoom location directly.
        """
        if not self._is_pressed:
            if not is_normalized:
                img_h, img_w = self._full_image.shape[0:2]
                center_xy = (center_xy[0] / (img_w - 1), center_xy[1] / (img_h - 1))
            self._set_zoom_center_unlocked(center_xy)
        return self

    def set_zoom_factor(self, zoom_factor_0to1: float) -> SelfType:
        """
        Update zoom amount. Higher values zoom further into the image.
        The amount of zoom can be altered using set_zoom_range(...)
        """
        self._zoom_factor = min(1, max(0, zoom_factor_0to1))
        self._update_zoom_boundary()
        return self

    def set_zoom_range(self, min_zoom_norm: float = 0.0, max_zoom_norm: float = 0.8, min_zoom_px: int = 32) -> SelfType:
        """
        Adjusts zoom range. The min_zoom_norm setting determines the smallest fraction
        of the image that can be displayed, when zoomed in fully, while max_zoom_norm
        is the largest fraction of the image when zoomed fully out.
        The min_zoom_px setting sets an absolute bound on pixel sizing when
        zoomed in fully. If the min_zoom_norm is 0, for example, then the min_zoom_px
        setting will determine how many pixels are used in the fully zoomed in image.
        """

        min_zoom_norm, max_zoom_norm = sorted(min(1, max(0, value)) for value in (min_zoom_norm, max_zoom_norm))
        self._zoom_min_px = max(0, min_zoom_px)
        self._zoom_min_norm = min(min_zoom_norm, 0.99)
        self._zoom_delta_norm = max(max_zoom_norm - min_zoom_norm, 0.01)

        return self

    # .................................................................................................................

    def _set_zoom_center_unlocked(self, center_xy_norm: XYNORM) -> None:
        """
        Helper used to update zoom center location, without checks
        that prevent movement (e.g. when user is interacting)
        """
        self._zoom_xy_center_norm = np.float32(center_xy_norm)
        self._update_zoom_center()
        return None

    def _update_zoom_boundary(self) -> None:
        """Helper used to update cached zoom boundary info"""

        img_h, img_w = self._full_image.shape[0:2]
        min_side = min(img_h, img_w)
        zoom_norm = (1 - self._zoom_factor) * self._zoom_delta_norm + self._zoom_min_norm
        zoom_boundary_px = round(min_side * zoom_norm * 0.5)
        zoom_boundary_px = max(zoom_boundary_px, self._zoom_min_px // 2, 1)

        # Set up bounding xy on zoom center
        self._zoom_boundary_px = zoom_boundary_px
        self._zoom_xy_center_min_px = np.int32((zoom_boundary_px, zoom_boundary_px))
        self._zoom_xy_center_max_px = np.int32((img_w, img_h)) - self._zoom_xy_center_min_px

        # Changes to boundary require re-computing centering
        self._update_zoom_center()

        return

    def _update_zoom_center(self) -> None:
        """Helper used to update cached zoom centering & 'crop' coordinates"""

        # Make sure the zoom center point doesn't go 'off the edge' of the image
        img_h, img_w = self._full_image.shape[0:2]
        img_wh = np.float32((img_w, img_h))
        xy_center_px = np.int32(np.round(self._zoom_xy_center_norm * (img_wh - 1)))
        xy_center_px = np.clip(xy_center_px, self._zoom_xy_center_min_px, self._zoom_xy_center_max_px)
        x1, y1 = xy_center_px - self._zoom_boundary_px
        x2, y2 = xy_center_px + self._zoom_boundary_px

        # Store zoom crop coords
        self._zoom_hw = ((x2 - x1), (y2 - y1))
        self._zoom_xy1xy2_px = np.int32(((x1, y1), (x2, y2)))
        self._zoom_xy1xy2_norm = np.float32(self._zoom_xy1xy2_px) / img_wh

        # Update center norm coords. to prevent going way out of bounds
        # - not strictly needed, but helps give more natural behavior when dragging off far edges
        self._zoom_xy_center_norm = np.float32(xy_center_px) / (img_wh - 1)

        # Changes to centering require rerendering, generally
        self._need_zoom_rerender = True
        self._is_zoom_changed = True

        return

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        need_rerender = self._targ_h != h or self._targ_w != w or self._need_zoom_rerender

        # Re-compute zoomed region & create zoomed image
        if self._need_zoom_rerender:
            self._need_zoom_rerender = False
            (x1, y1), (x2, y2) = self._zoom_xy1xy2_px
            self._cached_zoom_img = self._full_image[y1:y2, x1:x2]

        # Re-scale zoomed image for rendering
        if need_rerender:
            self._targ_h = h
            self._targ_w = w
            scale_h, scale_w = get_image_hw_to_fit_region(self._cached_zoom_img.shape, (h, w))
            scale_wh = (max(1, scale_w), max(1, scale_h))
            self._render_image = cv2.resize(self._cached_zoom_img, scale_wh, interpolation=self.style.interpolation)

        return self._render_image

    # .................................................................................................................

    def _on_left_down(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:

        # Don't respond to clicks outside of zoom image
        if (not cbxy.is_in_region) or (not self._allow_panning):
            return

        self._is_pressed = True

        # Figure out which 'pixel' was clicked, for dragging purposes
        zoom_h, zoom_w = self._zoom_hw
        zoom_xy1 = self._zoom_xy1xy2_px[0]
        click_x = cbxy.xy_norm.x * (zoom_w - 1) + zoom_xy1[0]
        click_y = cbxy.xy_norm.y * (zoom_h - 1) + zoom_xy1[1]

        self._pan_pixel_xy = np.float32((click_x, click_y))
        self._pan_zoom_xy_center_norm = self._zoom_xy_center_norm
        self._pan_zoom_shape = (zoom_h, zoom_w)
        self._pan_zoom_xy1_px = zoom_xy1
        return

    def _on_left_up(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        self._is_pressed = False
        return

    def _on_move(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:

        # Don't do anything if the user isn't dragging the image
        if not self._is_pressed:
            return

        zoom_h, zoom_w = self._pan_zoom_shape[0:2]
        click_x = cbxy.xy_norm.x * (zoom_w - 1) + self._pan_zoom_xy1_px[0]
        click_y = cbxy.xy_norm.y * (zoom_h - 1) + self._pan_zoom_xy1_px[1]

        delta_x = click_x - self._pan_pixel_xy[0]
        delta_y = click_y - self._pan_pixel_xy[1]

        dx_norm = delta_x / self._full_image.shape[1]
        dy_norm = delta_y / self._full_image.shape[0]
        delta_norm = np.float32((dx_norm, dy_norm))

        new_center_xy_norm = self._pan_zoom_xy_center_norm - delta_norm

        self._set_zoom_center_unlocked(new_center_xy_norm)

        return
