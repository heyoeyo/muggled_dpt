#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import CachedBgFgElement
from .helpers.styling import UIStyle
from .helpers.colors import interpret_coloru8, pick_contrasting_gray_color, lerp_colors
from .helpers.text import TextDrawer
from .helpers.images import blank_image
from .helpers.drawing import draw_box_outline

# Typing
from typing import Iterable
from numpy import ndarray
from .helpers.types import COLORU8, SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class Slider(CachedBgFgElement):
    """
    Simple horizontal slider. Intended to replace built-in opencv trackbars
    """

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        value: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.05,
        color: COLORU8 | int = (40, 40, 40),
        indicator_width: int = 1,
        text_scale: float = 0.5,
        marker_step: float | None = None,
        marker_origin: float | None = None,
        typecast=None,
        enable_value_display: bool = True,
        height: int = 40,
        minimum_width: int = 64,
    ):

        # Make sure the given values make sense
        min_val, max_val = sorted((min_val, max_val))
        initial_value = min(max_val, max(min_val, value))

        # Figure out type casting, if not given, and apply to inputs
        if typecast is None:
            is_integer = all((int(val) == val for val in (initial_value, min_val, max_val, step)))
            typecast = int if is_integer else float
        initial_value, min_val, max_val, step = [typecast(val) for val in (initial_value, min_val, max_val, step)]

        # Storage for slider value
        self._label = label
        self._initial_value = initial_value
        self._slider_value = initial_value
        self._slider_min = min_val
        self._slider_max = max_val
        self._slider_step = step
        self._slider_delta = max(self._slider_max - self._slider_min, 1e-9)
        self._marker_x_norm = _get_norm_marker_positions(min_val, max_val, marker_step, marker_origin)
        self._max_precision = _get_step_precision(step)
        self._type = typecast

        # Storage for slider state
        self._is_changed = True
        self._enable_value_display = enable_value_display

        # Set up text drawing
        txt_h = height * 0.8
        color = interpret_coloru8(color)
        fg_color = pick_contrasting_gray_color(color)
        fg_text = TextDrawer(scale=text_scale, color=fg_color, max_height=txt_h)
        bg_text = TextDrawer(scale=text_scale, color=lerp_colors(fg_color, color, 0.55), max_height=txt_h)

        # Set up element styling
        self.style = UIStyle(
            color=color,
            indicator_width_fg=indicator_width,
            indicator_width_bg=indicator_width * 2,
            indicator_color_fg=fg_color,
            indicator_color_bg=pick_contrasting_gray_color(fg_color),
            marker_color=lerp_colors(fg_color, color, 0.85),
            marker_width=1,
            marker_pad=5,
            outline_color=(0, 0, 0),
            fg_text=fg_text,
            bg_text=bg_text,
            label_xy_norm=(0, 0.5),
            label_anchor_xy_norm=None,
            label_offset_xy_px=(0, 0),
            label_margin_xy_px=(5, 0),
        )

        # Inherit from parent
        _, label_w, _ = fg_text.get_text_size(self._label)
        min_w = max(label_w, minimum_width)
        super().__init__(height, min_w, is_flexible_h=False, is_flexible_w=True)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name} ({self._label})"

    # .................................................................................................................

    def read(self) -> tuple[bool, float | int]:
        """Read current slider value. Returns: is_changed, slider_value"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._slider_value

    def set(self, slider_value: int | float, use_as_default_value: bool = True) -> SelfType:
        new_value = self._type(max(self._slider_min, min(self._slider_max, slider_value)))
        if use_as_default_value:
            self._initial_value = new_value
        self._is_changed |= new_value != self._slider_value
        self._slider_value = new_value
        self.request_fg_repaint()
        return self

    def reset(self) -> SelfType:
        self.set(self._initial_value, use_as_default_value=False)
        return self

    def increment(self, num_increments: int = 1) -> SelfType:
        return self.set(self._slider_value + self._slider_step * num_increments, use_as_default_value=False)

    def decrement(self, num_decrements: int = 1) -> SelfType:
        return self.set(self._slider_value - self._slider_step * num_decrements, use_as_default_value=False)

    def set_is_changed(self, is_changed: bool = True) -> SelfType:
        """Helper used to artificially toggle is_changed flag, useful for forcing read updates (e.g. on startup)"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def _on_left_down(self, cbxy, cbflags) -> None:

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Update slider as if dragging
        self._on_drag(cbxy, cbflags)
        return

    def _on_drag(self, cbxy, cbflags) -> None:

        # Update slider value while dragging
        new_slider_value = self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])
        value_is_changed = new_slider_value != self._slider_value
        if value_is_changed:
            self._is_changed = True
            self._slider_value = new_slider_value
            self.request_fg_repaint()

        return

    def _on_right_click(self, cbxy, cbflags) -> None:
        self.reset()
        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_value(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""

        # Map normalized x position to slider range, snapped to step increments
        slider_x = self._slider_min + round(x_norm * self._slider_delta / self._slider_step) * self._slider_step

        # Finally, make sure the slider value doesn't go out of range
        return self._type(max(self._slider_min, min(self._slider_max, slider_x)))

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:

        # Draw marker lines
        new_img = blank_image(h, w, self.style.color)
        mrk_pad_offset = 1 + self.style.marker_pad
        for x_norm in self._marker_x_norm:
            x_px = round(w * x_norm)
            pt1, pt2 = (x_px, mrk_pad_offset), (x_px, h - 1 - mrk_pad_offset)
            cv2.line(new_img, pt1, pt2, self.style.marker_color, self.style.marker_width, cv2.LINE_4)

        # Draw label
        return self.style.bg_text.xy_norm(
            new_img,
            self._label,
            self.style.label_xy_norm,
            anchor_xy_norm=self.style.label_anchor_xy_norm,
            offset_xy_px=self.style.label_offset_xy_px,
            margin_xy_px=self.style.label_margin_xy_px,
        )

    def _rerender_fg(self, bg_image: ndarray) -> ndarray:

        # Get image sizing for norm-to-px conversions
        img_h, img_w = bg_image.shape[0:2]
        max_x_px = img_w - 1

        # Draw indicator line
        slider_norm = (self._slider_value - self._slider_min) / self._slider_delta
        line_x_px = round(slider_norm * max_x_px)
        pt1, pt2 = (line_x_px, 0), (line_x_px, img_h)
        new_img = cv2.line(bg_image, pt1, pt2, self.style.indicator_color_bg, self.style.indicator_width_bg)
        new_img = cv2.line(bg_image, pt1, pt2, self.style.indicator_color_fg, self.style.indicator_width_fg)

        # Draw text beside indicator line to show current value if needed
        if self._enable_value_display:
            value_str = f"{float(self._slider_value):.{self._max_precision}f}"
            _, txt_w, _ = self.style.fg_text.get_text_size(value_str)

            # Draw the text to the left or right of the indicator line, depending on where the image border is
            is_near_right_edge = line_x_px + txt_w + 10 > max_x_px
            anchor_xy_norm = (1, 0.5) if is_near_right_edge else (0, 0.5)
            offset_xy_px = (-5, 0) if is_near_right_edge else (5, 0)
            self.style.fg_text.xy_norm(new_img, value_str, (slider_norm, 0.5), anchor_xy_norm, offset_xy_px)

        return draw_box_outline(new_img, self.style.outline_color)

    # .................................................................................................................


class MultiSlider(CachedBgFgElement):
    """
    Variant of a horizontal slider which has more than 1 control point
    This can be useful for setting min/max limits on a single value, for example.

    The number of control points is determined by the number of initial values provided.
    """

    # .................................................................................................................

    def __init__(
        self,
        label: str,
        values: int | float | list[int | float],
        min_val: float = 0.0,
        max_val: float = 1.0,
        step: float = 0.05,
        color: COLORU8 | int = (40, 40, 40),
        indicator_width: int = 1,
        text_scale: float = 0.5,
        marker_step: float | None = None,
        marker_origin: float | None = None,
        typecast=None,
        enable_value_display: bool = True,
        fill_color: COLORU8 | int | None = None,
        height: int = 40,
        minimum_width: int = 64,
    ):

        # Force to list-type, so we can handle single values as if they are multi-values
        if isinstance(values, (int, float)):
            values = tuple([values])

        # Figure out type casting, if not given, and apply to inputs
        if typecast is None:
            values_are_ints = all(int(val) == val for val in values)
            range_is_int = all((int(val) == val for val in (min_val, max_val, step)))
            typecast = np.int32 if (values_are_ints and range_is_int) else np.float32
        if typecast is float:
            typecast = np.float32
        if typecast is int:
            typecast = np.int32
        values = [typecast(val) for val in values]
        min_val, max_val, step = [typecast(val) for val in (min_val, max_val, step)]

        # Make sure the given values make sense
        is_int = all(isinstance(var, int) for var in [min_val, max_val, step])
        data_dtype = np.int32 if is_int else np.float32
        min_val, max_val = sorted((min_val, max_val))
        initial_values = np.clip(np.array(sorted(values), dtype=data_dtype), min_val, max_val)

        # Storage for slider value
        self._label = label
        self._initial_values = initial_values
        self._slider_values = initial_values.copy()
        self._slider_min = min_val
        self._slider_max = max_val
        self._slider_step = step
        self._slider_delta = max(self._slider_max - self._slider_min, 1e-9)
        self._marker_x_norm = _get_norm_marker_positions(min_val, max_val, marker_step, marker_origin)
        self._max_precision = _get_step_precision(step)
        self._is_filled = fill_color is not None
        self._type = typecast

        # Storage for slider state
        self._is_changed = True
        self._enable_value_display = enable_value_display
        self._drag_idx = 0

        # Set up text drawing
        txt_h = height * 0.8
        color = interpret_coloru8(color)
        fg_color = pick_contrasting_gray_color(color)
        fg_text = TextDrawer(scale=text_scale, color=fg_color, max_height=txt_h)
        bg_text = TextDrawer(scale=text_scale, color=lerp_colors(fg_color, color, 0.55), max_height=txt_h)

        # Set up element styling
        self.style = UIStyle(
            color=color,
            indicator_width_fg=indicator_width,
            indicator_width_bg=indicator_width * 2,
            indicator_color_fg=fg_color,
            indicator_color_bg=pick_contrasting_gray_color(fg_color),
            marker_color=lerp_colors(fg_color, color, 0.85),
            marker_width=1,
            marker_pad=5,
            fill_color=interpret_coloru8(fill_color, (255, 255, 255)),
            fill_weight=0.5,
            fg_text=fg_text,
            bg_text=bg_text,
            label_xy_norm=(0, 0.5),
            label_anchor_xy_norm=None,
            label_offset_xy_px=(0, 0),
            label_margin_xy_px=(5, 0),
            outline_color=(0, 0, 0),
        )

        # Inherit from parent
        _, label_w, _ = fg_text.get_text_size(self._label)
        min_w = max(label_w, minimum_width)
        super().__init__(height, min_w, is_flexible_h=False, is_flexible_w=True)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name} ({self._label})"

    # .................................................................................................................

    def read(self) -> tuple[bool, float | int]:
        """Read slider values. Returns: is_changed, [slider_value1, slider_value2, ..., etc.]"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, np.sort(self._slider_values).tolist()

    def set(self, new_values: tuple[int | float], use_as_default_values: bool = True) -> SelfType:

        # Force new values to array-like data, within slider range
        if isinstance(new_values, (int, float)):
            new_values = [new_values]
        new_values = np.clip(np.array(sorted(new_values)), self._slider_min, self._slider_max)

        # Use new values as default if needed
        if use_as_default_values:
            self._initial_values = new_values.copy()

        # Check if new values are actually different and store
        self._is_changed |= not np.allclose(new_values, self._slider_values)
        self._slider_values = new_values
        self.request_fg_repaint()

        return self

    def reset(self) -> SelfType:
        self.set(self._initial_values, use_as_default_values=False)
        return self

    def set_is_changed(self, is_changed=True) -> SelfType:
        """Helper used to artificially toggle is_changed flag, useful for forcing read updates"""
        self._is_changed = is_changed
        return self

    # .................................................................................................................

    def _on_left_down(self, cbxy, cbflags) -> None:
        """Update closest slider point on click and record index for dragging"""

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Update closest click, redardless of whether we actually change values
        # (this is important for dragging to work properly)
        new_slider_value = self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])
        closest_idx = np.argmin((np.abs(self._slider_values - new_slider_value)))
        self._drag_idx = closest_idx

        # Update slider only if value changes
        is_value_changed = new_slider_value != self._slider_values[closest_idx]
        if is_value_changed:
            self._is_changed = True
            self._slider_values[closest_idx] = new_slider_value
            self.request_fg_repaint()

        return

    def _on_drag(self, cbxy, cbflags) -> None:
        """Update a single slider point on drag (determined by closest on left click)"""

        # Update slider value while dragging, only if values change
        new_slider_value = self._mouse_x_norm_to_slider_value(cbxy.xy_norm[0])
        is_value_changed = new_slider_value != self._slider_values[self._drag_idx]
        if is_value_changed:
            self._is_changed = True
            self._slider_values[self._drag_idx] = new_slider_value
            self.request_fg_repaint()

        return

    def _on_left_up(self, cbxy, cbflags) -> None:
        """For slight efficiency gain, sort values after modifications are complete"""
        self._slider_values = np.sort(self._slider_values)
        return

    def _on_right_click(self, cbxy, cbflags) -> None:
        """Reset slider position on right click"""
        self.reset()
        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_value(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""

        # Map normalized x position to slider range, snapped to step increments
        slider_x = (x_norm * self._slider_delta) + self._slider_min
        slider_x = round(slider_x / self._slider_step) * self._slider_step
        slider_x = max(self._slider_min, min(self._slider_max, slider_x))

        return self._type(slider_x)

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:

        # Draw marker lines
        new_img = blank_image(h, w, self.style.color)
        mrk_pad_offset = 1 + self.style.marker_pad
        for x_norm in self._marker_x_norm:
            x_px = round(w * x_norm)
            pt1, pt2 = (x_px, mrk_pad_offset), (x_px, h - 1 - mrk_pad_offset)
            cv2.line(new_img, pt1, pt2, self.style.marker_color, self.style.marker_width)

        # Draw label
        return self.style.bg_text.xy_norm(
            new_img,
            self._label,
            self.style.label_xy_norm,
            anchor_xy_norm=self.style.label_anchor_xy_norm,
            offset_xy_px=self.style.label_offset_xy_px,
            margin_xy_px=self.style.label_margin_xy_px,
        )

    def _rerender_fg(self, image: ndarray) -> ndarray:

        # Get image sizing for norm-to-px conversions
        img_h, img_w = image.shape[0:2]
        max_x_px = img_w - 1

        # Draw filled in region between min/max values, if needed
        if self._is_filled:

            # Figure out highlight region bounds
            x1_norm = (np.min(self._slider_values) - self._slider_min) / self._slider_delta
            x2_norm = (np.max(self._slider_values) - self._slider_min) / self._slider_delta
            x1_px = round(x1_norm * max_x_px)
            x2_px = round(x2_norm * max_x_px)

            # Mix in another color to indicator highlighted region
            fill_w = max(0, x2_px - x1_px)
            if fill_w > 0:
                fill_img = np.full((img_h, fill_w, 3), self.style.fill_color, dtype=np.uint8)
                orig_region = image[:, x1_px:x2_px, :]
                f_weight, inv_weight = self.style.fill_weight, 1 - self.style.fill_weight
                image[:, x1_px:x2_px, :] = cv2.addWeighted(orig_region, inv_weight, fill_img, f_weight, 0)

        # Draw indicator line(s)
        for value in self._slider_values:
            value_norm = (value - self._slider_min) / self._slider_delta
            line_x_px = round(value_norm * max_x_px)
            pt1, pt2 = (line_x_px, 0), (line_x_px, img_h)
            image = cv2.line(image, pt1, pt2, self.style.indicator_color_bg, self.style.indicator_width_bg)
            image = cv2.line(image, pt1, pt2, self.style.indicator_color_fg, self.style.indicator_width_fg)

            # Draw text beside indicator line to show current value if needed
            if self._enable_value_display:
                value_str = f"{float(value):.{self._max_precision}f}"
                _, txt_w, _ = self.style.fg_text.get_text_size(value_str)

                # Draw the text to the left or right of the indicator line, depending on where the image border is
                is_near_right_edge = line_x_px + txt_w + 10 > max_x_px
                anchor_xy_norm = (1, 0.5) if is_near_right_edge else (0, 0.5)
                offset_xy_px = (-5, 0) if is_near_right_edge else (5, 0)
                self.style.fg_text.xy_norm(image, value_str, (value_norm, 0.5), anchor_xy_norm, offset_xy_px)

        return draw_box_outline(image, self.style.outline_color)

    # .................................................................................................................


class ColorSlider(Slider):
    """
    Helper used to implement a slider that shows a colormap as a background
    The background can be provided directly as an Nx1 or Nx3 numpy array,
    where N is the number of colors and the x1 or x3 corresponds to grayscale
    or BGR values, respectively. Also works with colormap LUT shape (1xNx3).
    Note that if a non-none value of 'steps' is given, then the color array
    will be resized to have 'steps' number of entries!

    Alternatively, colors can be given as a list of grayscale or BGR values,
    in which case the slider will linearly interpolate to produce a background
    with 'steps' number of entries.
    """

    # .................................................................................................................

    def __init__(
        self,
        label: str | None,
        colors: ndarray | Iterable = ((0, 0, 0), (0, 128, 255)),
        initial_position_norm: float = 0.5,
        num_steps: int | None = 256,
        indicator_width: int = 1,
        text_scale: float = 0.5,
        height: int = 40,
        minimum_width: int = 64,
        interpolation=cv2.INTER_LINEAR,
    ):

        # Force into array type
        given_color_lut = isinstance(colors, ndarray)
        if not given_color_lut:
            colors = np.array(colors)

        # Try to force colors array into Nx1 or Nx3 shape
        if colors.ndim >= 3:
            colors = np.squeeze(colors)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=-1)
        assert colors.ndim == 2, "Error! Cannot interpret colors input shape. Expecting Nx3 or Nx1"
        assert colors.shape[1] in (1, 3), "Error! Cannot interpret colors input shape. Expecting Nx3 or Nx1"

        # Interpolate color steps, if needed
        num_colors, num_ch = colors.shape[0:2]
        num_steps = num_colors if num_steps is None else num_steps
        if num_colors != num_steps:
            x_interp = np.arange(num_steps)
            x_given = np.linspace(0, 255, num_colors)
            interp_ch = [np.interp(x_interp, x_given, colors[:, ch_idx]) for ch_idx in range(num_ch)]
            colors = np.stack(interp_ch, axis=-1)
        assert num_steps > 2, "Error! Color slider must have at least 2 color steps"

        # Force internal color array to BGR for use in display
        is_grayscale = num_ch == 1
        if is_grayscale:
            colors = colors.repeat(3, axis=-1)
        self._colors_1px_img = np.round(np.expand_dims(colors, axis=0)).astype(np.uint8)
        self._interpolation = interpolation
        self._is_grayscale = is_grayscale

        # Initialize from parent
        low_color = self._colors_1px_img[0, 0, :]
        label = label if label is not None else ""
        slider_step_size = 1.0 / (num_steps - 1)
        super().__init__(
            label,
            value=initial_position_norm,
            min_val=0,
            max_val=1,
            step=slider_step_size,
            color=low_color,
            enable_value_display=False,
        )

    # .................................................................................................................

    def read(self) -> tuple[bool, float, int | COLORU8]:
        """Read current slider value. Returns: is_changed, slider_position_norm, color_select"""
        is_changed, slider_pos_norm = super().read()
        idx_select = round(slider_pos_norm * (self._colors_1px_img.shape[1] - 1))
        color_select = self._colors_1px_img[0, idx_select, :].tolist()
        color_select = color_select[0] if self._is_grayscale else color_select
        return is_changed, slider_pos_norm, color_select

    # .................................................................................................................

    def _rerender_bg(self, h: int, w: int) -> ndarray:

        # Draw color bar with label
        new_img = cv2.resize(self._colors_1px_img, dsize=(w, h), interpolation=self._interpolation)
        return self.style.bg_text.xy_norm(
            new_img,
            self._label,
            self.style.label_xy_norm,
            anchor_xy_norm=self.style.label_anchor_xy_norm,
            offset_xy_px=self.style.label_offset_xy_px,
            margin_xy_px=self.style.label_margin_xy_px,
        )

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def _get_norm_marker_positions(
    min_value: int | float,
    max_value: int | float,
    marker_step: int | float | None,
    marker_origin: int | float | None = None,
) -> ndarray:
    """
    Helper used to compute the location of slider marker indicators.
    If marker_step is None, then no markers will be returned,
    if marker_origin is None, then the min_value will be used.

    As an example, assume we have min/max of: (3, 9), and a step of 2
    With a marker_origin of None, we fallback to the min value (3) and
    would get markers at: 3, 5, 7, 9
    By comparison, a marker_origin of 0, gives markers at: 4, 6, 8
    """

    # Bail if we don't need steps
    if marker_step is None:
        return np.float32([])

    # Use min value as origin if not given (generally most intuitive behavior)
    if marker_origin is None:
        marker_origin = min_value

    # Find left-/right-most marker values
    mrk_min = marker_origin - round((marker_origin - min_value) / marker_step) * marker_step
    mrk_max = marker_origin + round((max_value - marker_origin) / marker_step) * marker_step

    # Make sure marker min/max is equal or within the value min/max range
    if mrk_min < min_value:
        mrk_min += marker_step
    if mrk_max > max_value:
        mrk_max -= marker_step

    # Calculate normalized marker coordinates for drawing
    marker_pts = np.arange(mrk_min, mrk_max + marker_step, marker_step, dtype=np.float32)
    marker_x_norm = (marker_pts - min_value) / max(max_value - min_value, 1e-9)
    return marker_x_norm


def _get_step_precision(slider_step_size: int | float) -> int:
    """
    Helper used to decide how many digits to display when showing
    a slider value indicator, based on step sizing.

    For example, for a step size of 0.05, we would want a
    precision of 2 decimal places. For a step of 5, we would
    want a precision of 0. This function includes extra checks
    to try to handle weird floating points issues,
    for example a step size of (0.1 + 0.2) = 0.30000000000000004
    will return a precision of 1
    """

    step_as_str = str(slider_step_size)
    step_dec_str = step_as_str.split(".")[-1] if "." in step_as_str else ""
    num_dec_places = len(step_dec_str)
    if num_dec_places >= 7:
        num_trunc = 2
        num_places_truncated = len(step_dec_str[:-num_trunc].rstrip("0"))
        is_much_smaller = 0 < num_places_truncated < (num_dec_places - num_trunc)
        if is_much_smaller:
            num_dec_places = num_places_truncated
    return num_dec_places

    step_fractional = slider_step_size % 1
    if step_fractional == 0:
        return 0

    return int(np.ceil(-np.log10(max(slider_step_size, 1e-9))))
