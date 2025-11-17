#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .base import CachedBgFgElement
from .helpers.images import blank_image
from .helpers.sizing import get_image_hw_to_fit_by_ar
from .helpers.styling import UIStyle
from .helpers.colors import interpret_coloru8, pick_contrasting_gray_color
from .helpers.drawing import draw_box_outline
from .helpers.text import TextDrawer

# For type hints
from typing import Iterable
from numpy import ndarray
from .helpers.types import SelfType, COLORU8


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class SimpleHistogramPlot(CachedBgFgElement):
    """
    Element used to display a histogram plot where values
    are normalized to have a peak (y-axis) of 1.

    To update histogram plot, use .set_data(...)
    """

    # .................................................................................................................

    def __init__(
        self,
        title: str | None = "Histogram",
        bin_centers_or_range: ndarray | tuple = (0, 255),
        color_bg: COLORU8 | int = (50, 45, 40),
        color_line: COLORU8 | int = (75, 185, 225),
        aspect_ratio: float = 1.5,
        min_side_length: int = 128,
        include_y_axis: bool = True,
        use_bar_plot: bool = False,
        use_log_scale: bool = False,
        use_strict_ar: bool = False,
    ):
        """Bin centers should be either a numpy array or a tuple of the form:
            (start, end) or
            (start, end, num_bins)

        Note that additional color/styling options are available
        by accessing the element's .style attribute
        """

        # Storage for histogram plot data
        self._data = None
        self._bins = None
        self._histo_norm = np.float32((0, 0, 0))
        self.set_bins(bin_centers_or_range)

        # Store text
        self._title = title
        self._x_label = None
        self._y_label = None

        # Store plot config
        self._title_margin_px = 30
        self._label_margin_px = 20
        self._include_y_axis = include_y_axis
        self._use_bar_plot = use_bar_plot
        self._use_log_scale = use_log_scale
        self._use_strict_ar = use_strict_ar
        self._w_over_h = aspect_ratio
        self._h_over_w = 1.0 / self._w_over_h

        # Storage for cached plotting area (used when re-drawing updated data)
        self._plot_xy1 = (0, 0)
        self._plot_hw = (1, 1)

        # Set up element styling
        color_bg = interpret_coloru8(color_bg)
        color_axis = pick_contrasting_gray_color(color_bg)
        color_line = interpret_coloru8(color_line)
        self.style = UIStyle(
            label_text=TextDrawer(scale=0.35, color=(180, 180, 180), max_height=self._label_margin_px),
            title_text=TextDrawer(scale=0.5, color=(255, 255, 255), max_height=self._title_margin_px),
            color_bg=color_bg,
            color_plot=color_bg,
            color_line=color_line,
            color_axis=color_axis,
            thickness_line=1,
            thickness_axis=1,
            line_style=cv2.LINE_4,
            outline_color=(0, 0, 0),
        )

        # Inherit from parent
        is_tall = self._w_over_h < 1
        min_h = round(min_side_length * self._h_over_w) if is_tall else min_side_length
        min_w = min_side_length if is_tall else round(min_side_length * self._w_over_h)
        super().__init__(min_h, min_w, is_flexible_h=True, is_flexible_w=True)

    # .................................................................................................................

    def toggle_bar_plot(self, use_bar_plot: bool | None = None) -> SelfType:
        """Toggle bar plot vs. line plot"""
        self._use_bar_plot = (not self._use_bar_plot) if use_bar_plot is None else use_bar_plot
        self.request_fg_repaint()
        return self

    def toggle_log_scale(self, use_log_scale: bool | None = None) -> SelfType:
        """Toggle log-scale y-axis plotting"""
        self._use_log_scale = (not self._use_log_scale) if use_log_scale is None else use_log_scale
        self.request_fg_repaint()
        return self

    # .................................................................................................................

    def set_data(self, data: ndarray) -> SelfType:
        """Update data for which a histogram will be computed"""

        # Store data for reference, and compute normalized (peak value of 1) histogram
        self._data = data
        histo_counts, _ = np.histogram(self._data, self._bins)
        if self._use_log_scale:
            is_zero = histo_counts == 0
            histo_counts[is_zero] = 1
            histo_counts = np.log10(histo_counts, dtype=np.float32)
            histo_counts[is_zero] = 0
        self._histo_norm = np.float32(histo_counts) / max(1.0, np.float32(histo_counts.max()))

        # Force a re-draw of the plot when data changes
        self.request_fg_repaint()
        return self

    def set_bins(self, bin_centers_or_range=ndarray | tuple) -> SelfType:
        """Update bins used to compute histogram"""

        # If we get a numpy array, assume it is bin centers
        is_bin_centers = isinstance(bin_centers_or_range, ndarray)
        if is_bin_centers:
            # Shift each center point left/right half a step to generate bin edges
            prepend_val = 2 * bin_centers_or_range[0] - bin_centers_or_range[1]
            postpend_val = 2 * bin_centers_or_range[-1] - bin_centers_or_range[-2]
            bin_centers_or_range = np.concatenate((bin_centers_or_range, [postpend_val]))
            bin_edges = bin_centers_or_range - np.diff(bin_centers_or_range, prepend=prepend_val) * 0.5

        else:
            # If we're here, we assume user gave a (start, end) or (start, end, count) tuple
            assert isinstance(bin_centers_or_range, Iterable), "Must provide numpy array or (start, end, count) tuple"
            assert len(bin_centers_or_range) in (2, 3), "Must provide tuple of (start, end) or (start, end, count)"
            if len(bin_centers_or_range) == 2:
                bin_start, bin_end = bin_centers_or_range
                bin_count = bin_end - bin_start + 1
                bin_count = 100 if bin_count < 2 else bin_count
            else:
                bin_start, bin_end, bin_count = bin_centers_or_range

            # Sanity checks on bin range
            bin_start, bin_end = sorted((bin_start, bin_end))
            bin_edge_count = int(max(1, bin_count)) + 1

            # Create numpy array based on the provided range input
            bin_edges = np.linspace(bin_start, bin_end, bin_edge_count, dtype=np.float32)
            bin_edges_as_int = bin_edges.astype(np.int32)
            if np.allclose(bin_edges_as_int, bin_edges):
                bin_edges = bin_edges_as_int

        self._bins = bin_edges
        self.request_fg_repaint()
        return self

    def set_bins_advanced(self, bin_edges: ndarray) -> SelfType:
        """
        This is a more direct form of the .set_bins(...) function, which
        directly sets the 'bins' value used in: np.histogram(data, bins).
        Prefer the .set_bins(...) function, unless you know you need this
        """
        self._bins = bin_edges
        self.request_fg_repaint()
        return self

    def set_title(self, title: str | None) -> SelfType:
        """Update title of histogram"""
        self._title = title
        self.request_full_repaint()
        return self

    def set_axis_labels(self, x_axis: str | None = None, y_axis: str | None = None) -> SelfType:
        """
        Update x- and/or y-axis labels.
        - If 'None' is given, the existing label will not be modified
        - To clear an existing label, use empty string: ''
        """
        if x_axis is not None:
            self._x_label = x_axis if len(x_axis) > 0 else None
        if y_axis is not None:
            self._y_label = y_axis if len(y_axis) > 0 else None

        self.request_full_repaint()
        return self

    def set_margins(self, title_margin_px: int | None = None, label_margin_px: int | None = None) -> SelfType:
        """Update plot margins (can also resize text, if margins are reduced!)"""
        if title_margin_px is not None:
            self._title_margin_px = title_margin_px
            self.style.title_text.scale_to_hw(title_margin_px)

        if label_margin_px is not None:
            self._label_margin_px = label_margin_px
            self.style.label_text.scale_to_hw(label_margin_px)

        self.request_full_repaint()
        return self

    # .................................................................................................................

    def _get_width_given_height(self, h: int) -> int:
        return max(self._cb_rdr.min_w, round(h * self._w_over_h))

    def _get_height_given_width(self, w: int) -> int:
        return max(self._cb_rdr.min_h, round(w * self._h_over_w))

    def _get_dynamic_aspect_ratio(self):
        return self._w_over_h

    def _rerender_bg(self, h: int, w: int) -> ndarray:

        # Create blank plot image
        img_h, img_w = h, w
        if self._use_strict_ar:
            img_h, img_w = get_image_hw_to_fit_by_ar(self._w_over_h, (h, w))
        img = blank_image(img_h, img_w, self.style.color_bg)

        # Draw labels & title
        if self._label_margin_px > 0:
            if self._y_label is not None:
                x_slice = slice(0, self._label_margin_px)
                y_axis_block = np.rot90(img[:, x_slice, :], -1)
                y_axis_block = self.style.label_text.xy_centered(y_axis_block.copy(), self._y_label)
                img[:, x_slice, :] = np.rot90(y_axis_block, 1)
            if self._x_label is not None:
                y_slice = slice(-max(1, self._label_margin_px), None)
                x_axis_block = img[y_slice, :, :]
                self.style.label_text.xy_centered(x_axis_block, self._x_label)
        if self._title_margin_px > 0 and self._title is not None:
            y_slice = slice(0, self._title_margin_px)
            title_block = img[y_slice, :, :]
            self.style.title_text.xy_centered(title_block, self._title)

        # Figure out plotting region and fill with color
        dlabel = max(0, self._label_margin_px)
        plot_x1, plot_y1 = dlabel, max(0, self._title_margin_px)
        plot_w = max(1, (img_w - dlabel) - plot_x1)
        plot_h = max(1, (img_h - dlabel) - plot_y1)
        plot_x2, plot_y2 = plot_x1 + plot_w - 1, plot_y1 + plot_h - 1
        cv2.rectangle(img, (plot_x1, plot_y1), (plot_x2, plot_y2), self.style.color_plot, -1, cv2.LINE_4)
        # ^^^ This is significantly faster than: img[y1:y2,x1:x2,:] = color

        # Draw x/y axis (shift y-axis left/down 1 px, to avoid overlap with plot data)
        axis_xy1 = (plot_x1 - 1, plot_y1)
        axis_xy2 = (plot_x1 - 1, plot_y2 + 1)
        axis_xy3 = (plot_x2, plot_y2 + 1)
        if self._include_y_axis:
            cv2.line(img, axis_xy1, axis_xy2, self.style.color_axis, self.style.thickness_axis, cv2.LINE_4)
        cv2.line(img, axis_xy2, axis_xy3, self.style.color_axis, self.style.thickness_axis, cv2.LINE_4)

        # Store plot area coords for drawing data line/bars
        self._plot_xy1 = (plot_x1, plot_y1)
        self._plot_hw = (plot_h, plot_w)

        return draw_box_outline(img, self.style.outline_color)

    def _rerender_fg(self, bg_image: ndarray) -> ndarray:

        histo_w = self._histo_norm.shape[0]
        px1, py1 = self._plot_xy1
        ph, pw = self._plot_hw
        if self._use_bar_plot:
            # Generate triplet sequence like: [(x1, 0), (x1, y1), (x1, 0), (x2, 0), (x2, y2), (x2, 0), ...]
            # -> This will draw a single 'spike' at each histogram bin location (y1, y2, y3, etc.)
            # -> We then resize this image (with nearest-neighbour) to get a bar plot
            # -> This is faster than looping in python to draw each bar (tested)
            # -> Also faster than drawing many 2-point lines, e.g. Nx2x2 polyline (tested)
            x_px = np.arange(histo_w, dtype=np.int32)
            y1_px = np.round((1 - self._histo_norm) * (ph - 1)).astype(np.int32)
            y2_px = np.full_like(y1_px, ph - 1)
            bar_x = x_px.repeat(3)
            bar_y = np.vstack((y2_px, y1_px, y2_px)).T.flatten()
            xy_data = np.dstack((bar_x, bar_y))

            # Draw plot into small image & scale up to replace plot area
            bar_plot = blank_image(ph, histo_w, self.style.color_plot)
            cv2.polylines(
                bar_plot, xy_data, False, self.style.color_line, self.style.thickness_line, self.style.line_style
            )
            px2, py2 = px1 + pw, py1 + ph
            bg_image[py1:py2, px1:px2, :] = cv2.resize(bar_plot, (pw, ph), interpolation=cv2.INTER_NEAREST)
        else:
            # Map normalized histogram counts to plot area coords. and draw as a polyline
            y_px = py1 + np.round((1.0 - self._histo_norm) * (ph - 1)).astype(np.int32)
            x_px = px1 + np.round(np.linspace(0, 1, histo_w, dtype=np.float32) * (pw - 1)).astype(np.int32)
            xy_data = np.dstack((x_px, y_px))
            cv2.polylines(
                bg_image, xy_data, False, self.style.color_line, self.style.thickness_line, self.style.line_style
            )

        return bg_image

    # .................................................................................................................
