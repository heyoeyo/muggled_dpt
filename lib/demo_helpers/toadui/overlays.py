#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from time import perf_counter

import cv2
import numpy as np

from .base import BaseCallback, BaseOverlay, CBEventXY, CBEventFlags
from .helpers.colors import interpret_coloru8, pick_contrasting_gray_color
from .helpers.styling import UIStyle
from .helpers.drawing import draw_normalized_polygon, draw_circle_norm, draw_box_outline
from .helpers.text import TextDrawer

# Typing
from typing import NamedTuple, Callable, Iterable
from numpy import ndarray
from .helpers.types import COLORU8, IMGSHAPE_HW, XYPX, XYNORM, HWPX, XY1XY2NORM, XY1XY2PX, SelfType, IsLMR
from .helpers.ocv_types import OCVInterp, OCVLineType, OCVFont


# ---------------------------------------------------------------------------------------------------------------------
# %% Types


class CBEventAndFlags(NamedTuple):
    event: CBEventXY
    flags: CBEventFlags


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class DrawRectangleOverlay(BaseOverlay):
    """Simple overlay which draws a rectangle over top of base images"""

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 = (0, 255, 255),
        bg_color: COLORU8 | None = None,
        thickness: int = 2,
        max_rectangles: int = 12,
    ):
        self._rectangle_xy1xy2_norm_list: list[XY1XY2NORM] = []
        self.style = UIStyle(
            color_fg=color,
            color_bg=bg_color,
            thickness_fg=thickness,
            thickness_bg=thickness + 1,
        )
        super().__init__(base_item)

    # .................................................................................................................

    def clear(self) -> SelfType:
        self._poly_xy_norm_list = []
        return self

    # .................................................................................................................

    def set_rectangles(self, *rectangle_xy1xy2_norm_list: list[XY1XY2NORM]) -> SelfType:
        """
        Set or replace polygons. Polygons should be given as either a list/tuple
        of normalized xy coordinates or as an Nx2 numpy array, where N is the
        number of points in the polygon. More than one polygon can be provided.

        For example:
            poly1 = np.float32([(0.25,0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
            poly2 = [(0.25, 0.25), (0.85, 0.5), (0.25, 0.5)]
            set_polygons(poly1, poly2)
        """

        self._rectangle_xy1xy2_norm_list = tuple(*rectangle_xy1xy2_norm_list)

        return self

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        if len(self._rectangle_xy1xy2_norm_list) == 0:
            return frame

        out_frame = frame.copy()
        xscale = out_frame.shape[1] - 1
        yscale = out_frame.shape[0] - 1
        xy1_norm, xy2_norm = self._rectangle_xy1xy2_norm_list
        xy1_px = tuple((round(xy1_norm[0] * xscale), round(xy1_norm[1] * yscale)))
        xy2_px = tuple((round(xy2_norm[0] * xscale), round(xy2_norm[1] * yscale)))
        if self.style.color_bg is not None:
            cv2.rectangle(out_frame, xy1_px, xy2_px, self.style.color_bg, self.style.thickness_bg, cv2.LINE_4)
        cv2.rectangle(out_frame, xy1_px, xy2_px, self.style.color_fg, self.style.thickness_bg, cv2.LINE_4)

        return out_frame

    # .................................................................................................................


class DrawPolygonsOverlay(BaseOverlay):
    """Simple overlay which draws polygons over top of base images"""

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 = (0, 255, 255),
        bg_color: COLORU8 | None = None,
        thickness: int = 2,
        line_type: OCVLineType = cv2.LINE_AA,
        is_closed: bool = True,
    ):

        self._poly_xy_norm_list: list[XYNORM] = []

        self.style = UIStyle(
            color=color,
            color_bg=bg_color,
            thickness=thickness,
            line_type=line_type,
            is_closed=is_closed,
        )

        super().__init__(base_item)

    # .................................................................................................................

    def clear(self) -> SelfType:
        self._poly_xy_norm_list = []
        return self

    # .................................................................................................................

    def set_polygons(self, *polygon_xy_norm_list: list[XYNORM] | ndarray) -> SelfType:
        """
        Set or replace polygons. Polygons should be given as either a list/tuple
        of normalized xy coordinates or as an Nx2 numpy array, where N is the
        number of points in the polygon. More than one polygon can be provided.

        For example:
            poly1 = np.float32([(0.25,0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
            poly2 = [(0.25, 0.25), (0.85, 0.5), (0.25, 0.5)]
            set_polygons(poly1, poly2)
        """

        # if isinstance(polygon_xy_norm_list, ndarray):
        #     polygon_xy_norm_list = [polygon_xy_norm_list]
        self._poly_xy_norm_list = tuple([*polygon_xy_norm_list])

        return self

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        if self._poly_xy_norm_list is None:
            return frame

        if len(self._poly_xy_norm_list) == 0:
            return frame

        out_frame = frame.copy()
        for poly in self._poly_xy_norm_list:
            out_frame = draw_normalized_polygon(
                out_frame,
                poly,
                self.style.color,
                self.style.thickness,
                self.style.color_bg,
                self.style.line_type,
                self.style.is_closed,
            )

        return out_frame

    # .................................................................................................................


class DrawMaskOverlay(BaseOverlay):
    """Simple overlay which draws a binary mask over top of a base image"""

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        mask_color: COLORU8 = (90, 0, 255),
        scaling_interpolation: OCVInterp = cv2.INTER_NEAREST,
    ):

        # Storage for mask data
        self._inv_mask = None
        self._mask_bgr = None
        self._interpolation = scaling_interpolation
        self._color = np.uint8(mask_color)

        # Storage for cached data
        self._cached_h = None
        self._cached_w = None
        self._cached_mask_bgr = None
        self._cached_inv_mask = None

        super().__init__(base_item)

    # .................................................................................................................

    def clear(self) -> SelfType:
        """Clear all mask data"""
        self._inv_mask = None
        self._mask_bgr = None
        self._cached_inv_mask = None
        self._cached_mask_bgr = None
        self._cached_h = None
        self._cached_w = None
        return self

    # .................................................................................................................

    def set_mask(self, mask: ndarray, mask_threshold: int = 0) -> SelfType:
        """Update mask used for overlay"""

        mask_bin = np.uint8(mask > mask_threshold)
        self._mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR) * self._color
        self._inv_mask = cv2.cvtColor(cv2.bitwise_not(mask_bin * 255), cv2.COLOR_GRAY2BGR)

        # Update cached values, in case mask already matches target frame size
        self._cached_mask_bgr = self._mask_bgr
        self._cached_inv_mask = self._inv_mask
        self._cached_h, self._cached_w = mask_bin.shape[0:2]

        return self

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Skip overlay if we don't have a mask
        if self._mask_bgr is None:
            return frame

        # Re-build cached masks if sizing changes
        frame_h, frame_w = frame.shape[0:2]
        if (frame_h != self._cached_h) or (frame_w != self._cached_w):
            frame_wh = (frame_w, frame_h)
            self._cached_inv_mask = cv2.resize(self._inv_mask, dsize=frame_wh, interpolation=self._interpolation)
            self._cached_mask_bgr = cv2.resize(self._mask_bgr, dsize=frame_wh, interpolation=self._interpolation)
            self._cached_h, self._cached_w = frame_h, frame_w

        inv_frame = cv2.bitwise_and(frame, self._cached_inv_mask)
        return cv2.add(inv_frame, self._cached_mask_bgr)

    # .................................................................................................................


class DrawOutlineOverlay(BaseOverlay):
    """
    Simple overlay used to draw a box outline around the base element
    Includes support for having a separate color when hovered.

    Supports 'draw_in_place' option for greater efficiency.
    This allows for drawing the outline directly onto base item, rather
    than creating a copy of the image first. This is a destructive
    modification, but can be disabled if needed.
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 | int = (0, 0, 0),
        thickness_px: int = 1,
        hover_color: COLORU8 | int | None = None,
        hover_thickness_px: int | None = None,
        draw_in_place: bool = True,
    ):
        # Inherit from parent
        super().__init__(base_item)

        # Storage for state
        self._draw_in_place = draw_in_place
        self._is_hovered = False

        # Handle missing hover styling
        if hover_color is None:
            hover_color = color
        if hover_thickness_px is None:
            hover_thickness_px = thickness_px

        # Set up element styling
        self.style = UIStyle(
            color=interpret_coloru8(color),
            thickness=thickness_px,
            hover_color=interpret_coloru8(hover_color),
            hover_thickness=hover_thickness_px,
        )

    def _on_move(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        self._is_hovered = cbxy.is_in_region
        return

    def _render_overlay(self, frame: ndarray) -> ndarray:
        if self._is_hovered:
            color = self.style.hover_color
            thickness = self.style.hover_thickness
        else:
            color = self.style.color
            thickness = self.style.thickness
        outframe = frame if self._draw_in_place else frame.copy()
        return draw_box_outline(outframe, color, thickness)

    # .................................................................................................................


class DrawCustomOverlay(BaseOverlay):
    """
    Overlay which allows for defining custom drawing logic

    A custom drawing function must be provided which takes in
    a numpy array (the underlying frame to draw on) and an
    xy norm coordinate (0 to 1) corresponding to the most recent
    mouse positioning. The function must return a numpy array
    of the same size. For example:

        def custom_blank_out_func(frame: ndarray, xy_norm: tuple[float, float]) -> ndarray:
            return frame * 0

    Alternatively, 'None' can be given, which will disable the overlay.

    For cases where dynamic content is meant to be rendered, it is highly
    recommended that the custom render function be implemented as a
    method of a class. This will allow you to adjust parameters through
    the class, which are then reflected in the render method, without
    having to constantly re-assign a new render function. For example:

        class DimmerOverlay:
            def __init__(self, initial_dim: float = 0.5):
                self._dim = initial_dim
            def set_dim(self, dim:float) -> None:
                self._dim = dim
            def render(self, frame: ndarray, xy_norm: tuple[float, float]) -> ndarray:
                return np.uint8(frame * self._dim)

        # Assign dimmer render method as custom overlay
        dimmer = DimmerOverlay(0.25)
        custom_olay = DrawCustomOverlay(some_img_element, dimmer.render)
        ... later ...
        dimmer.set_dim(0.75) # <- this will affect the overlay at run-time
    """

    # .................................................................................................................

    def __init__(self, base_item: BaseCallback, custom_render_function: Callable[[ndarray, XYNORM], ndarray] | None):
        super().__init__(base_item)
        self._xy_norm = (-1.0, -1.0)
        self._custom_render_func: Callable[[ndarray, XYNORM], ndarray] | None = custom_render_function

    def set_render_function(self, custom_render_function: Callable[[ndarray, XYNORM], ndarray] | None) -> SelfType:
        """
        Set a new overlay rendering function
        This function is expected to take an image (numpy array) and mouse position
        (xy_norm) as input and return an image (numpy array) of the same size, but
        potentially with custom drawing done as an overlay.
        """
        self._custom_render_func = custom_render_function
        return self

    def _render_overlay(self, frame: ndarray) -> ndarray:
        if self._custom_render_func is not None:
            return self._custom_render_func(frame, self._xy_norm)
        return frame

    def _on_move(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        self._xy_norm = cbxy.xy_norm
        return

    # .................................................................................................................


class TextOverlay(BaseOverlay):
    """Overlay used to draw text over a base image"""

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        xy_norm: XYNORM = (0.5, 0.5),
        scale: float = 0.5,
        thickness: int = 1,
        color: COLORU8 = (255, 255, 255),
        bg_color: COLORU8 = (0, 0, 0),
        font: OCVFont = cv2.FONT_HERSHEY_SIMPLEX,
        line_type: OCVLineType = cv2.LINE_AA,
        anchor_xy_norm: XYNORM | None = None,
        offset_xy_px: XYPX = (0, 0),
        margin_xy_px: XYPX = (5, 5),
    ):
        super().__init__(base_item)
        self._text = None
        self._xy_norm = xy_norm
        self._anchor_xy_norm = anchor_xy_norm
        self._offset_xy_px = offset_xy_px
        self._margin_xy_px = margin_xy_px
        self.style = UIStyle(text=TextDrawer(scale, thickness, color, bg_color, font, line_type))
        self.style.text = TextDrawer(scale, thickness, color, bg_color, font, line_type)

    # .................................................................................................................

    def set_text(self, text: str | None) -> SelfType:
        self._text = text
        return self

    # .................................................................................................................

    def set_postion(
        self,
        xy_norm: XYNORM | None = None,
        anchor_xy_norm: XYNORM | None = None,
        offset_xy_px: XYPX | None = None,
        margin_xy_px: XYPX | None = None,
    ) -> SelfType:

        if xy_norm is not None:
            self._xy_norm = xy_norm
        if anchor_xy_norm is not None:
            self._anchor_xy_norm = anchor_xy_norm
        if offset_xy_px is not None:
            self._offset_xy_px = offset_xy_px
        if margin_xy_px is not None:
            self._margin_xy_px = margin_xy_px

        return self

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        if self._text is None:
            return frame

        return self.style.text.xy_norm(
            frame, self._text, self._xy_norm, self._anchor_xy_norm, self._offset_xy_px, self._margin_xy_px
        )

    # .................................................................................................................


class MousePaintOverlay(BaseOverlay):
    """
    Overlay used to allow for 'painting' over a base item.
    Supports left/middle/right click painting.

    Note that this overlay does not try to manage accumulated
    painting data. Instead, painting data is 'reset' every time
    a user finishes, and it is up to the code calling the overlay
    to manage the painted trail data as needed.
    This is handled through the 'read_trail()' function:

        # Read trail data
        is_trail_finished, trail_xy_norm, lmr_index = overlay.read_trail()

        # Handle storage/plotting of trail data
        if is_trail_finished:
            some_data_storage.append(trail_xy_norm)
        else:
            draw_trail_in_progress(trail_xy_norm)
        ...etc...

    Consider using an 'UndoRedoList' (from toadui data management) to
    store trail data, which provides support for undo/redo & clearing!
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        enable_hover_indicator: bool = True,
        enable_render: bool = True,
        allow_left_click: bool = True,
        allow_middle_click: bool = False,
        allow_right_click: bool = False,
        brush_radius_norm=0.1,
    ):

        # Allocate storage for mouse press/change state for left/middle/right click
        self._is_lmr_pressed = [False, False, False]
        self._is_lmr_changed = [False, False, False]
        self._allow_lmr = (allow_left_click, allow_middle_click, allow_right_click)

        # Storage for current paint trail details
        self._curr_cbxy = CBEventXY.default()
        self._trail = []
        self._trail_lmr = 0
        self._brush_rad_norm = brush_radius_norm
        self._is_trail_in_progress = False

        # Set rendering state
        self._enable_hover_indicator = enable_hover_indicator
        self.enable_render(enable_render)

        # Configure styling of overlay graphics
        self.style = UIStyle(
            color_left_paint=(0, 255, 255),
            color_right_paint=(0, 0, 255),
            color_middle_paint=(255, 0, 255),
            color_hover_fg=(0, 255, 255),
            color_hover_bg=(0, 115, 115),
            thickness_hover_fg=1,
            thickness_hover_bg=2,
            hover_line_type=cv2.LINE_AA,
            paint_line_type=cv2.LINE_AA,
        )

        # Inherit from parent
        super().__init__(base_item)

    # .................................................................................................................

    def read_mouse_xy(self) -> tuple[IsLMR, IsLMR, CBEventXY]:
        """Returns: is_dragging, current_event_xy, previous_event_xy"""
        is_lmr_pressed = IsLMR(*self._is_lmr_pressed)
        is_lmr_changed = IsLMR(*self._is_lmr_changed)
        self._is_lmr_changed = [False, False, False]
        return is_lmr_changed, is_lmr_pressed, self._curr_cbxy

    def read_trail(self) -> tuple[bool, list[XYNORM], int]:
        """
        Read current trail data, returns xy coordinates & left/middle/right click index.

        Note: if the user has has just 'finished' painting,
        then the trail data read from this function will be deleted.
        It is up to the code calling this function to manage/store the
        painted trail data over time!

        Returns:
            is_trail_finished, trail_xy_norm_list, lmr_index

        - lmr_index is a value of 0 (left), 1 (middle) or 2 (right), representing the mouse click used to paint
        """

        trail_xy_norm = self._trail
        is_trail_finished = len(trail_xy_norm) > 0 and (not self._is_trail_in_progress)
        if is_trail_finished:
            self._trail = []

        return is_trail_finished, trail_xy_norm, self._trail_lmr

    def clear(self) -> SelfType:
        """Wipe out any existing trail data and mouse-press state"""
        self._trail = []
        self._is_trail_in_progress = False
        self._is_lmr_changed = [pressed != False for pressed in self._is_lmr_pressed]
        self._is_lmr_pressed = [False, False, False]
        return self

    def set_brush_size(self, size_norm: float) -> SelfType:
        """Update paint brush sizing. Note that sizing cannot be changed mid-paint!"""
        self._brush_rad_norm = size_norm * 0.5
        return self

    # .................................................................................................................

    def _on_mouse_down(self, cbxy: CBEventXY, cbflags: CBEventFlags, lmr_index: int):

        # Don't register clicks outside of overlay
        if not cbxy.is_in_region:
            return

        # Don't begin new trail if we're already painting
        if self._is_trail_in_progress:
            return

        # Bail on left/middle/right click if not allowed
        if not self._allow_lmr[lmr_index]:
            return

        self._curr_cbxy = cbxy
        self._is_lmr_pressed[lmr_index] = True
        self._is_lmr_changed[lmr_index] = True
        self._is_trail_in_progress = True
        self._trail_lmr = lmr_index

        self._trail.append(cbxy.xy_norm)

        return

    def _on_mouse_up(self, cbxy: CBEventXY, cbflags: CBEventFlags, lmr_index: int):
        # Only register mouse-up if the mouse was recorded as down previously!
        # -> Need this because we ignore down-events when another button is already pressed
        # -> So not guaranteed down-up pairing
        is_changed = self._is_lmr_pressed[lmr_index]
        if is_changed:
            self._curr_cbxy = cbxy
            self._is_lmr_changed[lmr_index] = True
            self._is_lmr_pressed[lmr_index] = False
            self._is_trail_in_progress = False
        return

    def _on_left_down(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        self._on_mouse_down(cbxy, cbflags, 0)

    def _on_middle_down(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        self._on_mouse_down(cbxy, cbflags, 1)

    def _on_right_down(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        self._on_mouse_down(cbxy, cbflags, 2)

    def _on_left_up(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        self._on_mouse_up(cbxy, cbflags, 0)

    def _on_middle_up(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        self._on_mouse_up(cbxy, cbflags, 1)

    def _on_right_up(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        self._on_mouse_up(cbxy, cbflags, 2)

    def _on_move(self, cbxy: CBEventXY, cbflags: CBEventFlags):

        self._curr_cbxy = cbxy
        if self._is_trail_in_progress:
            self._trail.append(cbxy.xy_norm)

        return

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Copy frame so we don't modify original (in case original is being re-used)
        out_frame = frame.copy()

        # Get brush sizing for both circle (radius) and line (thickness) drawing
        brush_radius_px = self._brush_rad_norm * min(out_frame.shape[0:2])
        brush_thick_px = 2 * brush_radius_px
        brush_radius_px, brush_thick_px = [round(value) for value in (brush_radius_px, brush_thick_px)]

        # Draw trail
        have_trail_data = len(self._trail) > 0
        if have_trail_data:

            # Pick color based on which mouse button is painting
            color = self.style.color_left_paint
            if self._trail_lmr == 1:
                color = self.style.color_middle_paint
            elif self._trail_lmr == 2:
                color = self.style.color_right_paint

            # Draw current paint trail, if we have a valid color
            if color is not None:
                line_type = self.style.paint_line_type
                num_xy = len(self._trail)
                if num_xy == 1:
                    xy_cen = self._trail[0]
                    thickness = -1
                    draw_circle_norm(out_frame, xy_cen, brush_radius_px, color, thickness, line_type)
                elif num_xy > 1:
                    bg_col, is_closed = None, False
                    draw_normalized_polygon(out_frame, self._trail, color, brush_thick_px, bg_col, line_type, is_closed)

        # Draw hover indicator (i.e. circle used to indicate mouse painting position)
        if self._enable_hover_indicator and self._curr_cbxy.is_in_region:
            xy_cen = self._curr_cbxy.xy_norm
            line_type = self.style.hover_line_type
            if self.style.color_hover_bg is not None:
                bg_col = self.style.color_hover_bg
                bg_thick = self.style.thickness_hover_bg
                draw_circle_norm(out_frame, xy_cen, brush_radius_px, bg_col, bg_thick, line_type)
            fg_col = self.style.color_hover_fg
            fg_thick = self.style.thickness_hover_fg
            draw_circle_norm(out_frame, xy_cen, brush_radius_px, fg_col, fg_thick, line_type)

        return out_frame

    # .................................................................................................................


class HoverLabelOverlay(BaseOverlay):
    """
    Simple overlay used to display text when hovering an element.
    Text will disappear after some idle time if the user does not move their mouse.
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        label: str = "Hover Label",
        idle_timeout_ms: int = 750,
        xy_norm: XYNORM = (0.5, 0),
        scale: float = 0.35,
        thickness: int = 1,
        color: COLORU8 = (255, 255, 255),
        bg_color: COLORU8 = (0, 0, 0),
        anchor_xy_norm: XYNORM | None = None,
        offset_xy_px: XYPX = (0, 0),
        margin_xy_px: XYPX = (5, 5),
    ):
        self._label = label
        self._idle_timeout_ms = idle_timeout_ms
        self._timeout_target_ms = -1
        self._need_text = False
        self._is_in_region = False

        self.style = UIStyle(
            text=TextDrawer(scale, thickness, color, bg_color),
            idle_timeout_ms=idle_timeout_ms,
        )
        self._xy_norm = xy_norm
        self._anchor_xy_norm = anchor_xy_norm
        self._offset_xy_px = offset_xy_px
        self._margin_xy_px = margin_xy_px

        # Inherit from parent
        super().__init__(base_item)

    def set_label(self, label: str) -> SelfType:
        """Update the label shown on hover"""
        self._label = label
        return self

    def _on_move(self, cbxy, cbflags) -> None:

        self._is_in_region = cbxy.is_in_region
        if cbxy.is_in_region:
            self._need_text = True
            self._timeout_target_ms = round(perf_counter() * 1000) + self.style.idle_timeout_ms

        return

    def _render_overlay(self, frame: ndarray) -> ndarray:

        if self._need_text:
            curr_time_ms = round(perf_counter() * 1000)
            self._need_text = self._is_in_region and (curr_time_ms < self._timeout_target_ms)
            if self._need_text:
                return self.style.text.xy_norm(
                    frame, self._label, self._xy_norm, self._anchor_xy_norm, self._offset_xy_px, self._margin_xy_px
                )

        return frame

    # .................................................................................................................


class PointClickOverlay(BaseOverlay):
    """
    Overlay which allows for clicking to add points over top of a base image
    Multiple points can be added by shift clicking. Right click removes points.
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 = (0, 255, 255),
        radius: int = 4,
        bg_color: COLORU8 | None = (0, 0, 0),
        thickness: int = -1,
        line_type: OCVLineType = cv2.LINE_AA,
        max_points: int | None = None,
    ):
        # Inherit from parent
        super().__init__(base_item)

        # Store point state
        self._xy_norm_list: list[tuple[float, float]] = []
        self._is_changed = False
        self._max_points = int(max_points) if max_points is not None else 1_000_000
        assert self._max_points > 0, "Must have max_points > 0"

        self.style = UIStyle(
            color_fg=color,
            color_bg=bg_color,
            radius_fg=radius,
            radius_bg=radius if thickness > 0 else radius + 1,
            thickness_fg=thickness,
            thickness_bg=max(1 + thickness, 2 * thickness) if thickness > 0 else thickness,
            line_type=line_type,
        )

    # .................................................................................................................

    def clear(self, flag_is_changed: bool = True) -> SelfType:
        self._is_changed = (len(self._xy_norm_list) > 0) and flag_is_changed
        self._xy_norm_list = []
        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, tuple]:
        """Returns: is_changed, xy_norm_list"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, tuple(self._xy_norm_list)

    # .................................................................................................................

    def _on_left_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:

        # Add point if shift clicked or update point otherwise
        new_xy_norm = cbxy.xy_norm
        if cbflags.shift_key:
            self.add_points(new_xy_norm)
        else:
            if len(self._xy_norm_list) == 0:
                self._xy_norm_list = [new_xy_norm]
            else:
                self._xy_norm_list[-1] = new_xy_norm

        self._is_changed = True

        return

    def _on_right_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        self.remove_closest(cbxy.xy_norm, cbxy.hw_px)
        return

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Convert points to pixel coords for drawing
        frame_h, frame_w = frame.shape[0:2]
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))
        xy_px_list = [np.int32(xy_norm * norm_to_px_scale) for xy_norm in self._xy_norm_list]

        # Draw each point as a circle with a background if needed
        if self.style.color_bg is not None:
            for xy_px in xy_px_list:
                cv2.circle(
                    frame,
                    xy_px,
                    self.style.radius_bg,
                    self.style.color_bg,
                    self.style.thickness_bg,
                    self.style.line_type,
                )
        for xy_px in xy_px_list:
            cv2.circle(
                frame,
                xy_px,
                self.style.radius_fg,
                self.style.color_fg,
                self.style.thickness_fg,
                self.style.line_type,
            )

        return frame

    # .................................................................................................................

    def add_points(self, *xy_norm_points: XYNORM) -> SelfType:

        if len(xy_norm_points) == 0:
            return self

        # Remove earlier points if needed
        if len(self._xy_norm_list) >= self._max_points:
            self._xy_norm_list.pop(0)

        self._xy_norm_list.extend(xy_norm_points)
        self._is_changed = True

        return self

    # .................................................................................................................

    def remove_closest(self, xy_norm: XYNORM, frame_hw: HWPX | None = None) -> None | XYNORM:

        # Can't remove points if there aren't any!
        if len(self._xy_norm_list) == 0:
            return None

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (10, 10)
        frame_h, frame_w = frame_hw
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))

        # Find the point closest to the given (x,y) for removal
        xy_px_array = np.int32([np.int32(xy_norm * norm_to_px_scale) for xy_norm in self._xy_norm_list])
        input_array = np.int32(xy_norm * norm_to_px_scale)
        dist_to_pts = np.linalg.norm(xy_px_array - input_array, ord=2, axis=1)
        closest_pt_idx = np.argmin(dist_to_pts)

        # Remove the point closest to the click and finish
        closest_xy_norm = self._xy_norm_list.pop(closest_pt_idx)
        self._is_changed = True

        return closest_xy_norm

    # .................................................................................................................


class BoxSelectOverlay(BaseOverlay):

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 = (0, 255, 255),
        thickness: int = 1,
        bg_color: COLORU8 | None = (0, 0, 0),
    ):
        super().__init__(base_item)
        self._xy1xy2_norm_list: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._xy1xy2_norm_inprog = None
        self._is_changed = False

        # Store display config
        self._fg_color = color
        self._bg_color = bg_color
        self._fg_thick = thickness
        self._bg_thick = thickness + 1
        self._ltype = cv2.LINE_4

    # .................................................................................................................

    def style(self, color=None, thickness=None, bg_color=None, bg_thickness=None) -> SelfType:
        """Update box styling. Any settings given as None will remain unchanged"""

        if color is not None:
            self._fg_color = color
        if thickness is not None:
            self._fg_thick = thickness
        if bg_color is not None:
            self._bg_color = bg_color if bg_color != -1 else None
        if bg_thickness is not None:
            self._bg_thick = bg_thickness

        return self

    # .................................................................................................................

    def clear(self, flag_is_changed: bool = True) -> SelfType:
        had_boxes = (len(self._xy1xy2_norm_list) > 0) or (self._xy1xy2_norm_inprog is not None)
        self._is_changed = had_boxes and flag_is_changed
        self._xy1xy2_norm_list = []
        self._xy1xy2_norm_inprog = None
        return self

    # .................................................................................................................

    def read(self, include_in_progress_box: bool = True) -> tuple[bool, tuple]:
        """Returns: is_changed, box_xy1xy2_list"""

        # Toggle change state, if needed
        is_changed = self._is_changed
        self._is_changed = False

        # Get list of boxes including in-progress box if needed
        out_list = self._xy1xy2_norm_list
        if include_in_progress_box:
            is_valid, extra_tlbr = self._make_inprog_tlbr()
            extra_xy1xy2_list = [extra_tlbr] if is_valid else []
            out_list = self._xy1xy2_norm_list + extra_xy1xy2_list

        return is_changed, tuple(out_list)

    # .................................................................................................................

    def _on_left_down(self, cbxy: CBEventXY, cbflags: CBEventFlags):

        # Ignore clicks outside of region
        if not cbxy.is_in_region:
            return

        # Begin new 'in-progress' box
        self._xy1xy2_norm_inprog = [cbxy.xy_norm, cbxy.xy_norm]

        # Remove newest box if we're not shift-clicking
        if not cbflags.shift_key:
            if len(self._xy1xy2_norm_list) > 0:
                self._xy1xy2_norm_list.pop()

        self._is_changed = True

        return

    def _on_drag(self, cbxy: CBEventXY, cbflags: CBEventFlags):

        # Update second in-progress box point
        if self._xy1xy2_norm_inprog is not None:
            new_xy = np.clip(cbxy.xy_norm, 0.0, 1.0)
            self._xy1xy2_norm_inprog[1] = tuple(new_xy)
            self._is_changed = True

        return

    def _on_left_up(self, cbxy: CBEventXY, cbflags: CBEventFlags):

        is_valid, new_tlbr = self._make_inprog_tlbr()
        if is_valid:
            self._xy1xy2_norm_list.append(new_tlbr)
            self._is_changed = True
        self._xy1xy2_norm_inprog = None

        return

    def _on_right_click(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        self.remove_closest(cbxy.xy_norm, cbxy.hw_px)
        return

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Check if we need to draw an in-progress box
        is_valid, new_tlbr = self._make_inprog_tlbr()
        extra_tlbr = [new_tlbr] if is_valid else []
        boxes_to_draw = self._xy1xy2_norm_list + extra_tlbr

        frame_h, frame_w = frame.shape[0:2]
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))
        box_px_list = []
        for box in boxes_to_draw:
            box = np.int32([xy_norm * norm_to_px_scale for xy_norm in box])
            box_px_list.append(box)

        if self._bg_color is not None:
            for xy1_px, xy2_px in box_px_list:
                cv2.rectangle(frame, xy1_px, xy2_px, self._bg_color, self._bg_thick, self._ltype)
        for xy1_px, xy2_px in box_px_list:
            cv2.rectangle(frame, xy1_px, xy2_px, self._fg_color, self._fg_thick, self._ltype)

        return frame

    # .................................................................................................................

    def add_boxes(self, *xy1xy2_norm_list) -> SelfType:

        if len(xy1xy2_norm_list) == 0:
            return self

        self._xy1xy2_norm_list.extend(xy1xy2_norm_list)
        self._is_changed = True

        return self

    # .................................................................................................................

    def remove_closest(self, xy_norm: XYNORM, frame_hw: HWPX = None) -> None | XY1XY2NORM:

        # Can't remove boxes if there aren't any!
        if len(self._xy1xy2_norm_list) == 0:
            return None

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (10, 10)
        frame_h, frame_w = frame_hw
        norm_to_px_scale = np.float32((frame_w - 1, frame_h - 1))

        # For each box, find the distance to the closest corner
        input_array = np.int32(xy_norm * norm_to_px_scale)
        closest_dist_list = []
        for (x1, y1), (x2, y2) in self._xy1xy2_norm_list:
            xy_px_array = np.float32([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) * norm_to_px_scale
            dist_to_pts = np.linalg.norm(xy_px_array - input_array, ord=2, axis=1)
            closest_dist_list.append(min(dist_to_pts))

        # Among all boxes, remove the one with the closest corner to the given click
        closest_pt_idx = np.argmin(closest_dist_list)
        closest_xy1xy2_norm = self._xy1xy2_norm_list.pop(closest_pt_idx)
        self._is_changed = True

        return closest_xy1xy2_norm

    # .................................................................................................................

    def _make_inprog_tlbr(self) -> tuple[bool, XY1XY2NORM]:
        """
        Helper used to make a 'final' box out of in-progress data
        Includes re-arranging points to be in proper top-left/bottom-right order
        as well as discarding boxes that are 'too small'
        """

        new_tlbr = None
        is_valid = self._xy1xy2_norm_inprog is not None
        if is_valid:

            # Re-arrange points to make sure first xy is top-left, second is bottom-right
            xy1_xy2 = np.float32(self._xy1xy2_norm_inprog)
            tl_xy_norm = xy1_xy2.min(0)
            br_xy_norm = xy1_xy2.max(0)

            # Make sure the box is not infinitesimally small
            xy_diff = br_xy_norm - tl_xy_norm
            is_valid = np.all(xy_diff > 1e-4)
            if is_valid:
                new_tlbr = (tl_xy_norm.tolist(), br_xy_norm.tolist())

        return is_valid, new_tlbr

    # .................................................................................................................


class EditBoxOverlay(BaseOverlay):
    """
    Overlay used to provide a 'crop-box' or similar UI
    The idea being to have a single box that can be modified
    by clicking and dragging the corners or sides, or otherwise
    fully re-drawn by clicking far enough away from the box.
    It is always assumed that there is 1 box!

    This differs from the regular 'box select overlay' which
    re-draws boxes on every click and supports multiple boxes
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        color: COLORU8 = (0, 255, 255),
        thickness: int = 2,
        bg_color: COLORU8 | None = (0, 0, 0),
        indicator_radius: int = 3,
        interaction_distance_px: float = 50,
        minimum_box_area_norm: float = 5e-5,
        frame_shape: HWPX | None = None,
        allow_right_click_clear=True,
    ):
        # Inherit from parent
        super().__init__(base_item)

        # Store box points in format that supports 'mid points'
        self._x_norms = np.float32([0.2, 0.5, 0.8])
        self._y_norms = np.float32([0.2, 0.5, 0.8])
        self._prev_xy_norms = (self._x_norms, self._y_norms)
        self._is_changed = True
        self._allow_right_click_clear = allow_right_click_clear

        # Store indexing used to specify which of the box points is being modified, if any
        self._is_modifying = False
        self._xy_modify_idx = (2, 2)
        self._mouse_xy_norm = (0.0, 0.0)

        # Store sizing of frame being cropped, only use when 'nudging' the crop box
        self._frame_shape = None if frame_shape is None else frame_shape[0:2]

        # Store thresholding settings
        self._minimum_area_norm = minimum_box_area_norm
        self._interact_dist_px_threshold = interaction_distance_px

        self.style = UIStyle(
            color_fg=color,
            color_bg=bg_color,
            thickness_fg=thickness,
            thickness_bg=thickness + 1,
            indicator_radius_fg=indicator_radius + thickness,
            indicator_radius_bg=indicator_radius + thickness + 1,
            line_type=cv2.LINE_AA,
        )

    # .................................................................................................................

    def clear(self) -> SelfType:
        """Reset box back to entire frame size"""
        self._x_norms = np.float32([0.0, 0.5, 1.0])
        self._y_norms = np.float32([0.0, 0.5, 1.0])
        self._is_changed = True
        self._is_modifying = False
        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, bool, XY1XY2NORM]:
        """
        Read current box state
        Returns:
            is_changed, is_valid, box_xy1xy2_norm
            -> 'is_box_valid' is based on the minimum box area setting
            -> box_xy1xy2_norm is in format: ((x1, y1), (x2, y2))
        """

        # Toggle change state, if needed
        is_changed = self._is_changed
        self._is_changed = False

        # Get top-left/bottom-right output if it exists
        x1, _, x2 = sorted(self._x_norms.tolist())
        y1, _, y2 = sorted(self._y_norms.tolist())
        box_xy1xy2_norm = ((x1, y1), (x2, y2))
        is_valid = ((x2 - x1) * abs(y2 - y1)) > self._minimum_area_norm

        return is_changed, is_valid, box_xy1xy2_norm

    # .................................................................................................................

    def set_box(self, xy1xy2_norm: XY1XY2NORM) -> SelfType:
        """
        Update box coordinates. Input is expected in top-left/bottom-right format:
            ((x1, y1), (x2, y2))
        """

        (x1, y1), (x2, y2) = xy1xy2_norm
        x_mid, y_mid = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        self._x_norms = np.float32((x1, x_mid, x2))
        self._y_norms = np.float32((y1, y_mid, y2))
        self._is_changed = True
        self._is_modifying = False

        return self

    def set_frame_shape(self, frame_shape: HWPX | None) -> SelfType:
        """Update internal frame sizing. Used to properly 'nudge' by 1 pixel"""
        self._frame_shape = frame_shape
        return self

    # .................................................................................................................

    def nudge(
        self, left: int = 0, right: int = 0, up: int = 0, down: int = 0, frame_shape: HWPX | None = None
    ) -> SelfType:
        """Helper used to move the position of a point (nearest to the mouse) by some number of pixels"""

        # Figure out which point to nudge
        frame_shape = self._frame_shape if frame_shape is None else frame_shape
        frame_hw = frame_shape[0:2] if frame_shape is not None else self._base_item.get_render_hw()
        (x_idx, y_idx), _, _ = self._check_xy_interaction(self._mouse_xy_norm, frame_hw)

        # Handle left/right nudge
        is_leftright_nudgable = x_idx != 1
        leftright_nudge = right - left
        if is_leftright_nudgable and leftright_nudge != 0:
            _, w_px = frame_hw
            old_x_norm = self._x_norms[x_idx]
            old_x_px = old_x_norm * (w_px - 1)
            new_x_px = old_x_px + leftright_nudge
            new_x_norm = new_x_px / (w_px - 1)
            new_x_norm = np.clip(new_x_norm, 0.0, 1.0)

            # Update target x coord and re-compute midpoint
            self._x_norms[x_idx] = new_x_norm
            self._x_norms[1] = (self._x_norms[0] + self._x_norms[-1]) * 0.5

        # Handle up/down nudge
        is_updown_nudgable = y_idx != 1
        updown_nudge = down - up
        if is_updown_nudgable and updown_nudge != 0:
            h_px, _ = frame_hw
            old_y_norm = self._y_norms[y_idx]
            old_y_px = old_y_norm * (h_px - 1)
            new_y_px = old_y_px + updown_nudge
            new_y_norm = new_y_px / (h_px - 1)
            new_y_norm = np.clip(new_y_norm, 0.0, 1.0)

            # Update target x coord and re-compute midpoint
            self._y_norms[y_idx] = new_y_norm
            self._y_norms[1] = (self._y_norms[0] + self._y_norms[-1]) * 0.5

        # Assume we've changed the box
        self._is_changed = True

        return self

    # .................................................................................................................

    @classmethod
    def xy1xy2_norm_to_px(cls, image_shape: IMGSHAPE_HW, box_xy1xy2_norm: XY1XY2NORM) -> XY1XY2PX:
        """
        Helper used to convert from normalized xy1xy2 coordinates to pixel coordinates
        Returns:
            ((x1_px, y1_px), (x2_px, y2_px))
        """

        # For convenience
        img_h, img_w = image_shape[0:2]
        x_scale = img_w - 1
        y_scale = img_h - 1

        # Compute coords in pixel units
        xy1_norm, xy2_norm = box_xy1xy2_norm
        xy1_px = (round(xy1_norm[0] * x_scale), round(xy1_norm[1] * y_scale))
        xy2_px = (round(xy2_norm[0] * x_scale), round(xy2_norm[1] * y_scale))

        return (xy1_px, xy2_px)

    # .................................................................................................................

    def _on_move(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        # Record mouse position for rendering 'closest point' indicator on hover
        self._mouse_xy_norm = cbxy.xy_norm
        return

    def _on_left_down(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        """Create a new box or modify exist box based on left-click position"""

        # Ignore clicks outside of region
        if not cbxy.is_in_region:
            return

        # Record 'previous' box, in case we need to reset (happens if user draws invalid box)
        self._prev_xy_norms = (self._x_norms, self._y_norms)

        # Figure out if we're 'modifying' the box or drawing a new one
        xy_idx, _, is_interactive_dist = self._check_xy_interaction(cbxy.xy_norm, cbxy.hw_px)
        is_new_click = not is_interactive_dist or cbflags.shift_key

        # Either modify an existing point or reset/re-draw the box if clicking away from existing points
        self._xy_modify_idx = xy_idx
        if is_new_click:
            # We modify the 'last' xy coord on new boxes, by convention
            self._xy_modify_idx = (2, 2)
            new_x, new_y = cbxy.xy_norm
            self._x_norms = np.float32((new_x, new_x, new_x))
            self._y_norms = np.float32((new_y, new_y, new_y))

        self._is_modifying = True
        self._is_changed = True

        return

    def _on_drag(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        """Modify box corner or midpoint when dragging"""

        # Bail if no points are being modified (shouldn't happen...?)
        if not self._is_modifying:
            return

        # Don't allow dragging out-of-bounds!
        new_x, new_y = np.clip(cbxy.xy_norm, 0.0, 1.0)

        # Update corner points (if they're the ones being modified) and re-compute mid-points
        x_mod_idx, y_mod_idx = self._xy_modify_idx
        if x_mod_idx != 1:
            self._x_norms[x_mod_idx] = new_x
            self._x_norms[1] = (self._x_norms[0] + self._x_norms[2]) * 0.5
        if y_mod_idx != 1:
            self._y_norms[y_mod_idx] = new_y
            self._y_norms[1] = (self._y_norms[0] + self._y_norms[2]) * 0.5

        # Assume box is changed by dragging update
        self._is_changed = True

        return

    def _on_left_up(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        """Stop modifying box on left up"""

        # Reset modifier indexing
        self._is_modifying = False

        # Reset if the resulting box is too small
        h_px, w_px = cbxy.hw_px
        box_w = int(np.abs(self._x_norms[0] - self._x_norms[2]) * (h_px - 1))
        box_h = int(np.abs(self._y_norms[0] - self._y_norms[1]) * (w_px - 1))
        box_area_norm = (box_h * box_w) / (h_px * w_px)
        if box_area_norm < self._minimum_area_norm:
            self._x_norms, self._y_norms = self._prev_xy_norms
            self._is_changed = True

        return

    def _on_right_click(self, cbxy: CBEventXY, cbflags: CBEventFlags):
        if self._allow_right_click_clear:
            self.clear()
        return

    # .................................................................................................................

    def _check_xy_interaction(
        self,
        target_xy_norm: XYNORM,
        frame_hw: HWPX | None = None,
    ) -> tuple[tuple[int, int], tuple[float, float], bool]:
        """
        Helper used to check which of the box points (corners or midpoints)
        are closest to given target xy coordinate, and what the x/y distance
        ('manhattan distance') is to the closest point. Used to determine
        which points may be interacted with for dragging/modifying the box.

        Returns:
            closest_xy_index, closest_xy_distance_px, is_interactive_distance
            -> Indexing is with respect to self._x_norms & self._y_norms
        """

        # Default to 'fake' pixel count if not given (so we can re-use the same calculations)
        if frame_hw is None:
            frame_hw = (2.0, 2.0)
        h_scale, w_scale = tuple(np.float32(size - 1.0) for size in frame_hw)
        target_x, target_y = target_xy_norm

        # Find closest x point on box
        x_dists = np.abs(self._x_norms - target_x)
        closest_x_index = np.argmin(x_dists)
        closest_x_dist_px = x_dists[closest_x_index] * w_scale

        # Find closest y point on box
        y_dists = np.abs(self._y_norms - target_y)
        closest_y_index = np.argmin(y_dists)
        closest_y_dist_px = y_dists[closest_y_index] * h_scale

        # Check if the point is within interaction distance
        closest_xy_index = (closest_x_index, closest_y_index)
        closest_xy_dist_px = (closest_x_dist_px, closest_y_dist_px)
        is_interactive = all(dist < self._interact_dist_px_threshold for dist in closest_xy_dist_px)
        if is_interactive:
            is_center_point = all(idx == 1 for idx in closest_xy_index)
            is_interactive = not is_center_point

        return closest_xy_index, closest_xy_dist_px, is_interactive

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Get sizing info
        frame_hw = frame.shape[0:2]
        h_scale, w_scale = tuple(float(size - 1.0) for size in frame_hw)
        all_x_px = tuple(int(x * w_scale) for x in self._x_norms)
        all_y_px = tuple(int(y * h_scale) for y in self._y_norms)
        xy1_px, xy2_px = (all_x_px[0], all_y_px[0]), (all_x_px[-1], all_y_px[-1])

        # Figure out whether we should draw interaction indicator & where
        need_draw_indicator = True
        if self._is_modifying:
            # If user if modifying the box, choose the modified point for drawing
            # -> We want to always draw the indicator for the point being dragged, even if
            #    the mouse is closer to some other point (can happen when dragging mid points)
            close_x_px = all_x_px[self._xy_modify_idx[0]]
            close_y_px = all_y_px[self._xy_modify_idx[1]]

        else:
            # If user isn't already interacting, we'll draw an indicator if the mouse is
            # close enough to a corner or mid point on the box. But we have to figure
            # out which point that would be every time we re-render, in case the mouse moved!
            (x_idx, y_idx), _, is_interactive_dist = self._check_xy_interaction(self._mouse_xy_norm, frame_hw)
            close_x_px = all_x_px[x_idx]
            close_y_px = all_y_px[y_idx]
            is_inbounds = np.min(self._mouse_xy_norm) > 0.0 and np.max(self._mouse_xy_norm) < 1.0
            need_draw_indicator = is_interactive_dist and is_inbounds
        closest_xy_px = (close_x_px, close_y_px)

        # Draw all background coloring first, so it appears entirely 'behind' the foreground
        if self.style.color_bg is not None:
            if need_draw_indicator:
                cv2.circle(
                    frame, closest_xy_px, self.style.indicator_radius_bg, self.style.color_bg, -1, self.style.line_type
                )
            cv2.rectangle(frame, xy1_px, xy2_px, self.style.color_bg, self.style.thickness_bg, cv2.LINE_4)

        # Draw box + interaction indicator circle in foreground color
        if need_draw_indicator:
            cv2.circle(
                frame, closest_xy_px, self.style.indicator_radius_fg, self.style.color_fg, -1, self.style.line_type
            )
        cv2.rectangle(frame, xy1_px, xy2_px, self.style.color_fg, self.style.thickness_fg, cv2.LINE_4)

        return frame

    # .................................................................................................................


class GridSelectOverlay(BaseOverlay):
    """
    Overlay which allows for selecting tiles in a 'grid' layout over top of a base item.
    Grid selection follows the mouse position and can be locked by left-clicking and
    unlocked by right-clicking (or left-clicking the same point twice).
    By default, the selected grid cell is outlined with a rectangle, though this
    can be disabled by providing a color of 'None' or toggling visibility directly.

    For example, a grid with only 2 rows and 2 columns would allow for selecting
    quadrants of the underlying base item. Or, a grid with 1 row and 2 columns
    would provide a simple way of selecting between the left & right halves of
    the base item, etc.

    Also provides basic support for displaying a text overlay (use .set_text_overlay)
    next to the selected grid cell, meant for indicating values associated with cells.
    """

    # .................................................................................................................

    def __init__(
        self,
        base_item: BaseCallback,
        num_rows_columns: tuple[int, int] | int,
        color: COLORU8 | int | None = (0, 255, 255),
        thickness: int = 2,
        color_bg: COLORU8 | int | None = (0, 0, 0),
        initial_row_column_select: tuple[int, int] | None = None,
    ):
        # Interpret row/column count as tuple
        if isinstance(num_rows_columns, int):
            num_rows_columns = (num_rows_columns, num_rows_columns)
        num_rows, num_cols = [max(size, 1) for size in num_rows_columns]

        # Figure out initial setting
        initial_rowcol = initial_row_column_select
        if initial_rowcol is not None:
            init_row, init_col = initial_row_column_select
            init_row = max(0, min(num_rows - 1))
            init_col = max(0, min(num_cols - 1))
            initial_row_column_select = (init_row, init_col)

        # Allocate storage for element state
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._selected_row_col: tuple[int, int] | None = initial_row_column_select
        self._is_locked: bool = False
        self._is_visible: bool = color is not None
        self._is_changed: bool = True
        self._text_str: str | None = None

        # Set up element styling
        color_fg = interpret_coloru8(color)
        color_locked = pick_contrasting_gray_color(color_fg, contrast=0.5, color_lerp_weight=0.5)
        self.style = UIStyle(
            color_fg=color_fg,
            color_bg=interpret_coloru8(color_bg),
            color_locked=color_locked,
            thickness_fg=thickness,
            thickness_bg=max(1 + thickness, 2 * thickness) if thickness > 0 else 2,
            text=TextDrawer(0.5, 2 if thickness > 2 else 1, (255, 255, 255), (0, 0, 0)),
            text_margin_px=5,
        )

        # Inherit from parent
        super().__init__(base_item)

    # .................................................................................................................

    def read(self) -> tuple[bool, bool, tuple[int, int] | None]:
        """Returns: is_changed, is_locked, row_column_select"""
        is_changed = self._is_changed
        self._is_changed = False
        return is_changed, self._is_locked, self._selected_row_col

    def get_num_rows_columns(self) -> tuple[int, int]:
        """Get the current number of rows & columns as a tuple"""
        return (self._num_rows, self._num_cols)

    def set_num_rows_columns(self, num_rows_columns: tuple[int, int] | int) -> SelfType:
        """Update the number of rows and columns in the grid"""
        if isinstance(num_rows_columns, int):
            num_rows_columns = (num_rows_columns, num_rows_columns)
        num_rows, num_cols = [max(size, 1) for size in num_rows_columns]

        # Un-select row/column if it lies outside new grid
        if self._selected_row_col is not None:
            row_idx, col_idx = self._selected_row_col
            is_outside_new_grid = (row_idx > (num_rows - 1)) or (col_idx > (num_cols - 1))
            if is_outside_new_grid:
                self._selected_row_col = None
                self._is_changed = True

        # Store new row/column counts
        self._num_rows = num_rows
        self._num_cols = num_cols

        return self

    def set_selected_row_column(
        self,
        selected_row_column: tuple[int, int] | None,
        allow_wraparound: bool = True,
        is_relative_move: bool = False,
        ignore_lock: bool = False,
    ) -> SelfType:
        """
        Programmatically change the selected row/column (i.e. instead of relying on mouse input)
        If 'ignore_lock' is True, then the row/column can be adjusted even from the locked state
        """

        # Do nothing if locked
        if self._is_locked and (not ignore_lock):
            return self

        # Handle special disabling case
        if selected_row_column is None:
            if self._selected_row_col != None:
                self._selected_row_col = None
                self._is_changed = True
            return self

        # Sanity check
        assert isinstance(selected_row_column, Iterable), f"Must provide tuple (got: {selected_row_column})"
        assert len(selected_row_column) == 2, "Error selecting row/column. Must provide (row, column) tuple or None"
        row_idx, col_idx = selected_row_column

        # Handle relatie movement
        if is_relative_move and self._selected_row_col is not None:
            curr_row_idx, curr_col_idx = self._selected_row_col
            row_idx = curr_row_idx + row_idx
            col_idx = curr_col_idx + col_idx

        # Handle negative indexing
        if row_idx < 0:
            row_idx = self._num_rows + row_idx
        if col_idx < 0:
            col_idx = self._num_cols + col_idx

        # Handle wraparound vs. clamping
        if allow_wraparound:
            row_idx = row_idx % self._num_rows
            col_idx = col_idx % self._num_cols
        else:
            row_idx = max(0, min(row_idx, self._num_rows - 1))
            col_idx = max(0, min(col_idx, self._num_cols - 1))

        # Update row/column if it's different from current state
        new_row_col = (row_idx, col_idx)
        if new_row_col != self._selected_row_col:
            self._selected_row_col = new_row_col
            self._is_changed = True

        return self

    def set_text_overlay(self, text: str | None) -> SelfType:

        # Only update text if not-None. If given empty string, disable text
        if text is not None:
            new_text = str(text)
            self._text_str = new_text if len(new_text) > 0 else None

        return self

    # .................................................................................................................

    def toggle_visibility(self, is_visible: bool | None = None) -> bool:
        """Toggle visibilty (or set to True/False if given an input). Returns: is_visible"""
        self._is_visible = not self._is_visible if is_visible is None else is_visible
        return self._is_visible

    def toggle_lock(self, is_locked: bool | None = None) -> bool:
        """Toggle lock state (or set to True/False if given an input). Returns: is_locked"""
        self._is_locked = not self._is_locked if is_locked is None else is_locked
        return self._is_locked

    # .................................................................................................................

    def _on_move(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:

        # Locking stops movement
        if self._is_locked:
            return

        # Flag changes to the selected grid cell
        new_row_col = self._get_new_rowcol_index(cbxy)
        is_changed = new_row_col != self._selected_row_col
        if is_changed:
            self._selected_row_col = new_row_col
            self._is_changed = True

        return

    def _on_left_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:

        # Lock to clicked cell, unless clicking the same (already locked) cell, then unlock
        new_row_col = self._get_new_rowcol_index(cbxy)
        is_changed = new_row_col != self._selected_row_col
        if is_changed:
            self._selected_row_col = new_row_col
            self._is_locked = True
            self._is_changed = True
        else:
            # Toggle lock state when clicking on the same grid cell
            self._is_locked = not self._is_locked
        return

    def _on_right_click(self, cbxy: CBEventXY, cbflags: CBEventFlags) -> None:
        # Unlock on right click
        if self._is_locked:
            self._is_locked = False
            self._on_move(cbxy, cbflags)
        return

    # .................................................................................................................

    def _render_overlay(self, frame: ndarray) -> ndarray:

        # Skip drawing overlay if invisible or nothing is selected
        if not self._is_visible or self._selected_row_col is None or self.style.color_fg is None:
            return frame

        # Figure out grid cell bounds
        frame_h, frame_w = frame.shape[0:2]
        cell_w = (frame_w - 1) / self._num_cols
        cell_h = (frame_h - 1) / self._num_rows
        row_idx, col_idx = self._selected_row_col
        x1, x2 = round(col_idx * cell_w), round((col_idx + 1) * cell_w)
        y1, y2 = round(row_idx * cell_h), round((row_idx + 1) * cell_h)

        # Draw rectangle to indicate grid cell
        bg_color = self.style.color_bg
        fg_color = self.style.color_locked if self._is_locked else self.style.color_fg
        outframe = frame.copy()
        if bg_color is not None:
            cv2.rectangle(outframe, (x1, y1), (x2, y2), bg_color, self.style.thickness_bg, cv2.LINE_4)
        cv2.rectangle(outframe, (x1, y1), (x2, y2), fg_color, self.style.thickness_fg, cv2.LINE_4)

        # Draw text overlay, if present
        if self._text_str is not None:
            txt_h, txt_w, txt_base = self.style.text.get_text_size(self._text_str)
            margin = self.style.text_margin_px

            # Draw text 'x-centered' with selected grid cell (with clipping on left/right bounds)
            txt_half_w, txt_half_h = txt_w * 0.5, txt_h * 0.5
            x_px = (x1 + x2) * 0.5 - txt_half_w
            x_px = min(max(x_px, margin), frame_w - (1 + txt_w + margin))

            # Prefer to draw text above selected cell, unless selection is too close to the top of the frame
            # -> Drawing below cell tends to be obstructed by the mouse!
            y_px = y1 - txt_half_h - margin
            if y_px < txt_half_h:
                y_px = y2 + txt_h + margin

            # Draw text pixel position directly to ensure proper bounding
            self.style.text.xy_px(outframe, self._text_str, (round(x_px), round(y_px)))

        return outframe

    # .................................................................................................................

    def _get_new_rowcol_index(self, cbxy: CBEventXY) -> tuple[int, int] | None:
        """Helper used to get a grid row/column index from a mouse xy event"""
        new_xy_index = None
        if cbxy.is_in_region:
            x_norm, y_norm = cbxy.xy_norm
            row_idx = max(0, min(self._num_rows - 1, int(y_norm * self._num_rows)))
            col_idx = max(0, min(self._num_cols - 1, int(x_norm * self._num_cols)))
            new_xy_index = (row_idx, col_idx)
        return new_xy_index

    # .................................................................................................................
