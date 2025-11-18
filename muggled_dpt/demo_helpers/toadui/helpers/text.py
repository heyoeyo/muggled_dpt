#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from .styling import UIStyle, get_background_thickness

# For type hints
from typing import NamedTuple
from numpy import ndarray
from .types import XYPX, XYNORM, COLORU8, SelfType
from .ocv_types import OCVFont, OCVLineType


# ---------------------------------------------------------------------------------------------------------------------
# %% Data types


class TextSize(NamedTuple):
    h: int
    w: int
    base: int


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class TextDrawer:
    """
    Helper used to handle text-drawing onto images
    If a background color is given, text will be drawn with a thicker background for better contrast
    """

    # .................................................................................................................

    def __init__(
        self,
        scale: float = 0.5,
        thickness: int = 1,
        color: COLORU8 = (255, 255, 255),
        bg_color: COLORU8 | None = None,
        font: OCVFont = cv2.FONT_HERSHEY_SIMPLEX,
        line_type: OCVLineType = cv2.LINE_AA,
        max_width: float | None = None,
        max_height: float | None = None,
    ):

        self.style = UIStyle(
            color=color,
            bg_color=bg_color,
            fg_thickness=thickness,
            bg_thickness=get_background_thickness(thickness),
            font=font,
            scale=scale,
            line_type=line_type,
        )

        # Enforce sizing limits if needed
        if (max_height is not None) or (max_width is not None):
            self.scale_to_hw(max_height, max_width, allow_upscale=False)

        pass

    # .................................................................................................................

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(scale={self.style.scale}, thickness={self.style.fg_thickness}, color={self.style.color})"

    # .................................................................................................................

    @classmethod
    def from_existing(cls, other_text_drawer):
        assert isinstance(other_text_drawer, cls), "Must be created from another text drawer instance!"
        new_text_drawer = cls()
        new_text_drawer.style = other_text_drawer.style.copy()
        return new_text_drawer

    # .................................................................................................................

    def xy_px(
        self,
        image: ndarray,
        text: str,
        xy_px: XYPX,
        color: COLORU8 | None = None,
    ) -> ndarray:
        """Helper used to draw text at a give location using pre-configured settings"""

        # Fill in defaults
        if color is None:
            color = self.style.color

        if self.style.bg_color is not None:
            image = cv2.putText(
                image,
                text,
                xy_px,
                self.style.font,
                self.style.scale,
                self.style.bg_color,
                self.style.bg_thickness,
                self.style.line_type,
            )

        return cv2.putText(
            image,
            text,
            xy_px,
            self.style.font,
            self.style.scale,
            color,
            self.style.fg_thickness,
            self.style.line_type,
        )

    # .................................................................................................................

    def xy_norm(
        self,
        image: ndarray,
        text: str,
        xy_norm: XYNORM,
        anchor_xy_norm: XYNORM | None = None,
        offset_xy_px: XYPX = (0, 0),
        margin_xy_px: XYPX | int = (5, 5),
        color: COLORU8 | None = None,
    ) -> ndarray:
        """
        Helper used to draw text given normalized (0-to-1) xy coordinates
        An anchor point can be provided to change where the text is drawn, relative
        to the given xy_norm position. This can be used to get 'left/center/right justified' text.
        If an anchor point isn't given, then it will match the xy_norm value itself,
        which will lead to text always being drawn within the image, as long as 0-to-1 coordinates are given.

        For example, an anchor of (0.5, 0.5) means that the text will be centered on the given xy_norm position.
        - To draw text that is centered, but 'left-justified', use xy_norm=(0.5, 0.5), anchor=(0, 0)
        - To draw text in the top-left corner, use xy_norm=(0, 0)
        - To draw text in the bottom-right corner, use xy_norm=(1, 1)
        - To draw text at the bottom-center, use xy_norm=(0.5, 1)
        """

        # Figure out pixel coords for the given normalized position
        txt_h, txt_w, txt_base = self.get_text_size(text, self.style.scale, self.style.fg_thickness)
        img_h, img_w = image.shape[0:2]
        x_norm, y_norm = xy_norm

        # Handle integer margin inputs
        if isinstance(margin_xy_px, int):
            margin_xy_px = (margin_xy_px, margin_xy_px)
        x_margin, y_margin = margin_xy_px

        # If no anchor is given, match to positioning, which has a 'bounding' effect of text position
        if anchor_xy_norm is None:
            anchor_xy_norm = xy_norm

        # Figure out text positioning on image, in pixel coords
        anchor_x_norm, anchor_y_norm = anchor_xy_norm
        txt_x_px = x_norm * (img_w - 1 - 2 * x_margin) - txt_w * anchor_x_norm + x_margin
        txt_y_px = y_norm * (img_h - 1 - 2 * y_margin) + txt_h * (1 - anchor_y_norm) + y_margin

        # Apply offset before final drawing
        offset_x_px, offset_y_px = offset_xy_px
        txt_xy_px = (round(txt_x_px + offset_x_px), round(txt_y_px + offset_y_px))
        return self.xy_px(image, text, txt_xy_px, color)

    # .................................................................................................................

    def xy_centered(
        self,
        image: ndarray,
        text: str,
        color: COLORU8 | None = None,
        offset_xy_px: XYPX = (0, 0),
    ) -> ndarray:
        """Helper used to draw x/y centered text"""
        xy_norm, anchor_xy_norm, margin_xy = (0.5, 0.5), (0.5, 0.5), (0, 0)
        return self.xy_norm(image, text, xy_norm, anchor_xy_norm, offset_xy_px, margin_xy, color)

    # .................................................................................................................

    def draw_to_box_norm(
        self,
        image: ndarray,
        text: str,
        xy1_norm: XYNORM = (0.0, 0.0),
        xy2_norm: XYNORM = (1.0, 1.0),
        margin_xy_px: XYPX | int = (5, 5),
        scale_step_size: float = 0.05,
    ) -> ndarray:
        """
        Function used to draw text in order to 'fill' a given box region in the image.

        The scale of the text will be chosen so that the text fits into the
        box given by the top-left/bottom-right coords. (xy1, xy2), minus any
        margin specified and with a scaling limited to multiples of the
        given scale step size (rendering can be cleaner with certain multiples).
        """

        # Handle integer margin inputs
        if isinstance(margin_xy_px, int):
            margin_xy_px = (margin_xy_px, margin_xy_px)

        # Figure out how large of a drawing area we have
        img_h, img_w = image.shape[0:2]
        (x1, y1), (x2, y2) = xy1_norm, xy2_norm
        target_h = max(1, (abs(y2 - y1) * img_h) - margin_xy_px[1])
        target_w = max(1, (abs(x2 - x1) * img_w) - margin_xy_px[0])

        # Figure out how much to adjust scale to fit target size
        base_scale = 1
        txt_h, txt_w, _ = self.get_text_size(text, base_scale)
        h_scale, w_scale = target_h / txt_h, target_w / txt_w
        scale_adjust = min(h_scale, w_scale)

        # Draw text to new scale (with step size limiting)
        xy_mid = tuple((a + b) / 2 for a, b in zip(xy1_norm, xy2_norm))
        new_scale = np.floor((base_scale * scale_adjust) / scale_step_size) * scale_step_size

        # Cache previous scale, draw text, then restore scale (so we don't mess with original config)
        old_scale = self.style.scale
        self.style.scale = new_scale
        out_img = self.xy_norm(image, text, xy_mid, anchor_xy_norm=(0.5, 0.5), margin_xy_px=(0, 0))
        self.style.scale = old_scale
        return out_img

    # .................................................................................................................

    def get_text_size(
        self,
        text: str,
        scale: float | None = None,
        thickness: int | None = None,
    ) -> TextSize:
        """
        Helper used to check how big a piece of text will be when drawn
        Returns:
            text_height_px, text_width_px, text_baseline

        - Note the height/width returned is in 'matrix' order, reversed from cv2.getTextSize!
        """

        if scale is None:
            scale = self.style.scale
        if thickness is None:
            thickness = self.style.fg_thickness

        (txt_w, txt_h), txt_base = cv2.getTextSize(text, self.style.font, scale, thickness)

        return TextSize(txt_h, txt_w, txt_base)

    # .................................................................................................................

    def scale_to_hw(
        self,
        h: float | None = None,
        w: float | None = None,
        example_text: str | None = None,
        allow_upscale: bool = False,
    ) -> SelfType:
        """
        Function used to adjust text scaling (of an existing instance)
        such that the provided text fits within the provided height & width.

        If no example text is provided the text: 'Testing' will be used.
        If 'allow_upscale' is True, then the scaling is allowed to
        scale upwards to fit into the provided height/width.
        """

        # Sanity check
        targ_h, targ_w = h, w
        assert not (targ_h is None and targ_w is None), "Must provide at least one target height or width"

        # Make up dummy text to check, if needed
        if example_text is None:
            example_text = "Testing"

        # Make sure our text sizing fits in the given bar height
        txt_h, txt_w, txt_base = self.get_text_size(example_text)
        txt_total_h = txt_h + txt_base

        # Adjust scaling factor if needed
        h_scale = targ_h / txt_total_h if targ_h is not None else 1
        w_scale = targ_w / txt_w if targ_w is not None else 1
        downscale_factor = min(h_scale, w_scale)
        if downscale_factor < 1:
            self.style.scale *= downscale_factor
        elif allow_upscale:
            self.style.scale *= max(h_scale, w_scale)

        return self

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def find_minimum_text_width(text_drawer: TextDrawer, max_num_characters: int, padding_px: int = 6) -> int:
    """Helper used to find a minimum width of an image that needs to hold some number of characters"""
    txt_for_sizing = "M" * (max_num_characters)
    _, txt_w, _ = text_drawer.get_text_size(txt_for_sizing)
    return txt_w + 2 * padding_px


def find_minimum_text_height(text_drawer: TextDrawer, example_text=None, padding_px: int = 6) -> int:
    """Helper used to find a minimum height of an image that needs to hold some text"""
    txt_h, _, txt_base = text_drawer.get_text_size("Testing")
    return txt_h + txt_base + 2 * padding_px


# ---------------------------------------------------------------------------------------------------------------------
# %% Demo

if __name__ == "__main__":

    txt1 = TextDrawer()
    txt2 = TextDrawer(2, 2, color=(0, 0, 255))

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image = txt2.xy_norm(image, "X=0.25", (0.25, 0.1), color=(0, 255, 255))
    image = txt2.xy_norm(image, "AncX=0.75", (0.75, 0.9), color=(255, 255, 0))
    image = txt2.xy_centered(image, "**CENTERED**", color=(0, 255, 255))
    image = txt1.xy_norm(image, "LEFT-ANCHORED", (0.5, 0.25), (0, 0.5))
    image = txt1.xy_norm(image, "RIGHT-ANCHORED", (0.5, 0.75), (1, 0.5))

    image = txt1.xy_norm(image, "Top-Left", (0, 0))
    image = txt1.xy_norm(image, "Bot-Left", (0, 1))
    image = txt1.xy_norm(image, "Top-Right", (1, 0))
    image = txt1.xy_norm(image, "Bot-Right", (1, 1))

    while True:
        cv2.imshow("Example - esc to close", image)
        keypress = cv2.waitKey(0)
        if keypress == 27:
            break
    cv2.destroyAllWindows()
