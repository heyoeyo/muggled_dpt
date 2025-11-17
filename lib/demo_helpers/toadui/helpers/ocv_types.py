#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from typing import TypeAlias, Callable, Protocol
from enum import IntEnum

import cv2


# ---------------------------------------------------------------------------------------------------------------------
# %% Types


class InterpCode(IntEnum):
    nearest = cv2.INTER_NEAREST
    nearest_exact = cv2.INTER_NEAREST_EXACT
    linear = cv2.INTER_LINEAR
    linear_exact = cv2.INTER_LINEAR_EXACT
    cubic = cv2.INTER_CUBIC
    lanczos = cv2.INTER_LANCZOS4
    area = cv2.INTER_AREA


class FontCode(IntEnum):
    simplex = cv2.FONT_HERSHEY_SIMPLEX
    duplex = cv2.FONT_HERSHEY_DUPLEX
    triplex = cv2.FONT_HERSHEY_TRIPLEX
    complex = cv2.FONT_HERSHEY_COMPLEX
    small = cv2.FONT_HERSHEY_COMPLEX_SMALL
    script_simplex = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    script_complex = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    plain = cv2.FONT_HERSHEY_PLAIN
    italic = cv2.FONT_ITALIC

    # Alternate (more conventional) naming
    sans_serif = cv2.FONT_HERSHEY_SIMPLEX
    serif = cv2.FONT_HERSHEY_COMPLEX
    script = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    bold_sans_serif = cv2.FONT_HERSHEY_DUPLEX
    bold_serif = cv2.FONT_HERSHEY_TRIPLEX
    bold_script = cv2.FONT_HERSHEY_SCRIPT_COMPLEX


class LineTypeCode(IntEnum):
    connect_4 = cv2.LINE_4
    connect_8 = cv2.LINE_8
    antialiased = cv2.LINE_AA


class EventCode(IntEnum):
    left_double_click = cv2.EVENT_LBUTTONDBLCLK
    left_down = cv2.EVENT_LBUTTONDOWN
    left_up = cv2.EVENT_LBUTTONUP

    middle_double_click = cv2.EVENT_MBUTTONDBLCLK
    middle_down = cv2.EVENT_MBUTTONDOWN
    middle_up = cv2.EVENT_MBUTTONUP

    right_double_click = cv2.EVENT_RBUTTONDBLCLK
    right_down = cv2.EVENT_RBUTTONDOWN
    right_up = cv2.EVENT_RBUTTONUP

    mouse_hwheel = cv2.EVENT_MOUSEHWHEEL
    mouse_wheel = cv2.EVENT_MOUSEHWHEEL
    mouse_move = cv2.EVENT_MOUSEMOVE


# Helper types which catch general usage (i.e. support None/int so user doesn't need to use enums)
OCVInterp: TypeAlias = None | int | InterpCode
OCVFont: TypeAlias = None | int | FontCode
OCVLineType: TypeAlias = None | int | LineTypeCode
OCVEvent: TypeAlias = int | EventCode
OCVFlag: TypeAlias = int

# Helper used to hint opencv mouse callbacks (event, x_px, y_px, flags, params)
OCVCallback: TypeAlias = Callable[[OCVEvent, int, int, OCVFlag, None], None]


class HasOCVCallback(Protocol):
    def _on_opencv_event(self, event: OCVEvent, x: int, y: int, flags: int, params: None) -> None: ...
