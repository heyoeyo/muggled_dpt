#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

# For type hints
from typing import Any
from numpy import ndarray


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class WindowTrackbar:
    """
    Class used to simplify use of built-in window trackbars provided by opencv.
    This class is just a wrapper around the three functions:
        cv2.createTrackbar
        cv2.getTrackbarPos
        cv2.setTrackbarPos

    Note that these trackbars are limited to integer ranges, starting at 0,
    and may render differently on different platforms.
    """

    def __init__(self, window_title: str, trackbar_label: str, max_value: int, initial_value: int = 0):

        # Sanity checks. Opencv trackbars only allow for integer range, starting at 0
        assert max_value > 0, "Max value must be an integer > 0"
        assert initial_value >= 0, "Initial value must be an integer > 0"
        max_value = int(max_value)
        initial_value = int(min(initial_value, max_value))

        # Set up trackbar
        self.label = trackbar_label
        self._window_title = window_title
        cv2.createTrackbar(trackbar_label, window_title, initial_value, max_value, lambda x: None)

        # Store config
        self._prev_value = initial_value
        self._reset_value = initial_value
        self._max_value = max_value
        self._read_lambda = lambda x: x
        self._write_lambda: lambda x: x

    def reset(self) -> tuple[bool, int]:
        """Reset trackbar. Returns: is_changed, raw_read_value"""
        return self.write_raw(self._reset_value)

    def read_raw(self) -> tuple[bool, int]:
        """Read trackbar value without read-mapping. Returns: is_changed, raw_read_value"""
        raw_value = cv2.getTrackbarPos(self.label, self._window_title)
        is_changed = raw_value != self._prev_value
        self._prev_value = raw_value
        return is_changed, raw_value

    def write_raw(self, new_value: int) -> tuple[bool, int]:
        """Set trackbar value without write-mapping. Returns: is_changed, new_value"""
        safe_value = max(0, min(new_value, self._max_value))
        cv2.setTrackbarPos(self.label, self._window_title, safe_value)
        is_changed = safe_value != self._prev_value
        self._prev_value = safe_value
        return is_changed, safe_value

    def read(self) -> tuple[bool, Any]:
        """Read trackbar value. Returns: is_changed, read_value"""
        raw_value = self.read_raw()
        return self._read_lambda(raw_value)

    def write(self, new_value: Any) -> tuple[bool, int]:
        """Set trackbar value. Returns: is_changed, new_value"""
        new_value = int(self._write_lambda(new_value))
        return self.write_raw(new_value)

    def set_lambdas(self, read_lambda=None, write_lambda=None, verify=True):
        """
        Function which allows for setting functions which are applied when
        reading/writing values from/to the trackbar and can be used to map
        raw integer trackbar values to some other value range
        (including converting to different data types!).
        By default, trackbars use a simple: lambda x: x
        (e.g. values are written/read unchanged).

        An example of a read lambda which divides the raw value by 100:
            read_lambda = lambda raw_value: raw_value/100

        If this function is given 'None' for either lambda, then the
        existing lambda will not be modified.
        """

        if read_lambda is not None:
            assert callable(read_lambda), "read_lambda must be a function that takes a single integer argument"
            if verify:
                try:
                    read_lambda(0)
                except TypeError:
                    raise TypeError("read_lambda must take only a single argument!")
            self._read_lambda = read_lambda

        if write_lambda is not None:
            assert callable(write_lambda), "write_lambda must be a function that takes a single integer argument"
            if verify:
                try:
                    int(write_lambda(0))
                except TypeError:
                    raise TypeError("write_lambda must take in only a single argument!")
            self._write_lambda = write_lambda

        return self


class CallbackSequencer:
    """
    Simple wrapper used to execute more than one callback on a single opencv window

    Example usage:

        # Set up window that will hold callbacks
        winname = "Display"
        cv2.namedWindow(winname)

        # Create multiple callbacks and combine into sequence so they can both be added to the window
        cb_1 = MakeCB(...)
        cb_2 = MakeCB(...)
        cb_seq = CallbackSequence(cb_1, cb_2)
        cv2.setMouseCallback(winname, cb_seq)
    """

    def __init__(self, *callbacks):
        self._callbacks = [cb for cb in callbacks]

    def add(self, *callbacks):
        self._callbacks.extend(callbacks)

    def __call__(self, event, x, y, flags, param) -> None:
        for cb in self._callbacks:
            cb(event, x, y, flags, param)
        return

    def __getitem__(self, index):
        return self._callbacks[index]

    def __iter__(self):
        yield from self._callbacks


class MouseEventsCallback:
    """Simple helper for recording mouse events"""

    def __init__(self):

        # Storage for recording most recent event
        self.xy: ndarray = np.int32([-1, -1])
        self.event: int = 0
        self.flags: int = 0

        # Storage for checking for mouse movement
        self._last_move_xy: ndarray = self.xy.copy()

    def __repr__(self):
        return f"event: {self.event}, flags: {self.flags}, xy: ({self.xy[0]}, {self.xy[1]})"

    def __call__(self, event, x, y, flags, param) -> None:
        self.xy[0] = x
        self.xy[1] = y
        self.event = event
        self.flags = flags
        return

    def is_moved(self) -> bool:
        """
        Helper used to check for mouse movement. Works by directly
        checking for x/y changes (not using events!).

        Note that this check is stateful! Each call records the
        current mouse xy for use on the next check.
        """
        is_moved = not np.allclose(self.xy, self._last_move_xy)
        self._last_move_xy[0] = self.xy[0]
        self._last_move_xy[1] = self.xy[1]
        return is_moved


class WindowContextManager:
    """Simple helper which can be used as a context manager for auto-closing windows"""

    def __init__(self, window_title: str, *clean_up_functions):
        self._window_title = window_title
        self._cleanup_funcs = list(clean_up_functions)

    def __enter__(self):
        return

    def __exit__(self, exception_type, exception_value, exception_traceback):

        # Make sure we close the original window caller and run any other clean-up functions
        cv2.destroyWindow(self._window_title)
        for func in self._cleanup_funcs:
            func()

        suppress_error = exception_type is KeyboardInterrupt
        return suppress_error
