#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from enum import IntEnum
from time import perf_counter

import cv2

from .helpers.window import WindowContextManager, WindowTrackbar, CallbackSequencer, MouseEventsCallback
from .helpers.images import blank_image

# For type hints
from typing import Callable, TypeAlias, Iterable
from numpy import ndarray
from .helpers.types import SelfType, EmptyCallback
from .helpers.ocv_types import OCVCallback, HasOCVCallback


# ---------------------------------------------------------------------------------------------------------------------
# %% Types

KEYCODE: TypeAlias = int | str


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class DisplayWindow:
    """
    Class used to manage opencv window, mostly to make trackbars & callbacks easier to organize.
    The most recent mouse events can be accessed from: window.mouse (e.g. window.mouse.xy)
    """

    WINDOW_CLOSE_KEYS_SET = {ord("q"), 27}  # q, esc

    def __init__(self, window_title: str = "Display - esc to close", display_fps: float = 60):

        # Clear any existing window with the same title
        # -> This forces the window to 'pop-up' when initialized, in case a 'dead' window was still around
        # -> Without this, rendering will resume inside the existing window, which remains hidden
        try:
            cv2.destroyWindow(window_title)
        except cv2.error:
            pass

        # Store window state
        self.title = window_title
        self._frame_delay_ms = int(1000 // display_fps)
        self._last_display_sec = -self._frame_delay_ms
        self.dt = 1.0 / display_fps

        # Variables used for storing window size + changes
        self.size = None
        self._is_size_changed = False

        # Allocate variables for use of keypress callbacks
        self._keypress_callbacks_dict: dict[int, Callable] = {}
        self._keypress_keys_per_descriptions: dict[str | None, str] = {}

        # Fill in blank image to begin (otherwise errors before first image can cause UI to freeze!)
        cv2.namedWindow(self.title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        self.show(blank_image(50, 50), 100)

        # Set up callbacks on the window
        self.mouse = MouseEventsCallback()
        self._mouse_cbs = CallbackSequencer(self.mouse)
        cv2.setMouseCallback(self.title, self._mouse_cbs)

    def __repr__(self) -> str:
        return f"{self.title}  |  event: {self.mouse.event}, flags: {self.mouse.flags}"

    def move(self, x: int | None = None, y: int | None = None) -> SelfType:
        """Wrapper around cv2.moveWindow. If None is given for x or y, will try to re-use existing value"""
        if x is None or y is None:
            old_x, old_y, _, _ = self.get_xywh()
            x = old_x if x is None else x
            y = old_y if y is None else y
        cv2.moveWindow(self.title, x, y)
        return self

    def get_xywh(self) -> tuple[int | None, int | None, int | None, int | None]:
        """
        Wrapper around cv2.getWindowImageRect, with exception handling.
        On success, returns:
            (x, y, width, height)
        On failure (e.g. window is closed), returns None for all values
        """
        try:
            x, y, w, h = cv2.getWindowImageRect(self.title)
        except cv2.error:
            x, y, w, h = None, None, None, None
        return x, y, w, h

    def add_trackbar(self, trackbar_name: str, max_value: int, initial_value: int = 0) -> WindowTrackbar:
        """
        Add built-in (opencv) trackbar to the window
        - These have limited capability and may render inconsistently across platforms
        - Not recommended, except for simple experimentation
        - Use a slider UI element instead
        """
        return WindowTrackbar(self.title, trackbar_name, max_value, initial_value)

    def attach_mouse_callbacks(self, *callbacks: OCVCallback | HasOCVCallback) -> SelfType:
        """
        Attach callbacks for handling mouse events
        Callback functions should have a call signature as folows:

            def callback(event: int, x: int, y: int, flags: int, params: Any) -> None:

                # Example to handle left-button down event
                if event == EVENT_LBUTTONDOWN:
                    print("Mouse xy:", x, y)

                return
        """

        # Sanity check. Make sure we're given callbacks
        # -> We assume we're given objects with a ._on_opencv_event(...) function first
        # -> Otherwise assume we're given a direct function/callable
        actual_cbs = []
        for cb in callbacks:
            if hasattr(cb, "_on_opencv_event"):
                actual_cbs.append(cb._on_opencv_event)
            elif callable(cb):
                actual_cbs.append(cb)
            else:
                print(f"Invalid callback, cannot attach to window ({self.title}):\n{cb}")

        self._mouse_cbs.add(*actual_cbs)
        return self

    def attach_one_keypress_callback(
        self,
        keycode: KEYCODE,
        callback: EmptyCallback | None,
        description: str | None = None,
    ) -> SelfType:
        """
        Attach a callback for handling a keypress event
        Keycodes can be given as strings (i.e. the actual key, like 'a') or for
        keys that don't have simple string representations (e.g. the Enter key),
        the raw keycode integer can be given. To figure out what these are,
        print out the window keypress result while pressing the desired key,
        for example:
            req_break, keypress = window.show(frame)
            print(keypress)

        Callbacks should have no input arguments and no return values.
        """

        # Allow bailing on 'None' callbacks
        if callback is not None:
            self.attach_keypress_callbacks({description: {keycode: callback}})
        return self

    def attach_keypress_callbacks(self, desc_key_callback_dict: dict[str, dict[KEYCODE, EmptyCallback]]) -> SelfType:
        """
        Function used to attach multiple keypress/callback/descriptions at the same time.
        Descriptions, key codes and callbacks must be provided as a nested dictionary when
        using the method to add callbacks. The structure is as follows:

            {
                'description_A': {
                    'w': callback_func1,
                    'e': callback_func2,
                    etc.
                },
                'description_B': {'q': callback_func3} if enable_B else None,
                etc.
            }

        This would add 3 keypress callbacks ('w', 'e' and 'q'), which would be
        listed (when reporting callbacks) under 2 descriptions.

        Note that 'None' can be given in place of the {'key': callback} entry, which
        will cause the entry to be skipped. This is meant to be used for
        conditonally disabling callbacks.

        A 'None' description can also be used, but this just means that no
        description is provided (e.g. callbacks are still enabled).

        Returns self
        """

        for desc, key_to_cb_dict in desc_key_callback_dict.items():

            # Skip None entries (e.g. disabled callbacks)
            if key_to_cb_dict is None:
                continue

            nice_key_names = []
            assert not isinstance(key_to_cb_dict, set), f"Got set instead of dict.\n'{desc}': {key_to_cb_dict}"
            assert isinstance(key_to_cb_dict, dict), "Must provide {'description': {'key': callback}} dictionary!"
            for key_code, callback in key_to_cb_dict.items():

                # Convert list of callback functions into a single function
                if isinstance(callback, Iterable):
                    callback = _many_callbacks_to_one(callback)

                # Convert from string to int (code) if needed
                kcode = key_code
                if isinstance(kcode, str):
                    lowered_kcode = kcode.lower()
                    if len(kcode) > 1:
                        is_known_code = lowered_kcode in KEYNAMES_TO_CODES.keys()
                        if not is_known_code:
                            cv2.destroyAllWindows()
                            valid_names_str = ", ".join(KEYNAMES_TO_CODES.keys())
                            raise NameError(f"Unknown key name ({kcode}), must be one of: {valid_names_str}")
                        kcode = KEYNAMES_TO_CODES[lowered_kcode]
                    else:
                        kcode = ord(lowered_kcode)
                    pass

                # Get user-friendly name for key, if possible
                is_known_key, name = KEY.get_keycode_name(kcode)
                nice_name = name if is_known_key else chr(kcode)
                nice_key_names.append(nice_name)

                # Store key-to-callback entry
                if kcode in self._keypress_callbacks_dict:
                    raise KeyError("Key code ({nice_name}) already in use!")
                self._keypress_callbacks_dict[kcode] = callback

            # Allocate storage for keys per description, and store
            if desc not in self._keypress_keys_per_descriptions.keys():
                self._keypress_keys_per_descriptions[desc] = []
            self._keypress_keys_per_descriptions[desc].extend(nice_key_names)

        return self

    def report_keypress_descriptions(
        self,
        print_directly: bool = True,
        print_header: str | None = "***** Keyboard Controls: *****",
        print_trailing_blank_line: bool = True,
    ) -> dict[str, list[str]]:
        """Helper used to print out a list of all keypress callback descriptions"""

        if print_directly:
            strs_to_print = [] if print_header is None else ["", print_header]
            for desc, keys_list in self._keypress_keys_per_descriptions.items():
                joiner = ", " if "," not in keys_list else " or "
                keys_str = joiner.join(keys_list)
                full_str = f"{desc}: {keys_str}"
                if len(full_str) > 30 and len(keys_list) > 1:
                    strs_to_print.append(f"{desc}:")
                    strs_to_print.append(f"    {keys_str}")
                else:
                    strs_to_print.append(full_str)

            if print_trailing_blank_line:
                strs_to_print.append("")
            print(*strs_to_print, sep="\n", flush=True)

        return {desc: keys_list for desc, keys_list in self._keypress_keys_per_descriptions.items()}

    def run_keypress_callbacks(self, keypress: int) -> SelfType:
        """
        Helper used to run any attached keypress callbacks. This happens automatically
        when calling window.show(...). The only reason to use this function is if
        manually using cv2.imshow(...) & cv2.waitKey(...)
        """
        for cb_keycode, cb in self._keypress_callbacks_dict.items():
            if keypress == cb_keycode:
                cb()
        return self

    def show(self, image: ndarray, frame_delay_ms: float | None = None) -> [bool, int]:
        """
        Function which combines both opencv functions: 'imshow' and 'waitKey'
        This is meant as a convenience function in cases where only a single window is being displayed.
        If more than one window is displayed, it is better to use 'imshow' and 'waitKey' separately,
        so that 'waitKey' is only called once!
        Returns:
            request_close, keypress
        """

        # Figure out frame delay (to achieve target FPS) if we're not given one
        if frame_delay_ms is None:
            time_elapsed_ms = round(1000 * (perf_counter() - self._last_display_sec))
            frame_delay_ms = max(self._frame_delay_ms - time_elapsed_ms, 1)

        cv2.imshow(self.title, image)
        keypress = cv2.waitKey(int(frame_delay_ms)) & 0xFF
        curr_time_sec = perf_counter()
        self.dt, self._last_display_sec = curr_time_sec - self._last_display_sec, curr_time_sec

        self.run_keypress_callbacks(keypress)
        request_close = keypress in self.WINDOW_CLOSE_KEYS_SET

        return request_close, keypress

    def imshow(self, image: ndarray) -> SelfType:
        """
        Wrapper around opencv imshow, fills in 'winname' with the window title.
        Doesn't include any of the additional checks/features of using .show(...)
        """
        cv2.imshow(self.title, image)
        return self

    @classmethod
    def waitKey(cls, frame_delay_ms: float = 1) -> [bool, int]:
        """
        Wrapper around opencv waitkey (triggers draw to screen)
        Returns:
            request_close, keypress
        """

        keypress = cv2.waitKey(int(frame_delay_ms)) & 0xFF
        request_close = keypress in cls.WINDOW_CLOSE_KEYS_SET
        return request_close, keypress

    def close(self) -> bool:
        """Close window. Returns: was_open"""
        was_open = False
        try:
            cv2.destroyWindow(self.title)
            was_open = True
        except cv2.error:
            pass

        return was_open

    def auto_close(self, *clean_up_functions: EmptyCallback) -> WindowContextManager:
        """
        Context manager for auto-closing a window when finished.
        Accepts callback functions, which will be executed after the window is closed,
        even if an error occurs. This acts a bit like the 'finally' block of
        a try/except statement.

        Example usage:
            window = DisplayWindow("My Window")
            vreader = make_video_reader(...)
            # ... other setup code ...

            with window.auto_close(vreader.close):
                for frame in video:
                    window.show(frame)

            # ... Window & vreader will be closed ...
        """
        return WindowContextManager(self.title, *clean_up_functions)

    def enable_size_control(
        self,
        initial_size: int = 900,
        minimum: int = 350,
        maximum: int = 4000,
        step: int = 50,
        decrement_key: KEYCODE = "-",
        increment_key: KEYCODE = "=",
    ) -> SelfType:
        """
        For convenience, enables use of the .size attribute on the window for render sizing.
        Also adds keypress callbacks for adjusting the sizing.

        Note that this alone does not auto-handle sizing!
        This is intended to be used to set sizing when rendering
        a final display, for example:

            window.enable_size_control(...)
            # ...
            # During render loop:
            ui_result.render(h=window.size)

        This is a bit of an ugly hack to help with a common use-case.
        For more involved size control, it is much better to manually
        manage the size variable (e.g. outside of the window object!).
        """

        initial_size, minimum, maximum, step = [int(val) for val in (initial_size, minimum, maximum, step)]
        self.size = initial_size
        self._is_size_changed = True

        def increase_size():
            new_size = min(self.size + step, maximum)
            self._is_size_changed = self.size != new_size
            self.size = new_size

        def decrease_size():
            new_size = max(self.size - step, minimum)
            self._is_size_changed = self.size != new_size
            self.size = new_size

        self.attach_keypress_callbacks(
            {
                "Adjust window size": {
                    decrement_key: decrease_size,
                    increment_key: increase_size,
                }
            }
        )

        return self

    def is_size_changed(self) -> bool:
        """Check if window size has changed (only makes sense if size control has been enabled!)"""
        is_changed = self._is_size_changed
        self._is_size_changed = False
        return is_changed


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def _many_callbacks_to_one(callbacks: Iterable[EmptyCallback]) -> EmptyCallback:
    """
    Helper used to create a 'single' callback function which
    itself calls multiple callbacks in sequence
    """

    def _one_callback():
        for cb in callbacks:
            cb()

    return _one_callback


# ---------------------------------------------------------------------------------------------------------------------
# %% Define window key codes


class KEY(IntEnum):

    L_ARROW = 81
    U_ARROW = 82
    R_ARROW = 83
    D_ARROW = 84

    ESC = 27
    ENTER = 13
    BACKSPACE = 8
    SPACEBAR = ord(" ")
    TAB = ord("\t")

    SHIFT = 225
    ALT = 233
    CAPSLOCK = 229
    # CTRL = None # No key code for this one surprisingly!?

    @classmethod
    def get_keycode_name(cls, code: int) -> tuple[bool, str | None]:
        """Returns a 'nice name' given a key code value. Returns: is_known_code, name_of_keycode"""
        contains_kcode = code in tuple(cls)
        name = cls(code).name if contains_kcode else None
        return contains_kcode, name


KEYNAMES_TO_CODES: dict[str, int] = {
    "l_arrow": KEY.L_ARROW,
    "u_arrow": KEY.U_ARROW,
    "r_arrow": KEY.R_ARROW,
    "d_arrow": KEY.D_ARROW,
    "esc": KEY.ESC,
    "escape": KEY.ESC,
    "enter": KEY.ENTER,
    "return": KEY.ENTER,
    "backspace": KEY.BACKSPACE,
    "spacebar": KEY.SPACEBAR,
    "tab": KEY.TAB,
    "shift": KEY.SHIFT,
    "alt": KEY.ALT,
    "capslock": KEY.CAPSLOCK,
}
