#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

from typing import Protocol
import os.path as osp

import cv2
import numpy as np

from .base import BaseCallback
from .helpers.images import blank_image
from .helpers.sizing import get_image_hw_for_max_side_length
from .helpers.drawing import draw_box_outline
from .helpers.icons import draw_play_pause_icons
from .helpers.styling import UIStyle
from .helpers.colors import interpret_coloru8, pick_contrasting_gray_color, lerp_colors
from .helpers.truchet_patterns import draw_truchet, make_dot_tiles

# For type hints
from numpy import ndarray
from .helpers.types import COLORU8, SelfType


# ---------------------------------------------------------------------------------------------------------------------
# %% Protocols


class PauseableVideoReader(Protocol):

    def get_frame_count(self) -> int: ...

    def toggle_pause(self, new_pause_state: bool | None = None) -> bool: ...

    def get_pause_state(self) -> bool: ...

    def get_playback_position(self, normalized=True) -> int | float: ...

    def set_playback_position(self, position: int | float, is_normalized=False) -> int: ...

    def next_frame(self, num_frames: int) -> SelfType: ...

    def prev_frame(self, num_frames=int) -> SelfType: ...


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class LoopingVideoReader(PauseableVideoReader):
    """
    Helper used to provide looping frames from video, along with helpers
    to control playback & frame sizing
    Example usage:

        vreader = LoopingVideoReader("path/to/video.mp4")
        for is_paused, frame_idx, frame in vreader:
            # Do something with frames...
            if i_want_to_stop:
                break
    """

    # .................................................................................................................

    def __init__(self, video_path: str, display_size_px: int | None = None, initial_position_0_to_1: float = 0.0):

        # Store basic video data
        self._video_path = video_path
        self._vcap = create_VideoCapture(self._video_path)
        self._total_frames = int(self._vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._max_frame_idx = self._total_frames - 1
        self._fps = self._vcap.get(cv2.CAP_PROP_FPS)

        # Enable rotated orientation fix (disabled by default on opencv v4.11 for reason)
        # See: https://github.com/opencv/opencv/issues/26795
        self._vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        # Jump ahead to a different starting position if needed
        if initial_position_0_to_1 > 1e-10:
            self._vcap.set(cv2.CAP_PROP_POS_FRAMES, self._max_frame_idx * initial_position_0_to_1)

        # Read sample frame & reset video
        rec_frame, first_frame = self._vcap.read()
        if not rec_frame:
            raise IOError(f"Can't read frames from video! ({video_path})")
        self._vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.sample_frame = first_frame

        # Set up display sizing
        self._interpolation: int | None = None
        self._need_resize: bool = False
        self._scale_wh: tuple[int, int] = (1, 1)
        self.shape: tuple[int, int, int] = (1, 1, 3)
        self.set_display_size(display_size_px)

        # Allocate storage for 'previous frame', which is re-used when paused &
        self._is_paused = False
        self._frame_idx = 0
        self._pause_frame = self.scale_to_display_wh(first_frame) if self._need_resize else first_frame

    # .................................................................................................................

    def scale_to_display_wh(self, image: ndarray, interpolation=None) -> ndarray:
        """Helper used to scale a given image to a target display size (if configured)"""
        interp = self._interpolation if interpolation is None else interpolation
        return cv2.resize(image, dsize=self._scale_wh, interpolation=interp)

    def release(self) -> SelfType:
        """Close access to video source"""
        self._vcap.release()
        return self

    def toggle_pause(self, new_pause_state: bool | None = None) -> bool:
        """Toggle pause state (or set to True/False if given an input). Returns: is_paused"""
        self._is_paused = (not self._is_paused) if new_pause_state is None else new_pause_state
        return self._is_paused

    # .................................................................................................................

    def read(self) -> [bool, int, ndarray]:
        """
        Read the next available frame
        Returns:
            is_paused, frame_index, frame_bgr
        """

        # Don't read video frames while paused
        if self._is_paused:
            return self._is_paused, self._frame_idx, self._pause_frame

        # Read next frame, or loop back to beginning if there are no more frames
        self._frame_idx += 1
        read_ok, frame = self._vcap.read()
        if not read_ok:
            self._frame_idx = 0
            self._vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            read_ok, frame = self._vcap.read()
            assert read_ok, "Error looping video! Unable to read first frame"

        # Scale frame for display & store in case we pause
        if self._need_resize:
            frame = self.scale_to_display_wh(frame)
        self._pause_frame = frame

        return self._is_paused, self._frame_idx, frame

    # .................................................................................................................

    def next_frame(self, num_frames=1) -> SelfType:
        self.toggle_pause(True)
        self.set_playback_position((self._frame_idx + num_frames) % self._total_frames)
        return self

    def prev_frame(self, num_frames=1) -> SelfType:
        return self.next_frame(-num_frames)

    # .................................................................................................................

    def is_open(self) -> bool:
        """Check if video is 'open' (i.e. if frames can be read from it)"""
        return self._vcap.isOpened()

    def open(self) -> SelfType:
        """Re-open the video if it has been closed"""
        if not self.is_open():
            self._vcap = create_VideoCapture(self._video_path)
        return self

    # .................................................................................................................

    def get_sample_frame(self) -> ndarray:
        """Helper used to retrieve a sample frame (the first frame), most likely for init use-cases"""
        return self.sample_frame.copy()

    def get_pause_state(self) -> bool:
        """Helper used to figure out if the video is paused (separately from the frame iteration)"""
        return self._is_paused

    def get_frame_delay_ms(self, max_allowable_ms: int = 1000) -> int:
        """Returns a frame delay (in milliseconds) according to the video's reported framerate"""
        frame_delay_ms = round(1000.0 / self._fps)
        return int(min(max_allowable_ms, frame_delay_ms))

    def get_framerate(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._total_frames

    # .................................................................................................................

    def get_playback_position(self, normalized=True) -> int | float:
        """Returns playback position either as a frame index or a number between 0 and 1 (if normalized)"""
        if normalized:
            return self._vcap.get(cv2.CAP_PROP_POS_FRAMES) / self._max_frame_idx
        return int(self._vcap.get(cv2.CAP_PROP_POS_FRAMES))

    def set_playback_position(self, position: int | float, is_normalized=False) -> int:
        """Set position of video playback. Returns frame index"""

        frame_idx = round(position * self._max_frame_idx) if is_normalized else position
        frame_idx = max(min(frame_idx, self._max_frame_idx), 0)

        self._vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._frame_idx = frame_idx

        # If we're paused, but set a new frame, then update the pause frame
        # -> This is important for paused 'timeline scrubbing' to work intuitively
        if self._is_paused:
            ok_read, frame = self._vcap.read()
            if ok_read and self._need_resize:
                frame = self.scale_to_display_wh(frame)
            self._pause_frame = frame if ok_read else self._pause_frame

        return frame_idx

    # .................................................................................................................

    def set_display_size(self, display_size_px: int | None, interpolation: int | None = None) -> SelfType:
        """Set maximum side-length of video frames"""

        # Check if we need to do display re-sizing
        frame_h, frame_w = self.sample_frame.shape[0:2]
        scaled_h, scaled_w = (frame_h, frame_w)
        if display_size_px is not None:
            scaled_h, scaled_w = get_image_hw_for_max_side_length(self.sample_frame.shape, display_size_px)

        # Store new settings
        self._interpolation = interpolation
        self._need_resize = (scaled_w != frame_w) or (scaled_h != frame_h)
        self._scale_wh = (scaled_w, scaled_h)
        self.shape = (scaled_h, scaled_w, 3)

        return self

    # .................................................................................................................

    def __iter__(self) -> SelfType:
        """Called when using this object in an iterator (e.g. for loops)"""
        self.open()
        return self

    def __next__(self) -> [bool, int, ndarray]:
        """
        Iterator that provides frame data from a video capture object.
        Returns:
            is_paused, frame_index, frame_bgr
        """
        return self.read()

    # .................................................................................................................


class ImageAsVideoReader(PauseableVideoReader):
    """
    Helper used to have a single image act like a video reader.
    This is meant to provide support for use cases where either
    a video or single image could be loaded, by having the
    image treated like a video that repeats a a single frame.

    Example usage:

        vreader = ImageAsVideoReader("path/to/image.jpg")
        for is_paused, frame_idx, frame in vreader:
            # Do something with frames...
            if i_want_to_stop:
                break

    Note that the 'video' will always remain paused, and always on frame 0
    when used in a loop this way.
    """

    # .................................................................................................................

    def __init__(
        self,
        image_path: str | ndarray,
        display_size_px: int | None = None,
        initial_position_0_to_1: float = 0.0,
    ):

        # Make sure we can read the image, and force to be 3-channel (like a typical video)
        is_array_data = isinstance(image_path, ndarray)
        image = cv2.imread(image_path) if not is_array_data else image_path.copy()
        assert image is not None, f"Error loading image: {image_path}"
        if image.ndim < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Store basic video data
        self._image_path = image_path if not is_array_data else None
        self._total_frames = 1
        self._max_frame_idx = self._total_frames - 1
        self._fps = 60
        self._is_paused = False
        self._frame_idx = 0

        # Create a sample frame, to mimic video usage
        self.sample_frame = image

        # Set up display sizing
        self._scaled_frame: ndarray = image
        self._interpolation: int | None = None
        self._need_resize: bool = False
        self._scale_wh: tuple[int, int] = (1, 1)
        self.shape: tuple[int, int, int] = (1, 1, 3)
        self.set_display_size(display_size_px)

    # .................................................................................................................

    def scale_to_display_wh(self, image: ndarray, interpolation=None) -> ndarray:
        """Helper used to scale a given image to the same size as the 'video' frames"""
        return cv2.resize(image, dsize=self._scale_wh, interpolation=interpolation)

    def release(self) -> SelfType:
        """Do nothing"""
        return self

    def toggle_pause(self, new_pause_state: bool | None = None) -> bool:
        """
        Toggle pause state. This has no effect on the image source, but will alter
        the reported value when reading frames. Returns: is_paused
        """
        self._is_paused = (not self._is_paused) if new_pause_state is None else new_pause_state
        return self._is_paused

    def read(self) -> [bool, int, ndarray]:
        """
        Read 'new frame' from video. Though this will always
        be the same repeating image for the image video reader!
        Returns:
            is_paused, frame_index, frame_bgr
        """
        return self._is_paused, self._frame_idx, self._scaled_frame

    def next_frame(self, num_frames=1) -> SelfType:
        return self

    def prev_frame(self, num_frames=1) -> SelfType:
        return self.next_frame(-num_frames)

    def get_sample_frame(self) -> ndarray:
        """Retrieve a copy of the original image (without scaling)"""
        return self.sample_frame.copy()

    def get_pause_state(self) -> bool:
        """Returns: is_paused"""
        return self._is_paused

    # .................................................................................................................

    def get_frame_delay_ms(self, max_allowable_ms=1000) -> int:
        """Returns a frame delay (in milliseconds) according to the hard-coded framerate"""
        frame_delay_ms = 1000.0 / self._fps
        return int(min(max_allowable_ms, frame_delay_ms))

    def get_framerate(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._total_frames

    # .................................................................................................................

    def get_playback_position(self, normalized=True) -> int | float:
        """Always returns 0, for still image 'video'"""
        return 0

    # .................................................................................................................

    def set_display_size(self, display_size_px: int | None, interpolation: int | None = None) -> SelfType:
        """Set maximum side-length of frames"""

        # Check if we need to do display re-sizing
        frame_h, frame_w = self.sample_frame.shape[0:2]
        scaled_h, scaled_w = (frame_h, frame_w)
        if display_size_px is not None:
            scaled_h, scaled_w = get_image_hw_for_max_side_length(self.sample_frame.shape, display_size_px)

        # Store new settings
        self._interpolation = interpolation
        self._need_resize = (scaled_w != frame_w) or (scaled_h != frame_h)
        self._scale_wh = (scaled_w, scaled_h)
        self.shape = (scaled_h, scaled_w, 3)

        # Update internal (pre-scaled) copy of frame
        self._scaled_frame = self.sample_frame.copy()
        if self._need_resize:
            self._scaled_frame = self.scale_to_display_wh(self.sample_frame, self._interpolation)

        return self

    # .................................................................................................................

    def set_playback_position(self, position: int | float, is_normalized=False) -> int:
        """Ignore playback control for still image 'video'"""
        return 0

    # .................................................................................................................

    def __iter__(self) -> SelfType:
        """Called when using this object in an iterator (e.g. for loops)"""
        return self

    # .................................................................................................................

    def __next__(self) -> [bool, int, ndarray]:
        """
        Iterator that provides frame data for loops (as if reading frames from a video)
        Returns:
            is_paused, frame_index, frame_bgr
        """

        return self.read()

    # .................................................................................................................


class ReversibleLoopingVideoReader(LoopingVideoReader):
    """
    Simple variant on the basic looping reader.
    This version supports reading frames in reverse, though
    the implementation is extremely inefficient!

    To use, simply call the '.toggle_reverse_state(True)' function.
    (This can be done inside of a frame reading loop)
    """

    # .................................................................................................................

    def __init__(self, video_path: str, display_size_px: int | None = None, initial_position_0_to_1: float = 0.0):

        # Inherit from parent
        super().__init__(video_path, display_size_px, initial_position_0_to_1)

        # Flag used to keep track of playback direction
        self._is_reversed = False

    # .................................................................................................................

    def toggle_reverse_state(self, set_is_reversed: bool | None = None) -> bool:
        """
        Used to switch from forward-to-reverse frame reading
        If the given target state is None, then the current
        state will be toggled, otherwise it wil be set to
        the given state (True to reverse, False for forward reading).

        Returns: updated_reversal_state
        """

        self._is_reversed = (not self._is_reversed) if set_is_reversed is None else set_is_reversed

        return self._is_reversed

    # .................................................................................................................

    def read(self) -> [bool, int, ndarray]:
        """
        Read the next available frame
        Returns:
            is_paused, frame_index, frame_bgr
        """

        # Don't read video frames while paused
        if self._is_paused:
            return self._is_paused, self._frame_idx, self._pause_frame.copy()

        if self._is_reversed:

            # Repeatedly 'rewind' the video backwards and read frames, looping to end if needed
            self._frame_idx = (self._frame_idx - 1) % self._max_frame_idx
            self._vcap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
            read_ok, frame = self._vcap.read()
            while not read_ok:
                self._vcap.set(cv2.CAP_PROP_POS_FRAMES, self._max_frame_idx)
                read_ok, frame = self._vcap.read()
                if not read_ok:
                    self._max_frame_idx -= 1
                    print("Error reading last frame. Will try with reduced indexing:", self._max_frame_idx)
                self._frame_idx = self._max_frame_idx

        else:

            # Read next frame, or loop back to beginning if there are no more frames
            self._frame_idx += 1
            read_ok, frame = self._vcap.read()
            if not read_ok:
                self._frame_idx = 0
                self._vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_ok, frame = self._vcap.read()
                assert read_ok, "Error looping video! Unable to read first frame"

        # Scale frame for display & store in case we pause
        if self._need_resize:
            frame = self.scale_to_display_wh(frame)
        self._pause_frame = frame

        return self._is_paused, self._frame_idx, self._pause_frame.copy()

    # .................................................................................................................

    def get_reverse_state(self) -> bool:
        """Used to check the current forward/reverse state"""
        return self._is_reversed

    # .................................................................................................................

    def __next__(self) -> tuple[bool, int, ndarray]:
        """
        Iterator that provides frame data from a video capture object.
        Returns:
            is_paused, frame_index, frame_bgr
        """
        return self.read()

    # .................................................................................................................


class VideoPlaybackSlider(BaseCallback):
    """
    Implements a 'playback slider' UI element that is specific to working with videos.
    After initializing with a reference to the video reader whose playback is to be controlled,
    the playback slider only needs a single call (inside of the video loop) to work:
        slider.update_state(is_paused, frame_index)

    This will update the slider position indicator (according to the given frame index) as well
    as internally keep track of changes to the slider (i.e. user adjustments).

    Adjustments to the slider can be detected using:
        is_adjusting_changed, is_adjusting = slider.read()
    This can be used to avoid running heavy processing code if the user is adjusting playback.
    For example:

        is_adjusting_changed, is_adjusting = slider.read()
        if not is_adjusting:
            # user isn't modifying slider
        if is_adjusting_changed and is_adjusting:
            # user just began modifying the slider
        if is_adjusting_changed and not is_adjusting:
            # user just finished modifying the slider
    """

    # .................................................................................................................

    def __init__(
        self,
        video_reader: PauseableVideoReader,
        color: COLORU8 | int = (0, 0, 0),
        indicator_line_width: int = 1,
        include_button: bool = True,
        pause_on_right_click: bool = True,
        height: int = 50,
        minimum_width: int = 350,
        is_flexible_w: bool = True,
    ):
        # Store reference to video capture
        self._vreader = video_reader
        self._total_frames = video_reader.get_frame_count()
        self._pause_on_right_click = pause_on_right_click
        self._is_paused = video_reader.get_pause_state()

        # Storage for slider value
        self._max_frame_idx = max(int(self._total_frames) - 1, 1)
        self._slider_idx = video_reader.get_playback_position(normalized=False)

        # Storage for slider state
        self._include_button = include_button
        self._is_pressed = False
        self._is_pressed_changed = False
        self._need_playback_position_change = False
        self._pause_state_to_restore = None

        # Store for cached drawing data
        self._cached_pause_img = blank_image(1, 1)
        self._cached_play_img = blank_image(1, 1)
        self._btn_hover_x_px = 0
        self._btn_w_norm = 0
        self._bar_w_norm = 1

        # Set up element style
        color = interpret_coloru8(color)
        ind_color = pick_contrasting_gray_color(color)
        color_fg = lerp_colors(color, ind_color, 0.25)
        self.style = UIStyle(
            color_bg=color,
            color_fg=color_fg,
            indicator_width=indicator_line_width,
            indicator_color=ind_color,
            outline_color=(0, 0, 0),
            button=None,
        )

        # Add styling for button if included
        if include_button:
            btn_style = UIStyle(
                color=(60, 60, 225),
                color_symbol=(255, 255, 255),
                hide=False,
                swap_icons=False,
            )
            self.style.update(button=btn_style)

        # Inherit from parent & set default helper name for debugging
        min_w = max(height, minimum_width)
        super().__init__(height, min_w, is_flexible_w=is_flexible_w)

    # .................................................................................................................

    def update_state(self, is_paused: bool, frame_index: int) -> SelfType:
        """
        Function used to update playback indicator position.
        Expects to get a frame index from the associated video reader.

        If the user interacts with the playback bar, this function
        call is what adjusts the position of the video reader.
        """

        if self._need_playback_position_change:
            self._need_playback_position_change = False
            self._vreader.set_playback_position(self._slider_idx)
        else:
            self._is_paused = is_paused
            self._slider_idx = frame_index

        return self

    # .................................................................................................................

    def read(self) -> tuple[bool, bool]:
        """
        Reads 'adjusting state' of the playback slider
        Returns:
            is_adjusting_changed, is_adjusting

        -> is_adjusting is True while the user is actively modifying the playback slider
        -> is_adjusting_changed is True whenever the is_adjusting state changes, otherwise False.
           For example, it will be True when the user begins interacting with the slider,
           and again when the stop interacting, otherwise it will be False.
        """
        is_adjusting_changed = self._is_pressed_changed
        self._is_pressed_changed = False
        return is_adjusting_changed, self._is_pressed

    # .................................................................................................................

    def _on_left_down(self, cbxy, cbflags) -> None:

        # Ignore clicks outside of the slider
        if not cbxy.is_in_region:
            return

        # Handle play/pause button press
        if cbxy.xy_px.x < self._btn_hover_x_px:
            self._is_paused = self._vreader.toggle_pause()
            return

        # Prevent video playback while adjusting slider
        self._pause_state_to_restore = self._vreader.get_pause_state()
        self._is_paused = self._vreader.toggle_pause(True)
        self._is_pressed = True
        self._is_pressed_changed = True

        # Update slider state as if dragging
        self._on_drag(cbxy, cbflags)

        return

    def _on_drag(self, cbxy, cbflags) -> None:

        if not self._is_pressed:
            return

        # Update slider value while dragging
        new_slider_idx = self._mouse_x_norm_to_slider_idx(cbxy.xy_norm[0])
        idx_is_changed = new_slider_idx != self._slider_idx
        if idx_is_changed:
            self._need_playback_position_change = True
            self._slider_idx = new_slider_idx

        return

    def _on_left_up(self, cbxy, cbflags) -> None:

        # Don't react to mouse up if we weren't being interacted with
        if not self._is_pressed:
            return
        self._is_pressed = False
        self._is_pressed_changed = True

        # Restore pause state (prior to modifying slider)
        # -> If the video was paused, this will keep it paused, otherwise unpause it
        self._is_paused = self._vreader.toggle_pause(self._pause_state_to_restore)
        self._pause_state_to_restore = None

        return

    def _on_right_click(self, cbxy, cbflags) -> None:
        if self._pause_on_right_click:
            self._is_paused = self._vreader.toggle_pause()
        return

    # .................................................................................................................

    def _mouse_x_norm_to_slider_idx(self, x_norm: float) -> float | int:
        """Helper used to convert normalized mouse position into slider values"""
        adj_x_norm = (x_norm - self._btn_w_norm) / self._bar_w_norm
        frame_idx = round(adj_x_norm * self._max_frame_idx)
        return max(0, min(self._max_frame_idx, frame_idx))

    # .................................................................................................................

    def _render_up_to_size(self, h: int, w: int) -> ndarray:

        # Re-render label & marker lines onto new blank background if render size changes
        base_h, base_w = self._cached_play_img.shape[0:2]
        if base_h != h or base_w != w:

            # Draw slider component
            btn_side_length = min(h, w) if self._include_button else 0
            bar_w = w - btn_side_length
            bar_hw = (h, bar_w)
            dot_tiles = make_dot_tiles(7, radius_norm=0.5, color_fg=self.style.color_fg, color_bg=self.style.color_bg)
            new_slider_img = draw_truchet(bar_hw, dot_tiles, fit_by_crop=True)

            new_play_img = new_slider_img
            new_pause_img = new_slider_img

            if self._include_button:
                # Draw button components
                btn_side_length = min(h, w)
                tri_img, dblbar_img = draw_play_pause_icons(
                    self.style.button.color,
                    self.style.button.color_symbol,
                    side_length_px=btn_side_length,
                )

                # Combine button & slider into a single image
                new_play_img = np.hstack((dblbar_img, new_slider_img))
                new_pause_img = np.hstack((tri_img, new_slider_img))
                if self.style.button.swap_icons:
                    new_play_img, new_pause_img = new_pause_img, new_play_img

            # Cache data for re-use
            self._cached_play_img = draw_box_outline(new_play_img)
            self._cached_pause_img = draw_box_outline(new_pause_img)
            self._bar_w_norm = bar_w / w if self._include_button else 1
            self._btn_w_norm = btn_side_length / w if self._include_button else 0
            self._btn_hover_x_px = btn_side_length if self._include_button else -1

        # Decide which image state to show
        # -> Want to show play/pause based on whether video is paused
        # -> However, if user is adjusting slider (which pauses the video), we want
        #    to show the image based on state prior to adjustment
        show_pause_state = self._pause_state_to_restore if self._is_pressed else self._is_paused
        img = self._cached_pause_img.copy() if show_pause_state else self._cached_play_img.copy()

        # Draw indicator line (don't cache, because we expect it to change frequently)
        # img = self._cached_pause_img.copy() if self._is_paused else self._cached_play_img.copy()
        slider_norm = (self._slider_idx / self._max_frame_idx) * self._bar_w_norm + self._btn_w_norm
        line_x_px = round(slider_norm * (w - 1))
        if self.style.indicator_width == 1:
            cv2.line(img, (line_x_px, 1), (line_x_px, h - 2), self.style.indicator_color, self.style.indicator_width)
        else:
            # For thicker lines, draw as a filled rectangle, since ocv lines get rounded otherwise
            dx1 = self.style.indicator_width // 2
            dx2 = self.style.indicator_width - dx1
            pt1 = (max(self._btn_hover_x_px, line_x_px - dx1), 1)
            pt2 = (min(w - 1, line_x_px + dx2), h - 2)
            cv2.rectangle(img, pt1, pt2, self.style.indicator_color, -1)

        # Draw button play/pause hover indicator
        if self.is_hovered():
            is_hovering_btn = self.get_event_xy().xy_px.x < self._btn_hover_x_px
            if is_hovering_btn:
                pt1, pt2 = (0, 0), (self._btn_hover_x_px - 1, h - 1)
                cv2.rectangle(img, pt1, pt2, (255, 255, 255), 1)

        return img

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def create_VideoCapture(video_path):
    """
    Helper used to prevent bug introduced in newer versions of opencv,
    which disabled orientation correction on v4.11 for some reason, see:
        https://github.com/opencv/opencv/issues/26795
    """

    vcap = cv2.VideoCapture(video_path)
    vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    return vcap


def check_is_image(file_path: str | ndarray) -> bool:
    """Helper used to decide if a path is pointing to an image file or not"""

    # Yes if given a direct image (e.g. array data)
    if isinstance(file_path, ndarray):
        return True

    # Bail on non-file paths
    if not osp.isfile(file_path):
        return False

    # Check if we recognize (common) supported image extensions
    _, file_ext = osp.splitext(file_path)
    known_file_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    if file_ext.lower() in known_file_exts:
        return True

    # Last resort, try to load the image to see if it's valid
    try:
        loaded_image = cv2.imread(file_path)
        is_image = loaded_image is not None
    except cv2.error:
        is_image = False

    return is_image


def load_looping_video_or_image(
    video_or_image_path: str | ndarray, display_size_px: int | None = None
) -> tuple[bool, LoopingVideoReader | ImageAsVideoReader]:
    """
    Helper used to load either a 'image as video' or looping video reader.
    This can be used to support having a video input while also supporting
    static images (as if they are repeating videos with a single frame).
    Returns:
        is_static_image, video_reader

    """

    is_static_image = check_is_image(video_or_image_path)
    ReaderClass = ImageAsVideoReader if is_static_image else LoopingVideoReader
    reader = ReaderClass(video_or_image_path, display_size_px)

    return is_static_image, reader


def read_webcam_string(input_source: str | int | None):
    """
    Helper function used to interpret input sources that indicate a webcam
    Returns:
        is_webcam, video_source

    - If given None, returns: (False, None)
    - If given an integer, returns: (True, input_source)
    - If given a path to a file, returns: (False, input_source)
    - Supports input strings like: 'cam0' or 'webcam2' or even just 'cam'
        -> If given 'cam1', returns: (True, 1)
        -> If given 'webcam0', returns: (True, 0)
        -> If given 'cam', returns: (True, 0)
    """

    # Handle blank case
    is_webcam, output_source = False, input_source
    if input_source is None:
        return is_webcam, output_source

    # Handle integer case
    is_webcam = isinstance(input_source, int)
    if is_webcam:
        return is_webcam, output_source

    # Handle string case
    assert isinstance(input_source, str), "Expecting input source of type: string, integer or None!"
    is_valid_path = osp.exists(input_source)
    if not is_valid_path:
        lower_input = input_source.lower().strip()
        int_str = lower_input.removeprefix("webcam").removeprefix("cam")
        is_webcam = int_str.isdigit() or (len(int_str) == 0)
        if is_webcam:
            output_source = int(int_str if len(int_str) > 0 else 0)

    return is_webcam, output_source
