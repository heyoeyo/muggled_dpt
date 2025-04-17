#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class LoopingVideoReader:
    
    '''
    Helper used to provide looping frames from video, along with helpers
    to control playback & frame sizing
    Example usage:
        
        vread = LoopingVideoReader("path/to/video.mp4")
        for frame in vread:
            # Do something with frames...
            if i_want_to_stop: break
    
    '''
    
    # .................................................................................................................
    
    def __init__(self, video_path, display_size_px = 800):
        
        self._video_path = video_path
        self.vcap = create_video_capture(self._video_path)
        self.total_frames = self.vcap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Read sample frame & reset video
        rec_frame, frame = self.vcap.read()
        if not rec_frame: raise IOError("Can't read frames from video!")
        self.vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Set up display sizing
        vid_h, vid_w = frame.shape[0:2]
        max_side = max(vid_w, vid_h)
        scale_factor = display_size_px / max_side
        self.disp_wh = list(int(round(size_px * scale_factor)) for size_px in [vid_w, vid_h])
        self.shape = (self.disp_wh[1], self.disp_wh[0], 3)
        
        # Allocate storage for 'previous frame', which is re-used when paused & 
        self._prev_frame = self.scale_to_display_wh(frame)
    
    # .................................................................................................................
    
    def scale_to_display_wh(self, image): return cv2.resize(image, dsize = self.disp_wh)
    def release(self): self.vcap.release()
    
    # .................................................................................................................
    
    def get_frame_delay_ms(self, max_allowable_ms = 1000):
        fps = self.vcap.get(cv2.CAP_PROP_FPS)
        frame_delay_ms = 1000.0 / fps
        return int(min(max_allowable_ms, frame_delay_ms))
    
    # .................................................................................................................
    
    def get_playback_position(self, normalized=True):
        ''' Returns playback position either as a frame index or a number between 0 and 1 (if normalized) '''
        if normalized:
            return self.vcap.get(cv2.CAP_PROP_POS_FRAMES) / self.total_frames
        return int(self.vcap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # .................................................................................................................
    
    def set_playback_position(self, position_0_to_1):
        ''' Set position of video playback. Returns frame index '''
        position_0_to_1 = max(position_0_to_1, 0)
        frame_idx = int(round(position_0_to_1 * (self.total_frames - 1)))
        self.vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        return frame_idx
    
    # .................................................................................................................
    
    def __iter__(self):
        ''' Called when using this object in an iterator (e.g. for loops) '''
        if not self.vcap.isOpened(): self.vcap = create_video_capture(self._video_path)
        return self
    
    # .................................................................................................................

    def __next__(self):

        ''' Iterator that provides frame data from a video capture object. Returns frame_bgr '''
        
        # Read next frame, or loop back to beginning if there are no more frames
        read_ok, frame = self.vcap.read()
        if not read_ok:
            self.vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            read_ok, frame = self.vcap.read()
        
        # Scale frame for display & store in case we're paused
        frame = self.scale_to_display_wh(frame)
        self._prev_frame = frame
        
        return frame
        
    # .................................................................................................................


class PlaybackIndicatorCB:
    
    '''
    Helper used to provide a simple playback 'timeline' control to an
    opencv window, for use when playing videos. Works as a window callback
    (within opencv callback interface) while also providing functions
    to draw a playback indicator onto display frames.
    Can be disabled (i.e. when using none rewindable source),
    which also disables UI input & drawing functionality
    '''
    
    # .................................................................................................................
    
    def __init__(self, vreader, bar_height = 60, enabled = True):
        
        # Store vreader so we can access it later to change playback positioning
        self._vreader = vreader
        self._enabled = enabled
        
        # Storage for mouse state
        self.mouse_x_px = 0
        self._is_pressed = False
        self._frame_h = 1
        self._interact_y1y2 = (-20, -10)
        
        # Storage for indicator bar image
        self._bar_h = bar_height
        self._img = None
        self._frame_w = 1
    
    # .................................................................................................................
    
    def __call__(self, event, x, y, flags, param) -> None:
        
        # Bail when disabled
        if not self._enabled:
            return
        
        # Keep track of mouse positioning and click state
        self.mouse_x_px = x
        
        # Always release mouse press, regardless of where the mouse is located
        if event == cv2.EVENT_LBUTTONUP:
            self._is_pressed = False
        
        # Only respond when mouse is over top of slider
        if event == cv2.EVENT_LBUTTONDOWN:
            y1, y2 = self._interact_y1y2
            is_interacting = y1 < y < y2
            self._is_pressed = is_interacting
        
        return
    
    # .................................................................................................................
    
    def change_playback_position_on_mouse_press(self) -> None:
        
        # Bail if when not interacting
        if not (self._is_pressed and self._enabled):
            return
        
        playback_pos = self.mouse_x_px / (self._frame_w - 1)
        self._vreader.set_playback_position(playback_pos)
        
        return
    
    # .................................................................................................................
    
    def append_to_frame(self, frame) -> np.ndarray:
        
        # Don't draw anything when disabled
        if not self._enabled:
            return frame
        
        # Create base bar image, if we don't already have one matching the given frame size
        frame_h, frame_w = frame.shape[0:2]
        got_new_width = (frame_w != self._frame_w)
        if got_new_width:
            bar_img = np.full((self._bar_h, frame_w, 3), 40, dtype=np.uint8)
            cv2.rectangle(bar_img, (-5,0), (frame_w + 5, self._bar_h - 1), (0,0,0), 1)
            self._img = bar_img
            self._frame_w = frame_w
        
        # Update height related info
        got_new_height = (frame_h != self._frame_h)
        if got_new_height:
            self._frame_h = frame_h
            self._interact_y1y2 = (frame_h, frame_h + self._bar_h)
        
        # Figure out the playback position as a pixel position to draw
        x_norm = self._vreader.get_playback_position()
        x_px = int(round(x_norm * (frame_w - 1)))
        
        # Draw indicator line onto a indicator bar
        bar_img = self._img.copy()
        cv2.line(bar_img, (x_px, 2), (x_px, self._bar_h-2), (255,255,255), 1)
        return np.vstack((frame, bar_img))
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def create_video_capture(video_path) -> cv2.VideoCapture:
    ''' Helper used to set up reading from videos, with fix for OpenCV orientation bug '''
    vcap = cv2.VideoCapture(video_path)
    assert vcap.isOpened(), f"Unable to open video: {video_path}"
    vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
    return vcap
