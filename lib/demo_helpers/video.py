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
        self.vcap = cv2.VideoCapture(self._video_path)
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
    
    def get_playback_position(self):
        ''' Returns playback position as a number between 0 and 1 '''
        return self.vcap.get(cv2.CAP_PROP_POS_FRAMES) / self.total_frames
    
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
        if not self.vcap.isOpened(): self.vcap = cv2.VideoCapture(self._video_path)
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
    '''
    
    # .................................................................................................................
    
    def __init__(self, vreader, bar_height = 40):
        
        # Store vreader so we can access it later to change playback positioning
        self._vreader = vreader
        
        # Storage for mouse state
        self.x_px = (0,0)
        self._is_pressed = False
        
        # Storage for indicator bar image
        self._img = None
        self._disp_w = None
        self._disp_h = bar_height
    
    # .................................................................................................................
    
    def __call__(self, event, x, y, flags, param) -> None:
        
        # Keep track of mouse positioning and click state
        self.x_px = x
        if event == cv2.EVENT_LBUTTONUP: self._is_pressed = False
        if event == cv2.EVENT_LBUTTONDOWN: self._is_pressed = True
        
        return
    
    # .................................................................................................................
    
    def change_playback_position_on_mouse_press(self):
        
        # Bail if mouse isn't pressed
        if not self._is_pressed: return
        
        # Only try to update positioning if we have a bar width (otherwise can't normalize mouse x coord)
        if self._disp_w is None: return
        playback_pos = self.x_px / (self._disp_w - 1)
        self._vreader.set_playback_position(playback_pos)
        
        return
    
    # .................................................................................................................
    
    def add_playback_indicator(self, frame):
        
        # Create base bar image, if we don't already have one matching the given frame size
        frame_w = frame.shape[1]
        got_new_width = (frame_w != self._disp_w)
        if got_new_width:
            bar_img = np.full((self._disp_h, frame_w, 3), 40, dtype=np.uint8)
            cv2.rectangle(bar_img, (-5,0), (frame_w + 5, self._disp_h - 1), (0,0,0), 1)
            self._img = bar_img
            self._disp_w = frame_w
        
        # Figure out the playback position as a pixel position to draw
        x_norm = self._vreader.get_playback_position()
        x_px = int(round(x_norm * (frame_w - 1)))
        
        # Draw indicator line onto a indicator bar
        bar_img = self._img.copy()
        cv2.line(bar_img, (x_px, 2), (x_px, self._disp_h-2), (255,255,255), 1)
        return np.vstack((frame, bar_img))
    
    # .................................................................................................................

# ---------------------------------------------------------------------------------------------------------------------
#%% Functions


