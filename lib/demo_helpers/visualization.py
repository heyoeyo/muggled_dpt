#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DisplayWindow:
    
    ''' Class used to manage opencv window, mostly to make trackbars easier to organize '''
    
    WINDOW_CLOSE_KEYS_SET = {ord("q"), 27} # q, esc
    
    def __init__(self, window_title):
        self.title = window_title
        cv2.namedWindow(self.title, flags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        
    def move(self, x, y):
        cv2.moveWindow(self.title, x, y)
        return self
    
    def add_trackbar(self, trackbar_name, max_value, initial_value = 0):
        return WindowTrackbar(self.title, trackbar_name, max_value, initial_value)
    
    def set_callback(self, callback):
        cv2.setMouseCallback(self.title, callback)
        return self
    
    def imshow(self, image):
        return cv2.imshow(self.title, image)
    
    def waitKey(self, frame_delay_ms = 1):
        
        '''
        Wrapper around opencv waitkey (triggers draw to screen)
        Returns:
            request_close, keypress
        '''
        
        keypress = cv2.waitKey(int(frame_delay_ms)) & 0xFF
        request_close = keypress in self.WINDOW_CLOSE_KEYS_SET
        return request_close, keypress
    
    def close(self):
        return cv2.destroyWindow(self.title)


class WindowTrackbar:
    
    ''' Class used to keep track of strings that opencv uses to reference trackbars on windows '''
    
    def __init__(self, window_name, trackbar_name, max_value, initial_value = 0):
        
        self.name = trackbar_name
        self._window_name = window_name
        self._prev_value = int(initial_value)
        cv2.createTrackbar(trackbar_name, window_name, int(initial_value), int(max_value), lambda x: None)
    
    def read(self):
        return cv2.getTrackbarPos(self.name, self._window_name)


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def draw_corner_text(display_frame, text, row_idx = 0):
    
    ''' Draw text into the top-left corner of an image '''
    
    # For clarity, hard-coded config to avoid complex text sizing/spacing checks
    xy_px = (5, 22 + (row_idx * 32))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    bg_color = (0,0,0)
    fg_color = (255,255,255)
    fg_thickness = 2
    bg_thickness = fg_thickness + 3
    linetype = cv2.LINE_AA
    
    # Draw text with background for better contrast
    cv2.putText(display_frame, text, xy_px, font, scale, bg_color, bg_thickness, linetype)
    cv2.putText(display_frame, text, xy_px, font, scale, fg_color, fg_thickness, linetype)
    
    return display_frame


def histogram_equalization(depth_uint8, min_pct = 0.0, max_pct = 1.0):
    
    '''
    Function used to perform histogram equalization on a depth image.
    This function uses the built-in opencv function: cv2.equalizeHist(...)
    When the min/max thresholds are not set (since it works faster),
    however this implementation also supports truncating the low/high
    end of the input.
    This means that equalization can be performed over a subset of
    the input value range, which makes better use of the value range
    when using thresholded inputs.
    
    Returns:
        depth_uint8_equalized
    '''
    
    # Make sure min/max are properly ordered & separated
    min_value, max_value = [int(round(255*value)) for value in sorted((min_pct, max_pct))]
    max_value = max(max_value, min_value + 1)
    if min_value == 0 and max_value == 255:
        return cv2.equalizeHist(depth_uint8)
    
    # Compute histogram of input
    num_bins = 1 + max_value - min_value
    bin_counts, _ = np.histogram(depth_uint8, num_bins, range = (min_value, max_value))
    
    # Compute cdf of histogram counts
    cdf = bin_counts.cumsum()
    cdf_min, cdf_max = cdf.min(), cdf.max()
    cdf_norm = (cdf - cdf_min) / float(max(cdf_max - cdf_min, 1))
    cdf_uint8 = np.uint8(255 * cdf_norm)
    
    # Extend cdf to match 256 lut sizing, in case we skipped min/max value ranges
    low_end = np.zeros(min_value, dtype=np.uint8)
    high_end = np.full(255 - max_value, 255, dtype=np.uint8)
    equalization_lut = np.concatenate((low_end, cdf_uint8, high_end))
    
    # Apply LUT mapping to input
    return equalization_lut[depth_uint8]

