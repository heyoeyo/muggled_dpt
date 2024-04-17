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
        
        # Allocate variables for use of callbacks
        self._cbs = CallbackSequencer()
        self._using_cb = False
        
    def move(self, x, y):
        cv2.moveWindow(self.title, x, y)
        return self
    
    def add_trackbar(self, trackbar_name, max_value, initial_value = 0):
        return WindowTrackbar(self.title, trackbar_name, max_value, initial_value)
    
    def set_callbacks(self, *callbacks):
        
        # Record all the given callbacks
        self._cbs.add(*callbacks)
        
        # Attach callbacks to window for the first time, if needed
        if not self._using_cb:
            cv2.setMouseCallback(self.title, self._cbs)
            self._using_cb = True
        
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
        self._max_value = max_value
    
    def read(self):
        return cv2.getTrackbarPos(self.name, self._window_name)

    def write(self, new_value):
        safe_value = max(0, min(new_value, self._max_value))
        return cv2.setTrackbarPos(self.name, self._window_name, safe_value)

class CallbackSequencer:
    
    '''
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
    '''
    
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


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def add_bounding_box(image_bgr, color=(0, 0, 0), thickness=1, inset_box = True):
    
    ''' Draws a rectangular outline around the given image '''

    h, w = image_bgr.shape[:2]
    bot_right = (w-1, h-1) if inset_box else (w, h)
    return cv2.rectangle(image_bgr, (0, 0), bot_right, color, thickness)

# .....................................................................................................................

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

# .....................................................................................................................

def grid_stack_by_columns_first(image_list, num_columns):
    
    '''
    Helper used to combine a list of images into a single (grid) image
    Works by combining images horizontally first to create row-images,
    then stacks these vertically to build final grid image.
    If the last row is not as long as the previous, then it will be padded with black!
    '''
    
    # Stack together each row
    num_imgs = len(image_list)
    img_stacks = [np.hstack(image_list[k:k+num_columns]) for k in range(0, num_imgs, num_columns)]
    
    # Pad final entry, if it doesn't match first entry size
    # (this is only needed if the image list length isn't even divisble by the number of columns!)
    first_h, first_w = img_stacks[0].shape[0:2]
    last_h, last_w = img_stacks[-1].shape[0:2]
    diff_h, diff_w = (first_h - last_h, first_w - last_w)
    mismatched_shape = (diff_h > 0) or (diff_w > 0)
    if mismatched_shape:
        img_stacks[-1] = cv2.copyMakeBorder(img_stacks[-1], 0, diff_h, 0, diff_w, borderType = cv2.BORDER_CONSTANT)
    
    return np.vstack(img_stacks)
