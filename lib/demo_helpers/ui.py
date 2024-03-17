#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class SliderCB:
    
    '''
    Class used to provide a simple slider control to an
    opencv window, as an alternative to using trackbars
    
    Works as a window callback (within opencv callback interface)
    while also providing functions to draw onto display frames
    
    Example usage:
        # Attach slider callback to cv2 window
        slider = SliderCB("S-Value")
        cv2.setMouseCallback(winname, slider)
        
        # Read value from slider
        s_value = slider.read()
        
        # Draw slider
        disp_frame = slider.append_to_frame(disp_frame)
        cv2.imshow(winname, disp_frame)
        cv2.waitKey(1)
    '''
    
    # .................................................................................................................
    
    def __init__(self, label: str, start_value = 0.5, min_value = 0.0, max_value = 1.0, step_size = 0.01,
                 marker_step_size = None, bar_height = 40, enable_value_display = True):
        
        # Storage for slider value
        min_value, max_value = sorted((min_value, max_value))
        self._slider_min = min_value
        self._slider_max = max_value
        self._slider_value = min(max_value, max(min_value, start_value))
        self._slider_step = step_size
        self._slider_label = label
        self._slider_delta = self._slider_max - self._slider_min
        
        # Display config
        self._enable_value_display = enable_value_display
        self._max_digits = max(1 + len(str(val)) for val in (min_value, max_value, step_size))
        self._initial_value = self._slider_value
        self._marker_step_size = marker_step_size
        
        # Storage for mouse state
        self._mouse_x_px = 0
        self._mouse_is_pressed = False
        self._interact_y1y2 = (-20, -10)
        
        # Storage for indicator bar image
        self._bar_h = bar_height
        self._img = None
        self._frame_w = 0
        self._frame_h = 0
        
        # Hard-coded text config
        self._txt_size_config = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.5, "thickness": 1}
        self._txt_label_config = {"color": (150,150,150), "lineType": cv2.LINE_AA, **self._txt_size_config}
        self._txt_value_config = {"color": (255,255,255), "lineType": cv2.LINE_AA, **self._txt_size_config}
    
    # .................................................................................................................
    
    def __call__(self, event, x, y, flags, param) -> None:
        
        # Keep track of mouse positioning and click state
        self._mouse_x_px = x
        
        # Always release on mouse up (i.e. allow release outside of input bounds)
        if event == cv2.EVENT_LBUTTONUP:
            self._mouse_is_pressed = False
        
        # Only respond when mouse is over top of slider
        y1, y2 = self._interact_y1y2
        is_interacting = y1 < y < y2
        if is_interacting:
            
            if event == cv2.EVENT_LBUTTONDOWN:
                self._mouse_is_pressed = True
            
            # Reset state on right click
            if event == cv2.EVENT_RBUTTONDOWN:
                self._slider_value = self._initial_value
                self._mouse_is_pressed = False
        
        # Update slider value
        self._mouse_x_to_slider_value(x)
        
        return
    
    # .................................................................................................................
    
    def read(self, normalize = False) -> float:
        if normalize:
            return (self._slider_value - self._slider_min) / self._slider_delta
        return self._slider_value
    
    # .................................................................................................................
    
    def set(self, slider_value, update_initial_value = True) -> None:
        if update_initial_value:
            self._initial_value = slider_value
        self._slider_value = slider_value
        return
    
    # .................................................................................................................
    
    def append_to_frame(self, frame) -> np.ndarray:
        
        ''' Draws a slider bar onto the bottom of the given frame '''
        
        # Figure out the slider position as a pixel position to draw
        frame_h, frame_w = frame.shape[0:2]
        x_norm = self.read(normalize=True)
        x_px = int(round(x_norm * (frame_w - 1)))
        
        # Draw text indicator, if needed
        bar_img = self._update_base_image(frame_h, frame_w)
        if self._enable_value_display:
            value_str = str(self._slider_value)[:self._max_digits]
            (txt_w, _), _ = cv2.getTextSize(value_str, **self._txt_size_config)
            x1, x2 = x_px + 5, x_px - txt_w - 5
            txt_pos = (x1, 24) if (x1 + txt_w) < frame_w else (x2, 24)
            cv2.putText(bar_img, value_str, txt_pos, **self._txt_value_config)
        
        # Draw indicator line onto the bar
        cv2.line(bar_img, (x_px, 2), (x_px, self._bar_h-2), (255,255,255), 1)
        return np.vstack((frame, bar_img))
    
    # .................................................................................................................

    def _update_base_image(self, frame_h, frame_w) -> np.ndarray:
        
        ''' Helper used to set up base image sizing & interaction region '''
        
        # Re-draw base image if width changes
        got_new_width = (frame_w != self._frame_w)
        if got_new_width:

            # Re-draw new 'blank' bar image
            bar_img = np.full((self._bar_h, frame_w, 3), 40, dtype=np.uint8)
            
            # Add markers to bar
            if self._marker_step_size is not None:
                mark_min = round(self._slider_min / self._marker_step_size)
                mark_max = round(self._slider_max / self._marker_step_size)
                num_marks = round((mark_max - mark_min) / self._marker_step_size)
                markers = [mark_min + k*self._marker_step_size for k in range(num_marks)]
                for marker_value in markers:
                    x_px = round((frame_w - 1) * (marker_value - self._slider_min) / self._slider_delta)
                    cv2.line(bar_img, (x_px, 3), (x_px, self._bar_h - 3), [60]*3, 1)
            
            # Draw label + separator line
            cv2.putText(bar_img, self._slider_label, (5, 24), **self._txt_label_config)
            cv2.line(bar_img, (-5,0), (frame_w + 5, 0), (0,0,0), 1)
            
            # Store new base image + width info
            self._img = bar_img
            self._frame_w = frame_w
        
        # Update height related info
        got_new_height = (frame_h != self._frame_h)
        if got_new_height:
            self._frame_h = frame_h
            self._interact_y1y2 = (frame_h, frame_h + self._bar_h)
        
        return self._img.copy()
    
    # .................................................................................................................
    
    def _mouse_x_to_slider_value(self, x_px: int) -> None:
        
        ''' Helper used to convert mouse position into slider values '''
        
        # Bail if the mouse isn't held down
        if not self._mouse_is_pressed: return
        
        # Map normalized x position to slider range, snapped to step increments
        x_norm = self._mouse_x_px / (self._frame_w - 1)
        slider_x = self._slider_min + x_norm * (self._slider_max - self._slider_min)
        slider_x = round(slider_x / self._slider_step) * self._slider_step
        
        # Finally, make sure the slider value doesn't go out of range
        self._slider_value = max(self._slider_min, min(self._slider_max, slider_x))
        
        return
    
    # .................................................................................................................
    
    @staticmethod
    def append_many_to_frame(frame, *sliders) -> np.ndarray:
        for s in sliders:
            frame = s.append_to_frame(frame)
        return frame
    
    # .................................................................................................................


class ColormapButtonsCB:
    
    '''
    Class used to provide a colormap selector UI that can
    be drawn in an opencv window.
    
    Works as a callback and provides function for drawing
    the UI element onto an existing image
    
    Example usage:
        # Attach callback to cv2 window (so buttons respond to mouse clicks)
        cmap_btns = ColormapButtonsCB([cv2.COLORMAP_AUTUMN, cv2.COLORMAP_INFERNO])
        cv2.setMouseCallback(winname, cmap_btns)
        
        # Apply selected colormap to a 1ch image
        colormapped_image = cmap_btns.apply_colormap(uint8_image_1ch)
        
        # Draw colormap buttons UI onto existing frame
        disp_frame = cmap_btns.append_to_frame(colormapped_image)
        cv2.imshow(winname, disp_frame)
        cv2.waitKey(1)
    '''
    
    # .................................................................................................................
    
    def __init__(self, *cv2_colormap_codes, bar_height = 40):
        
        # Set up left/right boundaries for selecting cmaps
        cmaps = [*cv2_colormap_codes, None]
        num_cmaps = len(cmaps)
        xnorm_bounds = [idx/(num_cmaps) for idx in range(num_cmaps + 1)]
        self._cmap_x1x2_norm_list = list(zip(cmaps, xnorm_bounds[:-1], xnorm_bounds[1:]))
        
        # Storage for indicator bar image
        self._bar_h = bar_height
        self._img = None
        self._frame_w = 1
        self._frame_h = 1
        self._interact_y1y2 = (-20, -10)
        
        # Set default colormap selection
        self._cmap_select = cmaps[0]
    
    # .................................................................................................................
    
    def __call__(self, event, x, y, flags, param) -> None:
        
        y1, y2 = self._interact_y1y2
        is_interacting = y1 < y < y2
        if is_interacting and event == cv2.EVENT_LBUTTONDOWN:
            
            mouse_x_norm = x / (self._frame_w - 1)
            
            # Update cmap selection based on which boundary we fall in to
            for cmap_code, x1_norm, x2_norm in self._cmap_x1x2_norm_list:
                is_selected = x1_norm < mouse_x_norm < x2_norm
                if is_selected:
                    self._cmap_select = cmap_code
                    break
        
        return
    
    # .................................................................................................................
    
    def read(self) -> int | None:
        return self._cmaps[self.idx]
    
    # .................................................................................................................
    
    def append_to_frame(self, frame) -> np.ndarray:
        frame_h, frame_w = frame.shape[0:2]
        bar_img = self.draw_bar_image(frame_w, frame_h)
        return np.vstack((frame, bar_img))
    
    # .................................................................................................................
    
    def draw_bar_image(self, frame_w, frame_h):
        
        ''' Helper used to set up base image sizing & interaction region '''
        
        # Re-draw base image if width changes
        got_new_width = (frame_w != self._frame_w)
        if got_new_width:

            # Re-draw new 'blank' bar image
            bar_img = np.zeros((self._bar_h, frame_w, 3), dtype=np.uint8)
            
            # Draw each colormap as a 1-row image
            color_1px_list = []
            for cmap_code, x1_norm, x2_norm in self._cmap_x1x2_norm_list:
                width_px = round((x2_norm - x1_norm) * (frame_w - 1))
                uint8_1px = np.expand_dims(np.linspace(0, 255, width_px, dtype=np.uint8), axis = 0)
                color_1px = self.apply_given_colormap(uint8_1px, cmap_code)
                color_1px[:, -1] = (40,40,40)
                color_1px[:, 0] = (40,40,40)
                color_1px_list.append(color_1px)
            
            # Combine 1-row colormaps and resize to target bar sizing
            bar_img = np.hstack(color_1px_list)
            bar_img = cv2.resize(bar_img, dsize = (frame_w, self._bar_h))
            
            # Draw divider/separator line before storing for re-use
            cv2.rectangle(bar_img, (-5,0), (frame_w + 5, self._bar_h - 1), (0,0,0), 1)
            self._img = bar_img
            self._frame_w = frame_w
        
        # Update height related info
        got_new_height = (frame_h != self._frame_h)
        if got_new_height:
            self._frame_h = frame_h
            self._interact_y1y2 = (frame_h, frame_h + self._bar_h)
        
        return self._img.copy()
    
    # .................................................................................................................
    
    def apply_colormap(self, image_uint8_1ch) -> np.ndarray:
        return self.apply_given_colormap(image_uint8_1ch, self._cmap_select)
    
    # .................................................................................................................
    
    @staticmethod
    def apply_given_colormap(image_uint8_1ch, opencv_colormap_code = None) -> np.ndarray:
        
        '''
        Converts a uint8 image (numpy array) into a bgr color image using opencv colormaps
        Expects an image of shape: HxWxC (with 1 or no channels, i.e. HxW only)
        Colormap code should be from opencv, which are accessed with: cv2.COLORMAP_{name}
        If the colormap code is None, then a grayscale (3ch) image is returned
        '''
        
        # Special case, if no colormap code is given, return 3ch grayscale image
        if opencv_colormap_code is None:
            return cv2.cvtColor(image_uint8_1ch, cv2.COLOR_GRAY2BGR)
        
        return cv2.applyColorMap(image_uint8_1ch, opencv_colormap_code)
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def make_message_header_image(message: str, frame_width: int,
                              header_height_px = 30, bg_color = (53,29,31)) -> np.ndarray:
    
    ''' Helper which makes a header image containing the given message '''
    
    # Figure out text scaling so that it fits in width of frame
    for scale in [0.5, 0.4, 0.3, 0.2, 0.1]:    
        (txt_w, txt_h), txt_base = cv2.getTextSize(message, 0, scale, 1)
        if txt_w < frame_width: break
    
    # Find centering position
    x_pos = (frame_width - txt_w) // 2
    y_pos = header_height_px//2 + txt_base
    xy_pos = (x_pos, y_pos)
    
    header_img = np.full((header_height_px, frame_width, 3), bg_color, dtype = np.uint8)
    return cv2.putText(header_img, message, xy_pos, 0, scale, (255,255,255), 1, cv2.LINE_AA)
