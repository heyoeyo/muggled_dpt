#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2
import numpy as np

from .text import TextDrawer
from .visualization import add_bounding_box

from typing import Protocol


# ---------------------------------------------------------------------------------------------------------------------
#%% Slider UI

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
                 marker_step_size = None, bar_height = 40, enable_value_display = True,
                 bar_bg_color = (40,40,40), indicator_line_width = 1):
        
        # Storage for slider value
        min_value, max_value = sorted((min_value, max_value))
        self._slider_min = min_value
        self._slider_max = max_value
        self._slider_value = min(max_value, max(min_value, start_value))
        self._slider_step = step_size
        self._slider_label = label
        self._slider_delta = max(self._slider_max - self._slider_min, 1e-9)
        
        # Display config
        self._enable_value_display = enable_value_display
        self._max_digits = max(1 + len(str(val)) for val in (min_value, max_value, step_size))
        self._initial_value = self._slider_value
        self._marker_step_size = marker_step_size
        self._bar_bg_color = bar_bg_color
        self._indicator_thickness = indicator_line_width
        
        # Storage for mouse state
        self._mouse_x_px = 0
        self._mouse_is_pressed = False
        self._interact_y1y2 = (-20, -10)
        
        # Storage for indicator bar image
        self._bar_h = bar_height
        self._img = None
        self._frame_w = 0
        self._frame_h = 0
        
        # Handler for drawing label + value indicators
        self._txt = TextDrawer(scale=0.5)
    
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
    
    def set(self, slider_value, use_as_default_value = True) -> None:
        new_value = max(self._slider_min, min(self._slider_max, slider_value))
        if use_as_default_value:
            self._initial_value = new_value
        self._slider_value = new_value
        return self
    
    # .................................................................................................................
    
    def on_keypress(self, keypress_code, decrement_keycode, increment_keycode):
        
        if keypress_code == decrement_keycode:
            self.set(self._slider_value - self._slider_step, use_as_default_value=False)
        if keypress_code == increment_keycode:
            self.set(self._slider_value + self._slider_step, use_as_default_value=False)
        
        return self
    
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
            txt_w, txt_h, txt_baseline = self._txt.get_text_size(value_str)
            x1, x2 = x_px + 5, x_px - txt_w - 5
            txt_x = x1 if (x1 + txt_w) < frame_w else x2
            txt_y = txt_baseline + self._bar_h//2
            txt_xy = (txt_x, txt_y)
            self._txt.xy_px(bar_img, value_str, txt_xy)
        
        # Draw indicator line onto the bar
        cv2.line(bar_img, (x_px, 2), (x_px, self._bar_h-2), (255,255,255), self._indicator_thickness)
        return np.vstack((frame, bar_img))
    
    # .................................................................................................................

    def _update_base_image(self, frame_h, frame_w) -> np.ndarray:
        
        ''' Helper used to set up base image sizing & interaction region '''
        
        # Re-draw base image if width changes
        got_new_width = (frame_w != self._frame_w)
        if got_new_width:

            # Re-draw new 'blank' bar image
            bar_img = np.full((self._bar_h, frame_w, 3), self._bar_bg_color, dtype=np.uint8)
            
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
            self._txt.xy_norm(bar_img, self._slider_label, (0, 0.5), (150,150,150), pad_xy_px=(5, 0))
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


# ---------------------------------------------------------------------------------------------------------------------
#%% Colormap UI

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
        
        # Check & store each provided colormap and interpret 'None' as grayscale colormap
        cmaps = []
        for cm in cv2_colormap_codes:
            if cm is None:
                cm = self.make_gray_colormap()
            elif isinstance(cm, np.ndarray):
                assert cm.shape == (1,256,3), "Bad colormap shape, must be: 1x256x3"
            elif not isinstance(cm, int):
                raise TypeError("Unrecognized colormap type! Must be a cv2 colormap code, None or an np array")
            cmaps.append(cm)
        
        # Set up left/right boundaries for selecting cmaps
        num_cmaps = len(cmaps)
        xnorm_bounds = [idx/(num_cmaps) for idx in range(num_cmaps + 1)]
        self._cmap_x1x2_norm_list = list(zip(cmaps, xnorm_bounds[:-1], xnorm_bounds[1:]))
        
        # Storage for indicator bar image
        self.height_px = bar_height
        self._img = None
        self._frame_w = 1
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
        return self._cmap_select
    
    # .................................................................................................................
    
    def append_to_frame(self, frame) -> np.ndarray:
        
        frame_h, frame_w = frame.shape[0:2]
        bar_img = self.draw_bar_image(frame_w)
        self._interact_y1y2 = (frame_h, frame_h + bar_img.shape[0])
        
        return np.vstack((frame, bar_img))
    
    # .................................................................................................................
    
    def prepend_to_frame(self, frame) -> np.ndarray:
        
        frame_h, frame_w = frame.shape[0:2]
        bar_img = self.draw_bar_image(frame_w)
        self._interact_y1y2 = (0, bar_img.shape[0])
        
        return np.vstack((bar_img, frame))
    
    # .................................................................................................................
    
    def draw_bar_image(self, frame_w):
        
        ''' Helper used to set up base image sizing & interaction region '''
        
        # Re-draw base image if width changes
        need_redraw = (frame_w != self._frame_w)
        if need_redraw:

            # Re-draw new 'blank' bar image
            bar_img = np.zeros((self.height_px, frame_w, 3), dtype=np.uint8)
            
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
            bar_img = cv2.resize(bar_img, dsize = (frame_w, self.height_px))
            
            # Draw divider/separator line before storing for re-use
            cv2.rectangle(bar_img, (-5,0), (frame_w + 5, self.height_px - 1), (0,0,0), 1)
            self._img = bar_img
            self._frame_w = frame_w
        
        return self._img.copy()
    
    # .................................................................................................................
    
    def apply_colormap(self, image_uint8_1ch) -> np.ndarray:
        return self.apply_given_colormap(image_uint8_1ch, self._cmap_select)
    
    # .................................................................................................................
    
    @staticmethod
    def apply_given_colormap(image_uint8_1ch, colormap_code_or_lut) -> np.ndarray:
        
        '''
        Converts a uint8 image (numpy array) into a bgr color image using opencv colormaps
        or using LUTs (numpy arrays of shape 1x256x3).
        Colormap code should be from opencv, which are accessed with: cv2.COLORMAP_{name}
        LUTs should be numpy arrays of shape 1x256x3, where each of the 256 entries
        encodes a bgr value which maps on to a 0-255 range.
        
        Expects an image of shape: HxWxC (with 1 or no channels, i.e. HxW only)
        '''
        
        if isinstance(colormap_code_or_lut, int):
            # Handle maps provided as opencv colormap codes (e.g. cv2.COLORMAP_VIRIDIS)
            return cv2.applyColorMap(image_uint8_1ch, colormap_code_or_lut)
        
        elif isinstance(colormap_code_or_lut, np.ndarray):
            # Handle maps provided as LUTs (e.g. 1x256x3 numpy arrays)
            image_ch3 = cv2.cvtColor(image_uint8_1ch, cv2.COLOR_GRAY2BGR)
            return cv2.LUT(image_ch3, colormap_code_or_lut)
        
        elif colormap_code_or_lut is None:
            # Return grayscale image if no mapping is provided
            return cv2.cvtColor(image_uint8_1ch, cv2.COLOR_GRAY2BGR)
        
        # Error if we didn't deal with the colormap above
        raise TypeError(f"Error applying colormap, unrecognized colormap type: {type(colormap_code_or_lut)}")
    
    # .................................................................................................................
    
    @staticmethod
    def make_gray_colormap(num_samples = 256):
        
        """ Makes a colormap in opencv LUT format, for grayscale output using cv2.LUT function """
        
        gray_ramp = np.round(np.linspace(0, 1, num_samples) * 255).astype(np.uint8)
        gray_ramp_img = np.expand_dims(gray_ramp, 0)
        
        return cv2.cvtColor(gray_ramp_img, cv2.COLOR_GRAY2BGR)
    
    # .................................................................................................................
    
    @staticmethod
    def make_spectral_colormap(num_samples = 256):
        
        """
        Creates a colormap for use with opencv which matches the (reversed) appearance of a
        colormap called 'Spectral' from the library matplotlib.
        The colormap is generated this way in order to avoid requiring the full matplotlib dependency!
        
        The original colormap definition can be found here:
        https://github.com/matplotlib/matplotlib/blob/30f803b2e9b5e237c5c31df57f657ae69bec240d/lib/matplotlib/_cm.py#L793
        -> The version here uses a slightly truncated copy of the values
        -> This version is also pre-reversed compared to the original
        -> Also, color keypoints are in bgr order (the original uses rgb ordering, opencv needs bgr)
        
        Returns a colormap which can be used with opencv, for example:
            
            spectral_colormap = make_spectral_colormap()
            gray_image_3ch = cv2.cvtColor(gray_image_1ch, cv2.COLOR_GRAY2BGR)
            colormapped_image = cv2.LUT(gray_image_3ch, spectral_colormap)
            
        The result has a shape of: 1xNx3, where N is number of samples (256 by default and required for cv2.LUT usage)
        """
        
        # Colormap keypoints from matplotlib. The colormap is produced by linear-interpolation of these points
        spectral_rev_bgr = np.float32(
            (
                (0.635, 0.310, 0.369),
                (0.741, 0.533, 0.196),
                (0.647, 0.761, 0.400),
                (0.643, 0.867, 0.671),
                (0.596, 0.961, 0.902),
                (0.749, 1.000, 1.000),
                (0.545, 0.878, 0.996),
                (0.380, 0.682, 0.992),
                (0.263, 0.427, 0.957),
                (0.310, 0.243, 0.835),
                (0.259, 0.004, 0.620),
            )
        )
        
        # Build out indexing into the keypoint array vs. colormap sample indexes
        norm_idx = np.linspace(0, 1, num_samples)
        keypoint_idx = norm_idx * (len(spectral_rev_bgr) - 1)
        
        # Get the start/end indexes for linear interpolation at each colormap sample
        a_idx = np.int32(np.floor(keypoint_idx))
        b_idx = np.int32(np.ceil(keypoint_idx))
        t_val = keypoint_idx - a_idx
        
        # Compute colormap as a linear interpolation between bgr keypoints
        bias = spectral_rev_bgr[a_idx]
        slope = spectral_rev_bgr[b_idx] - spectral_rev_bgr[a_idx]
        cmap_bgr_values = bias + slope * np.expand_dims(t_val, 1)
        cmap_bgr_values = np.round(cmap_bgr_values * 255).astype(np.uint8)
        
        return np.expand_dims(cmap_bgr_values, 0)
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Button bar UI

class ButtonBar:
    
    '''
    Class used to provide a simple set of button controls drawn
    as a single-row bar on in an  opencv window
    
    Works as a window callback (within opencv callback interface)
    while also providing functions to draw onto display frames
    
    Example usage:
        # Attach callback to cv2 window
        btn_bar = ButtonBar()
        cv2.setMouseCallback(winname, btn_bar)
        
        # Add controls to bar
        enable_toggle = btn_bar.add_toggle(label_true = "Enabled", label_false = "Disabled", default=True)
        save_btn = btn_bar.add_button(label = "Save")
        
        # Read control values
        is_enabled = enable_toggle.read()
        need_save = save_btn.read()
        
        # Draw bar onto displayed frame
        disp_frame = btn_bar.append_to_frame(disp_frame)
        cv2.imshow(winname, disp_frame)
        cv2.waitKey(1)
    '''
    
    class Control(Protocol):
        ''' Define what a single 'control' on the button bar should be able to do '''
        def read(self): ...
        def click(self): ...
        def on_keypress(self, keycode): ...
        def make_image(self, button_height, button_width): ...
    
    # .................................................................................................................
    
    def __init__(self, bar_height = 36):
        
        # Drawing config
        self.height_px = bar_height
        self._base_img = np.full((1,1,3), 0, dtype = np.uint8)
        self._requires_redraw = True
        
        # Mouse interaction settings
        self._interact_y_offset = 0
        self._interact_y1y2 = (-10, -10)
        self._enable = True
        
        # Storage for controls that get added
        self._ctrls_list = []
    
    # .................................................................................................................
    
    def __call__(self, event, x, y, flags, param) -> None:
        
        # Don't run callback when disabled
        if not self._enable:
            return
        
        # Bail if mouse is at wrong height
        y1, y2 = self._interact_y1y2
        is_interacting_y = y1 < (y - self._interact_y_offset) < y2
        if not is_interacting_y:
            return
        
        # Change button state on mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            
            # Bail if we have no controls!
            num_ctrls = len(self._ctrls_list)
            if num_ctrls == 0:
                return
            
            # Figure out which control was clicked
            bar_w = self._base_img.shape[1]
            x_norm = x / (bar_w - 1)
            idx_select = int(x_norm * num_ctrls)
            idx_select = min(num_ctrls - 1, idx_select)
            idx_select = max(0, idx_select)
            
            # Click the selected control!
            self._ctrls_list[idx_select].click()
        
        return
    
    # .................................................................................................................
    
    def _on_control_change(self):
        self._requires_redraw = True
        return
        
    # .................................................................................................................
    
    def add_toggle(self, label_true="On", label_false="Off", default=True, keypress=None):
        
        new_toggle_ctrl = _ToggleControl(self._on_control_change, label_true, label_false, default, keypress)
        self._ctrls_list.append(new_toggle_ctrl)
        
        return new_toggle_ctrl
    
    # .................................................................................................................
    
    def add_button(self, label="Button", keypress=None):
        
        new_btn_ctrl = _ButtonControl(self._on_control_change, label, keypress)
        self._ctrls_list.append(new_btn_ctrl)
        
        return new_btn_ctrl
    
    # .................................................................................................................
    
    def make_disabled_button(self, read_value = True):
        
        """
        Special function used to create a 'placeholder' button that isn't part of the UI,
        but can be 'read' (returns a constant value) as if it were in the UI. This is meant
        for cases where UI elements are disabled/hidden, but ideally still need to be
        available for reading in code.
        """
        on_change = lambda x: None
        new_toggle_ctrl = _ToggleControl(on_change, "_", "_", read_value, None)
        
        return new_toggle_ctrl
    
    # .................................................................................................................
    
    def enable(self, enable = True):
        self._enable = enable
        return self
    
    # .................................................................................................................
    
    def set_y_offset(self, y_offset_px):
        self._interact_y_offset = y_offset_px
        return self
    
    # .................................................................................................................
    
    def _make_base_image(self, bar_width):
        
        # For convenience
        num_ctrls = len(self._ctrls_list)
        if num_ctrls == 0 or not self._enable:
            return np.zeros((0, bar_width, 3), dtype=np.uint8)
        btn_width = round(bar_width / num_ctrls)
        
        # Stack all button images together & scale to reach final target size, if needed
        btn_imgs_list = [ctrl.make_image(self.height_px, btn_width) for ctrl in self._ctrls_list]
        new_base_img = np.hstack(btn_imgs_list)
        wrong_final_wdith = (new_base_img.shape[1] != bar_width)
        if wrong_final_wdith:
            new_base_img = cv2.resize(new_base_img, dsize=(bar_width, self.height_px))
        
        # Signal that we've handled the re-draw
        self._requires_redraw = False
        
        return new_base_img
    
    # .................................................................................................................
    
    def draw_standalone(self, frame_width) -> np.ndarray:
        
        '''
        Used to draw the bar by itself. Assumes bar will be the top-most part of the
        image in which it is displayed (if not, mouse clicks will be handled incorrectly!)
        '''
        
        base_w = self._base_img.shape[1]
        size_mismatch = frame_width != base_w
        if size_mismatch or self._requires_redraw:
            self._base_img = self._make_base_image(frame_width)
        
        base_h = self._base_img.shape[0]
        self._interact_y1y2 = (0, base_h)
        
        return self._base_img.copy()
    
    # .................................................................................................................
    
    def prepend_to_frame(self, frame) -> np.ndarray:
        
        '''
        Used to add bar 'above' the given frame
        Importantly, updates the expected y-location so that mouse clicks are handled properly
        '''
        
        frame_h, frame_w = frame.shape[0:2]
        base_w = self._base_img.shape[1]
        size_mismatch = frame_w != base_w
        if size_mismatch or self._requires_redraw:
            self._base_img = self._make_base_image(frame_w)
        
        base_h = self._base_img.shape[0]
        self._interact_y1y2 = (0, base_h)
        
        return np.vstack((self._base_img, frame))
    
    # .................................................................................................................
    
    def append_to_frame(self, frame):
        
        '''
        Used to add bar 'under' the given frame
        Importantly, updates the expected y-location so that mouse clicks are handled properly
        '''
        
        frame_h, frame_w = frame.shape[0:2]
        base_w = self._base_img.shape[1]
        size_mismatch = frame_w != base_w
        if size_mismatch or self._requires_redraw:
            self._base_img = self._make_base_image(frame_w)
        
        base_h = self._base_img.shape[0]
        self._interact_y1y2 = (frame_h, frame_h + base_h)
        
        return np.vstack((frame, self._base_img))
    
    # .................................................................................................................
    
    def on_keypress(self, keypress_code):
        keypress_match = [ctrl.on_keypress(keypress_code) for ctrl in self._ctrls_list]
        return keypress_match
        
    # .................................................................................................................


class _ToggleControl(ButtonBar.Control):
    
    ''' Class representing a single toggle-able button entry on a ButtonBar '''
    
    def __init__(self, on_change_callback, label_true="On", label_false="Off", default=True, keypress=None):
        self._on_change_cb = on_change_callback
        self._state = default
        self._label_true = label_true
        self._label_false = label_false
        self._keypress = ord(keypress) if isinstance(keypress, str) else keypress
        
        # For graphics
        self._txt = TextDrawer(0.5)
        self._txt_color = (255,255,255)
        self._on_bg_color = (60,60,60)
        self._off_bg_color = (40,40,40)
    
    def __repr__(self):
        label_state = self._label_true if self._state else self._label_false
        return f"ToggleControl: {label_state}"
    
    def get_label(self) -> str:
        return self._label_true if self._state else self._label_false
    
    def read(self) -> bool:
        return self._state
    
    def click(self):
        self.toggle()
        return self
    
    def toggle(self):
        self._state = not self._state
        self._on_change_cb()
        return self
    
    def set(self, new_state: bool):
        self._state = new_state
        self._on_change_cb()
        return self
    
    def on_keypress(self, keypress_code):
        
        # Bail if we don't have a target key to listen for!
        is_match = False
        if self._keypress is None: return is_match
        
        # Toggle if we get our target key
        is_match = (keypress_code == self._keypress)
        if is_match: self.toggle()
        
        return is_match
    
    def make_image(self, button_height, button_width):
        
        # Make (small) image just for our button
        bg_color = self._on_bg_color if self._state else self._off_bg_color
        toggle_img = np.full((button_height, button_width, 3), bg_color, dtype=np.uint8)
        
        # Draw text onto button image & store for combining
        label = self.get_label()
        toggle_img = self._txt.xy_centered(toggle_img, label, self._txt_color)
        toggle_img = add_bounding_box(toggle_img, thickness=2, inset_box=False)
        
        return toggle_img


class _ButtonControl(ButtonBar.Control):
    
    ''' Class representing a single button for use on a ButtonBar '''
    
    def __init__(self, on_change_callback, label="Button", keypress=None):
        self._on_change_cb = on_change_callback
        self._state = False
        self._label = label
        self._keypress = ord(keypress) if isinstance(keypress, str) else keypress
        
        # For graphics
        self._txt = TextDrawer(0.5)
        self._txt_color = (255,255,255)
        self._bg_color = (60,60,60)
    
    def __repr__(self):
        return f"ButtonControl: {self._label} ({self._state})"
    
    def get_label(self) -> str:
        return self._label
    
    def read(self) -> bool:
        ''' Read button state. The state returns to False afterwards! '''
        read_state = self._state
        self.reset()
        return read_state
    
    def click(self):
        self.set()
        return self
    
    def set(self):
        self._state = True
        self._on_change_cb()
        return self
    
    def reset(self):
        if self._state == True:
            self._state = False
            self._on_change_cb()
        return self
    
    def on_keypress(self, keypress_code):
        
        # Bail if we don't have a target key to listen for!
        is_match = False
        if self._keypress is None: return is_match
        
        # Change state on target keypress code
        is_match = (keypress_code == self._keypress)
        if is_match:
            # Note can't just do: self._state = is_match
            # -> It's possible state is True already but hasn't been read
            # -> setting if is_match is False, doing _state = is_match
            #    would clear the True state before it's read!
            self.set()
        
        return is_match
    
    def make_image(self, button_height, button_width):
        
        # Draw button image with label
        btn_img = np.full((button_height, button_width, 3), self._bg_color, dtype=np.uint8)
        btn_img = self._txt.xy_centered(btn_img, self._label, self._txt_color)
        btn_img = add_bounding_box(btn_img, thickness=2, inset_box=False)
        
        return btn_img


# ---------------------------------------------------------------------------------------------------------------------
#%% Scale keys

class ScaleByKeypress:
    
    '''
    Very simple helper which keeps track of a scaling factor that can
    be adjusted by pressing up/down arrow keys on OpenCV window updates.
    
    Example usage:
        
        # Create instance
        scaler = ScaleByKeypress()
        
        # ... do stuff ...
        
        # Get keypress code from opencv
        keypress = cv2.waitKey(1) & 0xFF
        
        # Update/listen for up/down arrow keys
        scale_factor = scaler.on_keypress(keypress)
        
        # Apply current scaling factor directly
        scaler.resize(numpy_image)
    '''
    
    KEY_UPARROW = 82
    KEY_DOWNARROW = 84
    
    def __init__(self):
        self._key_up = self.KEY_UPARROW
        self._key_down = self.KEY_DOWNARROW
        self._min_factor, self._max_factor = 0.1, 10
        self._step_factor = 1.05
        self._scale_factor = 1.0
    
    def resize(self, image):
        if abs(self._scale_factor - 1.0) < 0.001:
            return image
        return cv2.resize(image, dsize=None, fx=self._scale_factor, fy=self._scale_factor)
    
    def on_keypress(self, keypress_code):
        if keypress_code == self._key_up:
            self._scale_factor = min(self._scale_factor * self._step_factor, self._max_factor)
        elif keypress_code == self._key_down:
            self._scale_factor = max(self._scale_factor * (1/self._step_factor), self._min_factor)
        return self._scale_factor
