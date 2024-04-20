#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import cv2

# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class TextDrawer:
    
    '''
    Helper used to handle text-drawing onto images
    Holds fixed font + thickness & base scale
    If a background color is given, text will be drawn with a thicker background for better contrast
    '''
    
    def __init__(self, scale=1, thickness=1, bg_color=None, font=cv2.FONT_HERSHEY_SIMPLEX):
        
        self._bg_color = bg_color
        self._bg_thick = min(thickness + 3, thickness * 3)
        self._font = font
        self._scale = scale
        self._thick = thickness
        self._ltype = cv2.LINE_AA
    
    @classmethod
    def create_from_existing(cls, other_text_drawer):
        scale = other_text_drawer._scale
        thickness = other_text_drawer._thick
        bg_color = other_text_drawer._bg_color
        font = other_text_drawer._font
        return cls(scale=scale, thickness=thickness, bg_color=bg_color, font=font)
    
    def xy_px(self, image, text, xy_px, color=(255,255,255)):
        
        ''' Helper used to draw text at a give location using pre-configured settings '''
        
        if self._bg_color is not None:
            image = cv2.putText(image, text, xy_px, self._font, self._scale, self._bg_color, self._bg_thick, self._ltype)
            
        return cv2.putText(image, text, xy_px, self._font, self._scale, color, self._thick, self._ltype)
    
    def xy_norm(self, image, text, xy_norm, color=(255,255,255), pad_xy_px=(0,0)):
        
        ''' Helper used to draw text given normalized (0-to-1) xy coordinates '''
        
        # Figure out pixel coords for the given normalized position
        txt_w, txt_h, txt_base = self.get_text_size(text)
        img_h, img_w = image.shape[0:2]
        x_norm, y_norm = xy_norm
        txt_x = round((img_w - txt_w) * x_norm)
        txt_y = round(txt_base + img_h * y_norm + txt_h * (1 - 2.0*y_norm))
        # This comes from:
        #    x @ x_norm = 0: 0  and  x @ x_norm = 1: img_w - txt_w
        #    y @ y_norm = 0: txt_h + txt_base
        #    y @ y_norm = 1: img_h + txt_base - txt_h
        
        pad_x, pad_y = pad_xy_px
        txt_xy_px = (txt_x + pad_x, txt_y + pad_y)
        return self.xy_px(image, text, txt_xy_px, color)
    
    def xy_centered(self, image, text, color=(255,255,255)):
        ''' Helper used to draw x/y centered text '''
        return self.xy_norm(image, text, (0.5, 0.5), color)
    
    def adjust_scale(self, new_scale):
        self._scale = new_scale
        return self
    
    def check_will_fit_width(self, text, target_width, shrink_factor=0.9) -> bool:
        ''' Helper used to check if text could be written into given container width '''
        txt_w, _, _ = self.get_text_size(text)
        return txt_w < int(target_width * shrink_factor)
    
    def check_will_fit_height(self, text, target_height, shrink_factor=0.9) -> bool:
        ''' Helper used to check if text could be written into given container height '''
        txt_w, _, _ = self.get_text_size(text)
        return txt_w < int(target_height * shrink_factor)
    
    def get_text_size(self, text):
        (txt_w, txt_h), txt_base = cv2.getTextSize(text, self._font, self._scale, self._thick)
        return txt_w, txt_h, txt_base
