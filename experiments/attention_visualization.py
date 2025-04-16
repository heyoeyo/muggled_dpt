#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp
import argparse
import cv2
import torch
import numpy as np

# This is a hack to make this script work from inside the experiments folder!
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import sys
    parent_folder = osp.dirname(osp.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.saving import save_image
from lib.demo_helpers.visualization import DisplayWindow, add_bounding_box, grid_stack_by_columns_first
from lib.demo_helpers.text import TextDrawer
from lib.demo_helpers.ui import SliderCB, ColormapButtonsCB, ButtonBar
from lib.demo_helpers.model_capture import ModelOutputCapture
from lib.demo_helpers.misc import (
    get_default_device_string, make_device_config, print_config_feedback, reduce_overthreading
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_display_size = 1000
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to visualize internal attention values")
parser.add_argument("-i", "--image_path", default=default_image_path,
                    help="Path to image to run depth estimation on")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str,
                    help="Path to DPT model weights")
parser.add_argument("-s", "--display_size", default=default_display_size, type=int,
                    help="Controls size of displayed results (default: {})".format(default_display_size))
parser.add_argument("-d", "--device", default=default_device, type=str,
                    help="Device to use when running model (ex: 'cpu', 'cuda', 'mps')")
parser.add_argument("-f32", "--use_float32", default=False, action="store_true",
                    help="Use 32-bit floating point model weights. Note: this doubles VRAM usage")
parser.add_argument("-ar", "--use_aspect_ratio", default=False, action="store_true",
                    help="Process the image at it's original aspect ratio, if the model supports it")
parser.add_argument("-b", "--base_size_px", default=default_base_size, type=int,
                    help="Override base (e.g. 384, 512) model size")
parser.add_argument("-c", "--hide_cls", default=False, action="store_true",
                    help="Use to disable display of cls token on attention maps")

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
show_cls_token = not args.hide_cls

# Build pathing to repo-root, so we can search model weights properly
root_path = osp.dirname(osp.dirname(__file__))
save_folder = osp.join(root_path, "saved_images", "attention_images")

# Create history to re-use selected inputs
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Warning for unsupported swin models
if "swin" in model_path:
    raise NotImplementedError("Cannot handle swin models (yet)! Please try another model for now")

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)

# Hard-code no-cache usage (no benefit) or optimization (needed to access attention results)
use_cache = False
use_optimization = False

# Improve cpu utilization
reduce_overthreading(device_str)


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes

class PatchSelectCB:

    '''
    Class used to manage mouse-over UI for selecting image patches.
    Handles both the window-callback needed to determine mouse positioning,
    as well as handling the rendering of the selected patch.
    '''
    
    # .................................................................................................................

    def __init__(self, frame, grid_hw, cls_bg_color=(80,80,80), selection_color=(255,0,255)):

        # Store original frame for display
        self._frame = frame
        self._frame_hw = frame.shape[0:2]

        # Allocate storage for scaled 'display copy' of input frame
        self._disp_frame = frame.copy()
        self._disp_h, self._disp_w = self._disp_frame.shape[0:2]
        self._cls_h = 32
        
        # Store coloring info
        self._cls_bg_color = cls_bg_color
        self._selection_color = selection_color
        self._selection_color_locked = tuple((val + 127) //2 for val in selection_color)

        # Storage for grid sizing, needed to interpret patch selection
        self._grid_h, self._grid_w = grid_hw
        self._num_patch_tokens = grid_hw[0] * grid_hw[1]

        # Storage for mouse interactions
        self._locked_selection = False
        self._cls_or_grid_hw_select = (True, 0, 0)
        self._interact_x1x2 = (0, self._disp_w)
        self._interact_y1y2 = (0, self._disp_h)
        self._cls_y1y2 = (-10, -5)
        self._interact_xy_offsets = (0, 0)
        
        # Set up multiple text drawers, for writing at different scales
        font = cv2.FONT_HERSHEY_PLAIN
        self._txt_writers = [TextDrawer(scale=s, font=font) for s in [1, 0.8, 0.5, 0.35, 0.25]]
        self._xy_idx_txt = TextDrawer(scale=0.5, bg_color=(0,0,0))

    # .................................................................................................................

    def set_interaction_offsets(self, x_offset, y_offset):
        self._interact_xy_offsets = (x_offset, y_offset)
        return

    # .................................................................................................................

    def __call__(self, event, x, y, flags, param) -> None:

        # Apply x/y offsets so we interpret positioning correctly
        x_off, y_off = self._interact_xy_offsets
        x -= x_off
        y -= y_off

        # Bail if mouse isn't over top of interactive area (in x)
        x1, x2 = self._interact_x1x2
        is_interacting_x = (x1 <= x < x2)
        if not (is_interacting_x):
            return
        
        # Check interactive areas (in y)
        y1, y2 = self._interact_y1y2
        cls_y1, cls_y2 = self._cls_y1y2
        is_interacting_y = (y1 <= y < y2)
        is_cls = (cls_y1 <= y < cls_y2)
        if not (is_interacting_y or is_cls):
            return
        
        # Figure out which patch grid index (in x) we're hovering
        x_norm = (x - x1) / (self._disp_w - 1)
        x_idx = int(np.floor(x_norm * self._grid_w))
        x_idx = max(0, min(self._grid_w - 1, x_idx))
        
        # Figure out which patch grid index (in y) we're hovering
        y_norm = (y - y1) / ((self._disp_h - self._cls_h)- 1)
        y_idx = int(np.floor(y_norm * self._grid_h))
        y_idx = max(0, min(self._grid_h - 1, y_idx))

        # Toggle patch selection locking on mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            self._locked_selection = not self._locked_selection

        # Only record new grid location if we're not locked to another
        if not self._locked_selection:
            self._cls_or_grid_hw_select = (is_cls, y_idx, x_idx)

        return

    # .................................................................................................................
    
    def make_cls_bar_image(self, frame, cls_token_height = 40):
        
        # Figure out text sizing that will 'fit' into bar height
        txt = "(cls)"
        for txt_writer in self._txt_writers:
            if txt_writer.check_will_fit_height(txt, cls_token_height, 0.975):
                break
        
        frame_w = frame.shape[1]
        cls_bar_img = np.full((cls_token_height, frame_w, 3), self._cls_bg_color, dtype=np.uint8)
        cls_bar_img = txt_writer.xy_centered(cls_bar_img, txt)
        
        return add_bounding_box(cls_bar_img, inset_box=False)

    # .................................................................................................................

    def draw_selected_patch(self, frame):
        
        ''' Helper used to highlight a patch location on a given frame '''

        frame_h, frame_w = frame.shape[0:2]

        # Figure out bounding box coords, depending on selected token
        is_cls, y_idx, x_idx = self._cls_or_grid_hw_select
        if is_cls:
            y1, y2 = self._cls_y1y2
            x1, x2 = 0, (frame_w - 1)
            
        else:
            block_w = frame_w / self._grid_w
            block_h = (frame_h - self._cls_h) / self._grid_h
    
            x1 = int(np.floor(x_idx * block_w))
            x2 = int(np.floor((x_idx + 1) * block_w)) - 1
    
            y1 = int(np.floor(y_idx * block_h))
            y2 = int(np.floor((y_idx + 1) * block_h)) - 1

        # Draw rectangle around highlighted patch
        bg_color = (0, 0, 0)
        fg_color = self._selection_color_locked if self._locked_selection else self._selection_color
        cv2.rectangle(frame, (x1, y1-1), (x2, y2-1), bg_color, 4, cv2.LINE_4)
        cv2.rectangle(frame, (x1, y1-1), (x2, y2-1), fg_color, 2, cv2.LINE_4)
        
        # Include token coords if we're not hovering the class token
        # -> These get drawn beside selected patch, but adjusted so they 'fit' in frame
        if not is_cls:
            txt_str = f"({x_idx}, {y_idx})"
            txt_w, txt_h, txt_baseline = self._xy_idx_txt.get_text_size(txt_str)
            txt_x, txt_y = (x2 + 5), ((y1 + y2 + txt_baseline)//2)
            if (x2 + txt_w) > frame_w:
                txt_x -= txt_w + (frame_w // self._grid_w) + 10
            self._xy_idx_txt.xy_px(frame, txt_str, (txt_x, txt_y))

        return frame

    # .................................................................................................................

    def get_token_index(self):

        is_cls, y_idx, x_idx = self._cls_or_grid_hw_select
        token_idx = 0 if is_cls else 1 + (y_idx * self._grid_w) + x_idx

        return token_idx

    # .................................................................................................................

    def hstack(self, other_frame):

        # Figure out how much to scale input image for display
        oth_h, oth_w = other_frame.shape[0:2]
        if oth_h != self._disp_h:
            
            # Figure out sizing to match height of other frame
            cls_bar_height = self._cls_h
            frame_h, frame_w = self._frame_hw
            scaled_h = oth_h - cls_bar_height
            scaled_w = round(frame_w * (oth_h / frame_h))

            # Update display copy & interaction region
            disp_frame = cv2.resize(self._frame, dsize=(scaled_w, scaled_h))
            cls_bar_img = self.make_cls_bar_image(disp_frame, cls_bar_height)
            self._disp_frame = np.vstack((disp_frame, cls_bar_img))
            self._disp_h, self._disp_w = self._disp_frame.shape[0:2]
            self._cls_h = cls_bar_img.shape[0]
            self._interact_x1x2 = (0, scaled_w)
            self._interact_y1y2 = (0, scaled_h)
            self._cls_y1y2 = (scaled_h, scaled_h + self._cls_h)

        disp_frame = self._disp_frame.copy()
        disp_frame = self.draw_selected_patch(disp_frame)

        return np.hstack((disp_frame, other_frame))

    # .................................................................................................................

    def vstack(self, other_frame):

        # Figure out how much to scale input image, for display
        oth_h, oth_w = other_frame.shape[0:2]
        if oth_w != self._disp_w:

            # Figure out sizing to match width of other frame:
            cls_bar_height = self._cls_h
            frame_h, frame_w = self._frame_hw
            scaled_w = oth_w
            scaled_h = round(frame_h * (oth_w / frame_w))
            
            # Update display copy & interaction region
            disp_frame = cv2.resize(self._frame, dsize=(scaled_w, scaled_h))
            cls_bar_img = self.make_cls_bar_image(disp_frame, cls_bar_height)
            self._disp_frame = np.vstack((disp_frame, cls_bar_img))
            self._disp_h, self._disp_w = self._disp_frame.shape[0:2]
            self._cls_h = cls_bar_img.shape[0]
            self._interact_x1x2 = (0, scaled_w)
            self._interact_y1y2 = (0, scaled_h)
            self._cls_y1y2 = (scaled_h, scaled_h + self._cls_h)

        disp_frame = self._disp_frame.copy()
        disp_frame = self.draw_selected_patch(disp_frame)

        return np.vstack((disp_frame, other_frame))

    # .................................................................................................................
    
    def stack(self, other_frame, stack_horizontally = True):
        return self.hstack(other_frame) if stack_horizontally else self.vstack(other_frame)
    
    # .................................................................................................................


class AttentionTileRenderer:
    
    ''' Class used to manage rendering of attention maps '''
    
    # .................................................................................................................
    
    def __init__(self, patch_grid_hw, cls_token_index = None, show_cls = True,
                 max_cls_footer_height = 18, select_color = (255, 0, 255)):
        
        self._patch_grid_hw = patch_grid_hw
        self._cls_idx = cls_token_index
        self._has_cls_token = (cls_token_index is not None)
        self._max_cls_footer_height = max_cls_footer_height
        self._selection_color = select_color
        self._show_cls = show_cls
        
        # Rendering configs
        self._text = TextDrawer(scale = 0.35, bg_color=(0,0,0))
        self._interp = cv2.INTER_NEAREST_EXACT
        self._display_scale = 1
        self._cmap = ColormapButtonsCB(cv2.COLORMAP_VIRIDIS)
        self._use_log_scaling = False
        self._show_idx = True
    
    # .................................................................................................................
    
    def set_scale_factor(self, new_scale_factor):
        self._display_scale = new_scale_factor
        return self
    
    def set_colormap(self, colormap_handler):
        self._cmap = colormap_handler
        return self
    
    def set_ln_scaling(self, use_log_scaling = True):
        self._use_log_scaling = use_log_scaling
        return self
    
    def set_head_index_rendering(self, show_head_index = True):
        self._show_idx = show_head_index
        return self
    
    # .................................................................................................................
    
    def render_all_heads(self, attention_tensor, selected_token_index):
        
        '''
        Function used to render attention maps for all 'heads' of a vision transformers
        Returns a list of images, which still need to be stacked together for display!
        '''
        
        # Get display version of data
        all_heads_attn_disp = torch.log(attention_tensor + 1e-6) if self._use_log_scaling else attention_tensor
        all_heads_disp_mins = all_heads_attn_disp.min(dim=1).values.unsqueeze(-1)
        all_heads_disp_maxs = all_heads_attn_disp.max(dim=1).values.unsqueeze(-1)
        all_heads_disp_deltas = all_heads_disp_maxs - all_heads_disp_mins
        
        # Get attention values in uint8 format for display
        all_heads_attn_norm = (all_heads_attn_disp - all_heads_disp_mins) / all_heads_disp_deltas
        all_heads_attn_norm = torch.clamp(all_heads_attn_norm, 0.0, 1.0)
        all_heads_attn_uint8 = (255 * all_heads_attn_norm).cpu().byte().numpy()
        
        # Render each of the head entries as separate images
        head_imgs_list = []
        for h_idx in range(num_heads):
            one_attn_head_uint8 = all_heads_attn_uint8[h_idx, :]
            one_head_img = self._render_one_head(h_idx, one_attn_head_uint8, selected_token_index)
            head_imgs_list.append(one_head_img)
        
        return head_imgs_list
    
    # .................................................................................................................
    
    def _render_one_head(self, head_idx, attention_head_uint8, selected_token_index):
        
        '''
        Function used to render a single image (HxWx3 numpy array) representing an
        attention map for a single 'head' of a vision transformer
        '''
        
        # Apply colormap to all attention values (includes patches + cls entries)
        cmap_attn = self._cmap.apply_colormap(attention_head_uint8)
        
        # Separate cls and patch tokens for rendering
        # -> patches can be rendered as an image
        # -> cls is just a number and needs to be rendered differently
        grid_h, grid_w = self._patch_grid_hw
        cls_idx = self._cls_idx if self._has_cls_token else 0
        patch_idx = 1 + cls_idx if self._has_cls_token else 0
        tile_img = cmap_attn[patch_idx:].reshape(grid_h, grid_w, 3)
        cls_img = cmap_attn[:(1 + cls_idx)]
        is_cls_selected = selected_token_index == cls_idx
        
        # Create scaled copy of patch entries & highlight selected entry if needed
        scale = self._display_scale
        tile_img = cv2.resize(tile_img, dsize=None, fx=scale, fy=scale, interpolation=self._interp)
        if not is_cls_selected:
            patch_select_idx = selected_token_index - patch_idx
            x_select_idx = patch_select_idx % grid_w
            y_select_idx = (patch_select_idx // grid_w)
            xy_pt = [round(scale * (idx + 0.5)) for idx in [x_select_idx, y_select_idx]]
            circ_rad = max(1, int(scale // 3))
            tile_img = cv2.circle(tile_img, xy_pt, circ_rad, self._selection_color, -1)
        
        # Attach image representing cls value if needed
        if self._has_cls_token and self._show_cls:
            img_h, img_w = tile_img.shape[0:2]
            footer_height = min(self._max_cls_footer_height, int(img_h * 0.4))
            cls_img = cv2.resize(cls_img, dsize=(img_w, footer_height))
            cls_img = cv2.line(cls_img, (-5, 0), (img_w + 5, 0), (0,0,0), 1)
            
            # Draw selection highlight if needed
            if is_cls_selected:
                cls_img = add_bounding_box(cls_img, self._selection_color, thickness=2)
            
            # Combine with patch image
            tile_img = np.vstack((tile_img, cls_img))
        
        # Render head index, if needed
        if self._show_idx:
            idx_str = f"H{head_idx}"
            self._text.xy_norm(tile_img, idx_str, (0.5,1), pad_xy_px=(0,-1))
        
        return add_bounding_box(tile_img, inset_box=False)
    
    # .................................................................................................................


class AttentionDisplayArrangement:
    
    '''
    Class used to handle the sizing & arrangement of the displayed attention tiles
    More specifically, this class is responsible for:
        1. The (integer) scale factor applied to attention tiles
        2. Adjusting the number of rows/columns in the display
        3. Swapping between wide (horizontal) and tall (vertical) display stacking
        4. Picking the 'best' initial settings for display
    '''
    
    # .................................................................................................................
    
    def __init__(self, image_shape, patch_grid_hw, num_attention_heads, max_display_size):
        
        # Store config
        self._image_hw = image_shape[0:2]
        self._patch_grid_hw = patch_grid_hw
        self._num_heads = num_attention_heads
        self._max_display_size = max_display_size
        
        # Compute all display arrangement info
        rc_options = self._calculate_row_column_options(num_attention_heads)
        wide_scl_scr, tall_scl_scr = self._calculate_scales_and_score(rc_options)
        wide_scales, wide_scores = wide_scl_scr
        tall_scales, tall_scores = tall_scl_scr
        
        # Figure out the best initial display arrangement
        best_wide_score = min(wide_scores)
        best_tall_score = min(tall_scores)
        best_wide_idx = wide_scores.index(best_wide_score)
        best_tall_idx = tall_scores.index(best_tall_score)
        best_score_is_wide = best_wide_score < best_tall_score
        best_idx = best_wide_idx if best_score_is_wide else best_tall_idx
        
        # Set initial values
        self._rc_options = rc_options
        self._use_wide = best_score_is_wide
        self._rc_select = best_idx
        self._scales_listing_dict = {True: wide_scales, False: tall_scales}
    
    # .................................................................................................................
    
    def read(self):
        ''' Messy/hacky read-all function to get display settings '''
        idx = self._rc_select
        use_wide = self._use_wide
        num_rows, num_cols = self._rc_options[idx]
        disp_scale = self._scales_listing_dict[use_wide][idx]
        return (num_rows, num_cols), disp_scale, use_wide
    
    # .................................................................................................................
    
    def _calculate_row_column_options(self, num_heads):
        ''' Helper used to get all evenly disible row/column combinations for the number of tiles/heads '''
        return [(k, num_heads//k) for k in range(1, 1 + num_heads) if (num_heads % k) == 0]
    
    # .................................................................................................................
    
    def _calculate_scales_and_score(self, row_column_options_list):
        
        '''
        Helper used to get scaling factors for all row/column arrangements for
        both wide (horizontally) stacked display & tall (vertically) stacked displays.
        Also calculates a score for each arrangement, which indicates the 'best' choice
        as a default (based on arbitrary aesthetics)
        '''
        
        # For convenience
        orig_h, orig_w = self._image_hw
        grid_h, grid_w = self._patch_grid_hw
        
        wide_scale_factors_list, tall_scale_factors_list = [], []
        wide_scores_list, tall_scores_list = [], []
        for num_rows, num_cols in row_column_options_list:
                
            # Figure out how big the attention tile image would be, with no scaling
            attn_h, attn_w = (num_rows * grid_h, num_cols * grid_w)
            attn_hw = (attn_h, attn_w)
            
            # Figure out how big the display image would be if stacked wide/tall with attn tiles
            wide_srcimg_w = int(round(orig_w * (attn_h / orig_h)))
            tall_srcimg_h = int(round(orig_h * (attn_w / orig_w)))
            
            # Figure out wide-stack scale factor & scoring
            wide_stacked_hw = (attn_h, wide_srcimg_w + attn_w)
            wide_scale, wide_score = self._calculate_one_scale_and_score(attn_hw, wide_stacked_hw)
            wide_scale_factors_list.append(wide_scale)
            wide_scores_list.append(wide_score)
            
            # Figure out tall-stack scale factor & scoring
            tall_stacked_hw = (tall_srcimg_h + attn_h, attn_w)
            tall_scale, tall_score = self._calculate_one_scale_and_score(attn_hw, tall_stacked_hw)
            tall_scale_factors_list.append(tall_scale)
            tall_scores_list.append(tall_score)
        
        wide_scale_and_scores = (wide_scale_factors_list, wide_scores_list)
        tall_scale_and_scores = (tall_scale_factors_list, tall_scores_list)
        return wide_scale_and_scores, tall_scale_and_scores
    
    # .................................................................................................................
    
    def _calculate_one_scale_and_score(self, attn_hw, stacked_hw):
        
        '''
        Helper used to calculate the allowable scaling factor for a given attention
        tile image size & corresponding 'stacked' size (wide or tall), along with
        a related 'score' which indicates preferably stacking arrangements (smallest score is best)
        '''
        
        # Figure out how much we could scale attention tiles
        max_side_px = max(stacked_hw)
        raw_scale_factor = self._max_display_size / max_side_px
        scale_factor = max(1, int(np.floor(raw_scale_factor)))
        
        # Calculate an arrangement score. We want:
        # 1. Attention tiles are ~50% of stacked display area
        # 2. Attention tiles are scaled as much as possible
        attn_area = attn_hw[0] * attn_hw[1]
        stacked_area = stacked_hw[0] * stacked_hw[1]
        relative_attn_area = attn_area / stacked_area
        target_area_ratio = abs(0.5 - relative_attn_area)
        inv_scale_score = min(1 / raw_scale_factor, 100)
        arrangement_score = target_area_ratio + inv_scale_score
        
        return scale_factor, arrangement_score
    
    # .................................................................................................................
    
    def adjust_scale_on_keypress(self, keypress, scale_down_keycode, scale_up_keycode):
        if keypress == scale_down_keycode:
            idx = self._rc_select
            use_wide = self._use_wide
            self._scales_listing_dict[use_wide][idx] = max(1, self._scales_listing_dict[use_wide][idx] - 1)
        if keypress == scale_up_keycode:
            idx = self._rc_select
            use_wide = self._use_wide
            self._scales_listing_dict[use_wide][idx] = max(1, self._scales_listing_dict[use_wide][idx] + 1)
        return self
    
    # .................................................................................................................
    
    def adjust_tiling_on_keypress(self, keypress, prev_tiling_keycode, next_tiling_keycode):
        if keypress == prev_tiling_keycode:
            self._rc_select = (self._rc_select - 1) % len(self._rc_options)
        if keypress == next_tiling_keycode:
            self._rc_select = (self._rc_select + 1) % len(self._rc_options)
        return self
    
    # .................................................................................................................
    
    def wide_tall_toggle_on_keypress(self, keypress, toggle_keycode):
        if keypress == toggle_keycode:
            self._use_wide = not self._use_wide
        return self
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache, use_optimization)
if model_base_size is not None:
    dpt_imgproc.set_base_size(model_base_size)

# Move model to selected device
device_config_dict = make_device_config(device_str, use_float32)
dpt_model.to(**device_config_dict)
dpt_model.eval()

# Load image and apply preprocessing
orig_image_bgr = cv2.imread(image_path)
img_tensor = dpt_imgproc.prepare_image_bgr(orig_image_bgr, force_square=force_square_resolution)
print_config_feedback(model_path, device_config_dict, use_cache, img_tensor)


# ---------------------------------------------------------------------------------------------------------------------
# %% Run model

# Set up data capture after attention softmax
captures = ModelOutputCapture(dpt_model, torch.nn.Softmax)

# Run model (only up to image encoder, so we can get attention matrices)
try:
    with torch.inference_mode():
        img_tensor = img_tensor.to(**device_config_dict)
        tokens, patch_grid_hw = dpt_model.patch_embed(img_tensor)
        dpt_model.imgencoder(tokens, patch_grid_hw)

except torch.cuda.OutOfMemoryError as err:    
    print("", "ERROR CAPTURING ATTENTION MATRICES:",
          str(err), "", ""
          "Out of memory error!",
          "This script requires more VRAM than usual in order to store attention results!",
          "Try reducing the base image sizing or switch to using cpu",
          sep="\n")
    have_data = len(captures) > 0
    if not have_data:
        raise SystemExit()

    from time import sleep
    print("", "Some data was still captured, will display anyways...", sep="\n", flush=True)
    sleep(3)

# Bail if we didn't capture anything
num_layers = len(captures)
if num_layers == 0:
    raise AttributeError("No data captured! Model doesn't contain 'Softmax' module?")

# Get sizing info
_, num_heads, num_attn_tokens, _ = captures[0].shape
patch_h, patch_w = patch_grid_hw
num_patch_tokens = patch_h * patch_w
num_global_tokens = num_attn_tokens - num_patch_tokens

# UI is mostly hard-coded to look for cls tokens, so error out if they aren't found
has_cls_token = num_global_tokens > 0
if not has_cls_token:
    raise NotImplementedError("UI does not support models without cls tokens (yet), cannot display results")
cls_token_index = num_global_tokens - 1


# ---------------------------------------------------------------------------------------------------------------------
# %% Main interaction loop

# For clarity, some keypress codes
KEY_UPARROW, KEY_DOWNARROW = 82, 84
KEY_LEFTARROW, KEY_RIGHTARROW = 81, 83
KEY_COMMA, KEY_PERIOD = ord(","), ord(".")
KEY_SPACEBAR = ord(" ")

# Set up handlerS for managing the arrangement of attention tiles (display + sizing + grid rows/columns)
attn_tiler = AttentionTileRenderer(patch_grid_hw, cls_token_index, show_cls_token)
attn_disp_size = AttentionDisplayArrangement(orig_image_bgr.shape, patch_grid_hw, num_heads, display_size_px)

# Set up button controls
btnbar = ButtonBar()
toggle_rowwise_attn = btnbar.add_toggle("[a] Row-wise Attn", "[a] Col-wise Attn", keypress="a")
toggle_lin_scale = btnbar.add_toggle("[l] Linear-Scale", "[l] Log-Scale", keypress="l", default=False)
btn_save = btnbar.add_button("[s] Save", keypress="s")

# Set up other UI elements
gray_cmap = ColormapButtonsCB.make_gray_colormap()
spec_cmap = ColormapButtonsCB.make_spectral_colormap()
cmap_btns = ColormapButtonsCB(cv2.COLORMAP_VIRIDIS, spec_cmap, cv2.COLORMAP_TURBO, cv2.COLORMAP_HOT, gray_cmap)
patch_select_cb = PatchSelectCB(orig_image_bgr, patch_grid_hw)
layer_slider = SliderCB("Attention Layer Index", 0, 0, num_layers - 1, 1, marker_step_size=1, bar_bg_color=(10,10,10))

# Provide colormapping to the tiler for display updates
attn_tiler.set_colormap(cmap_btns)

# Set up display window + controls
cv2.destroyAllWindows()
dispwin = DisplayWindow("Self-Attention Maps (Per-Head) - q to quit")
dispwin.set_callbacks(cmap_btns, patch_select_cb, layer_slider, btnbar)

# Feedback about controls
print("", "Displaying attention maps",
      "  - Hover mouse over image to highlight a patch token",
      "   -> Attention map for selected patch token is shown on the right",
      "  - Click to lock/unlock patch selection",
      "  - Press spacebar to flip display orientation",
      "  - Press , or . to adjust tiling",
      "  - Use up/down arrows to change display scale",
      "  - Use left/right arrows to adjust layer selection",
      "  - Press esc or q to quit",
      "",
      sep="\n", flush=True)

while True:

    # Read inputs to decide what to display
    layer_idx = layer_slider.read()
    token_idx = patch_select_cb.get_token_index()
    cmap_select = cmap_btns.read()
    use_rowwise_attn = toggle_rowwise_attn.read()
    use_log_scale = not toggle_lin_scale.read()
    
    # Messy read-all display config
    (num_rows, num_cols), disp_scale, use_wide_display = attn_disp_size.read()

    # Pick attention orientation (row/query-centric or column/key-centric)
    # -> row-wise attention maps 'sum to 1' for each head (i.e. it is the direct softmax result)
    attn_result = captures[layer_idx].squeeze()
    one_token_attn = attn_result[:, token_idx, :] if use_rowwise_attn else attn_result[:, :, token_idx]
    
    # To confirm softmax, uncomment this line
    # print("Attention sum (per head):", one_token_attn.sum(-1).tolist())
    
    # Update tiler display settings
    attn_tiler.set_scale_factor(disp_scale)
    attn_tiler.set_ln_scaling(use_log_scale)
    
    # Draw attention tiles & combine with original input image
    attn_tiles_list = attn_tiler.render_all_heads(one_token_attn, token_idx)
    attn_heads_img = grid_stack_by_columns_first(attn_tiles_list, num_cols)
    sidebyside_img = patch_select_cb.stack(attn_heads_img, stack_horizontally = use_wide_display)

    # Build image for display
    # -> buttons | colormaps | side-by-side image + attn | layer select slider
    display_image = btnbar.draw_standalone(sidebyside_img.shape[1])
    display_image = cmap_btns.append_to_frame(display_image)
    patch_select_cb.set_interaction_offsets(0, display_image.shape[0])
    display_image = np.vstack((display_image, sidebyside_img))
    display_image = layer_slider.append_to_frame(display_image)
    dispwin.imshow(display_image)
    req_break, keypress = dispwin.waitKey(35)
    if req_break:
        break
    
    # Handle button keypresses
    btnbar.on_keypress(keypress)
    if btn_save.read():
        ok_save, save_path = save_image(display_image, image_path, save_folder=save_folder)
        if ok_save: print("", "SAVED:", save_path, "", sep="\n")
    
    # Handle remaining keypresses
    layer_slider.on_keypress(keypress, KEY_LEFTARROW, KEY_RIGHTARROW)
    attn_disp_size.adjust_scale_on_keypress(keypress, KEY_DOWNARROW, KEY_UPARROW)
    attn_disp_size.adjust_tiling_on_keypress(keypress, KEY_PERIOD, KEY_COMMA)
    attn_disp_size.wide_tall_toggle_on_keypress(keypress, KEY_SPACEBAR)
    

# Clean up
cv2.destroyAllWindows()
