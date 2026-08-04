#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import muggled_dpt  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "muggled_dpt" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to muggled_dpt folder!")

import os.path as osp
import argparse

import cv2
import torch
import numpy as np

from muggled_dpt.make_dpt import make_dpt_from_state_dict

import muggled_dpt.demo_helpers.toadui as ui
from muggled_dpt.demo_helpers.toadui.helpers.sizing import get_image_hw_for_max_side_length
from muggled_dpt.demo_helpers.toadui.helpers.data_management import ValueChangeTracker

from muggled_dpt.demo_helpers.crop_ui import run_crop_ui
from muggled_dpt.demo_helpers.history_keeper import HistoryKeeper
from muggled_dpt.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from muggled_dpt.demo_helpers.saving import save_image, save_numpy_array
from muggled_dpt.demo_helpers.model_capture import ModelOutputCapture
from muggled_dpt.demo_helpers.misc import (
    get_default_device_string,
    make_device_config,
    make_header_strings,
    print_config_feedback,
    reduce_overthreading,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_display_size = 600
default_base_size = None
default_log_contrast = 6
default_cls_border = 1

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to visualize image encoder attention values")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to image to run depth estimation on")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to DPT model weights")
parser.add_argument(
    "-s",
    "--display_size",
    default=default_display_size,
    type=int,
    help="Controls size of displayed results (default: {})".format(default_display_size),
)
parser.add_argument(
    "-d",
    "--device",
    default=default_device,
    type=str,
    help="Device to use when running model (ex: 'cpu', 'cuda', 'mps')",
)
parser.add_argument(
    "-f32",
    "--use_float32",
    default=False,
    action="store_true",
    help="Use 32-bit floating point model weights. Note: this doubles VRAM usage",
)
parser.add_argument(
    "-ar",
    "--use_aspect_ratio",
    default=False,
    action="store_true",
    help="Process the image at it's original aspect ratio, if the model supports it",
)
parser.add_argument(
    "-b", "--base_size_px", default=default_base_size, type=int, help="Override base (e.g. 384, 512) model size"
)
parser.add_argument(
    "-c",
    "--contrast",
    default=default_log_contrast,
    type=float,
    help="Contrast factor used in log-scaling. Higher contrast makes small attention scores easier to see",
)
parser.add_argument(
    "--cls_border",
    default=default_cls_border,
    type=int,
    help="Size of 'cls' token indicator. Set to 0 to disable visualization",
)
parser.add_argument(
    "-p",
    "--patch",
    default=False,
    action="store_true",
    help="Show selected patch row/column indexing",
)
parser.add_argument(
    "--crop",
    default=False,
    action="store_true",
    help="Crop image (interactively) before depth prediction",
)

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
log_contrast = args.contrast
cls_border_size = max(0, args.cls_border)
show_patch_index = args.patch
enable_crop_step = args.crop

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

# Set up device usage & data types
device_config_dict = make_device_config(device_str, use_float32)


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class AttentionTileRenderer:
    """Class used to manage (grayscale!) rendering of attention maps"""

    # .................................................................................................................

    def __init__(self, patch_grid_hw: tuple[int, int], cls_token_index: int | None = None, cls_border_size_px: int = 1):
        self._patch_grid_hw = patch_grid_hw
        self._cls_idx = cls_token_index
        self._has_cls_token = cls_token_index is not None
        self._cls_border_size = cls_border_size_px

    # .................................................................................................................

    def render_all_heads(
        self,
        attention_tensor: torch.Tensor,
        contrast_factor: int = 0,
    ) -> list[np.ndarray]:
        """
        Function used to render attention maps for all 'heads' of a vision transformers
        Returns a list of images, which still need to be stacked together for display.
        - Note that each image is grayscale
        - All images have HxW according to the token patch sizing itself
        - If a cls token is present, it is added as a border (1px by default) to the image
        """

        # Adjust contrast if needed (high contrast makes small values easier to see)
        all_heads_attn_disp = attention_tensor
        if contrast_factor > 0:
            contrast_scale = torch.pow(torch.tensor(10), torch.tensor(contrast_factor))
            all_heads_attn_disp = torch.log1p(contrast_scale * attention_tensor)

        # Get normalize version of data
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
            one_head_img = self._render_one_head(h_idx, one_attn_head_uint8)
            head_imgs_list.append(one_head_img)

        return head_imgs_list

    # .................................................................................................................

    def _render_one_head(self, head_idx: int, attention_head_uint8: np.ndarray) -> np.ndarray:
        """
        Function used to render a single grayscale image (HxW numpy array) representing an
        attention map for a single 'head' of a vision transformer
        """

        # Separate cls and patch tokens for rendering
        grid_h, grid_w = self._patch_grid_hw
        cls_idx = self._cls_idx
        patch_start_idx = 1 + cls_idx if self._has_cls_token else 0
        tile_img_uint8 = attention_head_uint8[patch_start_idx:].reshape(grid_h, grid_w)

        # Add border to indicate cls token value
        if self._has_cls_token:
            cls_val_uint8 = int(attention_head_uint8[cls_idx])
            tblr_size = [self._cls_border_size] * 4
            tile_img_uint8 = cv2.copyMakeBorder(tile_img_uint8, *tblr_size, cv2.BORDER_CONSTANT, value=cls_val_uint8)

        return tile_img_uint8

    # .................................................................................................................


class CustomTileOverlayDrawer:
    """Used to implement patch/cls highlighting on attention head tiles as a custom overlay"""

    def __init__(self, tile_hw: tuple[int, int], cls_border_size: int = 1):

        # Sizing variables
        self._tile_h, self._tile_w = tile_hw
        self._cls_border_size = cls_border_size

        # Selection state variables
        self._is_cls_selected: bool = True
        self._row_idx: int | None = None
        self._col_idx: int | None = None
        self._enable_highlighting = True

    def set_selected_token(self, row_column_index: tuple[int, int] | None) -> None:
        """Record selected tile for rendering as an overlay"""
        self._is_cls_selected = row_column_index is None
        self._row_idx, self._col_idx = (None, None) if self._is_cls_selected else row_column_index

    def toggle_highlighting(self, enable: bool | None = None) -> None:
        """Enable/disable highlighting overlay"""
        self._enable_highlighting = not self._enable_highlighting if enable is None else enable

    def render(self, frame: np.ndarray, xy_norm: tuple[float, float]) -> np.ndarray:
        """Custom overlay renderer. Highlights selected patch or cls token"""

        # Draw highlighting if needed
        img_h, img_w = frame.shape[0:2]
        if self._enable_highlighting:
            # Figure out how big 1 'patch pixel/step' is in terms of displayed pixels
            # (e.g. a 36x36 patch is displayed as 232x232px, so each patch is ~6.44px on display)
            x_step_size = (img_w - 1) / (2 * self._cls_border_size + self._tile_w)
            y_step_size = (img_h - 1) / (2 * self._cls_border_size + self._tile_h)
            if self._is_cls_selected:
                # Draw box around entire tile image, to indicate 'cls' is selected
                x1, y1 = x_step_size * self._cls_border_size, y_step_size * self._cls_border_size
                x2, y2 = (img_w - 1) - x1, (img_h - 1) - y1
                thickness = max(1, round(max(x1, y1) / 3))
            else:
                # Draw box around highlighted patch
                x1, x2 = [x_step_size * (self._cls_border_size + self._col_idx + x) for x in [0, 1]]
                y1, y2 = [y_step_size * (self._cls_border_size + self._row_idx + x) for x in [0, 1]]
                thickness = 1
            cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (255, 0, 255), thickness)

        # Draw black border outline, regardless
        return cv2.rectangle(frame, (0, 0), (img_w - 1, img_h - 1), (0, 0, 0), 1)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Load model
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model = make_dpt_from_state_dict(model_path, use_cache, use_optimization)
dpt_model.to(**device_config_dict)

# Load image
input_image_bgr = cv2.imread(image_path)
assert input_image_bgr is not None, f"Error loading image: {image_path}"

# Apply cropping if needed
crop_xy1xy2_norm = ((0, 0), (1, 1))
if enable_crop_step:
    _, crop_xy1xy2_norm = history.read("crop_xy1xy2_norm")
    (crop_y_slice, crop_x_slice), crop_xy1xy2_norm = run_crop_ui(input_image_bgr, crop_xy1xy2_norm)
    input_image_bgr = input_image_bgr[crop_y_slice, crop_x_slice]
    history.store(crop_xy1xy2_norm=crop_xy1xy2_norm)

# Get model info for feedback
model_name, devdtype_str, header_color = make_header_strings(model_path, device_config_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Run model

# Set up data capture after attention softmax
captures = ModelOutputCapture(dpt_model, torch.nn.Softmax)

# Run model (only up to image encoder, so we can get attention matrices)
try:
    with torch.inference_mode():
        img_tensor = dpt_model.prepare_image_bgr(input_image_bgr, model_base_size, force_square_resolution)
        tokens, patch_grid_hw = dpt_model.patch_embed(img_tensor)
        dpt_model.imgencoder(tokens, patch_grid_hw)

except torch.cuda.OutOfMemoryError as err:
    print(
        "",
        "ERROR CAPTURING ATTENTION MATRICES:",
        str(err),
        "",
        "" "Out of memory error!",
        "This script requires more VRAM than usual in order to store attention results!",
        "Try reducing the base image sizing or switch to using cpu",
        sep="\n",
    )
    have_data = len(captures) > 0
    if not have_data:
        raise SystemExit()

    from time import sleep

    print("", "Some data was still captured, will display anyways...", sep="\n", flush=True)
    sleep(1)

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

# Feedback on success
print_config_feedback(model_path, device_config_dict, use_cache, img_tensor)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up UI

# Create image display elements
block_ar = patch_w / patch_h
main_img = ui.FixedARImage(aspect_ratio=block_ar, resize_interpolation=cv2.INTER_NEAREST_EXACT)
cmap_bar = ui.ColormapsBar(cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TURBO, cv2.COLORMAP_HOT, None)

colwise_btn = ui.ToggleButton("[a] Column Attn", default_state=False, text_scale=0.5, color_on=(35, 75, 200))
logscale_btn = ui.ToggleButton("[l] Log-Scale", default_state=True, text_scale=0.5, color_on=(135, 105, 90))
save_btn = ui.ImmediateButton("[s] Save", text_scale=0.5, color=(80, 175, 0), text_color=255)
layer_slider = ui.Slider("Model Layer", 0, 0, num_layers - 1, 1, marker_step=1)

# Form grid of images, with aspect ratio to invert/balance the token aspect ratio
head_drawer = CustomTileOverlayDrawer(patch_grid_hw, cls_border_size=cls_border_size)
head_tiles_list = []
for idx in range(num_heads):
    head_img = ui.FixedARImage(aspect_ratio=block_ar, resize_interpolation=cv2.INTER_NEAREST_EXACT)
    head_tile = ui.OverlayStack(
        head_img,
        ui.overlays.DrawCustomOverlay(None, head_drawer.render),
        ui.HoverLabelOverlay(None, f"Head {idx}", xy_norm=(0.5, 0.5), scale=0.75, thickness=2),
    )
    head_tiles_list.append(head_tile)
tile_grid = ui.GridStack(*head_tiles_list, target_aspect_ratio=1 / block_ar)

# Build UI
main_olay = ui.GridSelectOverlay(main_img, (patch_h, patch_w), thickness=2, color=(255, 0, 255))
padded_main_img = ui.Padded(main_olay, pad_px=24, color=(30, 20, 40), outer_outline_color=(0, 0, 0))
display_swap = ui.Swapper(ui.HStack(padded_main_img, tile_grid), ui.VStack(padded_main_img, tile_grid))
header_bar = ui.MessageBar(model_name, devdtype_str, color=header_color)
ui_layout = ui.VStack(
    header_bar,
    ui.HStack(colwise_btn, logscale_btn, save_btn),
    cmap_bar,
    display_swap,
    layer_slider,
)

# Flip initial display orientation for very wide images
if block_ar > 2:
    display_swap.next()

# Given heavier text weight for patch index display
if show_patch_index:
    main_olay.style.text.style.update(fg_thickness=2, bg_thickness=4)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Main display loop ***

# Setup window & callbacks
window = ui.DisplayWindow(display_fps=30)
window.enable_size_control(display_size_px, minimum=ui_layout.get_min_hw().h)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Adjust layer selection": {"[": layer_slider.decrement, "]": layer_slider.increment},
        "Cycle colormapping": {"TAB": cmap_bar.next},
        "Adjust grid layout": {",": tile_grid.increment_num_rows, ".": tile_grid.decrement_num_rows},
        "Change display orientation": {"SPACEBAR": display_swap.next},
        "Toggle row-/column-wise attention": {"a": colwise_btn.toggle},
        "Toggle linear-/log-scale": {"l": logscale_btn.toggle},
        "Toggle highlighting on tiles": {"t": head_drawer.toggle_highlighting},
        "Toggle grid selection lock": {"k": main_olay.toggle_lock},
        "Save data": {"s": save_btn.click},
    }
).report_keypress_descriptions()
print(
    "- Hover mouse over image to highlight different patch tokens",
    "- Click to lock/unlock patch selection",
    "- Hover over the image border to select attention for the global 'cls' token",
    "- Use arrow keys to adjust token selection",
    "- Hover over attention maps to display head indexing",
    "- Press escape or q to close window",
    sep="\n",
    flush=True,
)

# Set up handler for rendering out attention images
attn_tiler = AttentionTileRenderer(patch_grid_hw, cls_token_index, cls_border_size)

# Initialize display data
scale_hw = get_image_hw_for_max_side_length(patch_grid_hw, display_size_px)
orig_img = cv2.resize(input_image_bgr, dsize=(scale_hw[1], scale_hw[0]))
main_img.set_image(orig_img)

# Set up special key group to listen for custom grid select movement logic
arrow_keys = (ui.KEY.L_ARROW, ui.KEY.R_ARROW, ui.KEY.U_ARROW, ui.KEY.D_ARROW)
move_lut = {ui.KEY.L_ARROW: (0, -1), ui.KEY.R_ARROW: (0, 1), ui.KEY.U_ARROW: (-1, 0), ui.KEY.D_ARROW: (1, 0)}

# Display loop
is_cls_select_changed = ValueChangeTracker(None)
token_idx = 0
with window.auto_close():

    while True:

        # Read inputs to decide what to display
        is_cmap_changed, _, _ = cmap_bar.read()
        is_colwise_changed, is_colwise = colwise_btn.read()
        is_logscale_changed, is_logscale = logscale_btn.read()
        is_layer_changed, layer_idx = layer_slider.read()
        is_token_select_changed, is_locked, row_col_select = main_olay.read()

        # Handle changes to selected 'patch' token
        if is_token_select_changed:
            token_idx = cls_token_index
            if row_col_select is not None:
                row_idx, col_idx = row_col_select
                token_idx = 1 + (row_idx * patch_w) + col_idx

            # Indicate 'cls' token highlighting
            is_cls_selected = token_idx == cls_token_index
            if is_cls_select_changed.is_changed(is_cls_selected):
                padded_main_img.style.thickness_inner_outline = 5
                padded_main_img.style.color_inner_outline = (200, 0, 200) if is_cls_selected else None
                padded_main_img.set_labels(bottom_label="CLS" if is_cls_selected else "")

            # Indicate patch co-ordinates
            if show_patch_index:
                main_olay.set_text_overlay("" if is_cls_selected else str(row_col_select))

            # Update highlighted token on head tiles
            head_drawer.set_selected_token(row_col_select)

        # Update displayed attention maps
        need_attn_map_update = is_token_select_changed or is_layer_changed or is_colwise_changed
        if need_attn_map_update:

            # Get attention tensor for the selected layer
            # -> Has shape: NumHeads x N x N (N is num tokens = patch_h*patch_w + 1 for 'cls')
            attn_result: torch.Tensor = captures[layer_idx].squeeze()

            # Pick attention orientation (row/query-centric or column/key-centric)
            # -> row-wise attention will 'sum to 1' for each head (i.e. it is the direct softmax result)
            selected_patch_attn: torch.Tensor = (
                attn_result[:, :, token_idx] if is_colwise else attn_result[:, token_idx, :]
            )

        # Update display visuals
        need_visual_update = need_attn_map_update or is_logscale_changed or is_cmap_changed
        if need_visual_update:
            contrast_factor = log_contrast if is_logscale else 0
            attn_tiles_gray_uint8_list = attn_tiler.render_all_heads(selected_patch_attn, contrast_factor)
            attn_tiles_color_list = [cmap_bar.apply_colormap(tile_img) for tile_img in attn_tiles_gray_uint8_list]
            for head_tile, tile_img in zip(head_tiles_list, attn_tiles_color_list):
                head_tile.get_base_item().set_image(tile_img)

        # Update displayed image
        display_frame = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_frame)
        if req_break:
            break

        # Allow grid selection to move with arrow keypress
        if keypress in arrow_keys:
            # If 'cls' is selected, jump to (0,0) token, so movement can work properly
            if row_col_select is None:
                main_olay.set_selected_row_column((0, 0), is_relative_move=False, ignore_lock=True)
            main_olay.toggle_lock(True)
            main_olay.set_selected_row_column(move_lut.get(keypress, (0, 0)), is_relative_move=True, ignore_lock=True)

        # Handling saving
        if save_btn.read():
            str_token = f"t{token_idx}_{patch_h}x{patch_w}"
            ok_img, save_img_path = save_image(display_frame, image_path, save_folder, f"_{str_token}")
            ok_npy, save_npy_path = save_numpy_array(selected_patch_attn.float().cpu().numpy(), save_img_path)
            if ok_img and ok_npy:
                print("", "SAVED:", save_img_path, save_npy_path, "", sep="\n")
            pass
        pass
    pass
