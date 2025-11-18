#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

import os.path as osp
import argparse
from time import perf_counter

import cv2
import numpy as np
import torch

from lib.make_dpt import make_dpt_from_state_dict

import lib.demo_helpers.toadui as ui
from lib.demo_helpers.toadui.helpers.images import load_valid_image
from lib.demo_helpers.toadui.helpers.sizing import get_image_hw_for_max_side_length

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.saving import save_image, save_numpy_array, save_uint16
from lib.demo_helpers.postprocess import scale_prediction, normalize_01, histogram_equalization
from lib.demo_helpers.misc import (
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
default_display_size = 800
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Adjsut DPT fusion scaling factors to see effects on depth prediction")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to image or folder of images")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to DPT model weights")
parser.add_argument(
    "-s",
    "--display_size",
    default=default_display_size,
    type=int,
    help="Controls initial size of displayed results (default: {})".format(default_display_size),
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
    "-u",
    "--prefer_unstable_f16",
    default=False,
    action="store_true",
    help="Prefer 'regular' 16-bit floating point model weights instead of bfloat16",
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
    "--noselect",
    default=False,
    action="store_true",
    help="Disable file selector UI. This also leads to allocating more display space to the loaded image",
)

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
prefer_bfloat16 = not args.prefer_unstable_f16
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
allow_file_selector = not args.noselect

# Hard-code no-cache usage (limited benefit for static images)
use_cache = False

# Set up device config
device_config_dict = make_device_config(device_str, use_float32, prefer_bfloat16)

# Build pathing to repo-root, so we can search model weights properly
root_path = osp.dirname(osp.dirname(__file__))
save_folder = osp.join(root_path, "saved_images", "fusion_scaling")

# Create history to re-use selected inputs
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image or folder", history_imgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)

# Improve cpu utilization
reduce_overthreading(device_str)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model = make_dpt_from_state_dict(model_path, use_cache)

# Move model to selected device
dpt_model.to(**device_config_dict)

# Set up loadable file paths
# -> *** This is a UI element! We're setting it up here because we need it to resolve initial file loading ***
is_folder_path = ui.PathCarousel.is_folder_path(image_path)
img_selector = ui.PathCarousel(image_path, search_parent_folder=True)
show_file_select = (len(img_selector) > 1 or is_folder_path) and allow_file_selector
if len(img_selector) == 0:
    print("", "No image files available!", f"@ {image_path}", "Quitting...", sep="\n")
    quit()

# Load (first) image
init_image_path, init_image_bgr = img_selector.load_next_valid(load_valid_image)


# ---------------------------------------------------------------------------------------------------------------------
# %% Initial model run

# Run model partially to get intermediate tokens for scaling
print("", "Computing reassembly results...", sep="\n", flush=True)
t1 = perf_counter()
with torch.inference_mode():
    img_tensor = dpt_model.prepare_image_bgr(init_image_bgr, model_base_size, force_square_resolution)
    patch_tokens, patch_grid_hw = dpt_model.patch_embed(img_tensor)
    imgenc_tokens = dpt_model.imgencoder(patch_tokens, patch_grid_hw)
    reasm_tokens = dpt_model.reassemble(*imgenc_tokens, patch_grid_hw)
t2 = perf_counter()
print("  -> Took", round(1000 * (t2 - t1), 1), "ms")
print_config_feedback(model_path, device_config_dict, use_cache, img_tensor)

# For feedback
model_name, devdtype_str, header_color = make_header_strings(model_path, device_config_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up UI

# Create image display elements
init_img_elem_args = {"image": init_image_bgr, "aspect_ratio": 1 if show_file_select else None}
img_elem = ui.FixedARImage(**init_img_elem_args)
depth_elem = ui.FixedARImage(**init_img_elem_args)

# Create upper control elements
spectral_cmap = ui.colormaps.make_spectral_colormap()
cmap_bar = ui.ColormapsBar(cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TWILIGHT, spectral_cmap, None)
reverse_colors_btn = ui.ToggleButton("[r] Reverse Colors", color_on=(110, 80, 95))
high_contrast_btn = ui.ToggleButton("[h] High Contrast", color_on=(100, 95, 80))
disable_scaling_btn = ui.ToggleButton("[d] Disable scaling", color_on=(75, 95, 105))
save_btn = ui.ImmediateButton("[s] Save", color=(80, 175, 0), text_color=255)

# Controls for adjusting processing size
initial_size = max(img_tensor.shape)
min_size, max_size = max(min(64, initial_size), 32), max(1280, initial_size)
imgsize_slider = ui.Slider("Image Size", initial_size, min_size, max_size, step=16, marker_step=256)
use_ar_btn = ui.ToggleButton(" AR ", default_state=not force_square_resolution)
resolution_txt = ui.TextBlock("0000x0000")

# Create lower control elements
f1_slider = ui.Slider("Fusion 1", 1, -5, 5, 0.01, marker_step=1)
f2_slider = ui.Slider("Fusion 2", 1, -5, 5, 0.01, marker_step=1)
f3_slider = ui.Slider("Fusion 3", 1, -5, 5, 0.01, marker_step=1)
f4_slider = ui.Slider("Fusion 4", 1, -5, 5, 0.01, marker_step=1)

# Build UI
header_bar = ui.MessageBar(model_name, devdtype_str, color=header_color)
ui_layout = ui.VStack(
    header_bar,
    ui.HStack(reverse_colors_btn, high_contrast_btn, disable_scaling_btn, save_btn, flex=(2, 2, 2, 1)),
    cmap_bar,
    ui.HStack(imgsize_slider, use_ar_btn, resolution_txt, flex=(1, 0, 0)),
    ui.HStack(img_elem, depth_elem),
    img_selector if show_file_select else None,
    f1_slider,
    f2_slider,
    f3_slider,
    f4_slider,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Main display loop ***


# Setup window & callbacks
window = ui.DisplayWindow()
window.enable_size_control(display_size_px)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Toggle processing aspect ratio": {"a": use_ar_btn.toggle},
        "Toggle reverse colors": {"r": reverse_colors_btn.toggle},
        "Toggle high contrast": {"h": high_contrast_btn.toggle},
        "Toggle fusion scaling": {"d": disable_scaling_btn.toggle},
        "Cycle colormapping": {"TAB": cmap_bar.next},
        "Cycle images": ({"L_ARROW": img_selector.prev, "R_ARROW": img_selector.next} if show_file_select else None),
        "Adjust image processing size": {"[": imgsize_slider.decrement, "]": imgsize_slider.increment},
        "Save results": {"s": save_btn.click},
    }
).report_keypress_descriptions()
print(
    "- Use up/down arrows to make fine adjustments to (hovered) fusion sliders",
    "- Right click sliders to reset values",
    "- Press escape or q to close window",
    sep="\n",
    flush=True,
)

# Get initial display sizing
depth_scaled_hw = get_image_hw_for_max_side_length(init_image_bgr.shape, display_size_px)
depth_scaled_wh = tuple(reversed(depth_scaled_hw))

# Initialize depth & display data
img_bgr = init_image_bgr
img_elem.set_image(init_image_bgr)

with window.auto_close():

    while True:

        # Read controls
        is_size_changed, img_size = imgsize_slider.read()
        is_ar_changed, use_aspect_ratio = use_ar_btn.read()
        _, use_high_contrast = high_contrast_btn.read()
        _, use_reverse_colors = reverse_colors_btn.read()
        is_disabled_changed, disable_fusion = disable_scaling_btn.read()
        is_file_changed, _, file_select_path = img_selector.read()
        is_f1_changed, f1_scale = f1_slider.read()
        is_f2_changed, f2_scale = f2_slider.read()
        is_f3_changed, f3_scale = f3_slider.read()
        is_f4_changed, f4_scale = f4_slider.read()

        # Load new image
        if is_file_changed:
            file_select_path, img_bgr = img_selector.load_next_valid(load_valid_image)
            img_elem.set_image(img_bgr)

        # Update display sizing (mostly a display optimization when handling large images)
        if is_file_changed or window.is_size_changed():
            depth_scaled_hw = get_image_hw_for_max_side_length(img_bgr.shape, window.size)
            depth_scaled_wh = tuple(reversed(depth_scaled_hw))

        # Re-run model up to reassembly steps
        need_new_reassembly = is_file_changed or is_size_changed or is_ar_changed
        if need_new_reassembly:
            with torch.inference_mode():
                img_tensor = dpt_model.prepare_image_bgr(img_bgr, img_size, not use_aspect_ratio)
                patch_tokens, patch_grid_hw = dpt_model.patch_embed(img_tensor)
                imgenc_tokens = dpt_model.imgencoder(patch_tokens, patch_grid_hw)
                reasm_tokens = dpt_model.reassemble(*imgenc_tokens, patch_grid_hw)
            resolution_txt.set_text(f"{img_tensor.shape[3]}x{img_tensor.shape[2]}")

        # Run fusion steps manually, so we can apply scaling factors
        is_scaling_changed = any((is_f1_changed, is_f2_changed, is_f3_changed, is_f4_changed))
        need_new_fusion = need_new_reassembly or is_scaling_changed or is_disabled_changed
        if need_new_fusion:
            if disable_fusion:
                f1_scale = f2_scale = f3_scale = f4_scale = 1.0
            with torch.inference_mode():
                fuse_3 = dpt_model.fusion.blocks[3](reasm_tokens[3] * f4_scale)
                fuse_2 = dpt_model.fusion.blocks[2](reasm_tokens[2], fuse_3 * f3_scale)
                fuse_1 = dpt_model.fusion.blocks[1](reasm_tokens[1], fuse_2 * f2_scale)
                fuse_0 = dpt_model.fusion.blocks[0](reasm_tokens[0], fuse_1 * f1_scale)
                prediction = dpt_model.head(fuse_0).squeeze(dim=1)

            # Post-processing for display
            scaled_prediction = scale_prediction(prediction, depth_scaled_wh)
            depth_norm = normalize_01(scaled_prediction).float().cpu().numpy().squeeze()

        # Produce colored depth image for display
        depth_uint8 = np.uint8(np.round(255.0 * depth_norm))
        if use_high_contrast:
            depth_uint8 = histogram_equalization(depth_uint8)
        if use_reverse_colors:
            depth_uint8 = 255 - depth_uint8
        depth_color = cmap_bar.apply_colormap(depth_uint8)
        depth_elem.set_image(depth_color)

        # Update displayed image
        display_frame = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_frame)
        if req_break:
            break

        # Handle adjustments to sliders using up/down arrows
        if keypress == ui.KEY.U_ARROW:
            for f_slider in (f1_slider, f2_slider, f3_slider, f4_slider):
                if f_slider.is_hovered():
                    f_slider.increment()
        elif keypress == ui.KEY.D_ARROW:
            for f_slider in (f1_slider, f2_slider, f3_slider, f4_slider):
                if f_slider.is_hovered():
                    f_slider.decrement()

        # Handle saving
        if save_btn.read():

            # Apply modifications to raw prediction for saving
            npy_prediction = normalize_01(prediction.clone()).float().cpu().numpy().squeeze()
            if use_reverse_colors:
                npy_prediction = 1.0 - npy_prediction

            # Save data!
            ok_img_save, save_img_path = save_image(depth_color, image_path, save_folder=save_folder)
            ok_npy_save, save_npy_path = save_numpy_array(npy_prediction, save_img_path)
            ok_uint16_save, save_uint16_path = save_uint16(npy_prediction, save_img_path)
            if any((ok_img_save, ok_npy_save, ok_uint16_save)):
                print("", "SAVED:", save_img_path, sep="\n")
                if ok_npy_save:
                    print(save_npy_path)
                if ok_uint16_save:
                    print(save_uint16_path)
                pass
            pass
        pass
    pass
