#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

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
from lib.demo_helpers.postprocess import scale_prediction, remove_inf_tensor, normalize_01, histogram_equalization
from lib.demo_helpers.plane_fit import estimate_plane_of_best_fit
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
parser = argparse.ArgumentParser(description="Script used to run depth-estimation on a single image")
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
    "-z",
    "--no_optimization",
    default=False,
    action="store_true",
    help="Disable attention optimizations (only effects DepthAnything models)",
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

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
prefer_bfloat16 = not args.prefer_unstable_f16
use_optimizations = not args.no_optimization
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px

# Hard-code no-cache usage (limited benefit for static images)
use_cache = False

# Set up device config
device_config_dict = make_device_config(device_str, use_float32, prefer_bfloat16)

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image or folder", history_imgpath)
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)

# Improve cpu utilization
reduce_overthreading(device_str)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model = make_dpt_from_state_dict(model_path, use_cache, use_optimizations)

# Move model to selected device
dpt_model.to(**device_config_dict)

# Set up loadable file paths
# -> *** This is a UI element! We're setting it up here because we need it to resolve initial file loading ***
is_folder_path = ui.PathCarousel.is_folder_path(image_path)
img_selector = ui.PathCarousel(image_path, search_parent_folder=True)
show_file_select = len(img_selector) > 1 or is_folder_path
if len(img_selector) == 0:
    print("", "No image files available!", f"@ {image_path}", "Quitting...", sep="\n")
    quit()

# Load (first) image
init_image_path, init_image_bgr = img_selector.load_next_valid(load_valid_image)


# ---------------------------------------------------------------------------------------------------------------------
# %% Run model


def post_process_prediction(prediction, scaled_wh):

    # Perform some post-processing to prepare for display
    scaled_prediction = scale_prediction(prediction, scaled_wh)
    depth_norm = remove_inf_tensor(scaled_prediction)
    depth_norm = normalize_01(scaled_prediction).float().cpu().numpy().squeeze()

    # Calculate a plane-of-best-fit, so we can (potentially) remove it during display
    plane_depth = estimate_plane_of_best_fit(depth_norm)

    return depth_norm, plane_depth


# Get display sizing
depth_scaled_hw = get_image_hw_for_max_side_length(init_image_bgr.shape, display_size_px)
depth_scaled_wh = tuple(reversed(depth_scaled_hw))

# Run model on initial file, so we can get some stats
print("", "Computing inverse depth...", sep="\n", flush=True)
t1 = perf_counter()
prediction = dpt_model.inference(init_image_bgr, model_base_size, force_square_resolution)
t2 = perf_counter()
print("  -> Took", round(1000 * (t2 - t1), 1), "ms")
print_config_feedback(model_path, device_config_dict, use_cache, prediction)

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
save_btn = ui.ImmediateButton("[s] Save", color=(80, 175, 0), text_color=255)

# Controls for adjusting processing size
initial_size = max(prediction.shape)
min_size, max_size = max(min(64, initial_size), 32), max(1280, initial_size)
imgsize_slider = ui.Slider("Image Size", initial_size, min_size, max_size, step=16, marker_step=256)
use_ar_btn = ui.ToggleButton(" AR ", default_state=not force_square_resolution)
resolution_txt = ui.TextBlock("0000x0000")

# Create lower control elements
plane_slider = ui.Slider("Remove plane", 0, -1, 2, 0.05, marker_step=0.25)
threshold_slider = ui.MultiSlider("Min/Max Threshold", (0, 1), 0, 1, 0.01, fill_color=(40, 110, 10), marker_step=0.25)

# Build UI
header_bar = ui.MessageBar(model_name, devdtype_str, color=header_color)
ui_layout = ui.VStack(
    header_bar,
    ui.HStack(reverse_colors_btn, high_contrast_btn, save_btn, flex=(3, 3, 1)),
    cmap_bar,
    ui.HStack(imgsize_slider, use_ar_btn, resolution_txt, flex=(1, 0, 0)),
    ui.HStack(img_elem, depth_elem),
    img_selector if show_file_select else None,
    plane_slider,
    threshold_slider,
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
        "Cycle colormapping": {"TAB": cmap_bar.next},
        "Cycle images": ({"L_ARROW": img_selector.prev, "R_ARROW": img_selector.next} if show_file_select else None),
        "Adjust plane removal": {"D_ARROW": plane_slider.decrement, "U_ARROW": plane_slider.increment},
        "Adjust image processing size": {"[": imgsize_slider.decrement, "]": imgsize_slider.increment},
        "Save results": {"s": save_btn.click},
    }
).report_keypress_descriptions()
print("Right click sliders to reset values")

# Initialize depth & display data
img_bgr = init_image_bgr
depth_norm, plane_depth = post_process_prediction(prediction, depth_scaled_wh)
depth_1ch = depth_norm.copy()
img_elem.set_image(init_image_bgr)

# Display loop
with window.auto_close():

    while True:

        # Read controls
        is_size_changed, img_size = imgsize_slider.read()
        is_ar_changed, use_aspect_ratio = use_ar_btn.read()
        is_plane_factor_changed, plane_removal_factor = plane_slider.read()
        _, (thresh_min, thresh_max) = threshold_slider.read()
        _, use_high_contrast = high_contrast_btn.read()
        _, use_reverse_colors = reverse_colors_btn.read()
        is_file_changed, _, file_select_path = img_selector.read()

        # Load new image
        if is_file_changed:
            file_select_path, img_bgr = img_selector.load_next_valid(load_valid_image)
            img_elem.set_image(img_bgr)

        # Update display sizing (mostly a display optimization when handling large images)
        if is_file_changed or window.is_size_changed():
            depth_scaled_hw = get_image_hw_for_max_side_length(img_bgr.shape, window.size)
            depth_scaled_wh = tuple(reversed(depth_scaled_hw))

        # Re-run depth estimate
        need_new_prediction = is_file_changed or is_size_changed or is_ar_changed
        if need_new_prediction:
            prediction = dpt_model.inference(img_bgr, img_size, not use_aspect_ratio)
            depth_norm, plane_depth = post_process_prediction(prediction, depth_scaled_wh)
            resolution_txt.set_text(f"{prediction.shape[2]}x{prediction.shape[1]}")

        # Re-calculate plane removal
        need_new_plane = is_plane_factor_changed or need_new_prediction
        if need_new_plane:
            depth_1ch = depth_norm - (plane_depth * plane_removal_factor)
            depth_1ch = normalize_01(depth_1ch)

        # Make sure we get non-zero delta to avoid divide-by-zero
        thresh_delta = max(0.001, thresh_max - thresh_min)
        depth_thresholded = np.clip((depth_1ch - thresh_min) / thresh_delta, 0.0, 1.0)

        # Produce colored depth image for display
        depth_uint8 = np.round(255.0 * depth_thresholded).astype(np.uint8)
        if use_high_contrast:
            depth_uint8 = histogram_equalization(depth_uint8, thresh_min, thresh_max)
        if use_reverse_colors:
            depth_uint8 = 255 - depth_uint8
        depth_color = cmap_bar.apply_colormap(depth_uint8)
        depth_elem.set_image(depth_color)

        # Update displayed image
        display_frame = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_frame)
        if req_break:
            break

        # Handle saving
        if save_btn.read():

            # Apply modifications to raw prediction for saving
            npy_prediction = remove_inf_tensor(prediction.clone())
            npy_prediction = normalize_01(npy_prediction).float().cpu().numpy().squeeze()
            npy_prediction = npy_prediction - (plane_removal_factor * estimate_plane_of_best_fit(npy_prediction))
            npy_prediction = normalize_01(npy_prediction)
            npy_prediction = np.clip((npy_prediction - thresh_min) / thresh_delta, 0.0, 1.0)
            if use_reverse_colors:
                npy_prediction = 1.0 - npy_prediction

            # Save data!
            ok_img_save, save_img_path = save_image(depth_color, image_path)
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
