#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import os.path as osp

import argparse
from time import perf_counter

import cv2
import numpy as np
import torch

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
from lib.demo_helpers.ui import SliderCB, ButtonBar, ScaleByKeypress
from lib.demo_helpers.visualization import DisplayWindow
from lib.demo_helpers.plane_fit import estimate_plane_of_best_fit

from lib.demo_helpers.saving import save_image
from lib.demo_helpers.misc import (
    get_default_device_string,
    make_device_config,
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
parser = argparse.ArgumentParser(description="Script used to mask parts of an image based on depth thresholds")
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

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px

# Build pathing to repo-root, so we can search model weights properly
root_path = osp.dirname(osp.dirname(__file__))
save_folder = osp.join(root_path, "saved_images", "depth_masking")

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)

# Hard-code no-cache usage, since there is no benefit if the model only runs once
use_cache = False

# Improve cpu utilization
reduce_overthreading(device_str)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache)
if model_base_size is not None:
    dpt_imgproc.set_base_size(model_base_size)

# Move model to selected device
dpt_model.to(**device_config_dict)
dpt_model.eval()

# Load image and apply preprocessing
orig_image_bgr = cv2.imread(image_path)
assert orig_image_bgr is not None, f"Error loading image: {image_path}"
img_tensor = dpt_imgproc.prepare_image_bgr(orig_image_bgr, force_square_resolution)
print_config_feedback(model_path, device_config_dict, use_cache, img_tensor)

# Prepare original image for display (and get sizing info)
scaled_input_img = dpt_imgproc.scale_to_max_side_length(orig_image_bgr, display_size_px)
disp_h, disp_w = scaled_input_img.shape[0:2]
disp_wh = (int(disp_w), int(disp_h))

# Create checker pattern, which will be tiled to indicate transparency when masking
checker_size = 64
checker_h, checker_w = 2 * checker_size, 2 * checker_size
checker_a_rgb, checker_b_rgb = [215] * 3, [170] * 3
checker_base = np.full((checker_h, checker_w, 3), checker_a_rgb, dtype=np.uint8)
cv2.rectangle(checker_base, (checker_size, 0), (checker_w, checker_size), checker_b_rgb, -1)
cv2.rectangle(checker_base, (0, checker_size), (checker_size, checker_h), checker_b_rgb, -1)

# Using wrap-padding to create nicely tiled checker pattern matching displayed image size
checker_x_pad = max(disp_w - checker_w, 0)
checker_y_pad = max(disp_h - checker_h, 0)
checker_img = cv2.copyMakeBorder(
    checker_base,
    top=checker_y_pad // 2,
    bottom=checker_y_pad - (checker_y_pad // 2),
    left=checker_x_pad // 2,
    right=checker_x_pad - (checker_x_pad // 2),
    borderType=cv2.BORDER_WRAP,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Run model

t1 = perf_counter()

# Run the model and move the result to the cpu (in case it was on GPU)
print("", "Computing inverse depth...", sep="\n", flush=True)
img_tensor = img_tensor.to(**device_config_dict)
prediction = dpt_model.inference(img_tensor)

# Perform some post-processing to prepare for display
scaled_prediction = dpt_imgproc.scale_prediction(prediction, disp_wh)
depth_norm = dpt_imgproc.remove_infinities(scaled_prediction)
depth_norm = dpt_imgproc.normalize_01(scaled_prediction).float().cpu().numpy().squeeze()

t2 = perf_counter()
print("  -> Took", round(1000 * (t2 - t1), 1), "ms")


# ---------------------------------------------------------------------------------------------------------------------
# %% Display results

# Calculate a plane-of-best-fit, so we can (potentially) remove it during display
plane_depth = estimate_plane_of_best_fit(depth_norm)

# Set up button controls
btnbar = ButtonBar()
toggle_invert_range = btnbar.add_toggle("[i] Invert", "[i] Invert", keypress="i", default=False)
btn_save = btnbar.add_button("[s] Save", keypress="s")

# Set up other UI elements
plane_slider = SliderCB("Remove plane", 0, -2, 2, 0.01, marker_step_size=0.5)
min_slider = SliderCB("Min Threshold", 0.0, 0, 1, 0.01, marker_step_size=0.1).set(0.5, use_as_default_value=False)
max_slider = SliderCB("Max Threshold", 1.0, 0, 1, 0.01, marker_step_size=0.1)
display_scaler = ScaleByKeypress()

# Set up window with controls
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit")
window.set_callbacks(btnbar, plane_slider, min_slider, max_slider)

# Pre-define parameters used inside conditionals
prev_plane_removal_factor = None
depth_1ch = depth_norm

# Feedback about controls
print(
    "",
    "Displaying results",
    "  - Click and drag bars to adjust display",
    "  - Right click on bars to reset values",
    "  - Use up/down arrow keys to adjust display size",
    "  - Press esc or q to quit",
    "",
    sep="\n",
    flush=True,
)

while True:

    # Read controls
    plane_removal_factor = plane_slider.read()
    thresh_min = min_slider.read()
    thresh_max = max_slider.read()
    use_inverted_range = toggle_invert_range.read()

    # Re-calculate depth image if plane removal changes
    removal_factor_changed = plane_removal_factor != prev_plane_removal_factor
    if removal_factor_changed:
        depth_1ch = depth_norm - (plane_depth * plane_removal_factor)
        depth_1ch = dpt_imgproc.normalize_01(depth_1ch)
        prev_plane_removal_factor = plane_removal_factor

    # Make sure we actually get min < max thresholds before thresholding
    thresh_min, thresh_max = sorted([thresh_min, thresh_max])
    depth_thresholded = np.bitwise_and(depth_1ch >= thresh_min, depth_1ch <= thresh_max)

    # Convert thresholded depth to binary & create inverted mask (used to mix in checkerboarding effect)
    depth_mask_uint8 = 255 * np.uint8(depth_thresholded)
    if use_inverted_range:
        depth_mask_uint8 = np.bitwise_not(depth_mask_uint8)
    inv_mask_uint8 = np.bitwise_not(depth_mask_uint8)

    # Mix color image with checkerboarding to indicate transparency/masking
    img_rgb_masked = np.bitwise_and(scaled_input_img, cv2.cvtColor(depth_mask_uint8, cv2.COLOR_GRAY2BGR))
    checker_masked = np.bitwise_and(checker_img, cv2.cvtColor(inv_mask_uint8, cv2.COLOR_GRAY2BGR))
    masked_color = cv2.add(img_rgb_masked, checker_masked)

    # Generate display image: button controls / side-by-side images / sliders
    sidebyside_display = display_scaler.resize(np.hstack((scaled_input_img, masked_color)))
    display_frame = btnbar.draw_standalone(sidebyside_display.shape[1])
    display_frame = np.vstack((display_frame, sidebyside_display))
    display_frame = SliderCB.append_many_to_frame(
        display_frame,
        plane_slider,
        min_slider,
        max_slider,
    )

    # Update displayed image
    window.imshow(display_frame)
    req_break, keypress = window.waitKey(20)
    if req_break:
        break

    # Handle keypresses
    display_scaler.on_keypress(keypress)
    btnbar.on_keypress(keypress)
    if btn_save.read():

        # Apply modifications to raw prediction for saving
        npy_prediction = dpt_imgproc.remove_infinities(prediction.clone())
        npy_prediction = dpt_imgproc.normalize_01(npy_prediction).float().cpu().numpy().squeeze()
        npy_prediction = npy_prediction - (plane_removal_factor * estimate_plane_of_best_fit(npy_prediction))
        npy_prediction = dpt_imgproc.normalize_01(npy_prediction)

        # Scale prediction to match original sizing
        orig_h, orig_w = orig_image_bgr.shape[0:2]
        scaled_prediction = cv2.resize(npy_prediction, dsize=(orig_w, orig_h))
        save_mask = np.bitwise_and(scaled_prediction >= thresh_min, scaled_prediction <= thresh_max)
        save_mask_uint8 = 255 * np.uint8(save_mask)
        if use_inverted_range:
            save_mask_uint8 = 255 - save_mask_uint8

        # Apply masking to original color image (at original scale!) & use mask as alpha channel
        # -> Even though alpha channel hides masked RGB, applying mask to RGB reduces the filesize!
        masked_image = np.bitwise_and(orig_image_bgr, cv2.cvtColor(save_mask_uint8, cv2.COLOR_GRAY2BGR))
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
        masked_image[:, :, 3] = save_mask_uint8

        # Save data!
        ok_img_save, save_img_path = save_image(masked_image, image_path, save_folder)
        ok_mask_save, save_mask_path = save_image(save_mask_uint8, image_path, save_folder, append_to_name="_mask")
        if any((ok_img_save, ok_mask_save)):
            print("", "SAVED:", save_img_path, sep="\n")
            if ok_mask_save:
                print(save_mask_path)

    pass

# Clean up windows
cv2.destroyAllWindows()
