#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import argparse
from time import perf_counter

import cv2
import numpy as np
import torch

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.ui import SliderCB, ColormapButtonsCB, ButtonBar, ScaleByKeypress
from lib.demo_helpers.visualization import DisplayWindow, histogram_equalization
from lib.demo_helpers.plane_fit import estimate_plane_of_best_fit
from lib.demo_helpers.saving import save_image, save_numpy_array, save_uint16
from lib.demo_helpers.misc import (
    get_default_device_string, make_device_config, print_config_feedback,
    reduce_overthreading, get_total_cuda_vram_usage_mb,
)


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_display_size = 800
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to run MiDaS DPT depth-estimation on a single image")
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
parser.add_argument("-z", "--no_optimization", default=False, action="store_true",
                    help="Disable attention optimizations (only effects DepthAnything models)")
parser.add_argument("-ar", "--use_aspect_ratio", default=False, action="store_true",
                    help="Process the image at it's original aspect ratio, if the model supports it")
parser.add_argument("-b", "--base_size_px", default=default_base_size, type=int,
                    help="Override base (e.g. 384, 512) model size")

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_optimizations = not args.no_optimization
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px

# Hard-code no-cache usage, since there is no benefit if the model only runs once
use_cache = False

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)

# Improve cpu utilization
reduce_overthreading(device_str)


# ---------------------------------------------------------------------------------------------------------------------
#%% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache, use_optimizations)
if (model_base_size is not None):
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


# ---------------------------------------------------------------------------------------------------------------------
#%% Run model

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
print("  -> Took", round(1000*(t2-t1), 1), "ms")

# Provide memory usage feedback, if using cuda GPU
if device_str == "cuda":
    total_vram_mb = get_total_cuda_vram_usage_mb()
    print("  VRAM:", total_vram_mb, "MB")


# ---------------------------------------------------------------------------------------------------------------------
#%% Display results

# Calculate a plane-of-best-fit, so we can (potentially) remove it during display
plane_depth = estimate_plane_of_best_fit(depth_norm)

# Set up button controls
btnbar = ButtonBar()
toggle_reverse_color = btnbar.add_toggle("[r] Reversed", "[r] Normal Order", keypress="r", default=False)
toggle_high_contrast = btnbar.add_toggle("[h] High Contrast", "[h] Normal Contrast", keypress="h", default=False)
btn_save = btnbar.add_button("[s] Save", keypress="s")

# Set up other UI elements
gray_cmap = ColormapButtonsCB.make_gray_colormap()
spec_cmap = ColormapButtonsCB.make_spectral_colormap()
cmap_btns = ColormapButtonsCB(cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TWILIGHT, spec_cmap, gray_cmap)
plane_slider = SliderCB("Remove plane", 0, -1, 2, 0.01, marker_step_size=0.5)
min_slider = SliderCB("Min Threshold", 0, 0, 1, 0.01, marker_step_size=0.1)
max_slider = SliderCB("Max Threshold", 1, 0, 1, 0.01, marker_step_size=0.1)
display_scaler = ScaleByKeypress()

# Set up window with controls
cv2.destroyAllWindows()
window = DisplayWindow("Inverse Depth Result - q to quit")
window.set_callbacks(btnbar, cmap_btns, plane_slider, min_slider, max_slider)

# Pre-define parameters used inside conditionals
prev_plane_removal_factor = None
depth_1ch = depth_norm

# Feedback about controls
print("", "Displaying results",
      "  - Click and drag bars to adjust display",
      "  - Right click on bars to reset values",
      "  - Use up/down arrow keys to adjust display size",
      "  - Press esc or q to quit",
      "",
      sep="\n", flush=True)

while True:
    
    # Read controls
    plane_removal_factor = plane_slider.read()
    thresh_min = min_slider.read()
    thresh_max = max_slider.read()
    use_high_contrast = toggle_high_contrast.read()
    use_reverse_colors = toggle_reverse_color.read()
    
    # Re-calculate depth image if plane removal changes
    removal_factor_changed = (plane_removal_factor != prev_plane_removal_factor)
    if removal_factor_changed:
        depth_1ch = depth_norm - (plane_depth * plane_removal_factor)
        depth_1ch = dpt_imgproc.normalize_01(depth_1ch)
        prev_plane_removal_factor = plane_removal_factor
    
    # Make sure we actually get min < max thresholds & non-zero delta to avoid divide-by-zero
    thresh_min, thresh_max = sorted([thresh_min, thresh_max])
    thresh_delta = max(0.001, thresh_max - thresh_min)
    depth_thresholded = np.clip((depth_1ch - thresh_min) / thresh_delta, 0.0, 1.0)
    
    # Produce colored depth image for display
    depth_uint8 = np.uint8(np.round(255.0*depth_thresholded))
    if use_high_contrast: depth_uint8 = histogram_equalization(depth_uint8, thresh_min, thresh_max)
    if use_reverse_colors: depth_uint8 = 255 - depth_uint8
    depth_color = cmap_btns.apply_colormap(depth_uint8)
    
    # Generate display image: button controls / colormaps / side-by-side images / sliders
    sidebyside_display = display_scaler.resize(np.hstack((scaled_input_img, depth_color)))
    display_frame = btnbar.draw_standalone(sidebyside_display.shape[1])
    display_frame = cmap_btns.append_to_frame(display_frame)
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

# Clean up windows
cv2.destroyAllWindows()
