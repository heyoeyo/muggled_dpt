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

from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.visualization import DisplayWindow
from lib.demo_helpers.plane_fit import estimate_plane_of_best_fit
from lib.demo_helpers.saving import save_image
from lib.demo_helpers.misc import (
    get_default_device_string, make_device_config, print_config_feedback, reduce_overthreading
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
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
override_base_size = (model_base_size is not None)

# Hard-code no-cache usage, since there is no benefit if the model only runs once
use_cache = False

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image")
model_path = ask_for_model_path_if_missing(__file__, arg_model_path)

# Improve cpu utilization
reduce_overthreading(device_str)


# ---------------------------------------------------------------------------------------------------------------------
#%% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache)
if override_base_size:
    dpt_imgproc.override_base_size(model_base_size)

# Move model to selected device
dpt_model.to(**device_config_dict)
dpt_model.eval()

# Load image and apply preprocessing
orig_image_bgr = cv2.imread(image_path)
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
depth_norm = dpt_imgproc.normalize_01(scaled_prediction).float().cpu().numpy().squeeze()

t2 = perf_counter()
print("  -> Took", round(1000*(t2-t1), 1), "ms")

# Provide memory usage feedback, if using cuda GPU
if device_str == "cuda":
    vram_bytes = torch.cuda.memory_allocated()
    print("  -> Using", vram_bytes // 1_000_000, "MB of VRAM total")


# ---------------------------------------------------------------------------------------------------------------------
#%% Display results

# Define colormaps for displaying depth map
cmaps_list = [cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TURBO, None]

# Set up window with trackbar controls
cv2.destroyAllWindows()
window = DisplayWindow("Inverse Depth Result")
contrast_tbar = window.add_trackbar("High contrast", 1)
reverse_tbar = window.add_trackbar("Reverse colors", 1)
cmap_tbar = window.add_trackbar("Color map", len(cmaps_list) - 1)
ramp_tbar = window.add_trackbar("Remove ramp", 100)
mint_tbar = window.add_trackbar("Min Depth Threshold", 1000)
maxt_tbar = window.add_trackbar("Max Depth Threshold", 1000, 1000)

# Calculate a plane-of-best-fit, so we can (potentially) remove it during display
plane_depth = estimate_plane_of_best_fit(depth_norm)

# Pre-define parameters used inside conditionals
prev_plane_removal_pct = None
depth_1ch = depth_norm

print("", "Displaying results",
      "  - Press s to save depth image",
      "  - Press esc or q to quit",
      "",
      sep="\n", flush=True)
while True:
    
    # Read window trackbars
    histo_equalize = contrast_tbar.read() > 0
    reverse_colors = reverse_tbar.read() > 0
    cmap_idx = cmap_tbar.read()
    plane_removal_pct = ramp_tbar.read()
    thresh_min = mint_tbar.read() / 1000.0
    thresh_max = maxt_tbar.read() / 1000.0
    
    # Re-calculate depth image if plane removal changes
    removal_factor_changed = (plane_removal_pct != prev_plane_removal_pct)
    if removal_factor_changed:
        depth_1ch = depth_norm - (plane_depth * (plane_removal_pct/100.0))
        depth_1ch = dpt_imgproc.normalize_01(depth_1ch)
        prev_plane_removal_pct = plane_removal_pct
    
    # Make sure we actually get min < max thresholds & non-zero delta to avoid divide-by-zero
    thresh_min, thresh_max = sorted([thresh_min, thresh_max])
    thresh_delta = max(0.001, thresh_max - thresh_min)
    depth_thresholded = np.clip((depth_1ch - thresh_min) / thresh_delta, 0.0, 1.0)
    
    # Produce colored depth image for display
    depth_uint8 = np.uint8(np.round(255.0*depth_thresholded))
    if histo_equalize: depth_uint8 = cv2.equalizeHist(depth_uint8)
    if reverse_colors: depth_uint8 = 255 - depth_uint8
    depth_color = dpt_imgproc.apply_colormap(depth_uint8, cmaps_list[cmap_idx])
    
    # Display original image along with colored depth result
    sidebyside_display = np.hstack((scaled_input_img, depth_color))
    window.imshow(sidebyside_display)
    req_break, keypress = window.waitKey(20)
    if req_break:
        break
    
    # Save depth image on 's' keypress
    if keypress == ord("s"):
        ok_save, save_path = save_image(depth_color, image_path)
        if ok_save: print("SAVED:", save_path)

# Clean up windows
cv2.destroyAllWindows()
