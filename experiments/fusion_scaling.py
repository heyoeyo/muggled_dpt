#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp

import argparse
from time import perf_counter

import cv2
import numpy as np
import torch

# This is a hack to make this script work from inside the experiments folder!
try:
    import lib # NOQA
except ModuleNotFoundError:
    import sys
    parent_folder = osp.dirname(osp.dirname(__file__))
    if "lib" in os.listdir(parent_folder): sys.path.insert(0, parent_folder)
    else: raise ImportError("Can't find path to lib folder!")

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.ui import SliderCB, ColormapButtonsCB, ButtonBar, ScaleByKeypress
from lib.demo_helpers.visualization import DisplayWindow, histogram_equalization
from lib.demo_helpers.saving import save_image, save_numpy_array, save_uint16
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

# Hard-code no-cache usage, since there is no benefit if the model only runs once
use_cache = False

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Build pathing to repo-root, so we can search model weights properly
root_path = osp.dirname(osp.dirname(__file__))
save_folder = osp.join(root_path, "saved_images", "fusion_scaling")

# Create history to re-use selected inputs
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)

# Improve cpu utilization
reduce_overthreading(device_str)


# ---------------------------------------------------------------------------------------------------------------------
#%% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache)
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

# Run model partially to get intermediate tokens for scaling
print("", "Computing reassembly results...", sep="\n", flush=True)
img_tensor = img_tensor.to(**device_config_dict)
with torch.inference_mode():
    patch_tokens, patch_grid_hw = dpt_model.patch_embed(img_tensor)
    imgenc_tokens = dpt_model.imgencoder(patch_tokens, patch_grid_hw)
    reasm_tokens = dpt_model.reassemble(*imgenc_tokens, patch_grid_hw)

t2 = perf_counter()
print("  -> Took", round(1000*(t2-t1), 1), "ms")


# ---------------------------------------------------------------------------------------------------------------------
#%% Display results

# Set up button controls
btnbar = ButtonBar()
toggle_reverse_color = btnbar.add_toggle("[r] Reversed", "[r] Normal Order", keypress="r", default=False)
toggle_high_contrast = btnbar.add_toggle("[h] High Contrast", "[h] Normal Contrast", keypress="h", default=False)
btn_save = btnbar.add_button("[s] Save", keypress="s")

# Set up other UI elements
gray_cmap = ColormapButtonsCB.make_gray_colormap()
spec_cmap = ColormapButtonsCB.make_spectral_colormap()
cmap_btns = ColormapButtonsCB(cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TWILIGHT, spec_cmap, gray_cmap)
sliders = [SliderCB(f"Fusion {1+idx}", 1, -5, 5, 0.01, marker_step_size=1) for idx in range(4)]
display_scaler = ScaleByKeypress()

# Set up window with controls
cv2.destroyAllWindows()
window = DisplayWindow("Fusion Scaling Result - q to quit")
window.set_callbacks(btnbar, cmap_btns, *sliders)

# Feedback about controls
print("", "Displaying results",
      "  - Drag bars to change fusion scaling factors",
      "  - Right click on bars to reset values",
      "  - Use up/down arrow keys to adjust display size",
      "  - Press esc or q to quit",
      "",
      sep="\n", flush=True)

while True:
    
    # Read controls
    scale_factors = [s.read() for s in sliders]
    use_high_contrast = toggle_high_contrast.read()
    use_reverse_colors = toggle_reverse_color.read()
    
    # Run remaining layers with scaling factors
    with torch.inference_mode():
        
        # Run fusion steps manually, so we can apply scaling factors
        fuse_3 = dpt_model.fusion.blocks[3](reasm_tokens[3] * scale_factors[3])
        fuse_2 = dpt_model.fusion.blocks[2](reasm_tokens[2], fuse_3 * scale_factors[2])
        fuse_1 = dpt_model.fusion.blocks[1](reasm_tokens[1], fuse_2 * scale_factors[1])
        fuse_0 = dpt_model.fusion.blocks[0](reasm_tokens[0], fuse_1 * scale_factors[0])
        depth_prediction = dpt_model.head(fuse_0).squeeze(dim=1)
    
    # Post-processing for display
    scaled_prediction = dpt_imgproc.scale_prediction(depth_prediction, disp_wh)
    depth_norm = dpt_imgproc.normalize_01(scaled_prediction).float().cpu().numpy().squeeze()
    
    # Produce colored depth image for display
    depth_uint8 = np.uint8(np.round(255.0*depth_norm))
    if use_high_contrast: depth_uint8 = histogram_equalization(depth_uint8)
    if use_reverse_colors: depth_uint8 = 255 - depth_uint8
    depth_color = cmap_btns.apply_colormap(depth_uint8)
    
    # Generate display image: buttons / colormaps / side-by-side images / sliders
    sidebyside_img = display_scaler.resize(np.hstack((scaled_input_img, depth_color)))
    display_frame = btnbar.draw_standalone(sidebyside_img.shape[1])
    display_frame = cmap_btns.append_to_frame(display_frame)
    display_frame = np.vstack((display_frame, sidebyside_img))
    display_frame = SliderCB.append_many_to_frame(display_frame, *sliders)
    
    # Display final image
    window.imshow(display_frame)
    req_break, keypress = window.waitKey(20)
    if req_break:
        break
    
    # Handle keypresses
    display_scaler.on_keypress(keypress)
    btnbar.on_keypress(keypress)
    if btn_save.read():
        
        # Apply modifications to raw prediction for saving
        npy_prediction = dpt_imgproc.normalize_01(depth_prediction.clone()).float().cpu().numpy().squeeze()
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

# Clean up windows
cv2.destroyAllWindows()
