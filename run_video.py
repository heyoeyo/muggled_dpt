#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import argparse
from time import perf_counter

import torch
import cv2
import numpy as np

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.ui import ColormapButtonsCB, make_message_header_image
from lib.demo_helpers.visualization import DisplayWindow, draw_corner_text, histogram_equalization
from lib.demo_helpers.video import LoopingVideoReader, PlaybackIndicatorCB
from lib.demo_helpers.misc import (
    DeviceChecker, get_default_device_string, make_device_config, print_config_feedback, reduce_overthreading
)


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_video_path = None
default_model_path = None
default_display_size = 800
default_display_ms = 1
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to run MiDaS DPT depth-estimation on a video")
parser.add_argument("-i", "--video_path", default=default_video_path,
                    help="Path to video to run depth estimation on")
parser.add_argument("-m", "--model_path", default=default_model_path,
                    help="Path to DPT model weights")
parser.add_argument("-s", "--display_size", default=default_display_size, type=int,
                    help="Controls size of displayed results (default: {})".format(default_display_size))
parser.add_argument("-t", "--display_ms", default=default_display_ms, type=int,
                    help="Time to display each frame. Set to 0 to use the video FPS")
parser.add_argument("-sync", "--force_sync", default=False, action="store_true",
                    help="Force synchronous GPU usage, so that every frame of video is processed")
parser.add_argument("-d", "--device", default=default_device, type=str,
                    help="Device to use when running model (ex: 'cpu', 'cuda', 'mps')")
parser.add_argument("-nc", "--no_cache", default=False, action="store_true",
                    help="Disable caching to reduce VRAM usage")
parser.add_argument("-f32", "--use_float32", default=False, action="store_true",
                    help="Use 32-bit floating point model weights. Note: this doubles VRAM usage")
parser.add_argument("-ar", "--use_aspect_ratio", default=False, action="store_true",
                    help="Process the video at it's original aspect ratio, if the model supports it")
parser.add_argument("-b", "--base_size_px", default=default_base_size, type=int,
                    help="Override base (e.g. 384, 512) model size. Must be multiple of 32")
parser.add_argument("-cam", "--use_webcam", default=False, action="store_true",
                    help="Use a webcam as the video input, instead of a file")

# For convenience
args = parser.parse_args()
arg_video_path = args.video_path
arg_model_path = args.model_path
display_size_px = args.display_size
display_ms_override = args.display_ms
force_sync = args.force_sync
device_str = args.device
use_cache = not args.no_cache
use_float32 = args.use_float32
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
use_webcam = args.use_webcam

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)
device_stream = DeviceChecker(device_str)

# Get pathing to resources, if not provided already
video_path = ask_for_path_if_missing(arg_video_path, "video") if not use_webcam else 0
model_path = ask_for_model_path_if_missing(__file__, arg_model_path)

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


# ---------------------------------------------------------------------------------------------------------------------
#%% Video setup & feedback

# Set up access to video
vreader = LoopingVideoReader(video_path, display_size_px)
video_frame_delay_ms = vreader.get_frame_delay_ms() if (display_ms_override == 0) else max(1, int(display_ms_override))
disp_wh = vreader.disp_wh
disp_w, disp_h = disp_wh

# Get example frame so we can provide sizing info feedback
example_frame = np.zeros(vreader.shape, dtype = np.uint8)
example_tensor = dpt_imgproc.prepare_image_bgr(example_frame, force_square_resolution)
print_config_feedback(model_path, device_config_dict, use_cache, example_tensor)


# ---------------------------------------------------------------------------------------------------------------------
#%% Run model & Display results

# Set up window with trackbar controls
cv2.destroyAllWindows()
window = DisplayWindow("Inverse Depth Result")

# Set up UI elements
cmap_btns = ColormapButtonsCB(cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TURBO)
playback_ctrl = PlaybackIndicatorCB(vreader)
window.set_callbacks(cmap_btns, playback_ctrl)

# Pre-define values that appear in async block, to make sure they exist before being used
depth_uint8 = np.zeros(vreader.shape[0:2], dtype = np.uint8)
depth_color = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
t_ready_last, time_ms_model = perf_counter(), 0

# Feedback about controls
info_msg = "[r to reverse colors]  [h for high contrast]  [n for sync]  [q to quit]"
info_img = make_message_header_image(info_msg, 2*disp_w)
use_async = not force_sync
use_reverse_colors = False
use_high_contrast = False
print("", "Displaying results",
      "  - Click & drag to move playback",
      "  - Press esc or q to quit",
      "", sep="\n", flush=True)
for frame in vreader:
    
    # Only process frame data when the device is ready
    if device_stream.is_ready():
        
        # Approximate time needed by the model by the time needed to get to this conditional check
        # Note: This ends up including frame display time! Can be very inaccurate with slower fps
        time_ms_model = 1000 * (perf_counter() - t_ready_last)
        t_ready_last = perf_counter()
        
        # Prepare image for model
        frame_tensor = dpt_imgproc.prepare_image_bgr(frame, force_square_resolution)
        frame_tensor = frame_tensor.to(**device_config_dict)
        
        # Run model and get prediction for display
        prediction = dpt_model.inference(frame_tensor)
        
        # Prepare depth data for display
        scaled_prediction = dpt_imgproc.scale_prediction(prediction, disp_wh)
        depth_tensor = dpt_imgproc.convert_to_uint8(scaled_prediction).to("cpu", non_blocking = use_async)
        depth_uint8 = depth_tensor.squeeze().numpy()
    
        # Provide more accurate timing when sync'd
        if not use_async: time_ms_model = 1000 * (perf_counter() - t_ready_last)
        
        # Produce colored depth image for display
        if use_reverse_colors: depth_uint8 = 255 - depth_uint8
        if use_high_contrast: depth_uint8 = histogram_equalization(depth_uint8)
        depth_color = cmap_btns.apply_colormap(depth_uint8)
    
    # Set up inference time text for display
    infer_txt = "inference: {:.1f}ms".format(time_ms_model)
    if not use_async: infer_txt = "{} (sync)".format(infer_txt)
    
    # Generate display image: info / colormaps / side-by-side images / playback control
    display_frame = cmap_btns.append_to_frame(info_img)
    sidebyside = draw_corner_text(np.hstack((frame, depth_color)), infer_txt)
    display_frame = np.vstack((display_frame, sidebyside))
    display_frame = playback_ctrl.append_to_frame(display_frame)
    
    # Display result
    window.imshow(display_frame)
    req_break, keypress = window.waitKey(video_frame_delay_ms)
    if req_break:
        break
    
    # Allow user to jump playback on mouse press
    playback_ctrl.change_playback_position_on_mouse_press()
    
    # Respond to keypresses
    if keypress == ord("n"):
        use_async = not use_async
        print(f"Synchronized: {not use_async}")
    if keypress == ord("r"):
        use_reverse_colors = not use_reverse_colors
        print(f"Reversed colors: {use_reverse_colors}")
    if keypress == ord("h"):
        use_high_contrast = not use_high_contrast
        print(f"High contrast: {use_high_contrast}")

# Clean up resources
vreader.release()
cv2.destroyAllWindows()

# Provide memory usage feedback, if using cuda GPU
if device_str == "cuda":
    vram_bytes = torch.cuda.memory_allocated()
    print("", f"Used {vram_bytes // 1_000_000} MB of VRAM total", "", sep = "\n")
