#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import argparse
from time import perf_counter

import torch
import cv2
import numpy as np

from lib.make_dpt import make_dpt_from_midas_v31

from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.visualization import DisplayWindow, draw_corner_text
from lib.demo_helpers.video import LoopingVideoReader, PlaybackIndicatorCB
from lib.demo_helpers.misc import DeviceChecker, get_default_device_string, make_device_config, print_config_feedback


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
override_base_size = (model_base_size is not None)
use_webcam = args.use_webcam

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)
device_stream = DeviceChecker(device_str)

# Get pathing to resources, if not provided already
video_path = ask_for_path_if_missing(arg_video_path, "video") if not use_webcam else 0
model_path = ask_for_model_path_if_missing(__file__, arg_model_path)

# Libraries make poor use of threading? Reduces cpu usage with no loss of speed
cv2.setNumThreads(1)
torch.set_num_threads(1)


# ---------------------------------------------------------------------------------------------------------------------
#%% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_midas_v31(model_path, enable_relpos_cache = use_cache)
if override_base_size:
    dpt_imgproc.override_base_size(model_base_size)

# Move model to selected device
dpt_model.to(**device_config_dict)
dpt_model.eval()


# ---------------------------------------------------------------------------------------------------------------------
#%% Video setup & feedback

# Set up access to video
vreader = LoopingVideoReader(video_path, display_size_px)
video_frame_delay_ms = vreader.get_frame_delay_ms() if (display_ms_override == 0) else max(1, int(display_ms_override))
disp_wh = vreader.disp_wh

# Get example frame so we can provide sizing info feedback
example_frame = np.zeros(vreader.shape, dtype = np.uint8)
example_tensor = dpt_imgproc.prepare_image_bgr(example_frame, force_square_resolution)
print_config_feedback(model_path, device_config_dict, use_cache, example_tensor)


# ---------------------------------------------------------------------------------------------------------------------
#%% Run model & Display results

# Define colormaps for displaying depth map
cmaps_list = [cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, None]

# Set up window with trackbar controls
cv2.destroyAllWindows()
window = DisplayWindow("Inverse Depth Result")
contrast_tbar = window.add_trackbar("High contrast", 1)
reverse_tbar = window.add_trackbar("Reverse colors", 1)
cmap_tbar = window.add_trackbar("Color map", len(cmaps_list) - 1)
sync_tbar = window.add_trackbar("Force Sync", 1, int(force_sync))

# Set up playback indicator, used to control video position
playback_ctrl = PlaybackIndicatorCB(vreader)
window.set_callback(playback_ctrl)

# Pre-define values that appear in async block, to make sure they exist before being used
depth_uint8 = np.zeros(vreader.shape[0:2], dtype = np.uint8)
t_ready_last, time_ms_model = perf_counter(), 0

print("", "Displaying results",
      "  - Click & drag to move playback",
      "  - Press esc or q to quit",
      "", sep="\n", flush=True)
for frame in vreader:
    
    # Read window trackbars
    histo_equalize = contrast_tbar.read() > 0
    reverse_colors = reverse_tbar.read() > 0
    cmap_idx = cmap_tbar.read()
    use_async = sync_tbar.read() == 0
    
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
        depth_tensor = dpt_imgproc.convert_to_uint8(scaled_prediction, use_async).to("cpu", non_blocking = use_async)
        depth_uint8 = depth_tensor.squeeze().numpy()
    
    # Produce colored depth image for display
    if histo_equalize: depth_uint8 = cv2.equalizeHist(depth_uint8)
    if reverse_colors: depth_uint8 = 255 - depth_uint8
    depth_color = dpt_imgproc.apply_colormap(depth_uint8, cmaps_list[cmap_idx])
        
    # Display results
    draw_corner_text(frame, "inference: {:.1f}ms".format(time_ms_model))
    sidebyside = np.hstack((frame, depth_color))
    sidebyside = playback_ctrl.add_playback_indicator(sidebyside)
    window.imshow(sidebyside)
    req_break, _ = window.waitKey(video_frame_delay_ms)
    if req_break:
        break
    
    # Allow user to jump playback on mouse press
    playback_ctrl.change_playback_position_on_mouse_press()

# Clean up resources
vreader.release()
cv2.destroyAllWindows()
