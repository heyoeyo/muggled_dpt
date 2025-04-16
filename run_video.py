#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import argparse
from time import perf_counter, sleep

import torch
import cv2
import numpy as np

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.ui import ColormapButtonsCB, ButtonBar, ScaleByKeypress
from lib.demo_helpers.visualization import DisplayWindow, histogram_equalization
from lib.demo_helpers.text import TextDrawer
from lib.demo_helpers.video import LoopingVideoReader, PlaybackIndicatorCB
from lib.demo_helpers.misc import (
    DeviceChecker, get_default_device_string, make_device_config, print_config_feedback,
    reduce_overthreading, get_total_cuda_vram_usage_mb,
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
parser.add_argument("-z", "--no_optimization", default=False, action="store_true",
                    help="Disable attention optimizations (only effects DepthAnything models)")
parser.add_argument("-ar", "--use_aspect_ratio", default=False, action="store_true",
                    help="Process the video at it's original aspect ratio, if the model supports it")
parser.add_argument("-b", "--base_size_px", default=default_base_size, type=int,
                    help="Override base (e.g. 384, 512) model size. Must be multiple of 32")
parser.add_argument("-cam", "--use_webcam", default=False, action="store_true",
                    help="Use a webcam as the video input, instead of a file")
parser.add_argument("-r", "--allow_recording", default=False, action="store_true",
                    help="Enables toggle-able recording of per-frame depth predictions")

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
use_optimizations = not args.no_optimization
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
use_webcam = args.use_webcam
allow_recording = args.allow_recording

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)
device_stream = DeviceChecker(device_str)

# Create history to re-use selected inputs
history = HistoryKeeper()
_, history_vidpath = history.read("video_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
video_path = ask_for_path_if_missing(arg_video_path, "video", history_vidpath) if not use_webcam else 0
model_path = ask_for_model_path_if_missing(__file__, arg_model_path, history_modelpath)

# Store history for use on reload
if use_webcam:
    # Don't save video pathing when using a webcam as input, since it isn't intuitive looking
    history.store(model_path=model_path)
else:
    history.store(video_path=video_path, model_path=model_path)


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

# Set up button controls
btnbar = ButtonBar()
toggle_normal_order_colors = btnbar.add_toggle("[r] Normal Order", "[r] Reversed", keypress="r")
toggle_normal_contrast = btnbar.add_toggle("[h] Normal Contrast", "[h] High Contrast", keypress="h")

# Use different UI if video recording is enabled
save_folder=  None
if not allow_recording:
    toggle_async = btnbar.add_toggle("[n] Async", "[n] Sync", keypress="n", default=not force_sync)
    toggle_record = btnbar.make_disabled_button(False)
    
else:
    toggle_async = btnbar.make_disabled_button(False)
    toggle_record = btnbar.add_toggle("[space] Recording", "[space] Not Recording", keypress=" ", default=False)
    
    # Create recording folder for saving video frames
    video_base_name, _ = os.path.splitext(os.path.basename(video_path))
    save_folder = os.path.join("saved_images", "video", video_base_name)
    os.makedirs(save_folder, exist_ok=True)
    
    print(
        "",
        "Recording support is enabled",
        "- Recording only occurs while the recording toggle is active!",
        "- Beware of excessive disk usage when recording for long times",
        "- Only the direct model output is recorded",
        "- Use script with -ar and -b flags to adjust sizing of saved frames",
        "- If colored data is needed, it's better to use a screen capture",
        "",
        "Results will be saved in:",
        f"  {save_folder}",
        sep="\n",
        flush=True,
    )
    sleep(3)

# Set up other UI elements
gray_cmap = ColormapButtonsCB.make_gray_colormap()
spec_cmap = ColormapButtonsCB.make_spectral_colormap()
cmap_btns = ColormapButtonsCB(cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TWILIGHT, spec_cmap, gray_cmap)
playback_ctrl = PlaybackIndicatorCB(vreader, enabled=(not use_webcam))
display_scaler = ScaleByKeypress()

# Set up window with controls
cv2.destroyAllWindows()
window = DisplayWindow("Inverse Depth Result - q to quit")
window.set_callbacks(btnbar, cmap_btns, playback_ctrl)

# Pre-define values that appear in async block, to make sure they exist before being used
depth_uint8 = np.zeros(vreader.shape[0:2], dtype = np.uint8)
depth_color = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
t_ready_last, time_ms_model = perf_counter(), 0

# Helper for drawing text
text_draw = TextDrawer(scale=0.75, thickness=2, bg_color=(0,0,0))

# Feedback about controls
print("", "Displaying results",
      "  - Click & drag to move playback",
      "  - Use up/down arrow keys to adjust display size",
      "  - Press esc or q to quit",
      "", sep="\n", flush=True)
for frame in vreader:
    
    # Read controls
    use_high_contrast = not toggle_normal_contrast.read()
    use_reverse_colors = not toggle_normal_order_colors.read()
    use_async = toggle_async.read()
    enable_video_recording = toggle_record.read()
    
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
        
        # Handle video recording
        if enable_video_recording:
            
            # Build save pathing
            frame_idx = vreader.get_playback_position(normalized=False)
            save_name = f"{frame_idx:0>8}.png"
            save_path = os.path.join(save_folder, save_name)
            
            # Create frame for saving (matched to some of the display settings)
            save_frame = dpt_imgproc.convert_to_uint8(prediction).to("cpu").squeeze().numpy()
            if use_reverse_colors: save_frame = 255 - save_frame
            if use_high_contrast: save_frame = histogram_equalization(save_frame)
            cv2.imwrite(save_path, save_frame)
    
    # Draw image/depth map with inference time
    infer_txt = "inference: {:.1f}ms".format(time_ms_model)
    if not use_async: infer_txt = "{} (sync)".format(infer_txt)
    sidebyside = display_scaler.resize(np.hstack((frame, depth_color)))
    sidebyside = text_draw.xy_norm(sidebyside, infer_txt, xy_norm=(0,0), pad_xy_px=(5,5))
    
    # Generate display image: buttons / colormaps / side-by-side images / playback control
    display_frame = btnbar.draw_standalone(sidebyside.shape[1])
    display_frame = cmap_btns.append_to_frame(display_frame)
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
    btnbar.on_keypress(keypress)
    display_scaler.on_keypress(keypress)

# Clean up resources
vreader.release()
cv2.destroyAllWindows()

# Provide memory usage feedback, if using cuda GPU
if device_str == "cuda":
    total_vram_mb = get_total_cuda_vram_usage_mb()
    print("  VRAM:", total_vram_mb, "MB")
