#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import os
import argparse
from time import perf_counter, sleep

import cv2
import numpy as np
import torch

from lib.make_dpt import make_dpt_from_state_dict

import lib.demo_helpers.toadui as ui
from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.postprocess import scale_prediction, convert_to_uint8, histogram_equalization
from lib.demo_helpers.misc import (
    DeviceChecker,
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
default_video_path = None
default_model_path = None
default_display_size = 600
default_display_ms = 1
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to run MiDaS DPT depth-estimation on a video")
parser.add_argument("-i", "--video_path", default=default_video_path, help="Path to video to run depth estimation on")
parser.add_argument("-m", "--model_path", default=default_model_path, help="Path to DPT model weights")
parser.add_argument(
    "-s",
    "--display_size",
    default=default_display_size,
    type=int,
    help="Controls initial size of displayed results (default: {})".format(default_display_size),
)
parser.add_argument(
    "-t",
    "--display_ms",
    default=default_display_ms,
    type=int,
    help="Time to display each frame. Set to 0 to use the video FPS",
)
parser.add_argument(
    "-sync",
    "--force_sync",
    default=False,
    action="store_true",
    help="Force synchronous GPU usage, so that every frame of video is processed",
)
parser.add_argument(
    "-d",
    "--device",
    default=default_device,
    type=str,
    help="Device to use when running model (ex: 'cpu', 'cuda', 'mps')",
)
parser.add_argument(
    "-nc", "--no_cache", default=False, action="store_true", help="Disable caching to reduce VRAM usage"
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
    help="Process the video at it's original aspect ratio, if the model supports it",
)
parser.add_argument(
    "-b",
    "--base_size_px",
    default=default_base_size,
    type=int,
    help="Override base (e.g. 384, 512) model size. Must be multiple of 32",
)
parser.add_argument(
    "-cam",
    "--use_webcam",
    default=False,
    action="store_true",
    help="Use a webcam as the video input, instead of a file",
)
parser.add_argument(
    "-r",
    "--allow_recording",
    default=False,
    action="store_true",
    help="Enables toggle-able recording of per-frame depth predictions",
)

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
prefer_bfloat16 = not args.prefer_unstable_f16

use_optimizations = not args.no_optimization
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
use_webcam = args.use_webcam
allow_recording = args.allow_recording

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

# Set up device config
device_config_dict = make_device_config(device_str, use_float32, prefer_bfloat16)
device_stream = DeviceChecker(device_str)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Load model
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model = make_dpt_from_state_dict(model_path, use_cache, use_optimizations)
dpt_model.to(**device_config_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Video setup & feedback

# Set up access to video
vreader = ui.LoopingVideoReader(video_path, display_size_px)
video_frame_delay_ms = vreader.get_frame_delay_ms() if (display_ms_override == 0) else max(1, int(display_ms_override))
sample_frame = vreader.get_sample_frame()

# Get example frame so we can provide sizing info feedback
example_prediction = dpt_model.inference(sample_frame, model_base_size, force_square_resolution)
print_config_feedback(model_path, device_config_dict, use_cache, example_prediction)

# For feedback
model_name, devdtype_str, header_color = make_header_strings(model_path, device_config_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Run model & Display results

# Control element for adjusting video playback
playback_slider = ui.VideoPlaybackSlider(vreader)

# Build image elements
img_elem = ui.FixedARImage(sample_frame)
depth_elem = ui.FixedARImage(sample_frame)
text_olay = ui.TextOverlay(img_elem, (0, 0), scale=0.75, thickness=2, offset_xy_px=(5, 5))

# Build upper control elements
spectral_cmap = ui.colormaps.make_spectral_colormap()
cmap_bar = ui.ColormapsBar(cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_TWILIGHT, spectral_cmap, None)
reverse_colors_btn = ui.ToggleButton("[r] Reverse Colors", color_on=(110, 80, 95))
high_contrast_btn = ui.ToggleButton("[h] High Contrast", color_on=(100, 95, 80))
async_btn = ui.ToggleButton("[n] Async", default_state=not force_sync, color_on=(95, 100, 110))
record_btn = ui.ToggleButton("[space] Record", default_state=False)

# Controls for adjusting processing size
initial_size = max(example_prediction.shape)
min_size, max_size = max(min(64, initial_size), 32), max(1280, initial_size)
imgsize_slider = ui.Slider("Image Size", initial_size, min_size, max_size, step=16, marker_step=256)
use_ar_btn = ui.ToggleButton(" AR ", default_state=not force_square_resolution)
resolution_txt = ui.TextBlock("0000x0000")

# Set up saving location if recording
save_folder = None
if allow_recording:
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


# Build full UI
header_bar = ui.MessageBar(model_name, devdtype_str, color=header_color)
img_swap = ui.Swapper(ui.HStack(text_olay, depth_elem), ui.VStack(text_olay, depth_elem))
ui_layout = ui.VStack(
    header_bar,
    ui.HStack(reverse_colors_btn, high_contrast_btn, record_btn if allow_recording else async_btn),
    cmap_bar,
    ui.HStack(imgsize_slider, use_ar_btn, resolution_txt, flex=(1, 0, 0)),
    img_swap,
    playback_slider,
    ui.MessageBar(
        "[space] Play/pause",
        "[-, =] To change resize",
        "[esc, q] To quit",
        color=header_color,
        text_scale=0.35,
        use_equal_width=True,
        height=20,
    ),
)

# Toggle layout for wide vs. tall images
if sample_frame.shape[0] * 2 < sample_frame.shape[1]:
    img_swap.next()


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Main loop ***

# Set up display window and attach UI for mouse interactions
window = ui.DisplayWindow(display_fps=120)
window.enable_size_control(display_size_px, minimum=ui_layout.get_min_hw().h)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Switch display layout": {"l": img_swap.next},
        "Play/Pause the video": {"SPACEBAR": vreader.toggle_pause},
        "Step video backwards/forwards": {"L_ARROW": vreader.prev_frame, "R_ARROW": vreader.next_frame},
        "Toggle recording": {"p": record_btn.toggle} if allow_recording else None,
        "Toggle reverse colors": {"r": reverse_colors_btn.toggle},
        "Toggle high contrast": {"h": high_contrast_btn.toggle},
        "Toggle async": {"n": async_btn.toggle} if not allow_recording else None,
        "Toggle processing aspect ratio": {"a": use_ar_btn.toggle},
        "Cycle colormapping": {"TAB": cmap_bar.next},
        "Adjust image processing size": {"[": imgsize_slider.decrement, "]": imgsize_slider.increment},
    }
).report_keypress_descriptions()

# Pre-define values that appear in async block, to make sure they exist before being used
depth_uint8 = np.zeros(vreader.shape[0:2], dtype=np.uint8)
depth_color = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
t_ready_last, time_ms_model, last_frame_idx = perf_counter(), 0, -1

with window.auto_close(vreader.release):

    for is_paused, frame_idx, frame in vreader:
        playback_slider.update_state(is_paused, frame_idx)

        # Read controls
        is_hc_changed, use_high_contrast = high_contrast_btn.read()
        is_rev_changed, use_reverse_colors = reverse_colors_btn.read()
        is_cmap_changed, _, _ = cmap_bar.read()
        is_size_changed, img_size = imgsize_slider.read()
        is_ar_changed, use_aspect_ratio = use_ar_btn.read()
        _, use_async = async_btn.read()
        _, enable_video_recording = record_btn.read()

        # Only process frame data when the device is ready
        is_settings_changed = any((is_hc_changed, is_rev_changed, is_cmap_changed, is_size_changed, is_ar_changed))
        is_new_frame = frame_idx != last_frame_idx
        if (is_new_frame or is_settings_changed) and device_stream.is_ready():

            # Approximate time needed by the model by the time needed to get to this conditional check
            # Note: This ends up including frame display time! Can be very inaccurate with slower fps
            time_ms_model = 1000 * (perf_counter() - t_ready_last)
            t_ready_last = perf_counter()

            # Run model and get prediction for display
            prediction = dpt_model.inference(frame, img_size, not use_aspect_ratio)

            # Prepare depth data for display
            scale_wh = img_elem.get_render_hw()[::-1]
            scaled_prediction = scale_prediction(prediction, scale_wh)
            depth_tensor = convert_to_uint8(scaled_prediction).to("cpu", non_blocking=use_async)
            depth_uint8 = depth_tensor.squeeze().numpy()

            # Provide more accurate timing when sync'd
            if not use_async:
                time_ms_model = 1000 * (perf_counter() - t_ready_last)

            # Produce colored depth image for display
            if use_reverse_colors:
                depth_uint8 = 255 - depth_uint8
            if use_high_contrast:
                depth_uint8 = histogram_equalization(depth_uint8)
            depth_color = cmap_bar.apply_colormap(depth_uint8)

            # Handle video recording
            if enable_video_recording:

                # Build save pathing
                frame_idx = vreader.get_playback_position(normalized=False)
                save_name = f"{frame_idx:0>8}.png"
                save_path = os.path.join(save_folder, save_name)

                # Create frame for saving (matched to some of the display settings)
                save_frame = convert_to_uint8(prediction).to("cpu").squeeze().numpy()
                if use_reverse_colors:
                    save_frame = 255 - save_frame
                if use_high_contrast:
                    save_frame = histogram_equalization(save_frame)
                cv2.imwrite(save_path, save_frame)

            # Report resolution
            if is_size_changed or is_ar_changed:
                resolution_txt.set_text(f"{prediction.shape[2]}x{prediction.shape[1]}")

            # Draw image/depth map with inference time
            infer_txt = f"{'async' if use_async else 'sync'}: {time_ms_model:.1f}ms"
            text_olay.set_text(infer_txt)
            last_frame_idx = frame_idx

        # Update displayed image & render
        img_elem.set_image(frame)
        depth_elem.set_image(depth_color)
        display_image = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_image, frame_delay_ms=None if is_new_frame else 25)
        if req_break:
            break

        pass
    pass
