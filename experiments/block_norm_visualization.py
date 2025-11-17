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
import torch
import numpy as np

from lib.make_dpt import make_dpt_from_state_dict

import lib.demo_helpers.toadui as ui
from lib.demo_helpers.toadui.helpers.sizing import get_image_hw_for_max_side_length
from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.saving import save_image, save_numpy_array
from lib.demo_helpers.model_capture import ModelOutputCapture
from lib.demo_helpers.misc import (
    get_default_device_string,
    make_device_config,
    make_header_strings,
    print_config_feedback,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_display_size = 640
default_base_size = None
default_histo_bins = 36

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to visualize internal transformer block norms")
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
    "-ar",
    "--use_aspect_ratio",
    default=False,
    action="store_true",
    help="Process the image at it's original aspect ratio, if the model supports it",
)
parser.add_argument(
    "-b", "--base_size_px", default=default_base_size, type=int, help="Override base (e.g. 384, 512) model size"
)
parser.add_argument("--h_bins", default=default_histo_bins, type=int, help="Number of bins to use for norm histograms")


# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
num_histo_bins = args.h_bins

# Hard-code f32 for accuracy and no-cache usage (limited benefit for static images)
use_float32 = True
use_cache = False

# Build pathing to repo-root, so we can search model weights properly
root_path = osp.dirname(osp.dirname(__file__))
save_folder = osp.join(root_path, "saved_images", "block_norm_images")

# Create history to re-use selected inputs
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath, allow_folders=False)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)


# ---------------------------------------------------------------------------------------------------------------------
# %% Helpers


class BlockData:
    """Convenience wrapper, used to hold block norm display data (assumes BxHxWxC data shape!)"""

    def __init__(self, block_tensor: torch.Tensor, max_token_hw: tuple[int, int]):
        block_norm = block_tensor.norm(dim=-1).squeeze(0)
        block_norm = block_norm.float().cpu().numpy()
        norm_min_val, norm_max_val = block_norm.min(), block_norm.max()
        block_norm_0to1 = (block_norm - norm_min_val) / (norm_max_val - norm_min_val)
        block_uint8 = np.round(block_norm_0to1 * 255).astype(np.uint8)

        # Special check (needed for swinv2) to force all tokens to same size, for nicer display
        blk_h, blk_w = block_uint8.shape[0:2]
        max_h, max_w = max_token_hw
        if blk_h != max_h or blk_w != max_w:
            block_uint8 = cv2.resize(block_uint8, dsize=(max_w, max_h), interpolation=cv2.INTER_NEAREST_EXACT)

        # Store data for re-use
        self._tensor: torch.Tensor = block_tensor
        self._block_hw = (int(blk_h), int(blk_w))
        self._ch_count: int = int(self._tensor.shape[-1])
        self._ch_min: float = block_tensor.min().float().cpu().numpy()
        self._ch_max: float = block_tensor.max().float().cpu().numpy()
        self._norm: np.ndarray = block_norm
        self._norm_min: float = norm_min_val
        self._norm_max: float = norm_max_val
        self._img_uint8: np.ndarray = block_uint8

    def get_hw(self) -> tuple[int, int]:
        return self._block_hw

    def get_channel_count(self) -> int:
        return self._ch_count

    def get_tensor_data(self) -> torch.Tensor:
        return self._tensor

    def get_rowcol_value(self, row_column_index: tuple[int, int] | None, channel_index: int | None) -> float:

        # Bail if invalid selection
        if row_column_index is None:
            return 0

        row_idx, col_idx = row_column_index
        if channel_index is None:

            return float(self._norm[row_idx, col_idx])

        return float(self._tensor[0, row_idx, col_idx, channel_index].float().cpu())

    def get_norm_data(self) -> tuple[np.ndarray, float, float]:
        return self._img_uint8, self._norm_min, self._norm_max

    def get_channel_data(self, channel_index: int) -> tuple[np.ndarray, float, float]:

        active_channel_data: np.ndarray = active_block_data._tensor[0, :, :, channel_index].float().cpu().numpy()
        active_channel_min: float = active_channel_data.min()
        active_channel_max: float = active_channel_data.max()

        ch_img_uint = (active_channel_data - active_channel_min) / (active_channel_max - active_channel_min)
        ch_img_uint = np.round(ch_img_uint * 255).astype(np.uint8)

        return ch_img_uint, active_channel_min, active_channel_max

    def get_histogram_data(self, channel_index: int | None = None, num_bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
        """Helper used to get histogram plot data. Returns: histogram_bins, histogram_data"""
        if channel_index is None:
            histo_bins = np.linspace(active_block_data._norm_min, active_block_data._norm_max, num_bins)
            histo_data = self._norm
        else:
            histo_bins = np.linspace(active_block_data._ch_min, active_block_data._ch_max, num_bins)
            histo_data = active_block_data._tensor[0, :, :, channel_index].float().cpu().numpy()
        return histo_bins, histo_data


def get_norm_2d(tokens: torch.Tensor, grid_hw: tuple[int, int], is_swin_v2: bool):
    """
    Takes a 'rows-of-tokens' input and outputs an image-like shape
    Note: This removes the cls/readout token, if present from index 0
          Will also downscale the target height & width if the tokens
          are sized as an integer multiple smaller than the given grid_hw
          (this is specifically meant to accomodate swinv2 block outputs)
    """

    h, w = grid_hw
    num_expected_tokens = h * w
    num_actual_tokens = tokens.shape[1]

    # Remove readout/cls token, if present
    has_readout_token = num_actual_tokens == (1 + num_expected_tokens)
    out = tokens[:, 1:, :] if has_readout_token else tokens

    # For swinv2
    if num_actual_tokens != (1 + num_expected_tokens):
        scale_factor = int(round((num_expected_tokens / num_actual_tokens) ** 0.5))
        grid_hw, orig_hw = [v // scale_factor for v in grid_hw], grid_hw
        if not is_swin_v2:
            print(f"Unexpected token size! Expected: {list(orig_hw)} using {list(grid_hw)} ")

    # Convert from rows-of-tokens to image-like tokens
    # -> Shape goes: BxNxC -> BxHxWxC
    return out.unflatten(1, grid_hw)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Load model
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model = make_dpt_from_state_dict(model_path, use_cache, strict_load=True)
dpt_model.to(**device_config_dict)

# Load image
input_image_bgr = cv2.imread(image_path)
assert input_image_bgr is not None, f"Error reading image: {image_path}"

# Get model info for feedback
model_name, devdtype_str, header_color = make_header_strings(model_path, device_config_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Capture model data

# Figure out which type of block we're looking to hook
is_swin_v2 = False
if "beit" in model_name:
    from lib.v31_beit.image_encoder_model import TransformerBlock as TargetBlock
elif "swin2" in model_name:
    from lib.v31_swinv2.image_encoder_model import SwinTransformerBlock as TargetBlock

    is_swin_v2 = True
elif "vit" in model_name:
    # Figure out if we're using v1 or v2
    from lib.v2_depthanything.image_encoder_model import TransformerBlock as TargetBlock

    is_v2 = any(isinstance(m, TargetBlock) for m in dpt_model.modules())
    if not is_v2:
        from lib.v1_depthanything.image_encoder_model import TransformerBlock as TargetBlock
else:
    raise NameError("Unknown model type! Expecting one of: {beit, swin2, vit} in model file path")

# Set up intermediate layer data capture
captures = ModelOutputCapture(dpt_model, TargetBlock)

# Run model (only up to image encoder)
print("", "Capturing image encoder results...", sep="\n", flush=True)
t1 = perf_counter()
with torch.inference_mode():
    img_tensor = dpt_model.prepare_image_bgr(input_image_bgr, model_base_size, force_square_resolution)
    tokens, grid_hw = dpt_model.patch_embed(img_tensor)
    dpt_model.imgencoder(tokens, grid_hw)
t2 = perf_counter()
print("  -> Took", round(1000 * (t2 - t1), 1), "ms")
print_config_feedback(model_path, device_config_dict, use_cache, img_tensor)

# Bail if we didn't capture anything
num_blocks = len(captures)
if num_blocks == 0:
    raise AttributeError(f"No data captured! Model doesn't contain '{TargetBlock.__name__}' module?")

# Convert tokens to 2D 'image-like' shape (otherwise, they are 'rows-of-tokens' shape)
captures = [get_norm_2d(result, grid_hw, is_swin_v2) for result in captures]


# ---------------------------------------------------------------------------------------------------------------------
# %% Compute (base) norm images

# Get information about captured token sizing (for display & control bounds)
max_token_h, max_token_w = -1, -1
channel_count_list = list()
for result_bhwc in captures:
    _, res_h, res_w, res_ch = result_bhwc.shape
    max_token_h = max(max_token_h, res_h)
    max_token_w = max(max_token_w, res_w)
    if res_ch not in channel_count_list:
        channel_count_list.append(res_ch)
max_channels = max(channel_count_list)
if len(channel_count_list) > 1:
    print("", "Multiple channel counts detected:", channel_count_list, sep="\n")

# Set up base data set for display
max_token_hw = (max_token_h, max_token_w)
norm_data_list: list[BlockData] = [BlockData(result, max_token_hw) for result in captures]


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up UI

# Create image display elements
block_ar = grid_hw[1] / grid_hw[0]
main_img = ui.FixedARImage(aspect_ratio=block_ar, resize_interpolation=cv2.INTER_NEAREST_EXACT)
grid_olay = ui.GridSelectOverlay(main_img, grid_hw, color=(255, 0, 255))
grid_olay.style.text.style.update(fg_thickness=2, bg_thickness=4)
histo_plot = ui.SimpleHistogramPlot(
    aspect_ratio=block_ar,
    include_y_axis=False,
    use_bar_plot=True,
)
cmap_bar = ui.ColormapsBar(cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_MAGMA, cv2.COLORMAP_TURBO, None)

# Set up text displays
block_txt = ui.PrefixedTextBlock("Block: ", 0)
min_txt = ui.PrefixedTextBlock("Min: ", "-")
max_txt = ui.PrefixedTextBlock("Min: ", "-")
show_orig_btn = ui.ToggleButton("Image", default_state=False, text_scale=0.35, color_on=(75, 120, 35))
show_norm_btn = ui.ToggleButton("Norm", default_state=True, text_scale=0.35, color_on=(35, 75, 200))
show_channels_btn = ui.ToggleButton("Channel", default_state=False, text_scale=0.35, color_on=(135, 105, 90))
left_text_section = ui.VStack(
    ui.HStack(block_txt, min_txt, max_txt),
    ui.HStack(show_orig_btn, show_norm_btn, show_channels_btn),
)

# Form grid of images, with aspect ratio to invert/balance the token aspect ratio
norm_btns_list = []
btns_olay_list = []
for bidx, data in enumerate(norm_data_list):
    norm_uint8, norm_min, norm_max = data.get_norm_data()
    norm_btn = ui.ToggleImageButton(
        norm_uint8,
        default_state=False,
        is_flexible_h=True,
        is_flexible_w=True,
        fill_to_fit_space=False,
        resize_interpolation=cv2.INTER_NEAREST_EXACT,
    )
    norm_btn_olay = ui.HoverLabelOverlay(norm_btn, f"B:{bidx}  [{norm_min:.0f}, {norm_max:.0f}]", xy_norm=(0.5, 0.5))
    norm_btns_list.append(norm_btn)
    btns_olay_list.append(norm_btn_olay)
norm_grid = ui.GridStack(*btns_olay_list, target_aspect_ratio=1 / block_ar)

# Create slider(s) to allow user to change which channel is being viewed
# -> Note different layers may have different channel counts, so multiple sliders are created and swapped as needed
channel_sliders_dict = {
    ch: ui.Slider("Channel Select", value=ch // 2, min_val=0, max_val=ch - 1, step=1, marker_step=ch // 4)
    for ch in channel_count_list
}
channel_slider_swap = ui.Swapper(*channel_sliders_dict.values(), keys=channel_count_list)

# Set up 'only 1 at a time' constraints on block & display buttons
radio_block = ui.RadioConstraint(*norm_btns_list)
radio_show = ui.RadioConstraint(show_orig_btn, show_norm_btn, show_channels_btn, initial_active_index=1)

# Set up toggle between histogram & feature view
feat_histo_labels = ("Features", "Histogram")
radio_histo = ui.RadioBar(*feat_histo_labels, height=40, color_on=0)
main_swap = ui.Swapper(grid_olay, histo_plot, keys=feat_histo_labels)

# Create title elements
header_bar = ui.MessageBar(model_name, devdtype_str, color=header_color)
msg_block_norms = ui.TextBlock(f"Block Norms ({num_blocks} layers)", color=(0, 0, 0), text_scale=0.5)
msg_sep = ui.HSeparator(4, color=(0, 0, 0))

# Build UI
ui_layout = ui.VStack(
    header_bar,
    cmap_bar,
    ui.HStack(
        ui.VStack(radio_histo, main_swap, left_text_section),
        msg_sep,
        ui.VStack(msg_block_norms, norm_grid),
    ),
    channel_slider_swap,
)

# Use hidden button for saving by keypress
hidden_save_btn = ui.ImmediateButton("Save")


# ---------------------------------------------------------------------------------------------------------------------
# %% *** Main display loop ***

# Setup window & callbacks
window = ui.DisplayWindow(display_fps=30)
window.enable_size_control(display_size_px, minimum=ui_layout.get_min_hw().h)
window.attach_mouse_callbacks(ui_layout)
window.attach_keypress_callbacks(
    {
        "Cycle image preview": {"SPACEBAR": radio_show.next},
        "Cycle colormapping": {"TAB": cmap_bar.next},
        "Adjust channel selection": {
            "[": lambda: channel_slider_swap.get_active_element().decrement(),
            "]": lambda: channel_slider_swap.get_active_element().increment(),
        },
        "Adjust grid layout": {",": norm_grid.increment_num_rows, ".": norm_grid.decrement_num_rows},
        "Change block selection": {
            "L_ARROW": radio_block.prev,
            "R_ARROW": radio_block.next,
            "D_ARROW": lambda: radio_block.next(norm_grid.get_num_rows_columns()[1]),
            "U_ARROW": lambda: radio_block.prev(norm_grid.get_num_rows_columns()[1]),
        },
        "Toggle histogram": {"h": radio_histo.next, "f": radio_histo.next},
        "Toggle histogram bar/line plot": {"b": histo_plot.toggle_bar_plot},
        "Switch to original image view": {"i": lambda: show_orig_btn.toggle(True)},
        "Switch to norm view": {"n": lambda: show_norm_btn.toggle(True)},
        "Switch to channel view": {"c": lambda: show_channels_btn.toggle(True)},
        "Save selected block data": {"s": hidden_save_btn.click},
    }
).report_keypress_descriptions()
print(
    "- Hover the main image to show selected cell norm or channel value",
    "- Hover block tiles to show block index & norm [min, max]",
    "- Click block tiles to view as main image",
    sep="\n",
)

# Get initial display sizing
depth_scaled_hw = get_image_hw_for_max_side_length(input_image_bgr.shape, display_size_px)
depth_scaled_wh = tuple(reversed(depth_scaled_hw))

# Initialize display data
orig_img = cv2.resize(input_image_bgr, dsize=(20 * grid_hw[1], 20 * grid_hw[0]))
main_img.set_image(orig_img)
show_orig, show_norm, show_channels = False, False, False

# Initialize block & channel selection
is_block_changed, block_idx, _ = radio_block.read()
_, _, active_channel_slider = channel_slider_swap.read()
_, channel_idx = active_channel_slider.read()
is_histogram_active = False

# Set initial data state
active_block_data: BlockData = norm_data_list[block_idx]
active_channel_count = active_block_data.get_channel_count()

# Disable initial change state for all sliders, so they don't trigger updates
radio_block.set_is_changed(True)
for chslider in channel_sliders_dict.values():
    chslider.set_is_changed(False)

# Display loop
with window.auto_close():

    while True:

        # Flag used to update data that is displayed
        is_data_changed = False

        # Read block control first (has context effects on other UI state)
        is_block_changed, block_idx, _ = radio_block.read()
        if is_block_changed:
            is_data_changed = True
            block_txt.set_text(block_idx)
            active_block_data = norm_data_list[block_idx]
            active_channel_count = active_block_data.get_channel_count()

            # Toggle to show norm image, if display was showing rgb
            if show_orig:
                radio_show.set_item(show_norm_btn)

            # Swap channel slider, if needed (e.g. for swinv2)
            channel_slider_swap.set_swap_key(active_channel_count)

            # Adjust grid selection sizing (can change on swinv2 models)
            new_block_hw = active_block_data.get_hw()
            grid_olay.set_num_rows_columns(new_block_hw)

        # Read from the appropriate slider (can change with block change!)
        _, _, active_channel_slider = channel_slider_swap.read()
        is_channel_changed, channel_idx = active_channel_slider.read()
        if is_channel_changed:
            is_data_changed = True

            # Switch to channel view if channel slider is adjusted
            if not show_channels:
                radio_show.set_item(show_channels_btn)

            # Messy! Manually update all channel sliders to the same (relative) positioning
            # -> This only matters for model with multiple channel counts, only as a user-friendly feature
            if len(channel_sliders_dict) > 1:
                channel_idx_norm = channel_idx / active_channel_count
                for chcount, chslider in channel_sliders_dict.items():
                    if chslider is not active_channel_slider:
                        chslider.set(round(channel_idx_norm * (chcount - 1)), use_as_default_value=False)
                        chslider.set_is_changed(False)
                    pass
                pass

        # Toggle feature/histogram view
        is_histo_changed, _, active_histo_swap_label = radio_histo.read()
        if is_histo_changed:
            main_swap.set_swap_key(active_histo_swap_label)
            is_histogram_active = active_histo_swap_label = "Histogram"

        # Re-color images if needed
        is_cmap_changed, _, cmap_lut = cmap_bar.read()
        if is_cmap_changed:
            for btn, data in zip(norm_btns_list, norm_data_list):
                norm_uint8, _, _ = data.get_norm_data()
                btn.set_default_image(cmap_bar.apply_colormap(norm_uint8), set_toggle_image=True)

        # Adjust which display to use for main image
        is_showtype_changed, _, active_show_btn = radio_show.read()
        if is_showtype_changed:
            show_orig = active_show_btn is show_orig_btn
            show_norm = active_show_btn is show_norm_btn
            show_channels = active_show_btn is show_channels_btn
            is_data_changed = True

        # Replace current data
        if is_data_changed:
            if show_orig:
                data_uint8 = orig_img
                _, data_min, data_max = active_block_data.get_norm_data()
            elif show_norm:
                data_uint8, data_min, data_max = active_block_data.get_norm_data()
            elif show_channels:
                data_uint8, data_min, data_max = active_block_data.get_channel_data(channel_idx)

            # Update min/max text
            min_txt.set_text(f"{data_min:.3g}")
            max_txt.set_text(f"{data_max:.3g}")

        # Update grid hover text to indicate selected cell value
        is_grid_select_changed, _, row_col_select = grid_olay.read()
        if is_grid_select_changed or is_data_changed:
            cell_value = active_block_data.get_rowcol_value(row_col_select, channel_idx if show_channels else None)
            grid_olay.set_text_overlay(f"{cell_value:.3g}" if row_col_select is not None else "")

        # Update main display
        need_main_rerender = is_data_changed or is_cmap_changed
        if need_main_rerender:
            main_display = data_uint8 if data_uint8.ndim == 3 else cmap_bar.apply_colormap(data_uint8)
            main_img.set_image(main_display)

        # Update histogram display, if active
        need_histogram_update = is_histogram_active and (is_histo_changed or is_data_changed)
        if need_histogram_update:
            histo_title, histo_bin_count, histo_ch_idx = "Norm histogram", num_histo_bins, None
            if show_channels:
                histo_title, histo_bin_count, histo_ch_idx = "Channel histogram", 4 * num_histo_bins, channel_idx
            histo_plot.set_title(histo_title)
            histo_bins, histo_data = active_block_data.get_histogram_data(histo_ch_idx, histo_bin_count)
            histo_plot.set_bins(histo_bins).set_data(histo_data)

        # Update displayed image
        display_frame = ui_layout.render(h=window.size)
        req_break, keypress = window.show(display_frame)
        if req_break:
            break

        # Handle saving
        if hidden_save_btn.read():
            active_tensor_as_nparray = active_block_data.get_tensor_data().float().cpu().numpy()
            ok_save, img_save_path = save_image(display_frame, image_path, save_folder=save_folder)
            ok_np, npy_save_path = save_numpy_array(active_tensor_as_nparray, img_save_path)
            if ok_save:
                print("", "SAVED:", img_save_path, npy_save_path, sep="\n")
        pass
    pass
