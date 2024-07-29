#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp
import argparse

import cv2
import torch
import numpy as np

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
from lib.demo_helpers.saving import save_image
from lib.demo_helpers.visualization import add_bounding_box, grid_stack_by_columns_first
from lib.demo_helpers.text import TextDrawer
from lib.demo_helpers.model_capture import ModelOutputCapture
from lib.demo_helpers.misc import (
    get_default_device_string, make_device_config, print_config_feedback
)


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_display_size = 1000
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to visualize internal (DPT) transformer block norms")
parser.add_argument("-i", "--image_path", default=default_image_path,
                    help="Path to image to run depth estimation on")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str,
                    help="Path to DPT model weights")
parser.add_argument("-s", "--display_size", default=default_display_size, type=int,
                    help="Controls size of displayed results (default: {})".format(default_display_size))
parser.add_argument("-d", "--device", default=default_device, type=str,
                    help="Device to use when running model (ex: 'cpu', 'cuda', 'mps')")
parser.add_argument("-ar", "--use_aspect_ratio", default=False, action="store_true",
                    help="Process the image at it's original aspect ratio, if the model supports it")
parser.add_argument("-b", "--base_size_px", default=default_base_size, type=int,
                    help="Override base (e.g. 384, 512) model size")
parser.add_argument("-nocol", "--no_colormap", default=False, action="store_true",
                    help="Output black-and-white results (instead of colormap)")
parser.add_argument("-l", "--headless", default=False, action="store_true",
                    help="Turns off the display pop-up (helpful on headless systems)")
parser.add_argument("--save", default=False, action="store_true",
                    help="Save block norm image result")

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
display_size_px = args.display_size
device_str = args.device
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px
enable_display = not args.headless
enable_save = args.save
colormap_select = None if args.no_colormap else cv2.COLORMAP_VIRIDIS

# Hard-code no-cache usage, since there is no benefit if the model only runs once
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
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# Hard-code (shared) display styling
TXT_LARGE = TextDrawer(font=cv2.FONT_HERSHEY_PLAIN)
TXT_SMALL = TextDrawer.create_from_existing(TXT_LARGE).adjust_scale(0.8)
BG_BGR = (40,40,40)


def get_name(name_or_path):
    ''' Given a file/folder path, returns just the file or folder name, with no file extension '''
    return osp.splitext(osp.basename(name_or_path))[0]

def get_norm_2d(tokens, grid_hw):
    
    '''
    Takes a 'rows-of-tokens' input and outputs an image-like shape
    Note: This removes the cls/readout token, if present from index 0
          Will also downscale the target height & width if the tokens
          are sized as an integer multiple smaller than the given grid_hw
          (this is specifically meant to accomodate swinv2 block outputs)
    '''
    
    h, w = grid_hw
    num_expected_tokens = h*w
    num_actual_tokens = tokens.shape[1]
    
    # Remove readout/cls token, if present
    has_readout_token = (num_actual_tokens == (1 + num_expected_tokens))
    out = tokens[:, 1:, :] if has_readout_token else tokens
        
    # For swinv2
    if num_actual_tokens != (1 + num_expected_tokens):
        scale_factor = int(round((num_expected_tokens/num_actual_tokens) ** 0.5))
        grid_hw, orig_hw = [v//scale_factor for v in grid_hw], grid_hw
        print(f"Unexpected token size! Expected: {list(orig_hw)} using {list(grid_hw)} ")
    
    # Convert from rows-of-tokens to image-like tokens
    out = torch.transpose(out, 1, 2)
    out = torch.unflatten(out, 2, grid_hw).squeeze().float()
    
    return out.norm(dim=0)

def get_norm_image(tokens, grid_hw):
    
    '''
    Takes a 'rows-of-tokens' input and outputs the (0-to-1 normalized) 
    L2 norm of the tokens, in an image-like shape
    '''
    
    raw_norm = get_norm_2d(tokens, grid_hw)
    
    min_norm = raw_norm.min()
    max_norm = raw_norm.max()
    norm = (raw_norm - min_norm) / (max_norm - min_norm)
    norm = norm.cpu().numpy()
    norm_uint8 = np.uint8(np.round(255 * norm))
    
    return norm_uint8, min_norm, max_norm

def add_minmax_footer(image_bgr, min_norm, max_norm, footer_height=16):
    
    ''' Takes an image and adds an extra space with text: "n: [min norm, max norm]" to the bottom '''
    
    min_n, max_n = sorted([min_norm.cpu().numpy(), max_norm.cpu().numpy()])
    min_n = float(min_n)
    max_n = float(max_n)
    txt = f"n: [{round(min_n)}, {round(max_n)}]"
    
    img_w = image_bgr.shape[1]
    footer = np.full((footer_height, img_w, 3), BG_BGR, dtype=np.uint8)
    footer = TXT_SMALL.xy_norm(footer, txt, (0,0.5), pad_xy_px=(5,0))
    
    return np.vstack((image_bgr, footer))

def add_block_idx_footer(image_bgr, index, footer_height=20):
    
    ''' Takes an image and adds an extra space with text: "Block: #" to the bottom '''
    
    img_w = image_bgr.shape[1]
    footer = np.full((footer_height, img_w, 3), BG_BGR, dtype=np.uint8)
    footer = TXT_LARGE.xy_norm(footer, f"Block: {index}", (0,0.5), pad_xy_px=(5,0))
    
    return np.vstack((image_bgr, footer))

def add_model_info_header(image_bgr, model_path_or_name, grid_hw, header_height=40):
    
    ''' Takes in an image and adds an info header bar to the image '''
    
    model_name = get_name(model_path_or_name)
    grid_txt = f"{grid_hw[0]} x {grid_hw[1]}"
    header_txt = f"{model_name} ({grid_txt})"
    
    img_w = image_bgr.shape[1]
    header = np.full((header_height, img_w, 3), BG_BGR, dtype=np.uint8)
    header = TXT_LARGE.xy_norm(header, header_txt, (0,0.5), pad_xy_px=(10,0))
    
    return np.vstack((header, image_bgr))


# ---------------------------------------------------------------------------------------------------------------------
#%% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache, strict_load=True)
if model_base_size is not None:
    dpt_imgproc.set_base_size(model_base_size)

# Move model to selected device
device_config_dict = make_device_config(device_str, use_float32)
dpt_model.to(**device_config_dict)
dpt_model.eval()

# Load image and apply preprocessing
orig_image_bgr = cv2.imread(image_path)
img_tensor = dpt_imgproc.prepare_image_bgr(orig_image_bgr, force_square = force_square_resolution)
print_config_feedback(model_path, device_config_dict, use_cache, img_tensor)


# ---------------------------------------------------------------------------------------------------------------------
#%% Run model

# Figure out which type of block we're looking to hook
model_name = osp.basename(model_path)
if "beit" in model_name:
    from lib.v31_beit.image_encoder_model import TransformerBlock as TargetBlock
elif "swin2" in model_name:
    from lib.v31_swinv2.image_encoder_model import SwinTransformerBlock as TargetBlock
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
with torch.inference_mode():
    img_tensor = img_tensor.to(**device_config_dict)
    tokens, grid_hw = dpt_model.patch_embed(img_tensor)
    dpt_model.imgencoder(tokens, grid_hw)

# Bail if we didn't capture anything
num_blocks = len(captures)
if num_blocks == 0:
    module_name = TargetBlock.__name__
    raise AttributeError(f"No data captured! Model doesn't contain '{module_name}' module?")


# ---------------------------------------------------------------------------------------------------------------------
#%% Figure out display sizing

# Get one of the captured outputs, so we can use it as reference for sizing
example_image, _, _ = get_norm_image(captures[0], grid_hw)
tile_h, tile_w = example_image.shape[:2]

# Figure out the best row/col layout for display
target_ar = 2
row_col_options = [(k, num_blocks//k) for k in range(1, num_blocks) if (num_blocks % k) == 0]
ar_match = [abs(target_ar - (w*tile_w)/(h*tile_h)) for h,w in row_col_options]
best_match = min(ar_match)
best_match_idx = ar_match.index(best_match)
num_rows, num_cols = row_col_options[best_match_idx]

# Figure out (integer) tile scaling to give large/readable image result
target_max_side_px = display_size_px
max_side_px = max(num_rows * tile_h, num_cols * tile_w)
scale_factor = max(1, int(np.floor(target_max_side_px / max_side_px)))
out_wh = (int(scale_factor * tile_w), int(scale_factor * tile_h))


# ---------------------------------------------------------------------------------------------------------------------
#%% Generate block-norm images

# Render out a 'tile' image for each captured block
tile_imgs_list = []
for block_idx, result in enumerate(captures):
    
    # Convert tokens back to image representation
    norm_img, min_norm, max_norm = get_norm_image(result, grid_hw)
    norm_img = cv2.resize(norm_img, dsize=out_wh, interpolation=cv2.INTER_NEAREST_EXACT)
    
    # Convert to color image
    if colormap_select is None:
        color_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    else:
        color_img = cv2.applyColorMap(norm_img, colormap_select)
    
    # Add header/footer info, per block before storing
    color_img = add_block_idx_footer(color_img, block_idx)
    color_img = add_minmax_footer(color_img, min_norm, max_norm)
    color_img = add_bounding_box(color_img)
    tile_imgs_list.append(color_img)

# Combine each row along with title bar to form final image
all_tiles_img = grid_stack_by_columns_first(tile_imgs_list, num_cols)
final_img = add_model_info_header(all_tiles_img, model_path, grid_hw)

# Display results!
if enable_display:
    winname = "Block Norms - q to quit"
    cv2.namedWindow(winname, flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow(winname, 500, 50)
    cv2.imshow(winname, final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------------
#%% Saving results

if enable_save:
    model_name = get_name(model_path)
    img_name = get_name(image_path)
    img_size = img_tensor.shape[2]
    
    if input("Save block norm image? [y/N] ").strip().lower().startswith("y"):
        save_name = f"{model_name}-{img_name}-{img_size}"
        ok_save, save_path = save_image(final_img, save_name, save_folder=save_folder)
        if ok_save: print("", "SAVED:", save_path, sep = "\n")
    
    pass
