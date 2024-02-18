#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp

import cv2
import torch
import numpy as np

# This is a hack to make this script work from inside the analysis folder!
try:
    import lib # NOQA
except ModuleNotFoundError:
    import sys
    parent_folder = osp.dirname(osp.dirname(__file__))
    if "lib" in os.listdir(parent_folder): sys.path.insert(0, parent_folder)
    else: raise ImportError("Can't find path to lib folder!")

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.misc import make_device_config, print_config_feedback


# ---------------------------------------------------------------------------------------------------------------------
#%% Hard-coded config

# Set these to avoid being prompted for paths everytime
default_image_path = None
default_model_path = "vitl"

# Controls model execution
device_str = "cuda"
use_aspect_ratio = True
use_float32 = True
use_cache = False
model_base_size = None
colormap_select = cv2.COLORMAP_VIRIDIS

# Control script outputs
enable_display = False
save_blocknorm_image = True


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class OutputCapture:
    
    '''
    Helper used to store intermediate model results
    Example usage:
        
        from model_definition import TransformerBlock
        
        # Capture target module output
        cap = OutputCapture(TransformerBlock)
        cap.hook_outputs(model)
        
        # Run model (results are captured by 'cap')
        model(input_data)
        
        # Check out the results
        # (can also access using: cap.results)
        for result in cap:
            # do something with results...
    '''
    
    def __init__(self, target_module_type):
        self._target = target_module_type
        self.results = []
    
    def __call__(self, module, module_in, module_out):
        self.results.append(module_out)
    
    def __len__(self):
        return len(self.results)
    
    def __iter__(self):
        yield from self.results
    
    def __getitem__(self, index):
        return self.results[index]
    
    def hook_outputs(self, model):
        
        for module in model.modules():
            if isinstance(module, self._target):
                module.register_forward_hook(self)
            pass
        
        return len(self)
    
    pass


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

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

def add_minmax_footer(image_bgr, min_norm, max_norm):
    
    ''' Takes an image and adds an extra space with text: "n: [min norm, max norm]" to the bottom '''
    
    h, w = image_bgr.shape[:2]
    
    min_n, max_n = sorted([min_norm.cpu().numpy(), max_norm.cpu().numpy()])
    min_n = float(min_n)
    max_n = float(max_n)
    
    footer = np.full((16, w, 3), (50,50,0), dtype=np.uint8)
    cv2.putText(footer, f"n: [{round(min_n)}, {round(max_n)}]", (5, 10), 0, 0.35, (0,255,255))
    combined = np.vstack((image_bgr, footer))
    
    return combined

def add_block_idx_footer(image_bgr, index):
    
    ''' Takes an image and adds an extra space with text: "Block: #" to the bottom '''
    
    h, w = image_bgr.shape[:2]
    
    footer = np.full((20, w, 3), (50,50,0), dtype=np.uint8)
    cv2.putText(footer, f"Block: {index}", (5, 14), 0, 0.5, (0,255,255))
    combined = np.vstack((image_bgr, footer))
    
    return combined

def add_bounding_box(image_bgr):
    
    ''' Draws a rectangular outline around the given image '''
    
    h, w = image_bgr.shape[:2]
    return cv2.rectangle(image_bgr, (0,0), (w, h), (80,80,80), 1)

def add_model_info_header(image_bgr, model_path_or_name, grid_hw):
    
    ''' Takes in an image and adds an info header bar to the image '''
    
    model_name = get_name(model_path_or_name)
    grid_txt = f"{grid_hw[0]} x {grid_hw[1]}"
    header_txt = f"{model_name} ({grid_txt})"
    
    h, w = image_bgr.shape[:2]
    header = np.full((40, w, 3), (50,50,0), dtype=np.uint8)
    cv2.putText(header, header_txt, (10, 26), 0, 0.5, (255,255,255))
    
    return np.vstack((header, image_bgr))


# ---------------------------------------------------------------------------------------------------------------------
#%% Load resources

# Build pathing to repo-root, so we can search model weights properly
root_path = osp.dirname(osp.dirname(__file__))

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(default_image_path, "image")
model_path = ask_for_model_path_if_missing(root_path, default_model_path)

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache, strict_load=True)
if model_base_size is not None:
    dpt_imgproc.override_base_size(model_base_size)

# Move model to selected device
device_config_dict = make_device_config(device_str, use_float32)
dpt_model.to(**device_config_dict)
dpt_model.eval()

# Load image and apply preprocessing
orig_image_bgr = cv2.imread(image_path)
img_tensor = dpt_imgproc.prepare_image_bgr(orig_image_bgr, force_square = not use_aspect_ratio)
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
    from lib.v1_depthanything.image_encoder_model import TransformerBlock as TargetBlock
else:
    raise NameError("Unknown model type!")

# Set up intermediate layer data capture
capture = OutputCapture(TargetBlock)
capture.hook_outputs(dpt_model)

# Run model (only up to image encoder)
with torch.inference_mode():
    img_tensor = img_tensor.to(**device_config_dict)
    tokens, grid_hw = dpt_model.patch_embed(img_tensor)
    dpt_model.imgencoder(tokens, grid_hw)

# Bail if we didn't capture anything
num_blocks = len(capture)
if num_blocks == 0:
    module_name = TargetBlock.__name__
    raise AttributeError(f"No data captured! Model doesn't contain '{module_name}' module?")


# ---------------------------------------------------------------------------------------------------------------------
#%% Figure out display sizing

# Get one of the captured outputs, so we can use it as reference for sizing
example_image, _, _ = get_norm_image(capture[0], grid_hw)
tile_h, tile_w = example_image.shape[:2]

# Figure out the best row/col layout for display
target_ar = 2
row_col_options = [(k, num_blocks//k) for k in range(1, num_blocks) if (num_blocks % k) == 0]
ar_match = [abs(target_ar - (w*tile_w)/(h*tile_h)) for h,w in row_col_options]
best_match = min(ar_match)
best_match_idx = ar_match.index(best_match)
num_rows, num_cols = row_col_options[best_match_idx]

# Figure out (integer) tile scaling to give large/readable image result
target_max_side_px = 1000
max_side_px = max(num_rows * tile_h, num_cols * tile_w)
scale_factor = int(np.floor(target_max_side_px / max_side_px))
out_wh = (int(scale_factor * tile_w), int(scale_factor * tile_h))


# ---------------------------------------------------------------------------------------------------------------------
#%% Generate block-norm images

img_rows = []
data_chunks = [capture[k:(num_cols+k)] for k in range(0, num_blocks, num_cols)]
for row_idx, data_chunk in enumerate(data_chunks):
    
    block_idx_offset = row_idx * num_cols
    img_cols = []
    for col_idx, res in enumerate(data_chunk):
        
        # Convert tokens back to image representation
        norm_img, min_norm, max_norm = get_norm_image(res, grid_hw)
        norm_img = cv2.resize(norm_img, dsize=out_wh, interpolation=cv2.INTER_NEAREST_EXACT)
        norm_img = dpt_imgproc.apply_colormap(norm_img, colormap_select)
        
        # Add header/footer info, per block before storing
        block_idx = col_idx + block_idx_offset
        norm_img = add_block_idx_footer(norm_img, block_idx)
        norm_img = add_minmax_footer(norm_img, min_norm, max_norm)
        norm_img = add_bounding_box(norm_img)
        img_cols.append(norm_img)
    
    one_row_image = np.hstack(img_cols)
    img_rows.append(one_row_image)

# Combine each row along with title bar to form final image
all_rows_image = np.vstack(img_rows)
final_img = add_model_info_header(all_rows_image, model_path, grid_hw)

# Display results!
if enable_display:
    cv2.imshow("Block Norms", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------------
#%% Saving results

if save_blocknorm_image:
    model_name = get_name(model_path)
    img_name = get_name(image_path)
    img_size = img_tensor.shape[2]
    
    save_name = f"{model_name}-{img_name}-{img_size}.png"
    save_folder = osp.join(root_path, "saved_images", "block_norm_images")
    save_path = osp.join(save_folder, save_name)
    if input("Save block norms? [y/N] ").strip().lower().startswith("y"):
        os.makedirs(save_folder, exist_ok = True)
        cv2.imwrite(save_path, final_img)
        print("Saved block norm image:", save_path, sep="\n")
    
    pass
