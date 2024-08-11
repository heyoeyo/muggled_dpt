#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import torch
import cv2
from lib.make_dpt import make_dpt_from_state_dict


# Define pathing
image_path = "/path/to/image.jpg"
model_path = "/path/to/model.pth"

# Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Error loading image from: {image_path}")

# Load model & prepare input image
print("Loading model...")
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path)
img_tensor = dpt_imgproc.prepare_image_bgr(img_bgr)

# Process image using each of the major model components
print("Processing image data...")
with torch.inference_mode():
    patch_tokens, patch_grid_hw = dpt_model.patch_embed(img_tensor)
    imgenc_1, imgenc_2, imgenc_3, imgenc_4 = dpt_model.imgencoder(patch_tokens, patch_grid_hw)
    reasm_1, reasm_2, reasm_3, reasm_4 = dpt_model.reassemble(imgenc_1, imgenc_2, imgenc_3, imgenc_4, patch_grid_hw)
    fused_feature_map = dpt_model.fusion(reasm_1, reasm_2, reasm_3, reasm_4)
    inverse_depth_tensor = dpt_model.inference(img_tensor)

# Feedback
print("")
print("Input image shape:", tuple(img_bgr.shape))
print("Pre-encoded image shape:", tuple(img_tensor.shape))
print("")
print("Patch grid height & width", tuple(patch_grid_hw))
print("Patch embedding shape:", tuple(patch_tokens.shape))
print("Image encoding stage 1 shape:", tuple(imgenc_1.shape))
print("Image encoding stage 4 shape:", tuple(imgenc_4.shape))
print("")
print("Reassembly 1 result shape:", tuple(reasm_1.shape))
print("Reassembly 4 result shape:", tuple(reasm_4.shape))
print("")
print("Fused feature map shape:", tuple(fused_feature_map.shape))
print("Final output shape:", tuple(inverse_depth_tensor.shape))
print("")
print("Model config:")
print(*[f"  {k}: {v}" for k, v in model_config_dict.items()], sep="\n")
