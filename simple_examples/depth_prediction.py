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

import cv2
from lib.make_dpt import make_dpt_from_state_dict


# Define pathing
image_path = "/path/to/image.jpg"
model_path = "/path/to/model.pth"

# Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Error loading image from: {image_path}")

# Process data
print("Loading model & computing inverse depth...")
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path)
img_tensor = dpt_imgproc.prepare_image_bgr(img_bgr)
inverse_depth_prediction = dpt_model.inference(img_tensor)

# Feedback
print("")
print("Input image shape:", tuple(img_bgr.shape))
print("Pre-encoded image shape:", tuple(img_tensor.shape))
print("Result shape:", tuple(inverse_depth_prediction.shape))
print("Result min:", float(inverse_depth_prediction.min()))
print("Result max:", float(inverse_depth_prediction.max()))
print("")
print("Model config:")
print(*[f"  {k}: {v}" for k, v in model_config_dict.items()], sep="\n")
