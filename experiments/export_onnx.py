#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp
import argparse
import io

from time import perf_counter

import cv2
import numpy as np
import torch

try:
    import onnx
    import onnxruntime
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException as ONNXRuntimeError

except ImportError:
    print("",
          "Error, missing onnx dependencies!",
          "Onnx export requires installing onnx and the onnx runtime (for testing)",
          "To install, use:",
          "pip install onnx==1.15.* onnxruntime==1.17.*",
          "",
          sep = "\n")
    raise SystemExit("Please install missing dependencies")

# This is a hack to make this script work from inside the experiments folder!
try:
    import lib # NOQA
except ModuleNotFoundError:
    import sys
    parent_folder = osp.dirname(osp.dirname(__file__))
    if "lib" in os.listdir(parent_folder): sys.path.insert(0, parent_folder)
    else: raise ImportError("Can't find path to lib folder!")

from lib.make_dpt import make_dpt_from_state_dict

from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from lib.demo_helpers.misc import print_config_feedback


# ---------------------------------------------------------------------------------------------------------------------
#%% Handle script args

# Set argparse defaults
default_image_path = None
default_model_path = None
default_base_size = None

# Define script arguments
parser = argparse.ArgumentParser(description="Script used to DPT depth-estimation onnx models")
parser.add_argument("-i", "--image_path", default=default_image_path,
                    help="Path to sample image for onnx export")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str,
                    help="Path to DPT model weights")
parser.add_argument("-ar", "--use_aspect_ratio", default=False, action="store_true",
                    help="Process the image at it's original aspect ratio, if the model supports it")
parser.add_argument("-b", "--base_size_px", default=default_base_size, type=int,
                    help="Override base (e.g. 384, 512) model size")

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
force_square_resolution = not args.use_aspect_ratio
model_base_size = args.base_size_px

# Hard-coded settings
device_config_dict = {"device": "cpu", "dtype": torch.float32}
use_cache = False

# Build pathing to repo-root, so we can search model weights properly
root_path = osp.dirname(osp.dirname(__file__))

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image")
model_path = ask_for_model_path_if_missing(root_path, arg_model_path)


# ---------------------------------------------------------------------------------------------------------------------
#%% Load resources

# Load model & image pre-processor
print("", "Loading model weights...", "  @ {}".format(model_path), sep="\n", flush=True)
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict(model_path, use_cache)
if (model_base_size is not None):
    dpt_imgproc.set_base_size(model_base_size)

# Load image and apply preprocessing
orig_image_bgr = cv2.imread(image_path)
img_tensor = dpt_imgproc.prepare_image_bgr(orig_image_bgr, force_square_resolution)
print_config_feedback(model_path, device_config_dict, use_cache, img_tensor)

# Move model & image to selected device
dpt_model.eval()
dpt_model.to(**device_config_dict)
img_tensor = img_tensor.to(**device_config_dict)


# ---------------------------------------------------------------------------------------------------------------------
#%% Config for export

# Tell the export which input/output axes can be resized
input_key, output_key = "input", "output"
dynamic_axes_dict = {0: "batch_size", 2: "height", 3: "width"}
io_config = {
    "input_names": [input_key],
    "output_names": [output_key],
    "dynamic_axes": {input_key: dynamic_axes_dict, output_key: dynamic_axes_dict},
}

misc_config = {
    "export_params": True,
    "opset_version": 13,
    "do_constant_folding": True,
}


# ---------------------------------------------------------------------------------------------------------------------
#%% Export

# Some feedback, since onnx can take some time
orig_model_name, _ = osp.splitext(osp.basename(model_path))
print("",
      f"Creating onnx model! ({orig_model_name})",
      "This may take a while..." ,
      "", sep = "\n", flush = True)

# Export the model (in memory only, don't want to save a file yet)
file_in_memory = io.BytesIO()
torch.onnx.export(
        model = dpt_model,
        args = img_tensor,
        f = file_in_memory,
        **io_config,
        **misc_config,
)

# Confirm export result is ok
file_in_memory.seek(0)
onnx_model = onnx.load(file_in_memory)
onnx.checker.check_model(onnx_model)

# Set up onnx model for inference test
ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------------------------------------------------
#%% Inference tests

# Try using onnx model
_, _, orig_h, orig_w = img_tensor.shape
onnx_input = {input_key: img_tensor.cpu().numpy()}
print("", f"Running onnx model... ({orig_h}x{orig_w})", sep = "\n")
ort_outs = ort_session.run(None, onnx_input)
depth_onnx = ort_outs[0]
print("  -> Success!")

# Try onnx on different aspect ratio
orig_image_bgr_diff_ar = cv2.resize(orig_image_bgr, dsize=None, fx=1.0, fy=0.75)
img_tensor_diff_ar = dpt_imgproc.prepare_image_bgr(orig_image_bgr_diff_ar, force_square=False)
_, _, alt_h, alt_w = img_tensor_diff_ar.shape
print("", f"Running on different image size... ({alt_h}x{alt_w})", sep = "\n")
try:
    ort_session.run(None, {input_key: img_tensor_diff_ar.cpu().numpy()})
    print("  -> Success!")
except ONNXRuntimeError:
    print("  -> FAILED!! Onnx model does not support alternate input sizes...")

# Display inference result using onnx model
enable_display = True
if enable_display:
    
    # Run pytorch-based model for comparison
    depth_pytorch = dpt_model.inference(img_tensor).cpu().numpy()
    
    # Create uint8 images from pytorch/onnx results for display
    to_uint8 = lambda x: np.uint8(255 * (x - x.min()) / (x.max() - x.min())).squeeze()    
    uint8_depth_ox = to_uint8(depth_onnx)
    uint8_depth_pt = to_uint8(depth_pytorch)
    
    # Show results side-by-side
    comparison_img = np.hstack((uint8_depth_pt, uint8_depth_ox))
    cv2.imshow("Depth (Pytorch vs. ONNX)", comparison_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Time model execution speed
print("", "Timing onnx model run-time...", sep = "\n")
max_test_time_sec = 2.0
t1 = perf_counter()
for iter_idx in range(100):
    ort_outs = ort_session.run(None, onnx_input)
    t2 = perf_counter()
    if (t2 - t1) > max_test_time_sec: break
depth_onnx = ort_outs[0]
t2 = perf_counter()
time_per_img_sec = (t2 - t1) / (1 + iter_idx)
print(f"  -> Took {round(1000*time_per_img_sec)} ms per image")


# ---------------------------------------------------------------------------------------------------------------------
#%% Save result

print("", "", sep = "\n", flush = True)
ask_user_save = input("Save onnx model? [y]/n: ")
enable_save = ask_user_save.strip().lower() in {"", "y", "yes"}
if enable_save:
    
    # Build save pathing
    save_name = f"{orig_model_name}.onnx"
    save_folder = osp.dirname(model_path)
    save_path = osp.join(save_folder, save_name)
    os.makedirs(save_folder, exist_ok = True)
    
    # Write in-memory data to file
    file_in_memory.seek(0)
    with open(save_path, 'wb') as file:
        file.write(file_in_memory.getvalue())
    
    # Give feedback about saving
    print("",
          "Saved onnx model!",
          f"@ {save_path}",
          sep = "\n")
    
    pass
