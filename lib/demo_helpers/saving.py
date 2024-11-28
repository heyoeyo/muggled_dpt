#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp
import datetime as dt

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def save_image(image_bgr, save_name, save_folder="saved_images", append_to_name=None):
    
    # Strip off pathing/ext, in case we get a full path for the input name
    file_name = osp.basename(save_name)
    name_only, _ = osp.splitext(file_name)
    if append_to_name is not None:
        name_only = f"{name_only}{append_to_name}"
    
    # Generate timestamp, so user can save the same image name multiple times without overwriting
    save_time = dt.datetime.now().isoformat(timespec = "seconds").replace(":", "").replace("T", "_")
    save_name = "{}_{}.png".format(name_only, save_time)
    
    # Try saving the image if possible
    ok_save = False
    save_path = osp.join(save_folder, save_name)
    try:
        os.makedirs(save_folder, exist_ok=True)
        cv2.imwrite(save_path, image_bgr)
        ok_save = True
        
    except Exception as err:
        print("",
              "Error trying to save depth image...",
              str(err),
              sep="\n", flush=True)
    
    return ok_save, save_path


def save_numpy_array(numpy_array, save_path):
    
    """
    Helper used to save full numpy data (as opposed to uint8 image data)
    Expects to get a full save path (likely from saving an image first),
    automatically adds the .npy file extension.
    """
    
    # Strip off any provided file extension and add .npy
    save_path_noext = os.path.splitext(save_path)[0]
    save_path_npy = f"{save_path_noext}.npy"
    
    # Try to save the numpy data (as-is)
    ok_save = False
    try:
        np.save(save_path_npy, numpy_array)
        ok_save = True
        
    except Exception as err:
        print("",
              "Error trying to save numpy array...",
              str(err),
              sep="\n", flush=True)
    
    return ok_save, save_path_npy


def save_uint16(numpy_array, save_path):
    
    """
    Helper used to save a uint16 'image' as a .png file, which provides
    higher resolution than the normal uint8 format. Expects to get
    a full save path (from saving a normal image first). Automatically
    suffixes with '_uint16.png' and handles 0-65535 value scaling.
    """
    
    # Strip off any provided file extension and add uint16 indicator
    save_path_noext = os.path.splitext(save_path)[0]
    save_path_uint16 = f"{save_path_noext}_uint16.png"
    
    # Normalize into uint16 range
    min_val, max_val = np.min(numpy_array), np.max(numpy_array)
    norm_array = ((numpy_array - min_val) / max(max_val - min_val, 1E-3))
    uint16_data = (norm_array * 65535.0).astype(np.uint16)
    
    # Try saving the uint16 image if possible
    ok_save = False
    try:
        cv2.imwrite(save_path_uint16, uint16_data)
        ok_save = True
        
    except Exception as err:
        print("",
              "Error trying to save uint16 image...",
              str(err),
              sep="\n", flush=True)
    
    return ok_save, save_path_uint16
