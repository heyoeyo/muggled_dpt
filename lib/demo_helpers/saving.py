#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp
import datetime as dt

import cv2


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def save_image(image_bgr, save_name, save_folder="saved_images"):
    
    # Strip off pathing/ext, in case we get a full path for the input name
    file_name = osp.basename(save_name)
    name_only, _ = osp.splitext(file_name)
    
    # Generate timestamp, so user can save the same image name multiple times without overwriting
    save_time = dt.datetime.now().isoformat(timespec = "seconds")
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


