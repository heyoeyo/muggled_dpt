#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def ask_for_path(path = None, file_type = "file"):
    
    # Bail if we get a good path
    path = "" if path is None else path
    if osp.exists(path):
        return path
    
    # Keep asking for path until it points to something
    try:
        while True:
            print("", flush=True)
            path = input("Enter path to {}: ".format(file_type))
            if osp.exists(path): break
            print("",
                  "",
                  "Invalid {} path!".format(file_type),
                  sep="\n", flush=True)
        
    except KeyboardInterrupt:
        quit()
    
    return path

# .....................................................................................................................

def ask_for_model_path(file_dunder, model_path = None):
    
    # Get all existing model weight files (in weights folder)
    model_file_paths = get_model_weights_paths(file_dunder)
    
    # If we're given a model path but it doesn't exist, check if it matches a name in model weights folder
    if model_path is not None and not osp.exists(model_path):
        for each_path in model_file_paths:
            each_model_name = osp.basename(each_path)
            contains_name = model_path in each_model_name
            if contains_name:
                model_path = each_path
                break
            pass
        pass
    
    # If a model path isn't given, try to auto-load the smallest model from the weights folder
    if model_path is None and len(model_file_paths) > 0:
        model_path = min(model_file_paths, key=osp.getsize)
    
    # If we still don't have a model path, ask the user
    model_path = ask_for_path(model_path, "model weights")
    
    return model_path

# .....................................................................................................................

def get_model_weights_paths(file_dunder, model_weights_folder_name = "model_weights"):
    
    # Build path to model weight folder (and create if missing)
    script_caller_folder_path = osp.dirname(file_dunder)
    model_weights_path = osp.join(script_caller_folder_path, model_weights_folder_name)
    os.makedirs(model_weights_path, exist_ok=True)
    
    # Get only the paths to files with specific extensions
    valid_exts = {".pt", ".pth"}
    all_files_list = os.listdir(model_weights_path)
    model_files_list = [file for file in all_files_list if osp.splitext(file)[1].lower() in valid_exts]
    model_file_paths = [osp.join(model_weights_path, file) for file in model_files_list]
    
    return model_file_paths
