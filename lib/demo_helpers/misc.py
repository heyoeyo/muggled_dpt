#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp

import cv2
import torch


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DeviceChecker:
    
    '''
    Helper used to set up 'stream' which can be used to check if a device is 'ready'
    for more work (i.e. GPU is not busy). Helps to provide async use of device!
    This class provides a work-around in case the cuda-specific stream check fails
    '''
    
    def __init__(self, device_str):
        self._stream = None
        self._have_stream = False
        try:
            self._stream = torch.cuda.current_stream(device_str)
            self._have_stream = True
            
        except (AssertionError, AttributeError, ValueError):
            # Assume this happens when cuda isn't available, so use dummy that is always ready
            pass
    
    def is_ready(self):
        return self._stream.query() if self._have_stream else True


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def get_default_device_string():
    
    ''' Helper used to select a default device depending on available hardware '''
    
    # Figure out which device we can use
    default_device = "cpu"
    if torch.backends.mps.is_available():
        default_device = "mps"
    if torch.cuda.is_available():
        default_device = "cuda"
    
    return default_device

# .....................................................................................................................

def make_device_config(device_str, use_float32, use_channels_last = True, prefer_bfloat16 = True):
    ''' Helper used to construct a dict for device usage. Meant to be used with 'model.to(**config)' '''
    
    # Torch v2.0 doesn't support bfloat16 interpolation
    if torch.__version__[0:3] == "2.0":
        prefer_bfloat16 = False
    
    f16_type = torch.bfloat16 if prefer_bfloat16 else torch.float16
    fallback_dtype = torch.float32 if (device_str == "cpu") else f16_type
    dtype = torch.float32 if use_float32 else fallback_dtype
    memory_format = torch.channels_last if use_channels_last else None
    return {"device": device_str, "dtype": dtype, "memory_format": memory_format}

# .....................................................................................................................

def print_config_feedback(model_path, device_config_dict, using_cache, preproc_tensor):
    
    ''' Simple helper used to print info about model execution '''
    
    # Get config info, with graceful fail so feedback attempt doesn't crash entire system
    model_name = osp.basename(model_path)
    device_str = device_config_dict.get("device", "unknown")
    nice_dtype_str = device_config_dict.get("dtype", "unknown data type")
    nice_dtype_str = str(nice_dtype_str).split(".")[-1]
    
    # Read shape of preprocessed tensor
    try:
        _, _, model_img_h, model_img_w = preproc_tensor.shape
    except AttributeError:
        model_img_h = "?"
        model_img_w = "?"
    
    # Provide some feedback about how the model is running
    print("",
          "Config ({}):".format(model_name),
          "  Device: {} ({})".format(device_str, nice_dtype_str),
          "  Cache: {}".format(using_cache),
          "  Resolution: {} x {}".format(model_img_w, model_img_h),
          sep = "\n", flush = True)
    
    return

# .....................................................................................................................

def reduce_overthreading(device_str):
    
    '''
    Both opencv & pytorch seem to make questionable use of threading...
    This function limits the use of threading in opencv & pytorch
    
    Opencv in particular makes poor use of threads, so reducing its usage can
    help reduce CPU load, without noticable slow down.
    
    Pytorch makes good use of threading when running models on CPU. It doesn't do
    well with threading alongside GPU usage, where it seems to only increase
    CPU load without any performance benefit
    
    This function will only give threads, and only to pytorch, if the cpu
    is being used to run models
    '''
    
    # Only use threading if user if running on cpu
    is_using_cpu = device_str == "cpu"
    max_threads_available = os.cpu_count()
    target_max_threads = max_threads_available // 2
    
    # Limit thread usage between libraries
    cv2.setNumThreads(1)
    torch.set_num_threads(target_max_threads if is_using_cpu else 2)
    
    return

# .....................................................................................................................

def get_total_cuda_vram_usage_mb():
    '''Helper used to measure the total VRAM usage when using CUDA. Returns 0 if not using CUDA'''
    if torch.cuda.is_available():
        free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
        return (total_vram_bytes - free_vram_bytes) // 1_000_000
    return 0
