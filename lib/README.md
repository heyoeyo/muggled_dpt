# Library

This folder contains all the code needed to load & run various DPT models.

## DPT Model

The most important file is [dpt_model.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/dpt_model.py) which is shared by all DPT implementations and is a very literal code interpretation of the model architecture described in the original DPT paper: ["Vision Transformers for Dense Prediction"](https://arxiv.org/abs/2103.13413). 

The [make_dpt.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/make_dpt.py) script is a helper used to build DPT models given model weights, without having to explicitly say which model you're trying to load. Each of the model variants have their own dedicated 'make_{variant}_dpt' script which can be used to create models more directly: [make_beit_dpt.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/make_beit_dpt.py), [make_beit_dpt.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/make_beit_dpt.py) and [make_depthanything_dpt](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/make_depthanything_dpt.py).

All models share the same DPT model structure code, but they implement their own versions of the DPT model components (e.g. patch embedding, image encoder, fusion etc.). The model-specific code can be found in their respective folders.


## Folders

### v3.1 BEiT

The [v31_beit](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_beit) folder contains code needed to construct the DPT model components (i.e. the patch embedding, image encoder, reassembly, fusion and head models), specific to the BEiT model provided in the original [isl-org/MiDaS](https://github.com/isl-org/MiDaS) implementation. Much of this code is derived from the [timm](https://github.com/huggingface/pytorch-image-models/tree/7da34a999ab6f365be2ccd223c2cdcaa9a224849/timm) library, specifically the [beit.py](https://github.com/huggingface/pytorch-image-models/blob/7da34a999ab6f365be2ccd223c2cdcaa9a224849/timm/models/beit.py) code, though there are also many modifications made by the MiDaS implementation of BEiT, which [overrides many](https://github.com/isl-org/MiDaS/blob/bdc4ed64c095e026dc0a2f17cabb14d58263decb/midas/backbones/beit.py) of the BEiT layer implementations!


### v3.1 SwinV2

The [v31_swinv2](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_swinv2) folder, like the BEiT folder, contains code needed to construct the DPT model components specific to the SwinV2 models provided in the original [isl-org/MiDaS](https://github.com/isl-org/MiDaS) implementation. The SwinV2 models have more model-specific modifications due to the hierarchical design of the models. The implementation is this repo also modifies some of the internal components to allow for varying input image sizes as well as adjusting the internal window sizing.


### v1 Depth-Anything

The [v1_depthanything](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v1_depthanything) folder holds code needed to construct the DPT model components for the [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) implementation, which is separate from the original MiDaS models. This model variant is much newer and generally has better performance all around compared to the v3.1 MiDaS models. It has the simplest image encoder and overall adheres closely to the original DPT structure, with only a few modifications due to the use of a 14px input patch sizing (as opposed to the 16px size used by the other models).

### Demo Helpers

The [demo_helpers](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/demo_helpers) folder contains scripts that help implement the functionality seen in the demo scripts ([run_image.py](https://github.com/heyoeyo/muggled_dpt/blob/main/run_image.py) & [run_video.py](https://github.com/heyoeyo/muggled_dpt/blob/main/run_video.py)). These helpers are not needed to understand or use the DPT models, but may be helpful if you want to make your own image or video processing scripts.

## Note on code structure

The code for each model is written in a standalone manner, so that the code for each model is almost entirely contained within their respective folders. Import statements are also written in a relative format, which means that the model folders can be copy-pasted into other projects if you'd like to experiement with them.

This goes against the convention of having [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) code and leads to lots of duplication across models. However, it results in code that has much fewer _switches_ to toggle functionality on or off depending of the requirements of different models, and is overall easier to understand (imo).