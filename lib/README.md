# Library

This folder contains all the code needed to load & run the DPT models.

## DPT Model

The most important file is [dpt_model.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/dpt_model.py) which is shared by all DPT implementations and is a very literal code interpretation of the model architecture described in the original DPT paper: ["Vision Transformers for Dense Prediction"](https://arxiv.org/abs/2103.13413). 

The [make_dpt.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/make_dpt.py) script is a helper used to build DPT models given model weights, without having to explicitly say which model you're trying to load. Each of the model variants have their own dedicated 'make_{variant}_dpt' script which can be used to create models more directly.


## v3.1 BEiT

The [v31_beit](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_beit) folder contains code needed to construct the DPT model components (i.e. the image encoder, reassembly, fusion and head models), specific to the BEiT model provided in the original [isl-org/MiDaS](https://github.com/isl-org/MiDaS) implementation. Much of this code is derived from the [timm](https://github.com/huggingface/pytorch-image-models/tree/7da34a999ab6f365be2ccd223c2cdcaa9a224849/timm) library, specifically the [beit.py](https://github.com/huggingface/pytorch-image-models/blob/7da34a999ab6f365be2ccd223c2cdcaa9a224849/timm/models/beit.py) code, though there are also many modifications made by the MiDaS implementation of BEiT, which [overrides many](https://github.com/isl-org/MiDaS/blob/bdc4ed64c095e026dc0a2f17cabb14d58263decb/midas/backbones/beit.py) of the BEiT layer implementations!

There is also a corresponding [make_beit_dpt.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/make_beit_dpt.py) script which is a helper used to import and instantiate all of the BEiT-specific components needed create a DPT model.


## v3.1 SwinV2

The [v31_swinv2](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_swinv2) folder, like the beit folder, contains code needed to construct the DPT model components specific to the SwinV2 models provided in the original [isl-org/MiDaS](https://github.com/isl-org/MiDaS) implementation. The SwinV2 models have more model-specific modifications due to the hierarchical design of the models.

SwinV2 models can be directly created using code from the [make_swinv2_dpt.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/make_swinv2_dpt.py) script.

## Demo Helpers

The [demo_helpers](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/demo_helpers) folder contains scripts that help implement the functionality seen in the demo scripts ([run_image.py](https://github.com/heyoeyo/muggled_dpt/blob/main/run_image.py) & [run_video.py](https://github.com/heyoeyo/muggled_dpt/blob/main/run_video.py)). These helpers are not needed to understand or use the DPT models, but may be helpful if you want to make your own image or video processing scripts.