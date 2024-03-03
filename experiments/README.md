# Muggled DPT - Experiments

This folder contains random experiments using the DPT models, mostly out of interest or curiosity.

## Block Norm Visualization

This script was made to visualize the [L2 norms](https://builtin.com/data-science/vector-norms) (i.e. lengths) of the tokens from internal transformer blocks of the DPT models. This is specifically due to the paper: "[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)", which suggests that there are unusually high-norm 'patches' within vision transformers when they do not include additional global tokens (similar to a class token), called registers. None of the DPT models (so far) include these registers, and so we'd expect to see the high-norm tokens (which we do!).

<p align="center">
  <img src=".readme_assets/block_norm_example.webp" alt="Plot of image patch token norms output from each block of the Depth-Anything ViT-L model. The minimum and maximum norm values are printed below each block number. Easily visible outliers appear on block 17 onwards.">
</p>

Running this script generates an image showing a plot of the norms of the image patches (tokens) from each transformer block. The high-norm tokens are easily visible in multiple models, usually on the later internal blocks. The minimum and maximum norm values are also printed out for each block, showing norms often in the 100's (rule of thumb suggests they should be around 1).

## Export Onnx

This script was made to help export DPT models into the ONNX format. These files fully contain the model execution instructions along with the model weights, which makes them simpler to deploy in applications. They can also be used outside of Python/Pytorch.

So far, the script seems to support Depth-Anything & BEiT DPT models and works across variable input image sizes. However, the SwinV2 models can only be exported for fixed-sized input images, as support for different image sizes relies on python control flow statements that do not seem to be compatible with the onnx exporter (it may be possible to fix this at some point).

If you export an onnx model, you can view the onnx structure using tools, like [netron](https://netron.app/)