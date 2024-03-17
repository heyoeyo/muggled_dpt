# Muggled DPT - Experiments

This folder contains random experiments using the DPT models, mostly out of curiosity.

## Block Norm Visualization

This script was made to visualize the [L2 norms](https://builtin.com/data-science/vector-norms) (i.e. lengths) of the tokens from internal transformer blocks of the DPT models. This is specifically due to the paper: "[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)", which suggests that there are unusually high-norm 'patches' within vision transformers when they do not include additional global tokens (similar to a class token), called registers. None of the DPT models (so far) include these registers, and so we'd expect to see the high-norm tokens (which we do!).

<p align="center">
  <img src=".readme_assets/block_norm_example.webp" alt="Plot of image patch token norms output from each block of the Depth-Anything ViT-L model. The minimum and maximum norm values are printed below each block number. Easily visible outliers appear on block 17 onwards.">
</p>

Running this script generates an image showing a plot of the norms of the image patches (tokens) from each transformer block. The high-norm tokens are easily visible in multiple models, usually on the later internal blocks. The minimum and maximum norm values are also printed out for each block, showing norms often in the 100's (rule of thumb suggests they should be around 1).

## Fusion Scaling

This script originally started as a test to see what would happen if some of the [4 fusion layers](https://github.com/heyoeyo/muggled_dpt/tree/main/lib#fusion-model) of the DPT model were disabled. Based on the DPT structure and the fact that the upper fusion layers are downscaled, it stands to reason that the upper layers should encode large scale patterns, while lower layers encode finer details. In practice something more complicated seems to be happening, and it varies (significantly) by model.

<p align="center">
  <img src=".readme_assets/fusion_scaling_example.webp" alt="">
</p>

Rather than simply disabling fusion layers, this script allows for each layer to be individually scaled (potentially to 0, which disables the layer) while observing the depth result. Interestingly, some of the fusion layers seem to have a kind of 'blurring' effect on the output. Reducing the impact of these blurring layers can lead to results that have far greater details than the normal model output, though this does seem to come at the expense of the correctness of the depth mapping for larger-scale details.

This script can also generate some wild looking [glitch art](https://en.wikipedia.org/wiki/Glitch_art) when using larger and/or negative scaling factors!


## Export Onnx

This script was made to help export DPT models into the [ONNX](https://onnx.ai/onnx/intro/concepts.html) format. These files fully contain the model execution instructions along with the model weights, which makes them simpler to deploy in applications. They can also be used outside of Python/Pytorch.

So far, the script seems to support Depth-Anything & BEiT DPT models and works across variable input image sizes. However, the SwinV2 models can only be exported for fixed-sized input images, as support for different image sizes relies on python control flow statements that do not seem to be compatible with the onnx exporter (it may be possible to fix this at some point). Also note that in order to run this script, you need to install onnx and (to also test the model works) onnxruntime. This can be done using pip:

`pip install onnx==1.15.* onnxruntime==1.17.*`

If you export an onnx model, you can view the onnx structure using tools, like [netron](https://netron.app/)
