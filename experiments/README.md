# Muggled DPT - Experiments

This folder contains random experiments using the DPT models, mostly out of interest or curiosity.

# Block Norm Visualization

This script was made to visualize the [L2 norms](https://builtin.com/data-science/vector-norms) (i.e. lengths) of the tokens from internal transformer blocks of the DPT models. This is specifically due to the paper: "[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)", which suggests that there are unusually high-norm 'patches' within vision transformers when they do not include additional global tokens (similar to a class token), called registers. None of the DPT models (so far) include these registers, and so we'd expect to see the high-norm tokens (which we do!).

<p align="center">
  <img src=".readme_assets/block_norm_example.webp" alt="Plot of image patch token norms output from each block of the Depth-Anything ViT-L model. The minimum and maximum norm values are printed below each block number. Easily visible outliers appear on block 17 onwards.">
</p>

Running this script generates an image showing a plot of the norms of the image patches (tokens) from each transformer block. The high-norm tokens are easily visible in multiple models, usually on the later internal blocks. The minimum and maximum norm values are also printed out for each block, showing norms often in the 100's (rule of thumb suggests they should be around 1).