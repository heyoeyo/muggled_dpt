# v3.1 BEiT

This folder contains the major pieces of the BEiT DPT model, in correspondence with figure 1 from the preprint: ["Vision Transformers for Dense Prediction"](https://arxiv.org/abs/2103.13413), along with BEiT-specific modifications described in the follow-up paper: ["MiDaS v3.1 -- A Model Zoo for Robust Monocular Relative Depth Estimation"](https://arxiv.org/abs/2307.14460).

The [BEiT model](https://arxiv.org/abs/2106.08254) is almost identical to the original [ViT](https://arxiv.org/abs/2010.11929) implementation. Rather than adjusting the model architecture, the BEiT paper is more to do with improvements to how the model was trained. As a result, BEiT slots nicely into the existing DPT structure with almost no modifications.

## Differences from Original DPT


### Image Encoder Model

The only difference between the BEiT image encoder and the original DPT implementation is in how the positional encodings are generated, and where they are used.

Unlike the original ViT model, BEiT uses learned [relative positional encodings](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_beit/components). It also adds these encodings to the attention matrix at _every_ transformer block, which greatly complicates it's implementation. This changes the attention calculation within each transformer block compared to the original ["Attention is all you need"](https://arxiv.org/abs/1706.03762) implementation, as follows:

$$\text{Attention}(Q, K, V) = \text{SoftMax} \left (\frac{QK^T}{\sqrt{d_{k}}} + B \right ) \times V$$

$$\text{(where B is the relative positional encoding 'bias' term)}$$

The BEiT model implementation in this repo is largely derived from the [timm](https://github.com/huggingface/pytorch-image-models/tree/7da34a999ab6f365be2ccd223c2cdcaa9a224849/timm) library, specifically the [beit.py](https://github.com/huggingface/pytorch-image-models/blob/7da34a999ab6f365be2ccd223c2cdcaa9a224849/timm/models/beit.py) code, but there also are a number of MiDaS-specific modifications which [override many](https://github.com/isl-org/MiDaS/blob/bdc4ed64c095e026dc0a2f17cabb14d58263decb/midas/backbones/beit.py) of the BEiT layers. The main reason for these modifications (which are present in this repo as well) is to allow the BEiT model to support different sized input images.