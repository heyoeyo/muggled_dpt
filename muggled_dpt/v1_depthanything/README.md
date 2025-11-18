# v1 Depth-Anything

This folder contains the major pieces of the Depth-Anything DPT model, in correspondence with figure 1 from the preprint: ["Vision Transformers for Dense Prediction"](https://arxiv.org/abs/2103.13413), along with modifications specific to the Depth-Anything implementation, from ["Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"](https://arxiv.org/abs/2401.10891). This model uses DINOv2 from ["DINOv2: Learning Robust Visual Features without Supervision"](https://arxiv.org/abs/2304.07193) as it's image encoder, which has a few unique differences compared to other DPT models, and seems to produce much higher quality results.

For a more compreshensive description of the DPT model components, please see the existing documentation describing the original [DPT implementation](https://github.com/heyoeyo/muggled_dpt/tree/main/lib). The focus here is on describing the details which are unique to the Depth-Anything models.


## Differences from Original DPT

### Patch Embedding Model

The patch embedding for Depth-Anything is notably different from the original MiDaS implementation in that it uses a patch size of 14 pixels, instead of 16 pixels. It also uses a base sizing of 518 pixels, which results in an odd number of patches: 37x37. While these sound like small details, the differences neccessitate a change to the fusion and head models.


### Image Encoder Model

Compared to other DPT models, the Depth-Anything implementation has, by far, the simplest image encoder. The encoder is based off of the [DINOv2](https://arxiv.org/abs/2304.07193) models ([without registers](https://arxiv.org/abs/2309.16588)). This model is seemingly identical to the original ViT structure, introduced in "[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)", just with a different (improved?) approach to training the model.

<p align="center">
  <img src=".readme_assets/image_encoder_depth_anything.svg" alt="Diagram of the Depth-Anything image encoder, showing different distribution of stage outputs">
</p>

For use within the DPT structure, the image encoder needs to be modified to output 4 sets of tokens. For other implementations, these 4 outputs come from intermediate layers of the transformer, more or less evenly spaced throughout. Unique to the Depth-Anything variant, the outputs come from the very last 4 layers of the transformer instead of being spread out like in other models. This implies that the first stage includes far more unique processing steps than the other stages.

### Reassembly Model

The Depth-Anything reassembly model very closely matches the original implementation, except for one small change. In the original [MiDaS preprint](https://arxiv.org/abs/2103.13413) (see pages 3 & 4), the authors experimented with different ways of handling the 'readout' token (also called the cls token) and found that a special projection mapping gave the best results. While the Depth-Anything implementation _does_ include a readout token, the reassembly model simply ignores it instead of combining it in with other tokens. It's worth noting that the original MiDaS preprint reported only small differences in model performance when comparing the different ways of handling the readout token (see table 7 of the paper), with the 'ignore' approach even being superior on some datasets.


### Fusion Model

Structurally, the fusion model exactly matches the original DPT implementation. However, the (odd-valued) default patch sizing of 37x37 is not neatly divisble by a factor of 2 (which occurs within the reassembly model), and this causes problems within the fusion model, which can no longer assume that reassembly feature maps can be upscaled by 2 to match one another when fusing them together. In the original Depth-Anything implementation, this problem is handled by [interpolating the feature maps](https://github.com/LiheYoung/Depth-Anything/blob/e7ef4b4b7a0afd8a05ce9564f04c1e5b68268516/depth_anything/dpt.py#L127C1-L130C61) inside the fusion model. While this is a very flexible way to handle the problem, in the implementation in this repo, the patch embedding is simply forced to output evenly-sized patch grids (e.g. 36x36 or 38x38) so that they are divisible by two. This makes the code directly compatible with the other existing DPT implementations.


### Monocular Depth Head Model

The head of the Depth-Anything model again differs slightly from the original DPT implementation due to the 14 pixel patch size. In the original implementation, the fusion model output a feature map 8 times larger than the initial patch embedding size in both height and width. The head model further upscaled this by a factor of 2, resulting in a total of 16x upscaling which would undo the 16x downscaling effect of the initial patch embedding due to the use of a 16 pixel patch size. The Depth-Anything fusion model is still designed to output a feature map 8 times larger than the patch embedding, but the head model needs to produce an output 14 times larger in order to undo the effect of the 14 pixel patch sizing. As a result, the head model has to upscale by 1.75 times (8 * 1.75 = 14) rather than doubling (as in the original DPT implementation). Aside from the different scaling factor, the Depth-Anything head model is otherwise identical to the original implementation.