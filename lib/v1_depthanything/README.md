# v1 Depth-Anything

This folder contains the major pieces of the Depth-Anything DPT model, in correspondence with figure 1 from the preprint: ["Vision Transformers for Dense Prediction"](https://arxiv.org/abs/2103.13413), along with modifications specific to the Depth-Anything implementation, from ["Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"](https://arxiv.org/abs/2401.10891). This model uses DINOv2 from ["DINOv2: Learning Robust Visual Features without Supervision"](https://arxiv.org/abs/2304.07193) as it's image encoder, which has a few unique differences compared to other DPT models, and seems to produce much higher quality results.

For a more compreshensive description of the DPT model components, please see the existing [BEiT documentation](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_beit). The focus here is on describing the details which are unique to the Depth-Anything models.


## Patch Embedding Model

The patch embedding for Depth-Anything is notably different from both the original MiDaS implementation in that it uses a patch size of 14 pixels. It also uses a base sizing of 518 pixels, which results in an odd number of patches: 37x37. While these sound like small details, the differences have important ramifications for later parts of the model!

Aside from the different patch sizing, the patch embedding model is otherwise identical to the one found in the [BEiT implementation](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_beit#patch-embedding-model), where you can find a more detailed explanation of it's functionality.


## Image Encoder Model

Compared to other DPT models, the Depth-Anything implementation has, by far, the simplest image encoder. The encoder is based off of the [DINOv2](https://arxiv.org/abs/2304.07193) models (though [without registers](https://arxiv.org/abs/2309.16588)). This model closely follows the original ViT structure, introduced in "[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)".


## Reassembly Model

Depth-Anything uses a reassembly model that very closely matches the original implementation, with only one small change. In the original [MiDaS preprint](https://arxiv.org/abs/2103.13413), the authors experimented with different ways of handling the 'readout' token (also called the cls token) and found that a special projection mapping gave the best results. In the Depth-Anything reassembly model, the readout token is simply ignored. Given the small reported benefit in the original paper, this modification seems to make a lot of sense, leading to a simpler design and faster execution speed.


## Fusion Model

Please see the [BEiT implementation](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_beit#fusion-model) for more details on the fusion model, as the Depth-Anything version is structurally identical.

One (partial) difference is that due to the default patch sizing of 37x37, the 2x downscaling step of the reassembly model produces a feature map which, when doubled, does not return to the original 37x37 sizing. This causes problems inside the fusion model where progressively upscaling feature maps by a factor of 2 does not lead to compatible sizes needed when adding feature maps. In the original Depth-Anything implementation, this is handled by [interpolating the feature maps](https://github.com/LiheYoung/Depth-Anything/blob/e7ef4b4b7a0afd8a05ce9564f04c1e5b68268516/depth_anything/dpt.py#L127C1-L130C61) inside the fusion model. While this is a very flexible way to handle the problem, in the implementation in this repo, the patch embedding is simply forced to output evenly-sized patch grids (e.g. 36x36 or 38x38) so that they are divisible by two. This makes the code directly compatible with the other existing DPT implementations.


## Monocular Depth Head Model

The head of the Depth-Anything model differs slightly from the original DPT implementation due to the 14 pixel patch sizing used in the patch embedding step (as opposed to 16 pixels). In the original implementation, the fusion model combined with the head model have the effect of scaling the patches by a factor of 2 * 2 *2 * 2 = 16, which un-does the 16 pixel patch embedding leading to an output that matches the size of the original input. The 14 pixel patch size in Depth-Anything means that the scaling factor in the head model must be adjusted to 1.75, so that the effect of the fusion model and head are 2 * 2 * 2 * 1.75 = 14.