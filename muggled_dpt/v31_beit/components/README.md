# v3.1 BEiT Model Components

This folder contains smaller sub-components of the BEiT DPT model.

## Readout Projection

In addition to processing the image patch tokens, the transformer model also includes an extra token (often called the 'cls' token or in this case the 'readout' token), which is commonly meant to encode some sort of global information about the data, since it doesn't correspond to an image patch. This can be useful in classification tasks, for example. While the DPT models aren't performing classification, the original authors found that merging the readout token into the image patch tokens provided a slight accuracy improvement over other approaches. The preprint (["Vision Transformers for Dense Prediction"](https://arxiv.org/abs/2103.13413)) contains an **Ablations** section which details their testing (see table 7). 

The corresponding script in this folder implements only the projection approach, which involves concatenating the readout token to all other tokens (doubling the feature size), followed by a projection back to the original token feature size.


## Relative Position Encoder

The relative positional encoder is **by far the most complex part** of the entire BEiT DPT model! It's purpose is to provide the transformer model with information that indicates the (relative) positioning of each of the tokens with respect to one another.

The original transformer implementation (from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)) used hand-crafted 1D positional encodings, while the original ViT (from ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929)) used learned 1D positional encodings, with both models adding the positional encodings to the input tokens before processing by the transformer blocks. By contrast, the BEiT model uses learned 2D relative positional encodings which are added to the attention matrix, as if it were an 'attention bias', at every layer of the transformer model!

The original concept for relative positional encodings seems to come from the paper: ["Self-Attention with Relative Position Representations"](https://arxiv.org/abs/1803.02155), but the actual implementation found in the BEiT code uses a simplified formula which matches that of the Swin transformer, from: ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/pdf/2103.14030.pdf) (specifically equation 4):

$$Attention(Q, K, V) = SoftMax(\frac{QK^T}{\sqrt{d}} + B)V$$
$$\text{(where B is the positional encoding 'bias' term, which is shaped NxN, for N tokens)}$$

Somewhat surprisingly (to me at least), the encodings are not learned directly as an `NxN` 2D bias matrix, as depicted in the formula above. Instead, the encodings are stored (and learned) in a small 2D lookup table. Each _row_ of the table corresponds to a unique relative position offset, for example (0,0), (+1, 0), (+2, 0), (-1, -1), etc. Each _column_ of the lookup table holds a different set of encodings for the different heads of the transformer. The 2D matrix needed in the formula above is then generated (at runtime) from this lookup table by using an _indexing matrix_ which matches the shape of the attention matrix. Each entry in the _indexing matrix_ selects the appropriate row of the lookup table, depending on which offset is needed at every entry in the attention matrix. It's a bit messy and complicated! Hopefully the following example helps, it shows the lookup table (left), indexing matrix (middle) and final positional encoding bias matrix (right):

<table align="center">
<tr><th>RelPos. Lookup Table</th><th>Indexing Matrix</th><th>Bias Result (Only head 1 & 2 shown)</th></tr>
<tr><td>

| Pos. Offset | Head 1 | Head 2 | ... | 
|-------------|:------:|:------:|:---:|
|   ( 0, 0)   |  0.10  | -0.07  |  .  |
|   (+1, 0)   | -0.55  | -0.20  |  .  |
|   (+2, 0)   |  0.73  | -0.68  |  .  |
|     ...     |    .   |   .    |  .  |
|   (-1, 0)   |  0.81  |  0.21  |  .  |
|   (-2, 0)   | -0.43  |  0.51  |  .  |
|     ...     |    .   |   .    |  .  |
|   (0, +1)   | -0.32  |  0.09  |  .  |
|  ...etc...  |    .   |   .    |  .  |

</td><td>

| Tokens |    0   |    1   |    2   | ... |
|-------:|:------:|:------:|:------:|-----|
|  **0** | ( 0,0) | (-1,0) | (-2,0) |  .  |
|  **1** | (+1,0) | ( 0,0) | (-1,0) |  .  |
|  **2** | (+2,0) | (+1,0) | ( 0,0) |  .  |
|    ... |    .   |    .   |    .   |  .  |

</td><td>

| Head 1 |       |       |       |     |
|-------:|:-----:|:-----:|:-----:|-----|
|        |  0.10 |  0.81 | -0.43 |  .  |
|        | -0.55 |  0.10 |  0.81 |  .  |
|        |  0.73 | -0.55 |  0.10 |  .  |
|        |   .   |   .   |   .   |  .  |

| Head 2 |       |       |       |     |
|-------:|:-----:|:-----:|:-----:|-----|
|        | -0.07 |  0.21 |  0.51 |  .  |
|        | -0.20 | -0.07 |  0.21 |  .  |
|        | -0.68 | -0.20 | -0.07 |  .  |
|        |   .   |   .   |   .   |  .  |

</td></tr> </table>

Note that the indexing matrix may seem incorrect since there are no +/- y offsets shown, even when moving up/down rows in the matrix. This is because the token numbering is assumed to be in row order here, so 0, 1, 2, ... refer to the first 3 image patches coming from the first row of patches. Given each token is on the same row, they're all offset from each other only in the x-direction (this is one of the more confusing patterns when looking at the way the indexing is generated).

### Storage & Representation

It's important to note that the lookup table is stored/learned for a specific value of `w` and `h` (i.e. patch grid sizing), which depends on the model. For example a grid sizing of 32x32 may be used, which corresponds to a 512x512px input image. The maximum x-offset in the table will be `w-1` (this is as far as way as one patch can be from another if there are `w` patches horizontally), and likewise the maximum y-offset is `h-1`. These offsets can also occur in the opposite direction (i.e. negative offsets), which means there is a total of `2(w-1)+1` unique x-offset values (the +1 is to account for the 0 offset), and `2(h-1)+1` unique y-offsets. So finally, we can say that the total number of unique (x,y) offset pairs is:
$$\text{Number of (x,y) pairs } = (2(w-1)+1)(2(h-1)+1)$$
$$= (2w - 1)(2h - 1)$$

So the lookup table for a 32x32 sized patch grid will contain 3969 rows. In practice, the table (and index matrix) do not explicitly represent the (x,y) offset pairs, instead all of the indexing is handled using integers, as is standard practice when working with arrays/matrices.

### The Readout (cls) Token

In addition to the image patch tokens, there is also a special 'readout' token which is processed by the transformer model. This token is therefore represented within the attention matrix of the model, and needs to be accounted for by the position encoding 'bias' matrix that gets added. However, the readout token does not have a spatial interpretation and is not part of the patch grid that the lookup table is built from. To handle this discrepancy, 3 additional entries are appended to the end of the lookup table (so that, for example, the 32x32 table ends up with 3972 entries), these represent the 'relative offset' of the `patch-to-readout` encoding, `readout-to-patch` encoding and `readout-to-readout` encoding. During the attention computation within the transformer model, the readout token is the 0th-indexed token, so it ends up affecting the 0th row and 0th column of the attention matrix. More specifically, the top row represents the `readout-to-patch` attention, the left-most column is the `patch-to-readout` attention and the top-left entry is the `readout-to-readout` attention. The indexing matrix needs to be modified to reference these special readout positional encodings, which is not shown in the simple example above, but can be seen in the source code.

### Scaling & Interpolation

While the model learns and stores a lookup table for a fixed patch grid size and therefore a fixed input image size, the implementation supports scaling to larger image sizes as well as different aspect ratios. To handle a different patch size, the lookup table is first reshaped into the original patch size (e.g. 32x32), then scaled to the new desired size (e.g. 36x48), and then turned back into a single row (per head) lookup table. This can be thought of as introducing new intermediate non-integer offset tuples like (+1.5, -0.5), which are interpolated from the original learned values. Note that the indexing table must also be scaled up to properly index the newly sized lookup table.

### Advantages/Disadvantages
The benefit of this lookup table + indexing approach is that it enforces the uniqueness of each encoding. For example, the offset (+1, 0) will be re-used by many token pairs, but exists only once in the lookup table as a learnable parameter of the model. However, this adds complexity to an already complicated system and so simplifying the implementation of positional encodings seems like one obvious area for future improvements for this model architecture.


### Caching
In the original DPT implementation, the index table is re-computed at every layer every time the model is run, and then the lookup table is re-indexed to build the bias result. This process contributes significantly to the inference time of the model (though mostly due to memory access, not computation). Caching the bias results (at every layer) results in a 1.5-2x speedup, at the cost of quite a bit of VRAM.

## Miscellaneous Helpers

The miscellaneous helpers include a spatial upsampling layer and two convolution layers.

The upsampling layer is just an interpolation function wrapped in a module to allow it to be used inside of other sequential models. It is used by both the fusion & head components of the DPT model.

The convolution layers implement 3x3 and 1x1 (i.e. channel-only) 2D convolutions, which are used in several places throughout the DPT model. Defining/naming them this way (as opposed to just using the existing [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) module) helps to make the code using these layers easier to understand at a glance.