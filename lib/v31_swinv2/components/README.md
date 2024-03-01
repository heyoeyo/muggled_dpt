# v3.1 SwinV2 Model Components

This folder contains smaller sub-components of the SwinV2 DPT model. It also contains a folder of helper modules which are not specific to SwinV2, but are used throughout various model components.

## Patch Merging

Patch merging is a new idea introduced in the SwinV1 paper, which occurs at the start of each of the image encoder stages except for the very first. Merging has the effect of reducing the number of tokens, which speeds up model execution. This process also leads to the Swin model having a hierarchical output, where the outputs at different stages can be thought of as representing finer or coarser level details from the original input image.

Merging works somewhat unusually, but looks a bit similar to the patch embedding step. First, we group image patches into separate 2x2 tiles (so 4 patches each), tiling the whole image. For each of these tiles, we can say that there is a top-left, top-right, bottom-left and bottm-right patch.

From here, we construct 4 new 'images', one made out of only the top-left patches, one made of the top-right patches, one from the bottom-left and a last image from the bottom-right patches. Due to this construction, each of these 4 images is exactly half the height and width of the original image. We then take these 4 images and stack them on top of each other, forming a new image, now with 4 times as many channels as the original input had.

Finally, a linear layer is applied along the channel dimension (i.e. affecting each patch independently) to reduce the channel count by a factor of 2, though this halving occurs _after_ we performed the stacking, which itself quadrupled the channel count. The net effect is that we end up with 1/4 the number of patches (tokens) due to the halving of both spatial dimensions, while doubling the final channel count compared to the input.

## Shifted-Windowed Attention

The idea of **s**hifting **win**dows is the namesake of swin, though these are two separate ideas: shifting _and_ windowing.

### Windowing

Windowing refers to performing self-attention on a smaller subset of tokens, instead of all tokens together as is normally the case. In 'normal' (non-windowed) attention, every token is compared to every other token and this leads to a [quadratic scaling](https://en.wikipedia.org/wiki/Time_complexity#Table_of_common_time_complexities) of computation. By using smaller subsets (windowing) we can reduce the computational cost of the model. Though this also reduces the information exchange among the tokens, which may affect output accuracy.

It's important to understand that all tokens still undergo some sort of attention computation with windowing, it's just that tokens are split into smaller groups that are only compared to other tokens within these smaller groups. This specifically helps reduce the impact of the quadratic scaling. For example, if we have 100 tokens, then regular attention requires 100<sup>2</sup> (= 10000) comparisons. However, if we split our 100 tokens into 20 groups (windows) of 5 tokens each, then each group only requires 5<sup>2</sup> (= 25) comparisons and we're repeating this for each of the 20 groups, for a total of 20*25 (= 500) total comparisons, a dramatic reduction!

### Shifting

One of the obvious problems with windowing is that is prevents exchange of information outside of the different windows. Shifting is a technique introduced with the Swin models in order to iteratively share information among different windows as information passes through the model. One straightforward solution would be to simply 're-draw' the window boundaries at different layers within the model.

In practice, due to performance and implementation concerns, the authors implement shifting using _cyclic shifts_. Despite how fancy this term sounds, the idea is very simple, instead of re-drawing the window boundaries we instead alter the data itself, by shifting (or sliding) the values. Data at the edges of the image are wrapped around (this is the 'cyclic' part of the term), so that patches on the far left of the image wrap around to the far right, for example.

This ends up being more computationally efficient than re-drawing the window boundaries and having to deal with partial windows near the edges of the image. It's also very easy to implement, as it directly corresponds to the [roll](https://pytorch.org/docs/stable/generated/torch.roll.html) operation within pytorch.

### Masking

The use of _cyclic_ shifting means that some image patches are wrapped around to opposite sides of the image. These wrap-around patches end up adajacent to patches that would typically be unrelated as they come from opposite sides of the image. The original authors considered it problematic to be performing the attention computation involving these wrap-around patches, so they introduce a mask to block out the wrap-around patches in their respective windows.

Computing this mask is surprisingly complicated, since it must match the shape of the attention tensor. It works by first generating an 'image' in the shape of the patch grid, and marking each of the major areas affected by shifting with a unique number. This image is then split into windows and these windows are then converted from their image-like format into a 'rows of tokens' format, where each row corresponds to a single window.

It's worth nothing that in this case the 'features' of each token are actually just the unique region numbers, which indicates where that patch originated from in the original image. The final step is to build 2D matrix, matching the attention matrix sizing, which is formed by subtracting every region number (feature value) in a given token to all other region numbers in that same token. This is done for each token independently, so that there is one attention mask for each window.

Due to the initial construction, any sections of the mask that were not in the same marked region will have a subtraction result that is non-zero. This implies that the value should be masked in the attention calculation, since the pair of numbers correspond to non-adjacent regions! So these are the patches that get masked (set to a large negative value).

## Relative Position Encoder

Please see the position encoding of the [BEiT model](https://github.com/heyoeyo/muggled_dpt/tree/main/lib/v31_beit/components#relative-position-encoder) for more information, as the SwinV2 implementation is quite similar.