# State-Dict Conversion (Depth-Anything)

This folder contains scripts which help with loading model weights (a.k.a. [state dictionaries](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)) for Depth-Anything models. They provide two important capabilities. One is to infer model [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) (i.e. model size and structure) directly from model weights, so that the end-user doesn't need to supply this information. The second major capability is to re-format the model weights so that they are compatible with the code implementation within this repo, which is generally different from the original Depth-Anything codebase.

## Config from State Dict

[The config script](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/v1_depthanything/state_dict_conversion/config_from_original_state_dict.py) is responsible for figuring out model hyperparameters, or in other words, how the model is configured. These parameters are what determines whether a 'small' or 'large' size of the model is instantiated. This includes things like how many [transformer blocks](https://github.com/heyoeyo/muggled_dpt/tree/main/lib#image-encoder-model) are there in the image encoder? How many channels do the image-like tokens have going into the [fusion model](https://github.com/heyoeyo/muggled_dpt/tree/main/lib#fusion-model)? Or even, what [patch size](https://github.com/heyoeyo/muggled_dpt/tree/main/lib#patch-embedding-model) does the model use?

This functionality is very low-level and almost hacky, it works by searching for very specific model weights and looking at their size/shapes to infer different properties about the model itself. While messy, the advantage of this code is that it allows models of varying sizes to all load in the same way, potentially even including model sizes that haven't been released yet!

## Convert State Dict Keys

This repo does not contain it's own copies of the Depth-Anything models and instead loads the original Depth-Anything weights. However, the coded implementation of the model is different from the original, mostly for the sake of readability, and for making the model more directly compatible with other DPT models. For example, much of the model structure in this repo is named directly after the DPT components (e.g. reassembly and fusion models), whereas the original Depth-Anything (and MiDaS models) use names like [scratch](https://github.com/LiheYoung/Depth-Anything/blob/6e780749e7772e911754a4eb00965727987f92f7/depth_anything/blocks.py#L4) and [refinenet](https://github.com/LiheYoung/Depth-Anything/blob/6e780749e7772e911754a4eb00965727987f92f7/depth_anything/dpt.py#L78C9-L81C71) to describe some of these components, which therefore need to be renamed for compatibility.

Here are examples of some of the original model weight names and the new implementation (converted) names:

| Original | Converted |
| -------- | --------- |
| pretrained.blocks.0.ls1.gamma | imgencoder.blocks.0.scale_attn | 
| depth_head.resize_layers.0.weight | reassemble.spatial_upx4.resample.1.weight |
| depth_head.scratch.refinenet1.resConfUnit1.conv1.weight | fusion.blocks.0.conv_reassembly.resconv_seq.1.weight |
| depth_head.scratch.output_conv1.weight | head.spatial_upsampler.0.weight |

If the weights aren't named to match the new implementation names _exactly_, then the model won't load. Therefore, almost all of the functions inside the [conversion script](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/v1_depthanything/state_dict_conversion/convert_original_state_dict_keys.py) are dedicated solely to handling the renaming properly, and not just changing words, but also updating the numerical indexing! Like the configuration script, this is messy code, but the advantage is that it allows for re-use of existing model weights and potentially supports newer updates to models in the future without any additional code changes.

#### Positional Encoding Weights

When it comes to the positional encodings/embeddings of the model, this repo explicitly distinguishes between the positional encoding of the readout/cls token and the positional encodings associated with image patches. This is due to how image-patch positional encodings are spatially scaled to match different input image sizes, while the cls positional encodings are not, and so it is nice to have them separated so one can be scaled without involving the other.

This split is performed by removing the original positional encoding weights, which are stored under the key: `pretrained.pos_embed` and replacing them with two new weights under keys: `posenc.cls_embedding` and `posenc.base_patch_embedding`. The cls embedding is simply the 0-th index entry in the original position encodings, while the patch embeddings are the remaining entries. This is the only part of the conversion script which directly modifies the weights themselves, rather than just renaming the labels.

## Key Regex

The [key_regex.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/v1_depthanything/state_dict_conversion/key_regex.py) script contains a collection of helper [regex](https://www.computerhope.com/jargon/r/regex.htm) functions and other string parsing functions. These are used extensively to help identify and update weight labels as described in the conversion script section above.
