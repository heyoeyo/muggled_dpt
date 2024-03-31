# State-Dict Conversion (BEiT)

This folder contains scripts which help with loading model weights (a.k.a. [state dictionaries](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)) for BEiT MiDaS models. They provide two important capabilities. One is to infer model [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) (i.e. model size and structure) directly from model weights, so that the end-user doesn't need to supply this information. The second major capability is to re-format the model weights so that they are compatible with the code implementation within this repo, which is generally different from the original [MiDaS codebase](https://github.com/isl-org/MiDaS/tree/master/midas).

## Config from State Dict

[The config script](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/v31_beit/state_dict_conversion/config_from_midas_state_dict.py) is responsible for figuring out model hyperparameters, or in other words, how the model is configured. These parameters are what determines whether a 'small' or 'large' size of the model is instantiated. This includes things like the number of [features per image patch](https://github.com/heyoeyo/muggled_dpt/tree/main/lib#patch-embedding-model) and the number of channels used on tokens within the [reassembly model](https://github.com/heyoeyo/muggled_dpt/tree/main/lib#reassembly-model), among other things.

This functionality works by searching for specific model weights and looking at their size/shapes to infer different properties about the model itself. While messy, the advantage of this code is that it allows models of varying sizes to all load in the same way, potentially even including model sizes that haven't been released yet!

## Convert State Dict Keys

This repo does not contain it's own copies of the BEiT models and instead loads the original weights from the [MiDaS repo](https://github.com/isl-org/MiDaS/releases/tag/v3_1). However, the coded implementation of the model is different from the original, mostly for the sake of readability, and for making the model more directly compatible with other DPT models. For example, much of the model structure in this repo is named directly after the DPT components (e.g. reassembly and fusion models), whereas the original MiDaS models use names like [scratch](https://github.com/isl-org/MiDaS/blob/bdc4ed64c095e026dc0a2f17cabb14d58263decb/midas/blocks.py#L73) and [refinenet](https://github.com/isl-org/MiDaS/blob/bdc4ed64c095e026dc0a2f17cabb14d58263decb/midas/dpt_depth.py#L101C9-L105C75) to describe some of these components, which therefore need to be renamed for compatibility.

Here are examples of some of the original model weight names and the new implementation (converted) names:

| Original | Converted |
| -------- | --------- |
| pretrained.model.blocks.11.mlp.fc1.weight | imgencoder.stages.3.blocks.2.mlp.layers.0.weight |
| pretrained.act_postprocess1.0.project.0.weight | reassemble.spatial_upx4.readout_proj.1.weight |
| scratch.refinenet3.resConfUnit1.conv2.bias | fusion.blocks.2.conv_reassembly.conv_seq.3.bias |
| scratch.output_conv.4.bias | head.proj_1ch.2.bias |

If the weights aren't named to match the new implementation names _exactly_, then the model won't load. Therefore, almost all of the functions inside the [conversion script](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/v31_beit/state_dict_conversion/convert_midas_state_dict_keys.py) are dedicated solely to handling the renaming properly. Like the configuration script, this is messy code, but the advantage is that it allows for re-use of existing model weights and potentially supports newer updates to models in the future without any additional code changes.


#### Query/Value Bias

In the original implementation, the query-key-value bias tensor used in the attention calculation is [repeatedly generated](https://github.com/huggingface/pytorch-image-models/blob/ce4d3485b690837ba4e1cb4e0e6c4ed415e36cea/timm/models/beit.py#L185) for use in the linear transformation used to generate the query/key/value tokens. This is due to the fact that the 'key' component of the bias term is meant to be [all zeros](https://github.com/huggingface/pytorch-image-models/blob/ce4d3485b690837ba4e1cb4e0e6c4ed415e36cea/timm/models/beit.py#L152), rather than being a learned component, so it cannot be directly shared with the query and value bias terms (and isn't stored in the model weights).

Regenerating this term at runtime is a bit wasteful but also confusing looking in code. For the sake of readability, in this repo, the QKV linear layer is computed with _no bias_ term, and then the query and value biases are separately added afterwards. In order to perform the addition afterwards, the bias weights (called `q_bias` and `v_bias`) need to be reshaped to match the results of the QKV linear layer, and this is handled by the conversion script. Note that this a modification to the model weights themselves, not just renaming the labels!

## Key Regex

The [key_regex.py](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/v31_beit/state_dict_conversion/key_regex.py) script contains a collection of helper [regex](https://www.computerhope.com/jargon/r/regex.htm) functions and other string parsing functions. These are used to help with updating the model weight labels, as described in the sections above.
