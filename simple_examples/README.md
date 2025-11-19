# Simple Examples

This folder contains scripts with the minimal amount of code needed to use the DPT models. They include hard-coded pathing to an input image and/or model, typically at the top of the script, which will need to be setup for the scripts to work properly.


## depth_prediction.py

This script show the most basic usage of producing an inverse depth prediction from an image. Basic information about the output (and model config) is printed out.

## internal_features.py

This script runs each of the separate stages of the DPT model, showing how each stage feeds into the other. This isn't normally how the DPT models would be used, but may be useful if attempting to intercept or modifiy the internal results. For more information about the DPT model structure, see the [written walkthrough](https://github.com/heyoeyo/muggled_dpt/tree/main/muggled_dpt#dpt-structure).