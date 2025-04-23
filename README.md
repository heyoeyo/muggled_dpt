# Muggled DPT

This repo contains a simplified implementation of the very cool 'dense prediction transformer' (DPT) depth estimation model from [isl-org/MiDaS](https://github.com/isl-org/MiDaS), with the intention of [removing the magic](https://en.wikipedia.org/wiki/Muggle) from the original code. Most of the changes come from eliminating dependencies as well as adjusting the code to more directly represent the model architecture as described in the preprint: ["Vision Transformers for Dense Prediction"](https://arxiv.org/abs/2103.13413). It also supports [Depth-Anything V1](https://github.com/LiheYoung/Depth-Anything) and [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) models, which use the same DPT structure.

<p align="center">
  <img src=".readme_assets/turtle_anim.gif" style="height:280px">
</p>

While the focus of this implementation is on readability, there are also some performance improvements with MiDaS v3.1 models (40-60% on my GPU at least) due to caching of positional encodings, at the cost of higher VRAM usage (this can be disabled).

### Purpose

The purpose of this repo is to provide an easy to follow code base to understand how the DPT & image encoder models are structured. The scripts found in the [simple_examples](https://github.com/heyoeyo/muggled_dpt/blob/main/simple_examples) folder are a good starting point if you'd like to better understand how to make use of the DPT models. The [run_image.py](https://github.com/heyoeyo/muggled_dpt/blob/main/run_image.py) demo script is a good example of a more practical use of the models.

To understand the model structure, there's a [written walkthrough](https://github.com/heyoeyo/muggled_dpt/tree/main/lib#dpt-structure) explaining each of the DPT components. It's also worth checking out the code implementation of the [DPT module](https://github.com/heyoeyo/muggled_dpt/blob/main/lib/dpt_model.py), I'd recommended comparing this to the information in the [original preprint](https://arxiv.org/abs/2103.13413), particularly figure 1 in the paper.

To get a better sense of what these models are actually doing internally, check out the [experiments](https://github.com/heyoeyo/muggled_dpt/tree/main/experiments).


## Getting started

This repo includes three demo scripts, [run_image.py](https://github.com/heyoeyo/muggled_dpt/blob/main/run_image.py), [run_video.py](https://github.com/heyoeyo/muggled_dpt/blob/main/run_video.py) and [run_3dviewer.py](https://github.com/heyoeyo/muggled_dpt/blob/main/run_3dviewer.py) (along with several [experimental scripts](https://github.com/heyoeyo/muggled_dpt/tree/main/experiments)). To use these scripts, you'll need to first have [Python](https://www.python.org/) (v3.10+) installed, then set up a virtual environment and install some additional requirements.


### Install
First create and activate a virtual environment (do this inside the repo folder after [cloning/downloading](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) it):
```bash
# For linux or mac:
python3 -m venv .env
source .env/bin/activate

# For windows (cmd):
python -m venv .env
.env\Scripts\activate
```

Then install the requirements (or you could install them manually from the [requirements.txt](https://github.com/heyoeyo/muggled_dpt/blob/main/requirements.txt) file):
```bash
pip install -r requirements.txt
```
(if you have an existing virtual environment or conda env for Pytorch projects, it may work without requiring installation)

<details>
<summary>Additional info for GPU usage</summary>

If you're using Windows and want to use an Nvidia GPU or if you're on Linux and don't have a GPU, you'll need to use a slightly different install command to make use of your hardware setup. You can use the [Pytorch installer guide](https://pytorch.org/get-started/locally/) to figure out the command to use. For example, for GPU use on Windows it may look something like:
```bash
pip3 uninstall torch  # <-- Do this first if you already installed from the requirements.txt file
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

**Note**: With the Windows install as-is, you may get an error about a `missing c10.dll` dependency. Downloading and installing this [mysterious .exe file](https://aka.ms/vs/16/release/vc_redist.x64.exe) seems to fix the problem.

</details>

### Model Weights

Before you can run a model, you'll need to download it's weights.

This repo supports the [BEiT](https://arxiv.org/abs/2106.08254) and [SwinV2](https://arxiv.org/abs/2111.09883) models from [MiDaS v3.1](https://arxiv.org/abs/2307.14460) which can be downloaded from the [isl-org/MiDaS releases page](https://github.com/isl-org/MiDaS/releases/tag/v3_1). Additionally, [DINOv2](https://arxiv.org/abs/2304.07193) models are supported from [Depth-Anything V1](https://arxiv.org/abs/2401.10891) and [Depth-Anything V2](https://arxiv.org/abs/2406.09414), which can be downloaded from the [LiheYoung/Depth-Anything](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) and [Depth-Anything/Depth-Anything-V2](https://huggingface.co/collections/depth-anything/depth-anything-v2-666b22412f18a6dbfde23a93) repos on Hugging Face, respectively. Note that only the relative depth models are support by this repo (however, Depth-Anything V2 metric models can be loaded in most cases).

After downloading a model file, you can place it in the `model_weights` folder of this repo or otherwise just keep note of the file path, since you'll need to provide this when running the demo scripts. If you do place the file in the [model_weights](https://github.com/heyoeyo/muggled_dpt/tree/main/model_weights) folder, then it will auto-load when running the scripts.

<details>

<summary>Direct download links</summary>

The table below includes direct download links to all of the supported models. **Note:** These are all links to _other_ repos, none of these files belong to MuggledDPT!

| Depth-Anything V2 | Size (MB) |
| ----------------- | --------- |
| [depth-anything-v2-vit-small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) | 99 |
| [depth-anything-v2-vit-base](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) | 390 |
| [depth-anything-v2-vit-large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) | 1340 |

| Depth-Anything V1 | Size (MB) |
| ----------------- | --------- |
| [depth-anything-v1-vit-small](https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth?download=true) | 99 |
| [depth-anything-v1-vit-base](https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth?download=true) | 390 |
| [depth-anything-v1-vit-large](https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth?download=true) | 1340 |

| Swin V2 | Size (MB) |
| ------- | --------- |
| [swin2-tiny-256](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt) | 164 |
| [swin2-base-384](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt) | 416 |
| [swin2-large-384](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt) | 840 |

| BEiT | Size (MB) |
| ---- | --------- |
| [beit-base-384](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt) | 456 |
| [beit-large-384](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt) | 1340 |
| [beit-large-512](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt) | 1470 |

</details>

> [!Tip]
> If you're not sure which model to use, pick one of the Depth-Anything V2 models as they consistently outperform BEiT or SwinV2. The [base model](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#pre-trained-models) is a good balance between speed and accuracy.


### Simple example

Here is an [example](https://github.com/heyoeyo/muggled_dpt/blob/main/simple_examples/depth_prediction.py) of using the model to generate an inverse depth map from an image:

```python
import cv2
from lib.make_dpt import make_dpt_from_state_dict

# Load image & model
img_bgr = cv2.imread("/path/to/image.jpg")
model_config_dict, dpt_model, dpt_imgproc = make_dpt_from_state_dict("/path/to/model.pth")

# Process data
img_tensor = dpt_imgproc.prepare_image_bgr(img_bgr)
inverse_depth_prediction = dpt_model.inference(img_tensor)
```

## Run Image

<p align="center">
  <img src=".readme_assets/run_image_anim.gif">
</p>

The `run_image.py` script will run the depth prediction model on a single image. To use the script, make sure you've activated the virtual environment (from the installation step) and then, from the repo folder use:
```bash
python run_image.py
```
You can also add  `--help` to the end of this command to see a list of additional flags you can set when running this script. One especially interesting flag is `-b`, which allows for processing images at higher resolutions.

If you don't provide an image path (using the `-i` flag), then you will be asked to provide one when you run the script, likewise for a path to the model weights. Afterwards, a window will pop-up, with various sliders that can be used to modify the depth visualization. These let you adjust the contrast of the depth visualization, as well as remove a plane-of-best-fit, which can often remove the 'floor' from the depth prediction. You can press `s` to save the current depth image.

## Run Video (or webcam)

<p align="center">
  <img src=".readme_assets/run_video_anim.gif">
</p>

The `run_video.py` script will run the depth prediction model on individual frames from a video. To use the script, again make sure you're in the activated virtual environment and then from the repo folder use:
```bash
python run_video.py
```
As with the image script, you can add `--help` to the end of this command to see a list of additional modifiers flags you can set. For example, you can use a webcam as input using the flag `--use_webcam`. It's possible to record video results (per-frame) using the `--allow_recording` flag.

When processing video, depth predictions are made _asynchrounously_, (i.e. only when the GPU is ready to do more processing). This leads to faster playback/interaction, but the depth results may appear choppy. You can force synchrounous playback using the `-sync` flag or toggling the option within the UI (this also gives more accurate inference timing results).

**Note:** The original DPT implementation is not designed for consistency across video frames, so the results can be very noisy looking. If you actually need video depth estimation, consider [Consistent Depth of Moving Objects in Video](https://dynamic-video-depth.github.io/) and the listed related works.


## Run 3D Viewer

<p align="center">
  <img src=".readme_assets/run_3dviewer_anim.gif">
</p>

The `run_3dviewer.py` script will start a local server running a DPT model, which can provide image and depth data to the browser for 3D rendering. To use the script, make sure you're in the activated virtual environment, then from the repo folder run the command:
```
python run_3dviewer.py -l
```
The `-l` flag will launch a browser window, you can leave this out if you prefer to open the page manually. As with the other scripts, you can add `--help` to the end of this command to see the other modifier flags that are available. For example, you can change the `--host` to `"0.0.0.0"` to make the server available to other devices on your network (including smartphones), though this is only recommended if you're on a trusted network! The server supports images, video files and even webcams (using the `--use_webcam` flag when launching the script).

The web interface provides support for visualizing depth data as a 3D model by displacing a dense plane mesh using depth predictions. There are controls for setting the scaling factors needed to [properly interpret](https://github.com/heyoeyo/muggled_dpt?tab=readme-ov-file#note-on-depth-results) the (inverse relative) depth estimate. There are also some controls bound to keypresses, for example the `~` key, which enables recording. An `info` page is available (via a link on the UI) that explains the controls and other available options in more detail.

> [!Note]
> In an attempt to minimize the dependencies for this repo, the server is very simple and is written in plain python. As a result, it has minimal features, efficiency and security. It's built purely for use as a local-running demo and is not suitable for deployment in a production environment!


## Note on depth results

The DPT models output results which are related to the _multiplicative inverse_ (i.e. `1/d`) of the true depth! As a result, the closest part of an image will have the _largest_ reported value from the DPT model and the furthest part will have the _smallest_ reported value. Additionally, the reported values will not be distributed linearly, which will make the results look distorted if interpretted geometrically (e.g. as a 3D model).

If you happen to know what the _true_ minimum and maximum depth values are for a given image, you can compute the true depth from the DPT result using:

$$\text{True Depth} = \left [ V_{norm} \left ( \frac{1}{d_{min}} - \frac{1}{d_{max}} \right ) + \frac{1}{d_{max}} \right ] ^{-1}$$

Where d<sub>min</sub> and d<sub>max</sub> are the known minimum and maximum (respectively) true depth values and V<sub>norm</sub> is the DPT result normalized to be between 0 and 1 (a.k.a the normalized inverse depth).

For more information, please see the [results explainer](https://github.com/heyoeyo/muggled_dpt/blob/main/.readme_assets/results_explainer.md)


# Acknowledgements

The code in this repo is based on code from the following sources.

[isl-org/MiDaS](https://github.com/isl-org/MiDaS):
```bibtext
@article {Ranftl2022,
    author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}

@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ICCV},
	year      = {2021},
}

@article{birkl2023midas,
      title={MiDaS v3.1 -- A Model Zoo for Robust Monocular Relative Depth Estimation},
      author={Reiner Birkl and Diana Wofk and Matthias M{\"u}ller},
      journal={arXiv preprint arXiv:2307.14460},
      year={2023}
}
```

[rwightman/pytorch-image-models](https://github.com/huggingface/pytorch-image-models/tree/v0.6.12) (aka timm, specifically v0.6.12):
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

[LiheYoung/Depth-Anything (v1)](https://github.com/LiheYoung/Depth-Anything):
```bibtex
@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```

[DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2):
```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2):
```bibtext
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

# TODOs
- Inevitable bugfixes