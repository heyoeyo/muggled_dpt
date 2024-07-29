# Model weights folder

Model weights can be placed in this folder. If a single model is placed in this folder, it will auto-load when launching scripts. If multiple models are stored, then a menu will appear on start-up, for example:
```bash
Select model file:

  1: depth_anything_v2_vitl.pth (default)
  2: dpt_swin2_large_384.pt
  3: dpt_beit_large_384.pt

Enter selection: 
```
Entering an index (e.g. 1) will select that model from the menu. Alternatively, partial model names can also be entered (e.g 'vit_l"). If a model has been loaded previously, it will be marked as 'default' and entering nothing will result in the default being chosen. A full path to a model file can also be provided here, if loading a model that isn't available in the list.

Files in this folder can also be be referenced by partial name when using script flags. For example if you place model files: `depthanything_v2_vitl.pth`, `dpt_swin2_large_384.pt` and `dpt_beit_large_384.pt` in this folder, you can reference a specific model when launching the `run_image.py` script using something like:


```bash
python run_image.py -m thing
```

This will cause the script to load the `depthanything_v2_vitl.pth` file, since it contains 'thing' in the filename.

You can download model files from the original [isl-org/MiDaS](https://github.com/isl-org/MiDaS/releases/tag/v3_1) releases page (only _beit_ and _swin2_ files are supported) or from the [LiheYoung/Depth-Anything](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) repo on Hugging Face.
