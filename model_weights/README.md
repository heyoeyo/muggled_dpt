# Model weights folder

Model weights can be placed in this folder, which enables auto-loading of the smallest model file when launching the image/video scripts (without providing a model path). It also allows models to be referenced by partial name, instead of providing a full path.

For example, if you place a model file called `depth_anything_vitl14.pt` in this folder, you can reference it when launching the `run_image.py` script using something like:

```bash
python run_image.py -m vitl
```

This will cause the script to load the `depth_anything_vitl14.pt` file, since it contains 'vitl' in the filename.

You can download model files from the original [isl-org/MiDaS](https://github.com/isl-org/MiDaS/releases/tag/v3_1) releases page (only _beit_ and _swinv2_ variants are supported) or from the [LiheYoung/Depth-Anything](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) repo on Hugging Face.
