# Model weights folder

Model weights can be placed in this folder, which enables auto-loading of the smallest model file when launching the image/video scripts (without providing a model path). It also allows models to be referenced by partial name, instead of providing a full path.

For example, if you place a model file called `dpt_beit_large_512.pt` in this folder, you can reference it when launching the `run_image.py` script using:

```bash
python run_image.py -m large_512
```

This will select the model file (assuming no other models have filenames containing 'large_512').

You can download model files from the original [isl-org/MiDaS](https://github.com/isl-org/MiDaS/releases/tag/v3_1) releases page. Only the _beit_ variants are supported at this time.
