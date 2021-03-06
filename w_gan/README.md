# Training
Training is done using the `w_gan.py` script.

Simple usage:
`python3 w_gan.py --data_dir=data/digi_images/`


## arg params
### Directories
* `data_dir` - directory where the images are
  * Train pokemon example: `python3 w_gan.py --data_dir=data/poke_images/`
  * Train digimon example: `python3 w_gan.py --data_dir=data/digi_images/`
* `out_dir` - directory where the output images and model checkpoints are saved
  * default location: `./w_gan_out/`

### Params
* `num_epochs` - number of training iterations
  * default: 1000
* `batch_size` - number of images trained in every iterations
  * default: 64
* `print_freq`, `save_model_freq`, `save_images_freq` - How often to debug pring, save model, save images
  * default: `print_freq: 50`, `save_model_freq: 500`, `save_images_freq: 100`

## Script Outputs
During **training** the script frequently saves **sample images** and **model checkpoints**.
You can use the **model checkpoints** to run **inference** later.

```
w_gan_out/
├── images
│   └── DEBUG
│       └── epoch0.jpg
└── model
    └── DEBUG
        ├── checkpoint
        ├── model0.chkpt.data-00000-of-00001
        ├── model0.chkpt.index
        └── model0.chkpt.meta
```
# Inference
```
install python 3.x
pip3 install --upgrade tensorflow
pip3 install scipy
pip3 install Pillow
```

Once you have a checkpoint (model output) then you can continue to pull images from the model using the `inference.py` script.

Simple usage: `python3 inference.py --model_path=w_gan_out/model/DEBUG/model0.chkpt`

### Model checkpoint based on Digimon
 https://files.fm/u/k55255q4

```
  digi1000
├── digi1000.chkpt.data-00000-of-00001
├── digi1000.chkpt.index
└── digi1000.chkpt.meta

python3 inference.py --model_path=digi1000/digi1000.chkpt
```
### Model checkpoint based on 150 Pokemon
https://files.fm/u/z77ddrrr
```
poke5500
├── poke5500.chkpt.data-00000-of-00001
├── poke5500.chkpt.index
└── poke5500.chkpt.meta
 python3 inference.py --model_path=poke5500/poke5500.chkpt
 ```

### Example from w_gan output
```
└── model
    └── DEBUG
        ├── checkpoint
        ├── model0.chkpt.data-00000-of-00001
        ├── model0.chkpt.index
        └── model0.chkpt.meta
```

In this case theres a checkpoint `model0.chkpt` even though there's not a file named `model0.chkpt` it is made up of 3 files `[CHECKPOINT_NAME].data-00000-of-00001`, `[CHECKPOINT_NAME].index`, `[CHECKPOINT_NAME].meta`.

## arg Params
* `model_path` - where is the model (and include chkpt name like above)
* `output_path` - where to put output images
* `output_image_columns` - image output format
* `output_num_images` - image output format
