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

# Inference
