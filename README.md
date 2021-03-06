# LearningConceptualDimensions

The code in this repository is used to explore whether deep represenation learning is able to learn interpretable dimensions for the domain of shapes. If not noted otherwise, please use the conda environment `tensorflow-CS` from the [Utilities project](https://github.com/lbechberger/Utilities).

## create_data

The folder `create_data` contains two scripts for creating and inspecting a data set of simple rectangles.

### Creating the data set

The script `create_rectangles.py` creates a data set of rectangles. It has the following parameters:
```
usage: create_rectangles.py [-h] [--image_size IMAGE_SIZE] [--mean MEAN]
                            [--variance VARIANCE] [--sigma SIGMA]
                            [--type TYPE] [--seed SEED] [-p]
                            n file_name first_dim second_dim
```
Optional parameters:
* `-h`: display a help message
* `--image_size IMAGE_SIZE`: the width of the images to be generated in pixels (default: 28). Images will be quadratic. 
* `--mean MEAN`: mean of the additive Gaussian noise (default: 0.0)
* `--variance VARIANCE`: variance of the additive Gaussian noise (default: 0.05)
* `--sigma SIGMA`: variance of the Gaussian filter used for blurring the images (default: 0.5)
* `--type TYPE`: type of distribution from which the width and height values of the generated rectangles will be samples (default: uniform; can also be set to normal)
* `--seed SEED`: seed used to initialize the random number generator
* `-p` or `--plot`: plot histograms of the four dimensions and stores them in the same folder as the output pickle file

Required parameters:
* `n`: number of rectangles to generate
* `file_name`: file name for output (pickle file containing all the generated rectangles)
* `first_dim`: name of the first dimension from which to sample (any of: width, height, size, orientation)
* `second_dim`: name of the second dimension from which to sample (any of: width, height, size, orientation)


### Visualizing the data set
The script `show_images.py` takes as argument the file name of the data set to visualize. It randomly selects twelve images from this data set and visualizes them to the user. This script can be used to double-check that reasonable images are generated.

You can use it like this from the project's root directory:
```python create_data/show_images.py data/uniform.pickle```

## vanilla_InfoGAN
The original InfoGAN implementation ([see here](https://github.com/openai/infogan)) uses an outdated version of tensorflow. In order to get everything set up, please follow the instructions from [here](https://github.com/felixblind/InfoGAN-for-Shapes/). Our code in the vanilla_InfoGAN folder is based on the code from the latter repository, which in turn modified the original repository.

Currently, the code requires the following libraries to be installed:
* numpy
* scipy

The code runs under Python 2.7.

For convenience, the `vanilla_InfoGAN` folder contains a script `infogan_setup.sge` which sets up a conda environment with the correct versions of all dependencies. The file `run_rectangle.sge` can be used to run the simple rectangle example. It requires as a parameter the number of epochs for which the network is trained (in the original MNIST experiment, 50 epochs were used). One can of course also manually execute the following lines:

```
source activate infogan
PYTHONPATH='.' python launchers/run_rectangle_exp.py $1
source deactivate infogan
```
The .sge scripts are used for running these jobs on the sun grid engine.

Further convenience scripts are `clean.sh` (which removes all logs and checkpoints) and `submit_jobs.sh` (which submits the job specified by the first argument with all the parameters found in a file indicated by the second argument).

Please note that the training is very unstable -- losses might become NaN which causes the network to crash. After some initial experimentation, we have stopped working with this code as it proved to be not very useful.

## InfoGAN-tf

This is a reimplementation of the InfoGAN network by Chen et al. using the tfgan library and TensorFlow 1.4. In our implementation, we only use continuous latent dimensions.

### run_infogan.py

This file does the heavy lifting: It trains and evaluates a given InfoGAN configuration. The script takes a single parameter: The name of the configuration to be executed. It assumes that there exists a file `grid_search.cfg` (e.g., created by the `grid_search.py` script from the [Utilities project](https://github.com/lbechberger/Utilities) using the `template.cfg` file as a starting point) which contains a configuration with that name. This configuration can look as follows (the given values are the default values used if no valid configuration is given to the script):

```
[default]
output_dir = output			# directory for all evaluation output (images, raw pickle files, and csv summary)
train_log_dir = logs			# directory for tensorflow logs (currently unused)
noise_dims = 62				# size of the noise vector
latent_dims = 2				# size of the latent vector
gen_lr = 1e-3				# learning rate of the generator network
dis_lr = 2e-04				# learning rate of the discriminator network
lambda = 1.0				# weight of the mutual information term in the overall loss function
epochs = 50				# number of epochs to train
batch_size = 128			# batch size used during training
training_file = ../data/uniform.pickle	# input data set file
type_latent = uniform			# distribution for the latent variables used during training (uniform or normal)
```

You can execute the script as follows: `python run_infogan.py default`
In order to execute this script on the Sun Grid Engine, another script called `run_infogan.sge` has been created which can be run as an array job. The preferred way of executing this array job is by using the script `submit_jobs.sh` from the [Utilities project](https://github.com/lbechberger/Utilities):
```
submit_array_job.sh run_infogan.sge params.txt 10 2
```
This will submit 10 array jobs of the InfoGAN code, where each array job instance executes two instances of the infogan pyhton script.

The program will create three types of outputs (all in the directory indicated by `output_dir` in the config file):
* **Images**: Generates a matrix of images with 20 rows and 13 columns for each latent dimension. The colums correspond to different values of the latent dimension of interest whereas the rows correspond to different values for all the other latent and noise variables.
* **Raw pickle files**: A file contining all latent values for all images as well as the expected values on interpretable dimensions. Also contains additional information (e.g., observed ranges of the latent variables and correlations between the latent dimensions and the expected interpretable dimensions)
* **CSV summary**: Each run of the InfoGAN script adds a single row to this matrix. Information in this row includes the pair-wise correlation values between the latent variables and the interpretable dimensions, ranges of latent variables, an overall interpretability score, and the best mapping between latent variables and interpretable dimensions.

### plot_mapping.py

This script visualizes the correlation between the latent variables and the expected interpretable dimensions. It creates a scatter plot for each combination of latent variable and interpretable dimension, including a line of best fit. It is used as follows, where `pickle_file` is the path to a raw pickle file generated by one of the InfoGAN runs:
```python plot_mapping.py pickle_file```

## BetaVAE-tf

This is an implementation of the Beta-VAE proposed by Burgess et al. compatible with TensorFlow 1.4. 

### run_betavae.py

Basically the same as run_infogan.py, but for BetaVAE instead of InfoGAN. This file uses slightly modified code from [wiseodd](https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_tensorflow.py). 
More information will be added in the future
