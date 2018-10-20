# LearningConceptualDimensions

The code in this repository is used to explore whether deep represenation learning is able to learn interpretable dimensions for the domain of shapes. If not noted otherwise, please use the conda environment `tensorflow-CS` from the [Utilities project](https://github.com/lbechberger/Utilities).

## create_data

The folder `create_data` contains two scripts for creating and inspecting a data set of simple rectangles.

### Creating the data set

The script `create_rectangles.py` creates a data set of rectangles. It has the following parameters:
```
create_rectangles.py [-h] [--image_size IMAGE_SIZE] [--mean MEAN]
                            [--variance VARIANCE] [--sigma SIGMA]
                            [--type TYPE]
                            n file_name
```
Optional parameters:
* `-h`: display a help message
* `--image_size IMAGE_SIZE`: the width of the images to be generated in pixels (default: 28). Images will be quadratic. 
* `--mean MEAN`: mean of the additive Gaussian noise (default: 0.0)
* `--variance VARIANCE`: variance of the additive Gaussian noise (default: 0.05)
* `--sigma SIGMA`: variance of the Gaussian filter used for blurring the images (default: 0.5)
* `--type TYPE`: type of distribution from which the width and height values of the generated rectangles will be samples (default: uniform; can also be set to normal)

Required parameters:
* `n`: number of rectangles to generate
* `file_name`: file name for output (pickle file containing all the generated rectangles)

In order to create the rectangles data, please execute the following command from the project root directory:
```
python create_data/create_rectangles.py 10000 data/uniform.pickle
python create_data/create_rectangles.py 10000 --type normal data/normal.pickle
```

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

### compress_results.py

This script is to be used on the CSV file output of the `run_infogan.py` script in order to average across multiple runs of the same configuration. It takes three arguments (where the last one is optional):
```python compress_results.py input_file_name output_file_name [grouping]```
* `input_file_name`: path to the CSV file generated by `run_infogan.py`
* `output_file_name`: file name for the output CSV file
* `grouping`: hyperparameter used for the aggregation (optional)

If `grouping` is not set, the script will aggregate all lines with the same configuration name into a single line by averaging across their values. This can be used to get a more robust estimation of a configuration's performance by averaging over a certain number of independent runs.

If `grouping` is set for example to `la`, there will be as many groups as the `la` hyperparameter takes different values. If we have three possible values of `la`, namely 1.0, 2.0, and 3.0, then this script will aggregate all lines containing `la1.0` in their configuration name into a single line, all configurations containing `la2.0` into another row, and all configurations containing `la3.0` into a third row. This can be used to get a first idea about the overall influence of a given hyperparameter.


### plot_mapping.py

This script visualizes the correlation between the latent variables and the expected interpretable dimensions. It creates a scatter plot for each combination of latent variable and interpretable dimension, including a line of best fit. It is used as follows, where `pickle_file` is the path to a raw pickle file generated by one of the InfoGAN runs:
```python plot_mapping.py pickle_file```


### statistical_analysis.py

This script conducts some statistical analyses on the individual hyperparameters with respect to a set of given metrics. In short, it analyzes whether there are any statistically significant differences between the results obtained for different values of a given hyperparameter. After having printed out the respective statistics, some box plots are generated in order to illustrate the findings. The script can be run as follows, where `input_file_name` contains the path to the CSV summary file generated by multiple InfoGAN runs:
```python statistical_analysis.py input_file_name```


