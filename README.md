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
```python create_data/show_iamges.py data/uniform.pickle```

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
