# LearningConceputalDimensions

The code in this repository is used to explore whether deep represenation learning is able to learn interpretable dimensions for the domain of shapes.

Currently, the code requires the following libraries to be installed:
* numpy
* scipy

The code runs under Python 2.7.

## Creating Data
In order to create the rectangles data, please execute the following command from the project root directory:
```
python create_data/create_rectangles.py 10000 data/rectancles.pickle
```

## vanilla_InfoGAN
The original InfoGAN implementation ([see here](https://github.com/openai/infogan)) uses an outdated version of tensorflow. In order to get everything set up, please follow the instructions from [here](https://github.com/felixblind/InfoGAN-for-Shapes/). Our code in the vanilla_InfoGAN folder is based on the code from the latter repository, which in turn modified the original repository.

For convenience, the `vanilla_InfoGAN` folder contains a script `infogan_setup.sge` which sets up a conda environment with the correct versions of all dependencies. The file `run_rectangle.sge` can be used to run the simple rectangle example. It requires as a parameter the number of epochs for which the network is trained (in the original MNIST experiment, 50 epochs were used). One can of course also manually execute the following lines:

```
source activate infogan
PYTHONPATH='.' python launchers/run_rectangle_exp.py $1
source deactivate infogan
```
The .sge scripts are used for running these jobs on the sun grid engine.

Further convenience scripts are `clean.sh` (which removes all logs and checkpoints) and `submit_jobs.sh` (which submits the job specified by the first argument with all the parameters found in a file indicated by the second argument).
