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
The original InfoGAN implementation [see here](https://github.com/openai/infogan) uses an outdated version of tensorflow. In order to get everything set up, please follow the instructions from [here](https://github.com/felixblind/InfoGAN-for-Shapes/). Our code in the vanilla_InfoGAN folder is based on the code from the latter repository, which in turn modified the original repository.
