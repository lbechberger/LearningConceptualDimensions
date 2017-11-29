# -*- coding: utf-8 -*-
"""
File to create a simple data set of rectangles that are centered in the image.

Basically, create a bunch of 28x28 matrices with values between 0 and 1 and store them as a pickle file.

Created on Wed Nov 29 11:12:51 2017

@author: lbechberger
"""

import sys, pickle
import numpy as np

# read command-line arguments
number_of_examples = int(sys.argv[1])
output_filename = sys.argv[2]

image_size = 10#28 # should always be an even number

dataset = []
for i in range(number_of_examples):
    # initialize the image with zeroes
    matrix = np.zeros(shape=[image_size,image_size])
    
    # randomly draw width and height 
    width = np.random.choice(range(1,image_size/2))
    height = np.random.choice(range(1,image_size/2))
    
    # now set all the pixels inside the rectangle to 1
    start_row = (image_size - 2 * height) / 2
    start_column = (image_size - 2 * width) / 2
    
    for row in range(2 * height):
        for column in range(2 * width):
            matrix[start_row + row][start_column + column] = 1.0
    
    # TODO: add Gaussian blur (to smooth edges) and/or Gaussian noise (to make image more noisy)    
    
    dataset.append(matrix)

# dump everything into a pickle file
with open(output_filename, "wb") as f:
    pickle.dump(dataset, f)