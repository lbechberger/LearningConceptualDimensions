# -*- coding: utf-8 -*-
"""
File to create a simple data set of rectangles that are centered in the image.

Basically, create a bunch of 28x28 matrices with values between 0 and 1 and store them as a pickle file.

Created on Wed Nov 29 11:12:51 2017

@author: lbechberger
"""

import sys, pickle
import numpy as np
import scipy.ndimage.filters as filters

# read command-line arguments
number_of_examples = int(sys.argv[1])
output_filename = sys.argv[2]

image_size = 28 # should always be an even number
half_img_size = int(image_size/2)
mean = 0.0      # mean of the Gaussian noise
variance = 0.05 # variance of the Gaussian noise
sigma = 0.5     # variance of the Gaussian filter

dataset = []
for i in range(number_of_examples):
    # initialize the image with zeroes
    matrix = np.zeros(shape=[image_size, image_size])
    
    # randomly draw width and height 
    width = np.random.choice(range(1,half_img_size))
    height = np.random.choice(range(1,half_img_size))
    
    # now set all the pixels inside the rectangle to 1
    start_row = int((image_size - 2 * height) / 2)
    start_column = int((image_size - 2 * width) / 2)
    
    for row in range(2 * height):
        for column in range(2 * width):
            matrix[start_row + row][start_column + column] = 1.0

    # add Gaussian blur to make edges a bit less crisp
    blurred = filters.gaussian_filter(matrix, sigma)
    
    # let's add some noise to make the images a bit more realistic & to avoid having the same matrix appear over and over again    
    noise = np.random.normal(mean, variance, [image_size, image_size])
    added = blurred + noise
    clipped = np.clip(added, 0.0, 1.0)
        
    dataset.append(clipped)
    
# dump everything into a pickle file
with open(output_filename, "wb") as f:
    pickle.dump(dataset, f)