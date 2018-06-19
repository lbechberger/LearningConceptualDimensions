# -*- coding: utf-8 -*-
"""
File to create a simple data set of rectangles that are centered in the image.

Basically, create a bunch of 28x28 matrices with values between 0 and 1 and store them as a pickle file.

Created on Wed Nov 29 11:12:51 2017

@author: lbechberger
"""

import pickle
import numpy as np
import scipy.ndimage.filters as filters
import argparse

# read command-line arguments
parser = argparse.ArgumentParser(description='Rectangle generator')
parser.add_argument("n", type = int, help = 'number of images to generate')
parser.add_argument("file_name", help = 'output file name')
parser.add_argument("--image_size", type = int, default = 28, help = 'size of the images')
parser.add_argument("--mean", type = float, default = 0.0, help = 'mean of the Gaussian noise')
parser.add_argument("--variance", type = float, default = 0.05, help = 'variance of the Gaussian noise')
parser.add_argument("--sigma", type = float, default = 0.5, help = 'variance of the Gaussian filter')
args = parser.parse_args()

half_img_size = int(args.image_size/2)

dataset = []

for i in range(args.n):
    # initialize the image with zeroes
    matrix = np.full(shape=[args.image_size, args.image_size], fill_value=-1.0)
    
    # randomly draw width and height 
    width = np.random.choice(range(1,half_img_size))
    height = np.random.choice(range(1,half_img_size))
    size = 4 * width * height
    orientation = width / height
    
    # now set all the pixels inside the rectangle to 1
    start_row = int((args.image_size - 2 * height) / 2)
    start_column = int((args.image_size - 2 * width) / 2)
    
    for row in range(2 * height):
        for column in range(2 * width):
            matrix[start_row + row][start_column + column] = 1.0

    # add Gaussian blur to make edges a bit less crisp
    blurred = filters.gaussian_filter(matrix, args.sigma)
    
    # let's add some noise to make the images a bit more realistic & to avoid having the same matrix appear over and over again    
    noise = np.random.normal(args.mean, args.variance, [args.image_size, args.image_size])
    added = blurred + noise
    clipped = np.clip(added, -1.0, 1.0)
        
    dataset.append((clipped, width, height, size, orientation))

# dump everything into a pickle file
with open(args.file_name, "wb") as f:
    pickle.dump(dataset, f)