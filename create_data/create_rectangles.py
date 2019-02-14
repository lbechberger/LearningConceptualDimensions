# -*- coding: utf-8 -*-
"""
File to create a simple data set of rectangles that are centered in the image.

Basically, create a bunch of 28x28 matrices with values between 0 and 1 and store them as a pickle file.

Created on Wed Nov 29 11:12:51 2017

@author: lbechberger
"""

import pickle, argparse
import numpy as np
import scipy.ndimage.filters as filters

# read command-line arguments
parser = argparse.ArgumentParser(description='Rectangle generator')
parser.add_argument("n", type = int, help = 'number of images to generate')
parser.add_argument("file_name", help = 'output file name')
parser.add_argument("first_dim", help = 'first dimension to sample over')
parser.add_argument("second_dim", help = 'second dimension to sample over')
parser.add_argument("--image_size", type = int, default = 28, help = 'size of the images')
parser.add_argument("--mean", type = float, default = 0.0, help = 'mean of the Gaussian noise')
parser.add_argument("--variance", type = float, default = 0.05, help = 'variance of the Gaussian noise')
parser.add_argument("--sigma", type = float, default = 0.5, help = 'variance of the Gaussian filter')
parser.add_argument("--type", default = 'uniform', help = 'type of distribution to use')
parser.add_argument("-p", "--plot", action = "store_true", help = 'triggers plotting of histograms')
args = parser.parse_args()

half_img_size = int(args.image_size/2)
min_rectangle_size = 2
max_rectangle_size = args.image_size - 2
step_size = 2

factor_names = ['width', 'height', 'size', 'orientation']

borders = {'width' : (min_rectangle_size, max_rectangle_size), 
                  'height' : (min_rectangle_size, max_rectangle_size),
                  'size' : (min_rectangle_size**2, max_rectangle_size**2),
                  'orientation' : (min_rectangle_size / min_rectangle_size + max_rectangle_size, max_rectangle_size / min_rectangle_size + max_rectangle_size)} 

distributions = {'uniform' : lambda x: np.random.choice(range(x[0], x[1] + 1, step_size)), 
                 'normal' : lambda x: np.random.normal(loc = np.mean(x), scale = np.mean(x)/3)}

dataset = []

def discretize(value):
    result = min(max_rectangle_size, max(min_rectangle_size, value))
    result = step_size * int(round(result/step_size))
    return result

drawn_factors = [args.first_dim, args.second_dim]

if args.plot:
    factor_history = { 'width' : [], 'height' : [], 'size' : [], 'orientation' : [] }

for i in range(args.n):
    
    # initialize everything
    generating_factors = { 'width' : None, 'height' : None, 'size' : None, 'orientation' : None}

    # get the two generating factors   
    generating_factors[args.first_dim] = distributions[args.type](borders[args.first_dim])
    generating_factors[args.second_dim] = distributions[args.type](borders[args.second_dim])
    
    # make sure that width and height are given    
    if 'width' in drawn_factors and 'height' in drawn_factors: 
        pass
    elif 'width' in drawn_factors and 'size' in drawn_factors:
        generating_factors['height'] = generating_factors['size'] / generating_factors['width']
    elif 'width' in drawn_factors and 'orientation' in drawn_factors:
        fraction = (1 - generating_factors['orientation']) / generating_factors['orientation']
        generating_factors['height'] = generating_factors['width'] * fraction 
    elif 'height' in drawn_factors and 'size' in drawn_factors:
        generating_factors['width'] = generating_factors['size'] / generating_factors['height']
    elif 'height' in drawn_factors and 'orientation' in drawn_factors:
        numerator = generating_factors['orientation'] * generating_factors['height']
        denominator = 1 - generating_factors['orientation']
        generating_factors['width'] = numerator / denominator
    elif 'size' in drawn_factors and 'orientation' in drawn_factors:
        numerator = 1 - generating_factors['orientation']
        denominator = generating_factors['orientation'] * generating_factors['size']
        generating_factors['width'] = numerator / denominator
        generating_factors['height'] = generating_factors['size'] / generating_factors['width']
        
    # make width and height conform to what we need later            
    generating_factors['width'] = discretize(generating_factors['width'])
    generating_factors['height'] = discretize(generating_factors['height'])

    # recompute remaining generating factors based on new width and height    
    generating_factors['size'] = generating_factors['width'] * generating_factors['height']
    generating_factors['orientation'] = generating_factors['width'] / (generating_factors['width'] + generating_factors['height'])

    
    # now draw the rectangle: start with -1 everywhere
    matrix = np.full(shape=[args.image_size, args.image_size], fill_value=-1.0)
    
    # set all the pixels inside the rectangle to 1
    start_row = int((args.image_size - generating_factors['height']) / 2)
    start_column = int((args.image_size - generating_factors['width']) / 2)
    
    for row in range(generating_factors['height']):
        for column in range(generating_factors['width']):
            matrix[start_row + row][start_column + column] = 1.0

    # add Gaussian blur to make edges a bit less crisp
    blurred = filters.gaussian_filter(matrix, args.sigma)
    
    # let's add some noise to make the images a bit more realistic & to avoid having the same matrix appear over and over again    
    noise = np.random.normal(args.mean, args.variance, [args.image_size, args.image_size])
    added = blurred + noise
    clipped = np.clip(added, -1.0, 1.0)
    
    # add to the data set
    dataset.append((clipped, [generating_factors[factor] for factor in factor_names]))
    
    # add to plotting info
    if args.plot:
        for factor in factor_names:
            factor_history[factor].append(generating_factors[factor])

output = {'data' : dataset, 'dimensions': factor_names}

# dump everything into a pickle file
with open(args.file_name, "wb") as f:
    pickle.dump(output, f)
    
if args.plot:
    from matplotlib import pyplot as plt
    import os
    
    folder_name = os.path.dirname(args.file_name)
    file_name = os.path.splitext(os.path.basename(args.file_name))[0]
    
    for factor in factor_names:
        plot_bins = range(int(borders[factor][0]), int(borders[factor][1]) + 3, step_size) if factor in ['width', 'height'] else None
        plt.hist(factor_history[factor], bins = plot_bins)
        plt.title('distribution of {0}'.format(factor))
        
        output_file_name = os.path.join(folder_name, '{0}-{1}.png'.format(file_name, factor))
        plt.savefig(output_file_name)
        plt.close()
