# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:40:59 2018

@author: lbechberger
"""

import sys
import pickle
import matplotlib.pyplot as plt

input_file = sys.argv[1]
input_data = pickle.load(open(input_file, 'rb'))

dimension_names = ['width', 'height', 'size', 'orientation']
fig = plt.figure(figsize=(16,10))

n_latent = input_data['n_latent']
        
for dimension in dimension_names:
    mappings = input_data[dimension]['bins']
    
    x = list(map(lambda x: x[0], mappings))    
    y = list(map(lambda x: x[1], mappings))
    var = list(map(lambda x: x[2], mappings))
    
    for latent in range(n_latent):
        index = dimension_names.index(dimension) + 1 + len(dimension_names)*latent
        ax = fig.add_subplot(n_latent, len(dimension_names), index)
        ax.scatter(x, list(map(lambda x: x[latent], y)))
        ax.set_xlabel(dimension)
        ax.set_ylabel("latent_{0}".format(latent))
        if dimension == 'orientation':
            ax.set_xscale('log')

plt.show()