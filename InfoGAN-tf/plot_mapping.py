# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:40:59 2018

@author: lbechberger
"""

import sys
import pickle
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

input_file = sys.argv[1]
input_data = pickle.load(open(input_file, 'rb'))

dimension_names = input_data['dimensions']
n_latent = input_data['n_latent']

fig = plt.figure('Visualization of InfoGAN performance', figsize=(16,10))

table = input_data['table']
        
for dim_idx, dimension in enumerate(dimension_names):
        
    for latent_dim in range(n_latent):
        index = dimension_names.index(dimension) + 1 + len(dimension_names)*latent_dim
        ax = fig.add_subplot(n_latent, len(dimension_names), index)
        x = table[:,n_latent + dim_idx]
        y = table[:,latent_dim]
        ax.scatter(x, y, marker='.', s = 1)
        ax.set_xlabel(dimension)
        if dim_idx == 0:
            ax.set_ylabel("latent_{0}".format(latent_dim))
#        if dimension == 'orientation_div':
#            ax.set_xscale('log')
        b, m = polyfit(x, y, 1)
        ax.plot(x, b + m * x, '-', color = "r", linewidth = 2)
        

plt.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.05, wspace = 0.3, hspace = 0.17)
plt.show()