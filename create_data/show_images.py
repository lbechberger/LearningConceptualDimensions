# -*- coding: utf-8 -*-
"""
Visualize 12 randomly picked images from a given pickle file.

Created on Thu Dec  7 08:49:27 2017

@author: lbechberger
"""

import sys, pickle, random
import matplotlib.pyplot as plt

pickle_file_name = sys.argv[1]

with open(pickle_file_name, "rb") as f:
    data_set = pickle.load(f)

random.shuffle(data_set)

rows = 3
columns = 4
fig = plt.figure(figsize=(16,10))
    
for i in range(12):
    ax = fig.add_subplot(rows, columns, i+1)
    ax.matshow(data_set[i][0], cmap="Greys")
    
plt.show()