# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:33:42 2018

@author: lbechberger
"""

import numpy as np
import matplotlib.pyplot as plt

size_mul = []
size_add = []
size_sqrt = []
orientation_div = []
orientation_sub = []
orientation_log = []
orientation_frac = []

for width in range(1,14):
    for height in range(1,14):
        size_mul.append(4 * width * height)
        size_add.append(2 * (width + height))
        size_sqrt.append(np.sqrt(4 * width * height))
        
        orientation_div.append(width/height)
        orientation_sub.append(2*(width-height))
        orientation_log.append(np.divide(np.log10(width/height), np.log10(13)))
        orientation_frac.append(width/(width+height))

plt.scatter(size_mul, size_add)
plt.xlabel('size_mul')
plt.ylabel('size_add')
plt.show()

plt.scatter(size_mul, size_sqrt)
plt.xlabel('size_mul')
plt.ylabel('size_sqrt')
plt.show()

plt.scatter(size_sqrt, size_add)
plt.xlabel('size_sqrt')
plt.ylabel('size_add')
plt.show()



plt.scatter(orientation_div, orientation_sub)
plt.xlabel('orientation_div')
plt.ylabel('orientation_sub')
plt.show()

plt.scatter(orientation_div, orientation_log)
plt.xlabel('orientation_div')
plt.ylabel('orientation_log')
plt.show()

plt.scatter(orientation_div, orientation_frac)
plt.xlabel('orientation_div')
plt.ylabel('orientation_frac')
plt.show()


plt.scatter(orientation_sub, orientation_log)
plt.xlabel('orientation_sub')
plt.ylabel('orientation_log')
plt.show()

plt.scatter(orientation_sub, orientation_frac)
plt.xlabel('orientation_sub')
plt.ylabel('orientation_frac')
plt.show()


plt.scatter(orientation_log, orientation_frac)
plt.xlabel('orientation_log')
plt.ylabel('orientation_frac')
plt.show()

