"""
CSV

"""
import numpy as np
from sklearn.linear_model import LinearRegression


"""
# TO-DO: Insert appropriate arguments into fit 
reg = LinearRegression().fit()
dann einfach die euklidische Distanz berechnen und r^2 kriegt man eh mit reg.score
"""

import csv
with open('eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])