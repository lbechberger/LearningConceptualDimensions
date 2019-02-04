"""
CSV

"""
import numpy as np
import sklearn as sk
import csv



"""
# TO-DO: Insert appropriate arguments into fit 

dann einfach die euklidische Distanz berechnen und r^2 kriegt man eh mit reg.score
"""

def reg_mse_and_score(inputs, labels):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(inputs, labels)
    pred = reg.predict(inputs)
    mseReg = sk.metrics.mean_squared_error(labels, pred)
    return mseReg, reg.score(inputs, pred)

