import numpy as np
import sklearn as sk
import csv

# TO-DO: Replace value later on
config_name = 'foo'
# TO-DO: Replace value later on
cats = ['f', 'o', 'o']
"""
csv_name = 'eggs.csv'
# Fail-fast if csv file does not exist or first line unequal to cats
with open(csv_name, 'r', newline='') as csvfile:
    if not (
            categories
            == next(csv.reader(csvfile))):
        raise ValueError("CSVFile's first line unequal config/metric-names")    
"""

def reg_mse_and_score(inputs, labels):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(inputs, labels)
    pred = reg.predict(inputs)
    mseReg = sk.metrics.mean_squared_error(labels, pred)
    return mseReg, reg.score(inputs, pred)

def add_to_csv(cats, csv_name, to_add):
    with open(csv_name, 'a', newline='') as csvfile:
        # If value for category missing, fill in with '-'
        keys = to_add.keys()
        for cat in cats:
            if not cat in keys:
                to_add[cat] = '-'
        csv.writer(csvfile).writerow([to_add[cat] for cat in cats])
