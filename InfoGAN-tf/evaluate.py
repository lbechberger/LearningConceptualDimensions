import numpy as np
import sklearn as sk
import csv

def load_np_pickle(ind):
    import pickle
    import sys
    pickle.load(open(sys.argv[ind], 'rb'), encoding='latin1')

input_data = load_np_pickle(0)
evaluation_data = load_np_pickle(1)


# TO-DO: Replace value later on
config_name = 'foo'

DEF = '-'
# Initialize to_add with DEFault value
to_add = {}
# TO-DO: Replace with correct categories later on
for cat in ['f', 'o']:
    to_add[cat] = DEF

CSV_NAME = 'eggs.csv'

# Fail-fast if csv file does not exist or first line unequal to to_add.keys()
with open(CSV_NAME, 'r', newline='') as csvfile:
    if not (
            to_add.keys()
            == next(csv.reader(csvfile))):
        raise ValueError("CSVFile's first line unequal config/metric-names")    


def reg_mse_and_score(inputs, labels):
    """

    :param inputs
    :param labels
    :return: mse and r^2 coefficient between the labels and predicted values based on the inputs
    """
    if not len(inputs) == labels(inputs):
        raise ValueError

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(inputs, labels)
    pred = reg.predict(inputs)
    mseReg = sk.metrics.mean_squared_error(labels, pred)
    return mseReg, reg.score(inputs, pred)

def add_to_csv(ordered_cats, csv_name, to_add):
    """
    adds to_add to csv with csvname. If values for a category are missing, '-' will be added

    :param ordered_cats: category names in the correct order
    :param csv_name: csv file's name
    :param to_add: dict that maps category names to their values
    :return: void
    """
    with open(csv_name, 'a', newline='') as csvfile:
        # If value for category missing, fill in with '-'
        keys = to_add.keys()
        for cat in ordered_cats:
            if not cat in keys:
                to_add[cat] = '-'
        csv.writer(csvfile).writerow([to_add[cat] for cat in ordered_cats])
