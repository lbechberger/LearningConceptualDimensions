import numpy as np
import sklearn as sk
import csv

def load_np_pickle(ind):
    import pickle
    import sys
    pickle.load(open(sys.argv[ind], 'rb'), encoding='latin1')

#input_data = load_np_pickle(0)
#evaluation_data = load_np_pickle(1)


# TO-DO: Replace value later on
config_name = 'foo'

DEF = '-'
to_add = {}
# TO-DO: Add the remaining categories later on
ORDERED_CATS = ['mseReg', 'r^2']
# Initialize to_add with DEFault value
for cat in ORDERED_CATS:
    to_add[cat] = DEF

# TO-DO: Replace with correct csv-name later on
CSV_NAME = 'bla.csv'

# Fail-fast if csv file does not exist or first line unequal to the categories
with open(CSV_NAME, 'r', newline='') as csvfile:
    if not (ORDERED_CATS == next(csv.reader(csvfile))):
        raise ValueError("CSVFile's first line unequal config/metric-names")

def reg_mse_and_score(inputs, labels):
    """

    :param inputs
    :param labels
    :return: mse and r^2 coefficient between the labels and predicted values based on the inputs
    """
    if not len(inputs) == len(labels):
        raise ValueError

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(inputs, labels)
    pred = reg.predict(inputs)
    mseReg = sk.metrics.mean_squared_error(labels, pred)
    return mseReg, reg.score(inputs, pred)

def add_to_csv(ordered_cats, csv_name, to_add):
    """
    adds to_add to csv with csv_name. If values for a category are missing, '-' will be added

    :param ordered_cats: category names in the correct order
    :param csv_name: csv file's name
    :param to_add: dict that maps category names to their values
    :return: void
    """
    import fcntl

    with open(csv_name, 'a', newline='') as csvfile:
        fcntl.flock(csvfile, fcntl.LOCK_EX)
        keys = to_add.keys()
        for cat in ordered_cats:
            # If value for a cat(egory) is missing, fill in with '-'
            if not cat in keys:
                to_add[cat] = '-'
        csv.writer(csvfile).writerow([to_add[cat] for cat in ordered_cats])
        fcntl.flock(f, fcntl.LOCK_UN)

def update_to_add(key, val):
    """
    Only allows val to be added to tp_add[key] if key is one of the categories in ORDERED_CATS.
    Use this function instead whenever you want to use "to_add[someKey] = someVal"

    :param key:
    :param val:
    :return:
    """
    if(key not in ORDERED_CATS):
        raise ValueError("IllegalKey")
    to_add[key] = val


# retrieve reconstruction errors from pickle file
def retrieve_rec_errors(ind):
    pickle_file = load_np_pickle(ind)
    codes_from_images = pickle_file[0]
    avg_manh_dist_images = pickle_file[1]
    avg_eucl_dist_images = pickle_file[2]
    avg_manh_dist_codes = pickle_file[3]
    avg_eucl_dist_codes = pickle_file[4]
    output_images_variing_lat_code = pickle_file[5]
    return (codes_from_images, avg_manh_dist_images, avg_eucl_dist_images, avg_manh_dist_codes, avg_eucl_dist_codes, output_images_variing_lat_code)

