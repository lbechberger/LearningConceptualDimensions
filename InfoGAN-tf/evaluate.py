import numpy as np
import sklearn as sk
import csv

# default values for options
options = {}
options['train_log_dir'] = 'logs'
options['output_dir'] = 'output'
options['training_file'] = '../data/uniform.pickle'
options['noise_dims'] = 62
options['latent_dims'] = 2
options['batch_size'] = 128
options['gen_lr'] = 1e-3
options['dis_lr'] = 2e-4
options['lambda'] = 1.0
options['epochs'] = '50'
options['type_latent'] = 'u'
options['g_weight_decay_gen'] = 2.5e-5
options['d_weight_decay_dis'] = 2.5e-5

# execute script: python evaluate.py path/to/dataset.pickle path/to/evaluation.pickle
def load_np_pickle(ind):
    import pickle
    import sys
    pickle.load(open(sys.argv[ind], 'rb'), encoding='latin1')

input_dataset = load_np_pickle(0)  # entweder normal.pickle oder uniform.pickle ??
evaluation_data = load_np_pickle(1)

evaluation_file = sys.argv(1)

config_name = str.split(evaluation_file,"-")[1]
epoch = str.split(evaluation_file,"-")[2].replace("ep","")
timestamp = str.split(evaluation_file,"-")[3].replace(".pickle","")

print(config_name)
print(epoch)
print(timestamp)

def parse_range(key):
    value = options[key]
    parsed_value = ast.literal_eval(value)
    if isinstance(parsed_value, list):
        options[key] = parsed_value
    else:
        options[key] = [parsed_value]


# overwrite default values for options
if config.has_section(config_name):
    options['train_log_dir'] = config.get(config_name, 'train_log_dir')
    options['output_dir'] = config.get(config_name, 'output_dir')
    options['training_file'] = config.get(config_name, 'training_file')
    options['noise_dims'] = config.getint(config_name, 'noise_dims')
    options['latent_dims'] = config.getint(config_name, 'latent_dims')
    options['batch_size'] = config.getint(config_name, 'batch_size')
    options['gen_lr'] = config.getfloat(config_name, 'gen_lr')
    options['dis_lr'] = config.getfloat(config_name, 'dis_lr')
    options['lambda'] = config.getfloat(config_name, 'lambda')
    options['epochs'] = config.get(config_name, 'epochs')
    options['type_latent'] = config.get(config_name, 'type_latent')
    options['g_weight_decay_gen'] = config.get(config_name, 'g_weight_decay_gen')
    options['d_weight_decay_dis'] = config.get(config_name, 'd_weight_decay_dis')


# Set up the input
input_data = pickle.load(open(options['training_file'], 'rb'), encoding='latin1')
rectangles = np.array(list(map(lambda x: x[0], input_data['data'])), dtype=np.float32)
labels = np.array(list(map(lambda x: x[1:], input_data['data'])), dtype=np.float32)
dimension_names = input_data['dimensions']
length_of_data_set = len(rectangles)
inp_images = rectangles.reshape((-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices((inp_images, labels))
dataset_training = dataset.shuffle(20480).repeat().batch(options['batch_size'])
dataset_evaluation = dataset.repeat().batch(options['batch_size'])
batch_images_training = dataset_training.make_one_shot_iterator().get_next()[0]
# batch_images_evaluation = dataset_evaluation.make_one_shot_iterator().get_next()[0]


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
codes_from_images = evaluation_data[0]
avg_manh_dist_images = evaluation_data[1]
avg_eucl_dist_images = evaluation_data[2]
avg_manh_dist_codes = evaluation_data[3]
avg_eucl_dist_codes = evaluation_data[4]
output_images_variing_lat_code = evaluation_data[5]


for item in input_dataset:
    mean_dif_vec = item[x] # feature: durchschnittlicher Differenz Vektor (pro batch)
    const_factor =  item [x] # label: der Konstant gehaltene Faktor (kategorielle Variable)
    training_set = ['feature': mean_dif_vec], ['label': const_factor]