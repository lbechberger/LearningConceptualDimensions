"""
This file uses slightly modified code from https://gist.github.com/danijar/1cb4d81fed37fd06ef60d08c1181f557#file-blog_tensorflow_variational_auto_encoder-py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from tensorflow.python import debug as tf_debug
# MF: weggeben?
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import pickle
import scipy.ndimage.filters as filters
import argparse
from six.moves import xrange
# Should be removed later on
from datetime import datetime
#from tensorflow.examples.tutorials.mnist import input_data
import functools
from configparser import RawConfigParser
import fcntl
import ast


tfd = tf.contrib.distributions
tfgan = tf.contrib.gan
l2_regul = tf.contrib.layers.l2_regularizer

timestamp = str(datetime.now()).replace(' ','-')

# default values for options
options = {}
options['train_log_dir'] = 'logs'
options['output_dir'] = 'output'
options['training_file'] = '../data/normal.pickle'
options['noise_dims'] = 62
options['latent_dims'] = 2
options['encoder_h_dim'] = 200
options['decoder_h_dim'] = 200 
options['beta'] = 1
options['lr'] = 1e-3
options['batch_size'] = 128
options['l2_regul_z_mu'] = 0.0
options['l2_regul_z_scale'] = 0.0
options['l2_regul_p_mu'] = 0.0
options['l2_regul_p_scale'] = 0.0
options['epochs'] = '50'

# read configuration file
config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("grid_search.cfg")


def parse_range(key):
    value = options[key]
    parsed_value = ast.literal_eval(value)
    if isinstance(parsed_value, list):
        options[key] = parsed_value
    else:
        options[key] = [parsed_value]
                
# overwrite default values for options
if config.has_section(config_name):
    for option_name in options.keys():
        options[option_name] = config.get(config_name, option_name)
  
      
parse_range('epochs')        

# Set up the input
input_data = pickle.load(open(options['training_file'], 'rb'), encoding='latin1')
rectangles = np.array(list(map(lambda x: x[0], input_data['data'])), dtype=np.float32)
rectangles = rectangles.reshape((-1, 28, 28))
dimension_names = input_data['dimensions']
length_of_data_set = len(rectangles)
labels = np.array(list(map(lambda x: x[1:], input_data['data'])), dtype=np.float32)
X_dim = list(rectangles.shape[1:])
Y_dim = list(labels.shape[1:])

print("Starting BetaVAE training. Here are my parameters:")
print(options)
print("Length of data set: {0}".format(length_of_data_set))

def make_encoder(data):
  x = tf.layers.flatten(data)
  x = tf.layers.dense(x, options['encoder_h_dim'], tf.nn.relu)
  x = tf.layers.dense(x, options['encoder_h_dim'], tf.nn.relu)
  z_mu = tf.layers.dense(x, options['latent_dims'], kernel_regularizer=l2_regul(options['l2_regul_z_mu']))
  z_scale = tf.layers.dense(x, options['latent_dims'], tf.nn.softplus, kernel_regularizer=l2_regul(options['l2_regul_z_scale']))
  z_scale = tf.sqrt(tf.exp(z_scale) - 1)
  return z_mu, z_scale


def make_prior():
  loc = tf.zeros(options['latent_dims'])
  scale = tf.ones(options['latent_dims'])
  return tfd.MultivariateNormalDiag(loc, scale)


def make_decoder(code):
  x = code
  x = tf.layers.dense(x, options['decoder_h_dim'], tf.nn.relu)
  x = tf.layers.dense(x, options['decoder_h_dim'], tf.nn.relu)
  p_mu = tf.layers.dense(x, np.prod(X_dim), kernel_regularizer=l2_regul(options['l2_regul_p_mu']))
  p_mu = tf.reshape(p_mu, [-1] + X_dim)
  p_scale = tf.layers.dense(x, np.prod(X_dim), tf.nn.softplus, kernel_regularizer=l2_regul(options['l2_regul_p_scale']))
  # Bound from below with 1e-5 to avoid NaNs
  p_scale = tf.sqrt(tf.exp(p_scale) - 1) + 1e-5
  p_scale = tf.reshape(p_scale, [-1] + X_dim)
  return tfd.Independent(tfd.MultivariateNormalDiag(p_mu, p_scale))

data = tf.placeholder(tf.float32, [None] + X_dim)
real_targets = tf.placeholder(tf.float32, [None] + Y_dim)

make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

# Define the model.
prior = make_prior()
z_mu, z_scale = make_encoder(data)
eps = tf.random_normal(shape=tf.shape(z_mu))
code = z_mu + z_scale * eps


# Define the loss.
recon_term = make_decoder(code).log_prob(data)

kl_term = options['beta'] * -0.5 * tf.reduce_sum(1 + tf.log(z_scale) - z_mu**2 - z_scale, 1)
vae_loss = -tf.reduce_mean(recon_term - kl_term)
total_loss = vae_loss + tf.losses.get_regularization_loss()
optimize = tf.train.AdamOptimizer(0.001).minimize(total_loss)

samples = make_decoder(prior.sample(10)).mean()

evaluation_output = tf.concat([code, real_targets], axis=1)



def get_batch(data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:options['batch_size']]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# May not be needed
def float_image_to_uint8(image):
    scaled = (image * 127.5) + 127.5
    return tf.cast(scaled, tf.uint8)

def get_eval_noise(noise_dims, continuous_sample_points, latent_dims, idx):
    """Create noise showing impact of first dim continuous noise in InfoGAN.
    First dimension of continuous noise is constant across columns. Other noise is
    constant across rows.
    Args:
        noise_samples: Number of non-categorical noise samples to use.
        continuous_sample_points: Possible continuous noise points to sample.
        unstructured_noise_dims: Dimensions of the unstructured noise.
        idx: Index of continuous dimension we want to evaluate.
    Returns:
        Unstructured noise, continuous noise numpy arrays."""

    rows, cols = 20, len(continuous_sample_points)

    # Take random draws for non-first-dim-continuous noise, making sure they are constant across columns.
    unstructured_noise = []
    for _ in xrange(rows):
        cur_sample = np.random.normal(size=[1, noise_dims])
        unstructured_noise.extend([cur_sample] * cols)
    unstructured_noise = np.concatenate(unstructured_noise)

    cont_noise_other_dim = []
    for _ in xrange(rows):
        cur_sample = np.random.choice(continuous_sample_points, size=[1, latent_dims-1])
        cont_noise_other_dim.extend([cur_sample] * cols)
    cont_noise_other_dim = np.concatenate(cont_noise_other_dim)

    # Increase evaluated dimension of continuous noise from left to right, making sure they are constant across rows.
    cont_noise_evaluated_dim = np.expand_dims(np.tile(continuous_sample_points, rows), 1)

    if idx == 0:
        # first dimension is the one to be evaluated
        continuous_noise = np.concatenate((cont_noise_evaluated_dim, cont_noise_other_dim), 1)
    elif idx == latent_dims - 1:
        # last dimension is the one to be evaluated
        continuous_noise = np.concatenate((cont_noise_other_dim, cont_noise_evaluated_dim), 1)
    else:
        # intermediate dimension is to be evaluated --> split other_dims_list in two parts
        other_dims_list = np.split(cont_noise_other_dim, latent_dims - 1, axis=1)
        first = np.concatenate(other_dims_list[:idx], 1)
        second = np.concatenate(other_dims_list[idx:], 1)
        # sneak cont_noise_evaluated_dim in the middle and glue everything back together
        continuous_noise = np.concatenate((first, cont_noise_evaluated_dim, second), 1)

    return unstructured_noise.astype('float32'), continuous_noise.astype('float32')






with tf.Session() as sess:
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  sess.run(tf.global_variables_initializer())
  for epoch in range(20):
    feed = {data: rectangles}
    test_vae_loss, test_codes, test_samples = sess.run([vae_loss, code, samples], feed)
    print('Epoch', epoch, 'vae_loss', test_vae_loss)
    
    CONT_SAMPLE_POINTS = np.linspace(-1.2, 1.2, 13)
    for i in range(options['latent_dims']):
        display_noise = get_eval_noise(options['noise_dims'], CONT_SAMPLE_POINTS, options['latent_dims'], i)
        continuous_image = tf.expand_dims(samples, -1)
        reshaped_continuous_image = tfgan.eval.image_reshaper(continuous_image, num_cols=len(CONT_SAMPLE_POINTS))
        uint8_continuous = float_image_to_uint8(reshaped_continuous_image)
        image_write_op = tf.write_file(os.path.join(options['output_dir'], "{0}-ep{1}-{2}_dim{3}.png".format(config_name, epoch, timestamp, i)),
        tf.image.encode_png(uint8_continuous[0]))
        sess.run(image_write_op)
    
    
    
    for _ in range(600):
        input_data, target_data = get_batch(rectangles, labels)
        feed = {data: input_data, real_targets: target_data}
        rows, _ = sess.run([evaluation_output, optimize], feed)

    
        # now evaluate the current latent codes
        num_eval_steps = int( (1.0 * length_of_data_set) / options['batch_size'] )
        epoch_name = "{0}-ep{1}".format(config_name, epoch)
        print(epoch_name)
    
        # compute all the outputs (= [latent_code, real_targets])
        table = []
        #for i in range(num_eval_steps):
        #rows = sess.run(evaluation_output)
        table.append(rows)
        table = np.concatenate(table, axis=0)
    
        # compute the ranges for each of the columns
        ranges = np.subtract(np.max(table, axis = 0), np.min(table, axis = 0))[:2]
        min_range = min(ranges)
    
        # compute correlations
        correlations = np.corrcoef(table, rowvar=False)
    
        output = {'n_latent' : options["latent_dims"], 'dimensions' : dimension_names, 'ranges' : ranges, 'table' : table}
    
        # store the so-far best matching interpretable dimension
        max_correlation_latent = [0.0]*options["latent_dims"]
        best_name_latent_correlation = [None]*options["latent_dims"]
    
        # iterate over all original dimensions
        for dim_idx, dimension in enumerate(dimension_names):
    
            # take Pearson's correlation coefficient for this dimension
            local_correlations = correlations[options["latent_dims"] + dim_idx][:options["latent_dims"]]
    
            output[dimension] = local_correlations
    
            # check whether we found a better interpretation for a latent variable...
            for latent_dim in range(options["latent_dims"]):
                if np.abs(local_correlations[latent_dim]) > max_correlation_latent[latent_dim]:
                    max_correlation_latent[latent_dim] = np.abs(local_correlations[latent_dim])
                    best_name_latent_correlation[latent_dim] = dimension
    
        # lower bound for best correlations
        interpretability_correlation = min(max_correlation_latent)
        # are no two latent variables best interpreted in the same way?
        all_different = (len(set(best_name_latent_correlation)) == options["latent_dims"])
    
        # dump all of this into a pickle file for later use
        with open(os.path.join(options['output_dir'], "{0}-ep{1}-{2}.pickle".format(config_name, epoch, timestamp)), 'wb') as f:
            pickle.dump(output, f)
    
        # some console output for debug purposes:
        print("\nOverall correlation-based interpretability: {0}".format(interpretability_correlation))
        print("Overall minimal range: {0}".format(min_range))
        print("Ended up with all different dimensions: {0}".format(all_different))
        for latent_dim in range(options["latent_dims"]):
            print("latent_{0} (range {1}): best interpreted as '{2}' ({3})".format(latent_dim, ranges[latent_dim], best_name_latent_correlation[latent_dim], max_correlation_latent[latent_dim]))
            for dimension in dimension_names:
                print("\t {0}: {1}".format(dimension, output[dimension][latent_dim]))
    
    
        # create output file if necessary
        file_name = os.path.join(options['output_dir'],'interpretabilities.csv')
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write("config,data_set,overall_cor,min_range,different")
                for latent_dim in range(options["latent_dims"]):
                    f.write(",range-{0}".format(latent_dim))
                    f.write(",max-cor-{0},name-cor-{0}".format(latent_dim))
                    for dimension in dimension_names:
                        f.write(",corr-{0}-{1}".format(dimension, latent_dim))
                f.write("\n")
                fcntl.flock(f, fcntl.LOCK_UN)
    
        # append information to output file
        with open(file_name, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write("{0},{1},{2},{3},{4}".format(epoch_name, "rectangles", interpretability_correlation, min_range, all_different))
            for latent_dim in range(options["latent_dims"]):
                f.write(",{0}".format(ranges[latent_dim]))
                f.write(",{0},{1}".format(max_correlation_latent[latent_dim], best_name_latent_correlation[latent_dim]))
                for dimension in dimension_names:
                    f.write(",{0}".format(output[dimension][latent_dim]))
    
            f.write("\n")
            fcntl.flock(f, fcntl.LOCK_UN)