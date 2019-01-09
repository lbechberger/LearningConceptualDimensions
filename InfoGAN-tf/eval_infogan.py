# -*- coding: utf-8 -*-
"""
Evaluation InfoGAN on our rectangle data.

Inspired by and based on
    https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb
and
    https://github.com/tensorflow/models/tree/master/research/gan/mnist

Created on Fri Sep 14 2018

@author: aabbood
"""


import tensorflow as tf
from helperfunctions import get_eval_noise,infogan_generator,float_image_to_uint8,infogan_discriminator
import tensorflow as tf
from configparser import RawConfigParser
from datetime import datetime
import numpy as np
import os, sys

tfgan = tf.contrib.gan
timestamp = str(datetime.now()).replace(' ','-')

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

# read configuration file
config_name = sys.argv[1]
to_evaluate = sys.argv[2]
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

parse_range('epochs')

# Set up the input
input_data = pickle.load(open(options['training_file'], 'rb'))
rectangles = np.array(list(map(lambda x: x[0], input_data['data'])), dtype=np.float32)
labels = np.array(list(map(lambda x: x[1:], input_data['data'])), dtype=np.float32)
dimension_names = input_data['dimensions']
length_of_data_set = len(rectangles)
images = rectangles.reshape((-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(20480).repeat().batch(options['batch_size'])
batch_images = dataset.make_one_shot_iterator().get_next()[0]

data_iterator = dataset.make_one_shot_iterator().get_next()
real_images = data_iterator[0]
real_targets = data_iterator[1]


# Function that runs the generator code from run_infogan.py based
# on a trained graph
def eval_generator():
    image_write_ops = None
    with tf.Session() as sess:
        CONT_SAMPLE_POINTS = np.linspace(-1.2, 1.2, 13)
        for i in range(options['latent_dims']):
            display_noise = get_eval_noise(options['noise_dims'], CONT_SAMPLE_POINTS, options['latent_dims'], i)
            with tf.variable_scope('Generator'):
                images = infogan_generator(
                      display_noise,
                      is_training=False)
            reshaped_images = tfgan.eval.image_reshaper(
              images, num_cols=len(CONT_SAMPLE_POINTS))
            uint8_images = float_image_to_uint8(reshaped_images)
            image_write_op = tf.write_file(os.path.join(options['output_dir'],
            "{0}-{1}_dim{2}.png".format(config_name, timestamp, i)),
            tf.image.encode_png(uint8_continuous[0]))


            tf.contrib.training.evaluate_repeatedly(
                './',
                hooks=[tf.contrib.training.StopAfterNEvalsHook()],
                eval_ops=image_write_ops,
                max_number_of_evaluations=2)

# Function that runs the discriminator code from run_infogan.py based
# on a trained graph
def eval_disciminator(real_images):
    num_eval_steps = int( (1.0 * length_of_data_set) / options['batch_size'] )
    compare = []
    for i in range(num_eval_steps):
        with tf.variable_scope('Discriminator', reuse=True):
            latent_code = (infogan_discriminator(real_images, None)[1][0]).loc
            evaluation_output = tf.concat([latent_code, real_targets], axis=1)

    compare.append(tf.contrib.training.evaluate_repeatedly('./',
        hooks=[tf.contrib.training.StopAfterNEvalsHook(1)],
        max_number_of_evaluations=1,
        final_ops=evaluation_output))

    return compare

# Run evaluation based on parsed argv
if to_evaluate == 'gen':
    eval_generator()
if to_evaluate == 'dis':
    eval_disciminator()
