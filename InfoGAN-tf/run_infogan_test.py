# -*- coding: utf-8 -*-
"""
Training InfoGAN on our rectangle data.

Inspired by and based on
    https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb
and
    https://github.com/tensorflow/models/tree/master/research/gan/mnist

This script is a copy of the training part of run_infogan.py with changes in order to
save and restore the trained weights and parameters for the model. 

Created on Thu Jan 25 2018,

@author: lbechberger

Edited on Fri Nov  9 by christina



"""
import pickle
import numpy as np
import os, sys
import ast

import tensorflow as tf
tfgan = tf.contrib.gan
layers = tf.contrib.layers
ds = tf.contrib.distributions

import functools
from configparser import RawConfigParser
from datetime import datetime
from helperfunctions import get_eval_noise,infogan_generator,float_image_to_uint8,infogan_discriminator
from tensorflow.python.tools import inspect_checkpoint as chkp

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
options['type_latent'] = 'uniform'

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

parse_range('epochs')

# Set up the input
input_data = pickle.load(open(options['training_file'], 'rb'), encoding='latin1')
rectangles = np.array(list(map(lambda x: x[0], input_data['data'])), dtype=np.float32)
labels = np.array(list(map(lambda x: x[1:], input_data['data'])), dtype=np.float32)
dimension_names = input_data['dimensions']
length_of_data_set = len(rectangles)
images = rectangles.reshape((-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(20480).repeat().batch(options['batch_size'])
batch_images = dataset.make_one_shot_iterator().get_next()[0]

print("Starting InfoGAN training. Here are my parameters:")
print(options)
print("Length of data set: {0}".format(length_of_data_set))



_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.1)




def get_training_noise(batch_size, structured_continuous_dim, noise_dims):
    """Get unstructured and structured noise for InfoGAN.
    Args:
        batch_size: The number of noise vectors to generate.
        structured_continuous_dim: The number of dimensions of the uniform continuous noise.
        total_continuous_noise_dims: The number of continuous noise dimensions. This number includes the structured and unstructured noise.
    Returns:
        A 2-tuple of structured and unstructured noise. First element is the
        unstructured noise, and the second is the continuous structured noise."""
    # Get unstructurd noise.
    unstructured_noise = tf.random_normal([batch_size, noise_dims])

    # Get continuous noise Tensor.
    if options['type_latent'] == 'u':
        continuous_dist = ds.Uniform(-tf.ones([structured_continuous_dim]), tf.ones([structured_continuous_dim]))
        continuous_noise = continuous_dist.sample([batch_size])
    elif options['type_latent'] == 'n':
        continuous_noise = tf.random_normal([batch_size, structured_continuous_dim], mean = 0.0, stddev = 0.5)
    else:
        raise Exception("Unknown type of latent distribution: {0}".format(options['type_latent']))

    return [unstructured_noise], [continuous_noise]


# Build the generator and discriminator.
discriminator_fn = functools.partial(infogan_discriminator, continuous_dim=options['latent_dims'])
unstructured_inputs, structured_inputs = get_training_noise(options['batch_size'], options['latent_dims'], options['noise_dims'])

# Create the overall GAN
gan_model = tfgan.infogan_model(
    generator_fn=infogan_generator,
    discriminator_fn=discriminator_fn,
    real_data=batch_images,
    unstructured_generator_inputs=unstructured_inputs,
    structured_generator_inputs=structured_inputs)

# Build the GAN loss.
gan_loss = tfgan.gan_loss(gan_model, gradient_penalty_weight=1.0, mutual_information_penalty_weight=options['lambda'], add_summaries=True)

# Create the train ops, which calculate gradients and apply updates to weights.
train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer=tf.train.AdamOptimizer(options['gen_lr'], 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(options['dis_lr'], 0.5),
    summarize_gradients=True)


# preparing everything for training
train_step_fn = tfgan.get_sequential_train_steps()
global_step = tf.train.get_or_create_global_step()
loss_values = []

# define the subsequent evaluation: ... first the data
data_iterator = dataset.make_one_shot_iterator().get_next()
real_images = data_iterator[0]
real_targets = data_iterator[1]

# ... and now the latent code
with tf.variable_scope('Discriminator', reuse=True):
    latent_code = (infogan_discriminator(real_images, None)[1][0]).loc

evaluation_output = tf.concat([latent_code, real_targets], axis=1)

# calculate the number of training steps
num_steps = {}
max_num_steps = 0
for epoch in options['epochs']:
    steps = int( (epoch * length_of_data_set) / options['batch_size'] )
    num_steps[steps] = epoch
    max_num_steps = max(max_num_steps, steps)

print("Number of training steps: {0}".format(max_num_steps))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(max_num_steps):
        # train the network
        cur_loss, _ = train_step_fn(sess, train_ops, global_step, train_step_kwargs={})
        loss_values.append((step, cur_loss))
        if step % 100 == 0:
            print('Current loss: %f' % cur_loss)

        if (step + 1) in num_steps.keys():
            # finished an epoch
            epoch = num_steps[step + 1]
            print("finished epoch {0}".format(epoch))

            #Save the graph https://www.tensorflow.org/api_docs/python/tf/train/Saver
            cwd = os.getcwd()  #returns current working directory of a process
            checkpoint_dir = os.path.join(cwd+'/graphs') #building a directory called 'graphs' 
                                                         #for the checkpoint data in the current directory
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model = timestamp + str(epoch) +'.model'
            checkpoint = os.path.join(checkpoint_dir, model)
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)
            
#retrieve graph            
with tf.Graph().as_default():
    assert len(tf.trainable_variables()) == 0
    reader = tf.train.NewCheckpointReader(checkpoint)
    restore_dict = dict()
    for v in tf.trainable_variables():
        tensor_name = v.name.split(':')[0]
        if reader.has_tensor(tensor_name):
            print('has tensor ', tensor_name)
            restore_dict[tensor_name] = v

    saver = tf.train.Saver(restore_dict)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint)
        print("Model restored") 
 
# define tf.Variable()s
       
# prints [array([ 0.], dtype=float32), array([ 2.], dtype=float32)]    
                
            # chkp.print_tensors_in_checkpoint_file(checkpoint, [latent_code, real_targets], all_tensors=True, all_tensor_names=True)  
            
            # Finds the filename of latest saved checkpoint file.
#            latest_checkpoint = (tf.train.latest_checkpoint(checkpoint,latest_filename=None))
#            
#            #Returns CheckpointReader for checkpoint found in latest_checkpoint.
#            print('NewCheckpointReader '+(tf.train.NewCheckpointReader(latest_checkpoint)))
#            
#            print('CheckpointReader.get_tensor(latent_code) '+(tf.train.CheckpointReader.get_tensor(latent_code)))
#            
#            #Returns CheckpointReader for checkpoint found in latest_checkpoint.
#            print('load_checkpoint '+(tf.train.load_checkpoint(latest_checkpoint)))
            
            
            
            
            