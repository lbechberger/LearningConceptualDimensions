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
from helperfunctions import get_eval_noise,float_image_to_uint8,infogan_discriminator, infogan_generator

timestamp = str(datetime.now()).replace(' ','-')
checkpointsaver = [] 
# prevent that only the latest checkpoint is saved in the txt file by saving the previous content and 
# including it together with the new checkpoints in the end
# comment out the following 5 lines if you do not want to save the old checkpoints

if os.path.exists('checkpoints.txt'):
    f = open("checkpoints.txt", "r")
    for line in f:
        checkpointsaver.append(line)
    f.close()

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

# # architecture of the generator network
# def infogan_generator(inputs, g_weight_decay_gen=9e-5):
#     with tf.contrib.framework.arg_scope(
#             [layers.fully_connected, layers.conv2d_transpose],
#             activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
#             weights_regularizer=layers.l2_regularizer(g_weight_decay_gen)):
#         unstructured_noise, cont_noise = inputs
#         noise = tf.concat([unstructured_noise, cont_noise], axis=1)
#         net = layers.fully_connected(noise, 1024)
#         net = layers.fully_connected(net, 7 * 7 * 128)
#         net = tf.reshape(net, [-1, 7, 7, 128])
#         net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
#         net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
#         # Make sure that generator output is in the same range as `inputs`, i.e. [-1, 1].
#         net = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.nn.tanh)
# 
#         return net

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
discriminator_fn = functools.partial(infogan_discriminator, continuous_dim=options['latent_dims'], d_weight_decay_dis=options['d_weight_decay_dis'])
generator_fn = functools.partial(infogan_generator, g_weight_decay_gen=options['g_weight_decay_gen'])
unstructured_inputs, structured_inputs = get_training_noise(options['batch_size'], options['latent_dims'], options['noise_dims'])

"""Returns an InfoGAN model outputs and variables.
  See https://arxiv.org/abs/1606.03657 for more details.
  Args:
    generator_fn: A python lambda that takes a list of Tensors as inputs and
      returns the outputs of the GAN generator.
    discriminator_fn: A python lambda that takes `real_data`/`generated data`
      and `generator_inputs`. Outputs a 2-tuple of (logits, distribution_list).
      `logits` are in the range [-inf, inf], and `distribution_list` is a list
      of Tensorflow distributions representing the predicted noise distribution
      of the ith structure noise.
    real_data: A Tensor representing the real data.
    unstructured_generator_inputs: A list of Tensors to the generator.
      These tensors represent the unstructured noise or conditioning.
    structured_generator_inputs: A list of Tensors to the generator.
      These tensors must have high mutual information with the recognizer.
    generator_scope: Optional generator variable scope. Useful if you want to
      reuse a subgraph that has already been created.
    discriminator_scope: Optional discriminator variable scope. Useful if you
      want to reuse a subgraph that has already been created.
  Returns:
    An InfoGANModel namedtuple.
      (generator_inputs, generated_data, generator_variables, gen_scope, generator_fn,
      real_data, dis_real_outputs, dis_gen_outputs, discriminator_variables, disc_scope,
      lambda x, y: discriminator_fn(x, y)[0],  # conform to non-InfoGAN API, 
      structured_generator_inputs, predicted_distributions, discriminator_fn)
  Raises:
    ValueError: If the generator outputs a Tensor that isn't the same shape as
      `real_data`.
    ValueError: If the discriminator output is malformed.
  """
# Create the overall GAN
gan_model = tfgan.infogan_model(
    generator_fn=generator_fn,
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

            #Save the graph 
            cwd = os.getcwd()
            checkpoint_dir = os.path.join(cwd+'/graphs')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model = timestamp + str(epoch) + config_name +'.model'
            #ganmodel = timestamp + str(epoch) + config_name +'.ganmodel'
            checkpoint = os.path.join(checkpoint_dir, model)
            #checkpointgan = os.path.join(checkpoint_dir, ganmodel)
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)
            checkpointsaver.append(model + " " + config_name)
                
print(checkpointsaver)
# save all checkpointfile/model names in a txt file to find them later for evaluation

f = open("checkpoints.txt", "w")
for modelname in checkpointsaver:
    f.write(modelname)
f.write('/n')
f.close()
