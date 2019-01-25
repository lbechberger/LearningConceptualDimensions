# -*- coding: utf-8 -*-
"""
Training InfoGAN on our rectangle data.
Inspired by and based on
    https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb
and
    https://github.com/tensorflow/models/tree/master/research/gan/mnist
Created on Thu Jan 25 2018
@author: lbechberger
"""

import pickle
import numpy as np
import os, sys
import fcntl
import ast
import functools
import tensorflow as tf
tfgan = tf.contrib.gan
layers = tf.contrib.layers
ds = tf.contrib.distributions

from six.moves import xrange

from configparser import RawConfigParser

from datetime import datetime

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
options['weight_decay_gen'] = 2.5e-5
options['weight_decay_dis'] = 2.5e-5

"""
import doctest
doctest.testmod(verbose=True)
"""

# False for normal running, start if you want it to enter the evaluation phase for each epoch
test = False
if(sys.argv[2] == "test"):
    test = True

# read configuration file
config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("grid_search.cfg")


def factorial(n):
    """

    >>> factorial(3)
    1
    """

    import math
    if not n >= 0:
        raise ValueError("n must be >= 0")
    if math.floor(n) != n:
        raise ValueError("n must be exact integer")
    if n+1 == n:  # catch a value like 1e300
        raise OverflowError("n too large")
    result = 1
    factor = 2
    while factor <= n:
        result *= factor
        factor += 1
    return result


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
    options['weight_decay_gen'] = config.get(config_name, 'weight_decay_gen')
    options['weight_decay_dis'] = config.get(config_name, 'weight_decay_dis')

parse_range('epochs')

# Set up the input
input_data = pickle.load(open(options['training_file'], 'rb'), encoding='latin1')
rectangles = np.array(list(map(lambda x: x[0], input_data['data'])), dtype=np.float32)
labels = np.array(list(map(lambda x: x[1:], input_data['data'])), dtype=np.float32)
dimension_names = input_data['dimensions']
length_of_data_set = len(rectangles)
images = rectangles.reshape((-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset_training = dataset.shuffle(20480).repeat().batch(options['batch_size'])
dataset_evaluation = dataset.repeat().batch(options['batch_size'])
batch_images_training = dataset_training.make_one_shot_iterator().get_next()[0]
# batch_images_evaluation = dataset_evaluation.make_one_shot_iterator().get_next()[0]

# MF
print(options['batch_size'])

print("Starting InfoGAN training. Here are my parameters:")
print(options)
print("Length of data set: {0}".format(length_of_data_set))


# architecture of the generator network
def infogan_generator(inputs, weight_decay_gen=9e-5):
    with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(weight_decay_gen)):
        unstructured_noise, cont_noise = inputs
        noise = tf.concat([unstructured_noise, cont_noise], axis=1)
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`, i.e. [-1, 1].
        net = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.nn.tanh)

        return net

_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.1)

# architecture of the discriminator network
def infogan_discriminator(img, unused_conditioning, weight_decay_dis=9e-5, continuous_dim=2):
    with tf.contrib.framework.arg_scope(
    [layers.conv2d, layers.fully_connected],
    activation_fn=_leaky_relu, normalizer_fn=None,
    weights_regularizer=layers.l2_regularizer(weight_decay_dis),
    biases_regularizer=layers.l2_regularizer(weight_decay_dis)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
    logits_real = layers.fully_connected(net, 1, activation_fn = None)

    with tf.contrib.framework.arg_scope([layers.batch_norm], is_training=False):
        encoder = layers.fully_connected(net, 128, normalizer_fn=layers.batch_norm, activation_fn=_leaky_relu)

    # Compute mean for Gaussian posterior of continuous latents.
    mu_cont = layers.fully_connected(encoder, continuous_dim, activation_fn=None)
    sigma_cont = tf.ones_like(mu_cont)
    q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

    return logits_real, [q_cont]


def get_training_noise(batch_size, structured_continuous_dim, noise_dims):
    
    # Get unstructurd noise.
    unstructured_noise = tf.random_normal([batch_size, noise_dims])

    # Get continuous noise Tensor.
    if options['type_latent'] == 'u':
        I = tf.ones([structured_continuous_dim]) + 1
        continuous_dist = ds.Uniform(-I, I)
        continuous_noise = continuous_dist.sample([batch_size])
    elif options['type_latent'] == 'n':
        continuous_noise = tf.random_normal([batch_size, structured_continuous_dim], mean = 0.0, stddev = 1.0)
    else:
        raise Exception("Unknown type of latent distribution: {0}".format(options['type_latent']))

    return [unstructured_noise], [continuous_noise]

def get_eval_noise(noise_dims, continuous_sample_points, latent_dims, idx):

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


# Build the generator and discriminator.
discriminator_fn = functools.partial(infogan_discriminator, continuous_dim=options['latent_dims'], weight_decay_dis=options['weight_decay_dis'])
generator_fn = functools.partial(infogan_generator, weight_decay_gen=options['weight_decay_gen'])
unstructured_inputs, structured_inputs = get_training_noise(options['batch_size'], options['latent_dims'], options['noise_dims'])

# Create the overall GAN
gan_model = tfgan.infogan_model(
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    real_data=batch_images_training,
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

def float_image_to_uint8(image):
    scaled = (image * 127.5) + 127.5
    return tf.cast(scaled, tf.uint8)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())
    for step in range(max_num_steps):
        #doctest: Still worked here
        # train the network
        cur_loss, _ = train_step_fn(sess, train_ops, global_step, train_step_kwargs={})
        #doctest: Stops working here
        loss_values.append((step, cur_loss))
        if step % 100 == 0:
            print('Current loss: %f' % cur_loss)

        #doctest. Does not work here
        eval_condition = True
        if(not test):
            eval_condition = (step + 1) in num_steps.keys()
        if eval_condition:
            # finished an epoch
            epoch = 1
            if(not test):
                epoch = num_steps[step + 1]
            print("finished epoch {0}".format(epoch))

            num_eval_steps = int((1.0 * length_of_data_set) / options['batch_size'])
            epoch_name = "{0}-ep{1}".format(config_name, epoch)
            print(epoch_name)

            # start - part relevant to get codes and images from input images
            # get a batch of input images
            data_iterator = dataset_evaluation.make_one_shot_iterator().get_next()
            real_images = data_iterator[0]

            # ... and now the latent code
            with tf.variable_scope(gan_model.discriminator_scope, reuse=True):
                latent_code = (discriminator_fn(real_images, None)[1][0]).loc

            # temp is only needed as an input to gan_model.generator_fn(temp
            temp = (tf.zeros([options['batch_size'], options['noise_dims']]), latent_code)

            with tf.variable_scope(gan_model.generator_scope, reuse=True):
                # get output images
                image_tensors_from_images = gan_model.generator_fn(temp)


            # Define helper function to get codes and images from input images
            def imagesInCodesAndImagesOut():
                return sess.run([latent_code, image_tensors_from_images])

            # list that will hold data generated from input images
            from_images = [[], []]

            # Get all the data generated from input images
            # Each loop gets us one batch of codes and output images
            for i in range(num_eval_steps):
                # get batch of codes and output images
                results_from_images = imagesInCodesAndImagesOut()
                assert (results_from_images[0].shape[0] == results_from_images[1].shape[0])
                # If j == 0, collect codes; if j == 1, collect output images
                for j in range(len(from_images)):
                    from_images[j].append(results_from_images[j])

            # lambda function that gets only used in the next 2 lines
            concat = lambda x: np.concatenate(x, axis=0)
            codes_from_all_images = concat(from_images[0])
            images_from_all_images = concat(from_images[1])


            def eval_shaped(a):
                assert a.shape[0] == length_of_data_set
                return np.reshape(a, (length_of_data_set, -1))

            eucl_dist_images = np.linalg.norm(eval_shaped(images_from_all_images) - eval_shaped(images), ord=2, axis=1)
            expected = eucl_dist_images.shape == (length_of_data_set, )
            if not expected:
                print(eucl_dist_images.shape)
                for i in range(3):
                    print("")
                assert expected

            avg_eucl_dist_images = np.mean(eucl_dist_images)
            print(avg_eucl_dist_images)

"""