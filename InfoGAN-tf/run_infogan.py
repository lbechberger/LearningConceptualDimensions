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

import tensorflow as tf
tfgan = tf.contrib.gan
layers = tf.contrib.layers
ds = tf.contrib.distributions

import functools
from six.moves import xrange

from configparser import RawConfigParser

from math import sqrt

options = {}
options['train_log_dir'] = 'logs'
options['output_dir'] = 'output'
options['training_file'] = '../data/rectangles_v0.05_s0.5.pickle'
options['batch_size'] = 128
options['num_epochs'] = 50
options['noise_dims'] = 64
options['latent_dims'] = 2
options['gen_lr'] = 1e-3
options['dis_lr'] = 9e-5

config_name = sys.argv[1]
config = RawConfigParser(options)
config.read("infogan.cfg")

if config.has_section(config_name):
    options['train_log_dir'] = config.get(config_name, 'train_log_dir')
    options['output_dir'] = config.get(config_name, 'output_dir')
    options['training_file'] = config.get(config_name, 'training_file')
    options['batch_size'] = config.getint(config_name, 'batch_size')
    options['num_epochs'] = config.getint(config_name, 'num_epochs')
    options['noise_dims'] = config.getint(config_name, 'noise_dims')
    options['latent_dims'] = config.getint(config_name, 'latent_dims')
    options['gen_lr'] = config.getfloat(config_name, 'gen_lr')
    options['dis_lr'] = config.getfloat(config_name, 'dis_lr')

# Set up the input.
#rectangles = np.array(pickle.load(open(options['training_file'], 'rb')), dtype=np.float32)
input_data = pickle.load(open(options['training_file'], 'rb'))
rectangles = np.array(list(map(lambda x: x[0], input_data)), dtype=np.float32)
labels = np.array(list(map(lambda x: [x[1], x[2]], input_data)), dtype=np.float32)
length_of_data_set = len(rectangles)
images = rectangles.reshape((-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(20000).repeat().batch(options['batch_size'])
batch_images = dataset.make_one_shot_iterator().get_next()[0]

print("Starting InfoGAN training. Here are my parameters:")
print(options)
print("Length of data set: {0}".format(length_of_data_set))

# architecture of the generator network
def infogan_generator(inputs):
    with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(2.5e-5)):
        unstructured_noise, cont_noise = inputs
        noise = tf.concat([unstructured_noise, cont_noise], axis=1)
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`, i.e. [-1, 1].
        net = layers.conv2d(
            net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.nn.tanh)
    
        return net    

_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)

# architecture of the discriminator network
def infogan_discriminator(img, unused_conditioning, weight_decay=2.5e-5, continuous_dim=2):
    with tf.contrib.framework.arg_scope(
    [layers.conv2d, layers.fully_connected],
    activation_fn=_leaky_relu, normalizer_fn=None,
    weights_regularizer=layers.l2_regularizer(2.5e-5),
    biases_regularizer=layers.l2_regularizer(2.5e-5)):
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
    continuous_dist = ds.Uniform(-tf.ones([structured_continuous_dim]), tf.ones([structured_continuous_dim]))
    continuous_noise = continuous_dist.sample([batch_size])

    return [unstructured_noise], [continuous_noise]

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


# Build the generator and discriminator.
discriminator_fn = functools.partial(infogan_discriminator, continuous_dim=options['latent_dims'])
unstructured_inputs, structured_inputs = get_training_noise(options['batch_size'], options['latent_dims'], options['noise_dims'])

gan_model = tfgan.infogan_model(
    generator_fn=infogan_generator,
    discriminator_fn=discriminator_fn,
    real_data=batch_images,
    unstructured_generator_inputs=unstructured_inputs,
    structured_generator_inputs=structured_inputs)

# Build the GAN loss.
gan_loss = tfgan.gan_loss(gan_model, gradient_penalty_weight=1.0, mutual_information_penalty_weight=1.0, add_summaries=True)

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
number_of_steps = int( (options['num_epochs'] * length_of_data_set) / options['batch_size'] )
print("Number of steps: {0}".format(number_of_steps))

with tf.Session() as sess:
    # train the network
    sess.run(tf.global_variables_initializer())
    for i in range(number_of_steps):
        cur_loss, _ = train_step_fn(sess, train_ops, global_step, train_step_kwargs={})
        loss_values.append((i, cur_loss))
        if i % 100 == 0:
            print('Current loss: %f' % cur_loss)
    print("done training")
    
    # now create some output images
    CONT_SAMPLE_POINTS = np.linspace(-1.2, 1.2, 13)
    for i in range(options['latent_dims']):
        display_noise = get_eval_noise(options['noise_dims'], CONT_SAMPLE_POINTS, options['latent_dims'], i)
        with tf.variable_scope('Generator', reuse=True):
            continuous_image = infogan_generator(display_noise)
        reshaped_continuous_image = tfgan.eval.image_reshaper(continuous_image, num_cols=len(CONT_SAMPLE_POINTS))

        def float_image_to_uint8(image):
            scaled = (image * 128.0) + 128.0
            return tf.cast(scaled, tf.uint8)
          
        uint8_continuous = float_image_to_uint8(reshaped_continuous_image)
        
        image_write_op = tf.write_file(os.path.join(options['output_dir'], "{0}_continuous{1}.png".format(config_name, i)), 
                                                    tf.image.encode_png(uint8_continuous[0]))
        sess.run(image_write_op)
    
    # now evaluate a regression of width and height on the latent space
    data_iterator = dataset.make_one_shot_iterator().get_next()
    real_images = data_iterator[0]
    real_targets = data_iterator[1]
    with tf.variable_scope('Discriminator', reuse=True):
        latent_code = (infogan_discriminator(batch_images, None)[1][0]).loc
    
    weights = tf.Variable(tf.truncated_normal([options['latent_dims'],options['latent_dims']]))
    bias = tf.Variable(tf.truncated_normal([options['latent_dims']]))
    prediction = tf.matmul(latent_code, weights, transpose_b=True) + bias
    mse_prediction = tf.reduce_sum(tf.square(real_targets - prediction))
    mse_latent = tf.reduce_sum(tf.square(real_targets - (14 * latent_code)))
    mse_baseline = tf.reduce_sum(tf.square(real_targets))
    global_step = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(1e-6).minimize(mse_prediction, var_list=[weights,bias], global_step = global_step)
    
    num_eval_steps = int( (1.0 * length_of_data_set) / options['batch_size'] )
    sess.run([global_step.initializer, weights.initializer, bias.initializer])
    for i in range(10 * num_eval_steps):
        # train the weights in the fully connected layer for 10 epochs
        sess.run([optimizer,mse_prediction])
    
    mse_latent_sum = 0.0
    mse_prediction_sum = 0.0
    mse_baseline_sum = 0.0
    for step in range(num_eval_steps):
        l, p, b = sess.run([mse_latent, mse_prediction, mse_baseline])
        mse_latent_sum += l
        mse_prediction_sum += p
        mse_baseline_sum += b
    rmse_baseline = sqrt(mse_baseline_sum / length_of_data_set)
    rmse_latent = sqrt(mse_latent_sum / length_of_data_set)
    rmse_prediction = sqrt(mse_prediction_sum / length_of_data_set)
    print(config_name)
    print("RMSE baseline: {0}".format(rmse_baseline))    
    print("RMSE latent:   {0}".format(rmse_latent))    
    print("RMSE lin_reg:  {0}".format(rmse_prediction))  
    with open("output/rmse.csv", 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("{0},{1},{2},{3}\n".format(config_name, rmse_baseline, rmse_latent, rmse_prediction))
        fcntl.flock(f, fcntl.LOCK_UN)
    
    print("done evaluating")