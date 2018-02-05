# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 2018

@author: lbechberger
"""

import pickle
import numpy as np

import tensorflow as tf
tfgan = tf.contrib.gan
flags = tf.flags
layers = tf.contrib.layers

flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')
flags.DEFINE_string('train_log_dir', 'logs', 'Directory where to write event logs.')
flags.DEFINE_integer('max_number_of_steps', 1000, 'The maximum number of gradient steps.')
flags.DEFINE_string('gan_type', 'unconditional', 'Either `unconditional`, `conditional`, or `infogan`.')
flags.DEFINE_integer('grid_size', 5, 'Grid size for image visualization.')
flags.DEFINE_integer('noise_dims', 64, 'Dimensions of the generator noise vector.')
flags.DEFINE_string('training_file', '../data/rectangles_v0.05_s0.5.pickle', 'Pickle file of images to use.')
flags.DEFINE_float('gen_lr', 1e-3, 'Learning rate for the generator.')
flags.DEFINE_float('dis_lr', 1e-4, 'Learning rate for the discriminator.')

FLAGS = flags.FLAGS

# Set up the input.
rectangles = np.array(pickle.load(open(FLAGS.training_file, 'rb')), dtype=np.float32)
print(rectangles.shape)
images = rectangles[:FLAGS.batch_size].reshape((-1, 28, 28, 1))
print(images.shape)

noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims], dtype=tf.float32)

def unconditional_generator(generator_inputs):
    with tf.contrib.framework.arg_scope(
      [layers.fully_connected, layers.conv2d_transpose],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
      weights_regularizer=layers.l2_regularizer(2.5e-5)):
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [0, 1].
        net = layers.conv2d(
            net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.nn.sigmoid)
    
        return net    

#    with tf.contrib.framework.arg_scope(
#      [layers.fully_connected, layers.conv2d_transpose],
#      activation_fn=tf.nn.relu, normalizer_fn=None,
#      weights_regularizer=layers.l2_regularizer(2.5e-5)):
#          net = tf.contrib.layers.fully_connected(generator_inputs, 28*28, activation_fn=tf.nn.sigmoid)
#          net = tf.cast(net, tf.float32)
#          return tf.reshape(net, shape=(-1,28,28,1))

_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)

def unconditional_discriminator(discriminator_inputs, generator_inputs):
    with tf.contrib.framework.arg_scope(
    [layers.conv2d, layers.fully_connected],
    activation_fn=_leaky_relu, normalizer_fn=None,
    weights_regularizer=layers.l2_regularizer(2.5e-5),
    biases_regularizer=layers.l2_regularizer(2.5e-5)):
        net = layers.conv2d(discriminator_inputs, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

        return net

#    with tf.contrib.framework.arg_scope(
#      [layers.conv2d, layers.fully_connected],
#      activation_fn=_leaky_relu, normalizer_fn=None,
#      weights_regularizer=layers.l2_regularizer(2.5e-5), biases_regularizer=layers.l2_regularizer(2.5e-5)):
#          net = tf.contrib.layers.fully_connected(discriminator_inputs, 1, activation_fn=tf.nn.sigmoid)
#          net = tf.cast(net, tf.float32)
#          return net

# Build the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=unconditional_generator,  
    discriminator_fn=unconditional_discriminator,  
    real_data=images,
    generator_inputs=noise)
tfgan.eval.add_gan_model_image_summaries(gan_model, FLAGS.grid_size)

# Build the GAN loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

# Create the train ops, which calculate gradients and apply updates to weights.
train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer=tf.train.AdamOptimizer(FLAGS.gen_lr, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.dis_lr, 0.5))

status_message = tf.string_join(['Starting train step: ', tf.as_string(tf.train.get_or_create_global_step())], name='status_message')

# Run the train ops in the alternating training scheme.
#tfgan.gan_train(
#    train_ops,
#    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps), tf.train.LoggingTensorHook([status_message], every_n_iter=1)],
#    logdir=FLAGS.train_log_dir,
#    get_hooks_fn = tfgan.get_joint_train_hooks())

train_step_fn = tfgan.get_sequential_train_steps()

global_step = tf.train.get_or_create_global_step()
loss_values, mnist_score_values  = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(FLAGS.max_number_of_steps):
        cur_loss, _ = train_step_fn(sess, train_ops, global_step, train_step_kwargs={})
        loss_values.append((i, cur_loss))
        if i % 100 == 0:
            print('Current loss: %f' % cur_loss)
    print("done training")
    
    # now create some output images
    with tf.variable_scope('Generator', reuse=True):
        images = unconditional_generator(tf.random_normal([100, FLAGS.noise_dims]))
    reshaped_images = tfgan.eval.image_reshaper(images, num_cols=10)
    def float_image_to_uint8(image):
      image = (image * 255.0)
      return tf.cast(image, tf.uint8)
      
    uint8_images = float_image_to_uint8(reshaped_images)
    image_write_ops = tf.write_file("./images.png", tf.image.encode_png(uint8_images[0]))
    sess.run(image_write_ops)
    print("done evaluating")

#with tf.variable_scope('Generator', reuse=True):
#    images = unconditional_generator(tf.random_normal([100, FLAGS.noise_dims]))
#reshaped_images = tfgan.eval.image_reshaper(images, num_cols=10)
#def float_image_to_uint8(image):
#  image = (image * 128.0)
#  return tf.cast(image, tf.uint8)
#  
#uint8_images = float_image_to_uint8(reshaped_images)
#image_write_ops = tf.write_file("./images.png", tf.image.encode_png(uint8_images[0]))
#tf.contrib.training.evaluate_repeatedly(FLAGS.train_log_dir, 
#                                        hooks=[tf.contrib.training.SummaryAtEndHook(FLAGS.train_log_dir), tf.contrib.training.StopAfterNEvalsHook(1)], 
#                                               eval_ops=image_write_ops, max_number_of_evaluations=1)
#
