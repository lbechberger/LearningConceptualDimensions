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

timestamp = str(datetime.now()).replace(' ', '-')

def check(expected, print_if_err):
    """Check if expected occurs.

    :param expected: true if correctly implemented
    :param error_msg: to print if expected is false
    :return: void
    """
    if not expected:
        print(print_if_err)
        for i in range(3):
            print("")
        assert expected



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

# False for normal running, start if you want it to enter the evaluation phase for each epoch
test = False

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
def infogan_generator(inputs, g_weight_decay_gen=9e-5):
    with tf.contrib.framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(g_weight_decay_gen)):
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
def infogan_discriminator(img, unused_conditioning, d_weight_decay_dis=9e-5, continuous_dim=2):
    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=_leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(d_weight_decay_dis),
            biases_regularizer=layers.l2_regularizer(d_weight_decay_dis)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
    logits_real = layers.fully_connected(net, 1, activation_fn=None)

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
    if options['type_latent'] == 'u':
        I = tf.ones([structured_continuous_dim]) + 1
        continuous_dist = ds.Uniform(-I, I)
        continuous_noise = continuous_dist.sample([batch_size])
    elif options['type_latent'] == 'n':
        continuous_noise = tf.random_normal([batch_size, structured_continuous_dim], mean=0.0, stddev=1.0)
    else:
        raise Exception("Unknown type of latent distribution: {0}".format(options['type_latent']))

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
    np.random.seed(seed=45)
    rows, cols = 20, len(continuous_sample_points) 

    # Take random draws for non-first-dim-continuous noise, making sure they are constant across columns.
    unstructured_noise = []
    for _ in xrange(rows):
        cur_sample = np.zeros((1, noise_dims))
        unstructured_noise.extend([cur_sample] * cols)
    unstructured_noise = np.concatenate(unstructured_noise)

    cont_noise_other_dim = []
    for _ in xrange(rows):
        if options['type_latent'] == 'u':
            cur_sample = np.random.uniform(low=-2, high=2, size=[1, latent_dims - 1])
        elif options['type_latent'] == 'n':
            cur_sample = np.random.normal(loc=0.0, scale=1.0, size=[1, latent_dims - 1])
        else:
            raise Exception("Unknown type of latent distribution: {0}".format(options['type_latent']))
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


def CodesInCodesOut(code):
    return sess.run(code)


def CodesInImageOut(image):
    return sess.run(image)


# Build the generator and discriminator.
discriminator_fn = functools.partial(infogan_discriminator, continuous_dim=options['latent_dims'],
                                     d_weight_decay_dis=options['d_weight_decay_dis'])
generator_fn = functools.partial(infogan_generator, g_weight_decay_gen=options['g_weight_decay_gen'])
unstructured_inputs, structured_inputs = get_training_noise(options['batch_size'], options['latent_dims'],
                                                            options['noise_dims'])

# Create the overall GAN
gan_model = tfgan.infogan_model(
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    real_data=batch_images_training,
    unstructured_generator_inputs=unstructured_inputs,
    structured_generator_inputs=structured_inputs)

# Build the GAN loss.
gan_loss = tfgan.gan_loss(gan_model, gradient_penalty_weight=1.0, mutual_information_penalty_weight=options['lambda'],
                          add_summaries=True)

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
    steps = int((epoch * length_of_data_set) / options['batch_size'])
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
        # train the network
        cur_loss, _ = train_step_fn(sess, train_ops, global_step, train_step_kwargs={})
        loss_values.append((step, cur_loss))
        if step % 100 == 0:
            print('Current loss: %f' % cur_loss)

        eval_condition = True
        if(not test):
            eval_condition = (step + 1) in num_steps.keys()
        if eval_condition:
            # finished an epoch
            epoch = 1
            if(not test):
                epoch = num_steps[step + 1]
            print("finished epoch {0}".format(epoch))

            # now evaluate the current latent codes

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

            # generate 'noise' (i.e. a white image) to feed it into the generator with the latent code
            eval_noise = tf.zeros((options['batch_size'], options['noise_dims']))

            #2) Image reconstruction Error

            # temp is only needed as an input to gan_model.generator_fn(temp
            temp = (eval_noise, latent_code)

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
            check(eucl_dist_images.shape == (length_of_data_set, ), eucl_dist_images.shape)

            avg_eucl_dist_images = np.mean(eucl_dist_images)
            print(avg_eucl_dist_images)


            #image_tensors_from_images = real_images
            
            
            # 3) Latent Code Reconstruction error

            # sample batch_size many latent codes either from a uniform or normal distribution
            if options['type_latent'] == 'u':
                latent_code_batch = tf.random_uniform([options['batch_size'], options['latent_dims']], minval=-2,
                                                      maxval=2, seed=45)
            elif options['type_latent'] == 'n':
                latent_code_batch = tf.random_normal([options['batch_size'], options['latent_dims']], mean=0.0,
                                                     stddev=1.0, seed=45)
            else:
                raise Exception("Unknown type of latent distribution: {0}".format(options['type_latent']))

            # prüfen ob random mit seed pro session gleiche werte ausgibt oder auch in session

            # generate an image with the sampled latent code batch
            # feed the generated image into the discriminator and save the reconstructed latent code
            with tf.variable_scope(gan_model.generator_scope, reuse=True):
                generated_image = generator_fn([eval_noise, latent_code_batch])
            with tf.variable_scope(gan_model.discriminator_scope, reuse=True):
                rec_lat_code = (discriminator_fn(generated_image, None)[1][0]).loc

            reconstructed_latCode = CodesInCodesOut(rec_lat_code)
            l1_recLC_error = 1 #calculate manh. distance between reconstructed_latCode and latent_code_batch
            l2_recLC_error = 1 #calculate eukl. distance between reconstructed_latCode and latent_code_batch



            # 4) create some output images for the current epoch using 20 sequences of variing fixed dimension in latent code
            CONT_SAMPLE_POINTS = np.linspace(-2, 2, 20)
            for i in range(options['latent_dims']):
                display_noise = get_eval_noise(options['noise_dims'], CONT_SAMPLE_POINTS, options['latent_dims'], i)
                with tf.variable_scope(gan_model.generator_scope, reuse=True):
                    continuous_image = gan_model.generator_fn(display_noise)
                reshaped_continuous_image = tfgan.eval.image_reshaper(continuous_image,
                                                                      num_cols=len(CONT_SAMPLE_POINTS))
            codeInImOut = CodesInImageOut(reshaped_continuous_image)

            '''
            uint8_continuous = float_image_to_uint8(reshaped_continuous_image)

            image_write_op = tf.write_file(os.path.join(options['output_dir'], "{0}-ep{1}-{2}_dim{3}.png".format(config_name, epoch, timestamp, i)),
                                               tf.image.encode_png(uint8_continuous[0]))
            sess.run(image_write_op)
            '''

            #dump all of this into a pickle file for later use --'l2_recIm_error': avg_eucl_dist_images,
            eval_outputs = {'codes_from_all_images': codes_from_all_images,
                            #'l1_LatCode_rec_error': l1_recLC_error,
                            #'l2_latCode_rec_error': l2_recLC_error,
                            'output_images_variing_lat_code': codeInImOut}
            with open(os.path.join(options['output_dir'], "eval-{0}-ep{1}-{2}.pickle".format(config_name, epoch, timestamp)), 'wb') as f:
               pickle.dump(eval_outputs, f)

            
            """
            Confirm latent_code's dim is 128, 2
            print(sess.run(latent_code).shape)
            """

            """
            Confirm iterator works as intended
            #print(sess.run(real_images))
            #print(sess.run(real_images)[0][0] == images[0][0])
            """
            """
            # Confirm first and last batch are the same
            # Can't be done if we let it run in the loop below, since the iterator will get called for evaluation_output
            for i in range(num_eval_steps):

                rows = sess.run(evaluation_output)
                table.append(rows)
            table = np.concatenate(table, axis=0)

                real = sess.run(real_images)
                if (i == 0):
                    real = real[0][0]
                    unmod = images[0][0]
                    expected = np.all(real == unmod)
                    if not expected:
                        print(real)
                        print(' ')
                        print(unmod)
                        assert expected
                if (i == num_eval_steps - 1):
                    real = real[-1][0]
                    unmod = images[-1][0]
                    expected = np.all(real == unmod)
                    if not expected:
                        print(real)
                        print(' ')
                        print(unmod)
                        assert expected
            """

            """
            #MF: Confirm concatenation went right
            expected = np.all(whole_images_from_images == images)
            if not expected:
                print(len(whole_images_from_images))
                assert expected
            """



