from __future__ import print_function
from __future__ import absolute_import
import sys; sys.path.append('')
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from infogan.misc.datasets import ShapeDataset
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime
import numpy as np
import pickle

if __name__ == "__main__":

    # the date is needed for the logs
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/shape"
    root_checkpoint_dir = "ckt/shape"

    # Batch-size and epochs to be used
    batch_size = 128
    updates_per_epoch = 100
    max_epoch = 10#100

    exp_name = "shape_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    # load the shape-data
    liste = pickle.load(open('images.pkl', 'rb'))

    trainMatrixSample = np.array(liste[0])
    trainLabelsSample = np.array(liste[1])
    testMatrixSample = np.array(liste[2])   # currently not used
    testLabelsSample = np.array(liste[3])   # currently not used

    print(len(trainMatrixSample), len(trainMatrixSample[0]))
    # ShapeDataset has a fixed height and widht of 28 pixels and only one 
    # matrix per image (grey-scale)
    dataset = ShapeDataset(trainMatrixSample, trainLabelsSample)


    # the variables used for the generation of shapes
    # Uniform variables are only supported with a value of 1
    # Possible would be also e.g. Categorical(10), True) for 
    # a categorical variable with 10 Steps.
    latent_spec = [
        (Uniform(62), False),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
    ]

    model = RegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist",
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
    )

    algo.train()
