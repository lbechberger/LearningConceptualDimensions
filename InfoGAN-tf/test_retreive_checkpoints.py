#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:48:19 2018

code is a modified version of code found here: 
   https://stackoverflow.com/questions/39137597/how-to-restore-variables-using-checkpointreader-in-tensorflow

@author: christina
"""

import tensorflow as tf
import os
from datetime import datetime

v1 = tf.Variable(tf.ones([1]), name='v1')
v2 = tf.Variable(2 * tf.ones([1]), name='v2')
v3 = tf.Variable(3 * tf.ones([1]), name='v3')

saver = tf.train.Saver({'v2': v2, 'v3': v3})
timestamp = str(datetime.now()).replace(' ','-')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    cwd = os.getcwd()  #returns current working directory of a process
    checkpoint_dir = os.path.join(cwd+'/graphs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model = timestamp +'.model'
    checkpoint = os.path.join(checkpoint_dir, model)
    saver.save(sess, checkpoint)
    
with tf.Graph().as_default():
    assert len(tf.trainable_variables()) == 0
    v1 = tf.Variable(tf.zeros([1]), name='v1')
    v2 = tf.Variable(tf.zeros([1]), name='v2')

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
        print(sess.run([v1, v2])) # prints [array([ 0.], dtype=float32), array([ 2.], dtype=float32)]    