# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.15
by Frank
BasicNetwork Architectures
"""

import numpy as np
from BasicTool.Define import *

def activation_summary(x):
	tensor_name = x.op.name
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
	regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
	new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)

	return new_variables

def output_layer(input_layer, num_labels):
	input_dim = input_layer.get_shape().as_list()[-1]
	fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
							initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
	fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())
	fc_h = tf.matmul(input_layer, fc_w) + fc_b

	return fc_h

def BN(input_layer, dimension):
	mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
	beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
	gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
	bn = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

	return bn

def Conv2d(input_layer, filter_shape, stride):
	out_channel = filter_shape[-1]
	filter = create_variables(name='conv', shape=filter_shape)

	conv = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
	bn = BN(conv, out_channel)
	relu = tf.nn.relu(bn)

	return relu

#def DeConv3d(input_layer, filter_shape, stride):
#	out_channel = filter_shape[-1]
#	filter = create_variables(name='deconv', shape=filter_shape)

#	deconv = tf.nn.

def Conv3d(input_layer, filter_shape, stride):
	out_channel = filter_shape[-1]
	filter = create_variables(name='conv3d', shape=filter_shape)

	conv3d = tf.nn.conv3d(input_layer, filter, strides=[1, stride, stride, stride, 1], padding='SAME')
	bn = BN(conv3d, out_channel)
	relu = tf.nn.relu(bn)

	return relu


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)

    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

    return conv_layer

def maxPooling2d(input_layer, filter_shape, stride):
	#filter = create_variables(name='pool', shape=filter_shape)
	maxPooling = tf.nn.max_pool(input_layer, ksize=filter_shape, strides=[1, stride, stride, 1], padding='VALID')

	return maxPooling

def maxPooling3d(input_layer, filter_shape, stride):
	maxPooling3d = tf.nn.max_pool3d(input_layer, ksize=filter_shape, strides=[1, stride, stride, 1], padding='VALID')

	return maxPooling3d

def avgPooling2d(input_layer, filter_shape, stride):
	avgPooling = tf.nn.avg_pool(input_layer, ksize=filter_shape, strides=[1, stride, stride, 1], padding='VALID')

	return avgPooling

def avgPooling3d(input_layer, filter_shape, stride):
	avgPooling3d = tf.nn.avg_pool3d(input_layer, ksize=filter_shape, strides=[1, stride, stride, 1], padding='VALID')

	return avgPooling3d

		