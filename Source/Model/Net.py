# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.15
by Frank
Network
"""

from BasicTool.Define import *
from Model.BasicNetArchitecture import *
from BasicTool.Switch import *
import math

class NetWorks(object):
	def __init__(self):
		pass

	def Inference(self, name, x, is_training):
		for case in Switch(name):
			if case('AlexNet'):
				res = AlexNet(x, is_training=is_training)
			else:
				print('NetWork Type Error~')

		return res

def AlexNet(x, is_training):
	layers = []
	with tf.variable_scope('conv0'):
		conv0 = Conv2d(x, [3, 3, 3, 16], 1)
		activation_summary(conv0)
		layers.append(conv0)

	with tf.variable_scope('pool0'):
		pool0 = maxPooling2d(conv0, [1, 3, 3, 1], 1)
		activation_summary(pool0)
		layers.append(pool0)

	with tf.variable_scope('conv1'):
		conv1 = Conv2d(pool0, [3, 3, 16, 32], 2)
		activation_summary(conv1)
		layers.append(conv1)

	with tf.variable_scope('conv2'):
		conv2 = Conv2d(conv1, [3, 3, 32, 64], 2)
		activation_summary(conv2)
		layers.append(conv2)

	with tf.variable_scope('fc'):
		in_channel = layers[-1].get_shape().as_list()[-1]
		bn = BN(layers[-1], in_channel)
		relu = tf.nn.relu(bn)
		global_pool = tf.reduce_mean(relu, [1, 2])

		output = output_layer(global_pool, 10)
		layers.append(output)

	return layers[-1]
