# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.15
by Frank
Evaluation
"""

from BasicTool.Define import *
from BasicTool.Switch import *

class AccFunction(object):
    def __init__(self):
        pass

    def Inference(self, name, result, labels, k=1):
        for case in Switch(name):
            if case('Acc_Cifar10'):
                acc = Acc_Cifar10(result, labels)
                break
            if case('top_k_error'):
                acc = top_k_error(result, labels, k)
                break
            if case('Acc_Mnist'):
                acc = Acc_Mnist(result, labels)
            else:
                print('Loss Type Error!!!!!!!')

        return acc

def Acc_Cifar10(res, labels):
    batch_size = res.get_shape().as_list()[0]
    in_top1 = tf.nn.in_top_k(res, labels, k=1)
    in_top1 = tf.to_float(in_top1)
    num_correct = tf.reduce_sum(in_top1)
    acc = float(num_correct / batch_size)

    return acc


def top_k_error(res, labels, k):
        '''
        Calculate the top-k error
        :param res: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = res.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(res, labels, k))
        num_correct = tf.reduce_sum(in_top1)
        acc = (batch_size - num_correct) / float(batch_size)

        return acc

def Acc_Mnist(res, labels):
    correct_prediction = tf.equal(tf.argmax(res, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy