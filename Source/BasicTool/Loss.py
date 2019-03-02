# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.15
by Frank
Loss
"""

from BasicTool.Define import *
from BasicTool.Switch import *

class LossFunction(object):
    def __init__(self):
        pass

    def Inference(self, name, result, labels):
        for case in Switch(name):
            if case('MAE'):
                res = MAE_Loss(result, labels)
                break
            if case('Cross_Softmax_Entropy'):
                res = Cross_Softmax_Entropy(result, labels)
                break
            if case('Cross_Sigmoid_Entropy'):
                res = Cross_Sigmoid_Entropy(result, labels)
                break
            if case('Cross_Weight_Entroy'):
                res = Cross_Weight_Entroy(result, labels)
                break
            if case('MSE'):
                res = MSE_Loss(result, labels)
                break
            else:
                print('Loss Type Error!!!!!!!')

        return res

def MAE_Loss(res, labels):
    mae = tf.reduce_mean(tf.abs(labels-res))

    return mae


def Cross_Softmax_Entropy(res, labels):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=res))

    return cross_entropy

def Cross_Sigmoid_Entropy(res, labels):
    cross_sigmoid_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=res))
    return cross_sigmoid_entropy

def MSE_Loss(res, labels):
    mse = tf.reduce_mean(tf.square(labels-res))
    print('res:', res)
    return mse

def Cross_Weight_Entroy(res, labels):
    cross_weight_entropy = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=res, pos_weight=2))
    return cross_weight_entropy