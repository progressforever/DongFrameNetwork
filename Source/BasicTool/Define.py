# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.15
by Frank
Define
"""

import os
import sys
import time
import tensorflow as tf
import numpy as np

#The root path of data, save model and result
DATA_DIR = "/home1/Documents/Database/cifar-10-batches-py"
TRAIN_DATA_DIR = "/home1/Documents/Database/cifar-10-batches-py"
TEST_DATA_DIR = "/home1/Documents/Database/cifar-10-batches-py"
OUTPUT_RESULT_LOG = "../Result/ResultLog/"
OUTPUT_RESULT = "../Result/ResultInfo/"
OUTPUT_IMG_DIR = "../Result/ResultImg/"
OUTPUT_MODEL_DIR = "../Result/ResultModel/"
PATH_PREFIX = "../Result/"

#The network and loss
MODEL_TYPE = "AlexNet"
LOSS_TYPE = "MSE"
ACC_TYPE = "Acc_Mnist"

#The batchsize, epoch, learning rate and gamma
BATCHSIZE_TRAIN = 128
BATCHSIZE_TEST = 128
EPOCH = 100
BASE_LEARNING_RATE = 0.01
GAMMA = 0.1
DELAY_SHOT = 10000

#Save the model
SAVE_SHOT = 10
PRINT_SHOT = 1000

#Restore the model
PRETRAIN_MODEL = "../Result/ResultModel/"
MODEL_NAME = "model.ckpt"

#Image size
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
CROP_WIDTH = 32
CROP_HEIGHT = 32

#Data process
IS_RANDOM_LABEL = False
IS_AUGMENT = True
PADDING_SIZE = 4

NUM_CLASS = 10
FILE_NUM = 5

#Net parameters
BN_EPSILON = 0.001
WEIGHT_DECAY = 0.0002

LOG_FILE = 'output.log'