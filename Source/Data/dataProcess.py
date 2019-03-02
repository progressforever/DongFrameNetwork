# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.15
by Frank
Data process
"""

import tarfile
from six.moves import urllib
import sys
import numpy as np
import pickle
import os
import cv2
from BasicTool.Define import *

def data_process(data, label, is_flip, is_random_crop, is_whiten, is_shuffle):
    if is_flip is True:
        process_data = horizontal_flip(data, axis=1)
    if is_random_crop is True:
        process_data = random_crop(process_data, padding_size=PADDING_SIZE)
    if is_whiten is True:
        process_data = whitening_image(process_data)
    if is_shuffle is True:
        process_data, process_label = shuffle_data(process_data, label)

    return process_data, process_label


#Horizontal flip
def horizontal_flip(batch_data, axis):
    for i in range(len(batch_data)):
        flip_prop = np.random.randint(low=0, high=2)
        if flip_prop == 0:
            batch_data[i, ...] = cv2.flip(batch_data[i, ...], axis)

    return batch_data

#Whitening image
def whitening_image(image_np):
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        std = np.max([np.std(image_np[i, ...]), 1.0 / np.sqrt(IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH)])
        image_np[i, ...] = (image_np[i, ...] - mean) / std

    return image_np

#Random crop
def random_crop(batch_data, padding_size):
    cropped_batch = np.zeros(len(batch_data) * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH)
    cropped_batch = cropped_batch.reshape(len(batch_data), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset : x_offset+CROP_HEIGHT, y_offset : y_offset+CROP_WIDTH, :]

    return cropped_batch

#Shuffle data
def shuffle_data(data, label):
    num_data = len(label)
    order = np.random.permutation(num_data)
    shuffle_data = data[order, ...]
    shuffle_label = label[order]

    return shuffle_data, shuffle_label

