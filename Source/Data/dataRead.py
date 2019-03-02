# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.12
by Frank
Data Read
"""

from BasicTool.Define import *
import pickle
import random
import cv2

def Load_Mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size

        return teX / 255., teY, num_te_batch


def Load_CIFAR_File(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')

    return dict


def Load_CIFAR_100(file, batch_size, is_training=True):
  
      if is_training:
          dict = Load_CIFAR_File(file)
          data = np.array(dict[b'data'])
          label = np.array(dict[b'fine_labels'])
          #data = data / 255.
          total = 50000
          trX = data[:total]
          trX = trX.reshape((total, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
          trX = trX.reshape(total, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
          trX = np.lib.pad(trX, ((0, 0), (4, 4), (4, 4),
                                 (0, 0)), mode='constant', constant_values=0)
          trY = label[:total]
  
          valX = data[45000:]
          valY = label[45000:]
  
          num_tr_batch = total // batch_size
          num_val_batch = 5000 // batch_size
  
          return trX, trY, num_tr_batch, valX, valY, num_val_batch
      else:
          dict = Load_CIFAR_File(file)
          data = np.array(dict[b'data'])
          label = np.array(dict[b'fine_labels'])
          #data = data / 255.
          total = 10000
          valX = data[:total]
          valX = valX.reshape((total, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
          valX = valX.reshape(total, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
  
          valY = label[:total]
  
          num_val_batch = total // batch_size
  
          return valX, valY, num_val_batch

def Load_CIFAR_10(file, batch_size, is_training=True):
    if is_training:
        data = []
        label = []
        for i in range(FILE_NUM):
            dict = Load_CIFAR_File(file + '/data_batch_%d' % (i+1))
            if i == 0:
                data = np.array(dict[b'data'])
            else:
                data = np.append(data, np.array(dict[b'data']), axis=0)
            label = np.append(label, np.array(dict[b'labels']))

        label = np.array(label)
        data = np.array(data)
        #data = data / 255.

        # the training and val set
        trX = data[:50000]
        trX = trX.reshape((50000, IMAGE_HEIGHT * IMAGE_WIDTH, IMAGE_DEPTH), order='F')
        trX = trX.reshape(50000, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)
        #trX = WhiteningImage(trX)
        trX = np.lib.pad(trX, ((0, 0), (4, 4), (4, 4),
                               (0, 0)), mode='constant', constant_values=0)
        trY = label[:50000]

        valX = data[45000:]
        valX = valX.reshape(5000, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)
        valY = label[45000:]

        num_tr_batch = 45000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch

    else:
        data = []
        label = []
        dict = Load_CIFAR_File(file + '/test_batch')
        data = np.array(dict[b'data'])
        label = np.append(label, np.array(dict[b'labels']))

        label = np.array(label)
        data = np.array(data)
        #data = data / 255.

        valX = data
        valX = valX.reshape((10000, IMAGE_HEIGHT * IMAGE_WIDTH, IMAGE_DEPTH), order='F')
        valX = valX.reshape(10000, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)
        #valX = WhiteningImage(valX)

        valY = label

        num_val_batch = 10000 // batch_size
        return valX, valY, num_val_batch