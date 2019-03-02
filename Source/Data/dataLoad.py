# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.12
by Frank
Data Load
"""
from BasicTool.Define import *
from Data.dataProcess import *
from Data.dataRead import *

def load_data(dataDir, batchsizeTrain, batchsizeTest):
	train_data, train_label, train_batch, val_data, val_label, val_batch = Load_CIFAR_10(dataDir, batchsizeTrain)
	test_data, test_label, test_batch = Load_CIFAR_10(dataDir, batchsizeTest, is_training=False)

	return train_data, train_label, train_batch, val_data, val_label, val_batch, test_data, test_label, test_batch

def load_processed_data_batch(data, label, batchsize, order, is_flip, is_random_crop, is_whiten, is_shuffle):
	offset = order * batchsize
	batch_data = data[offset : offset+batchsize, ...]
	batch_label = label[offset : offset+batchsize]
	data, label = data_process(batch_data, batch_label, is_flip=is_flip, is_random_crop=is_random_crop, is_whiten=is_whiten, is_shuffle=is_shuffle)

	return data, label

def load_test_batch(data, label, batchsize, order):
	offset = order * batchsize
	batch_data = data[offset : offset + batchsize, ...]
	batch_label = label[offset : offset + batchsize]

	return batch_data, batch_label