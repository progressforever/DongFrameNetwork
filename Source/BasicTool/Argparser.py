# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.12
by Frank
ArgParser
"""

#import the library
from argparse import ArgumentParser
from BasicTool.Define import *

def parse_args():
	"""
	Parse input arguments
	"""
	parser = ArgumentParser(description = "task of this work")
	#parser.add_argument('--phase', default = 'train', help = 'Select the phase of process, training or testing.')
	parser.add_argument('--gpuNumber', default = 1, type=int, help = 'Select the number of gpu to be used.')
	
	#The root path of data, save model and result
	parser.add_argument('--dataDir', default = DATA_DIR, help = 'Select the root path of dataset.')
	parser.add_argument('--trainDataDir', default = TRAIN_DATA_DIR, help = 'Select the root path of train dataset.')
	parser.add_argument('--testDataDir', default = TEST_DATA_DIR, help = 'Select the root path of test dataset.')
	parser.add_argument('--outputResultLog', default = OUTPUT_RESULT_LOG, help = 'The root path of output result.')
	parser.add_argument('--outputResultInfo', default = OUTPUT_RESULT)
	parser.add_argument('--outputImgDir', default = OUTPUT_IMG_DIR, help = 'The root path of output image.')
	parser.add_argument('--outputModel', default = OUTPUT_MODEL_DIR, help = 'The root path of saving models.')
	parser.add_argument('--pathPrefix', default = PATH_PREFIX, help = 'A common prefix to path of all the input images')
	
	#Select the network and loss
	parser.add_argument('--modelType', default = MODEL_TYPE, help = 'Select the network type.')
	parser.add_argument('--lossType', default = LOSS_TYPE, help = 'Select the loss type.')
	parser.add_argument('--accType', default = ACC_TYPE, help = 'Select the acc type.')
	
	#The batchsize, epoch, learning rate, gamma
	parser.add_argument('--batchsizeTrain', default = BATCHSIZE_TRAIN, type=int, help = 'The batch size of each step during training process.')
	parser.add_argument('--batchsizeTest', default = BATCHSIZE_TEST, type=int, help = 'The batch size of each step during testing process.')
	parser.add_argument('--epoch', default = EPOCH, type=int, help = 'The epoch of training process.')
	parser.add_argument('--baseLearningRate', default = BASE_LEARNING_RATE, type=float, help = 'The origin learning rate.')
	parser.add_argument('--gamma', default = GAMMA, type=float, help = 'Learning rate decay rate.')
	parser.add_argument('--delayShot', default = DELAY_SHOT, type=int, help = 'The shot step of delaying learning rate.')

	#Save the model
	parser.add_argument('--saveShot', default = SAVE_SHOT, type=int, help = 'The shot step of saving the model.')
	parser.add_argument('--printShot', default = PRINT_SHOT, type=int, help = 'The shot step of printing result once.')
	#Restore the model
	parser.add_argument('--pretrain', default = True, help = 'Whether to load the pretrain model.')
	parser.add_argument('--pretrainModel', default = PRETRAIN_MODEL, help = 'The path to restore the pretrain model.')

	#image size
	parser.add_argument('--maxDepth', default = 128, type=int, help = 'Maximum depth step when training.')
	parser.add_argument('--maxWidth', default = 640, type=int, help = 'Maximum image width when training.')
	parser.add_argument('--maxHeight', default = 512, type=int, help = 'Maximum image height when training.')

	#image post processing
	parser.add_argument('--cropWidth', default = CROP_WIDTH, type=int, help = 'The width of image random crop operation.')
	parser.add_argument('--cropHeight', default = CROP_HEIGHT, type=int, help = 'The height of image random crop operation.')
	parser.add_argument('--shuffle', default = True, help = 'Whether random label or not.')
	parser.add_argument('--isRandomLabel', default = IS_RANDOM_LABEL, help = 'Whether random label or not.')
	parser.add_argument('--isAugment', default = IS_AUGMENT, help = 'Whether augument or not.')
	parser.add_argument('--paddingSize', default = PADDING_SIZE, help = 'The size of padding.')


	args = parser.parse_args()
	return args



