# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.16
by Frank
Model run
"""

from BasicTool.Define import *
from BasicTool.Argparser import *
from BasicTool.Evaluation import *
from BasicTool.LogHandler import *
from BasicTool.Loss import *
from Data.dataLoad import *
from Model.Net import NetWorks
import random
from datetime import datetime
import time
import os
#import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#sys.path.append("../")

class TrainNetWork(object):
	def __init__(self, args):
		self.args = args
		self.checkpoint_path = os.path.join(self.args.outputModel, "model.ckpt")
		#self.train_acc, self.train_loss, self.val_acc, self.val_loss = CreateResultFile(args)
		self.placeholders()
		self.train_step, self.res, self.loss, self.acc = self.BuildNet(self.x, self.labels)

		#Session
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
		writer = tf.summary.FileWriter(self.args.outputResultLog, self.sess.graph)
		writer.close()
		Info("Finish training init work")

	def placeholders(self):
		self.x = tf.placeholder(tf.float32, shape=[self.args.batchsizeTrain, 32, 32, 3])
		self.y = tf.placeholder(tf.int32, shape=[self.args.batchsizeTrain])
		self.labels = tf.one_hot(self.y, depth=10, axis=1, dtype=tf.float32)

	def BuildNet(self, x, labels):
		global_step = tf.Variable(0, trainable=False)
		res = NetWorks().Inference(self.args.modelType, x, is_training = True)
		loss = LossFunction().Inference(self.args.lossType, res, labels)
		res = tf.nn.softmax(res)
		acc = AccFunction().Inference(self.args.accType, res, labels, k=1)
		opt = tf.train.AdamOptimizer(self.args.baseLearningRate)

		train_step = opt.minimize(loss, global_step=global_step)

		Info("The train graph builds completely.")

		return train_step, res, loss, acc

	def Count(self):
		total_parameters = 0
		for variable in tf.trainable_variables():
			shape = variable.get_shape()
			variable_parameters = 1

			for dim in shape:
				variable_parameters *= dim.value
			total_parameters += variable_parameters
		info = 'The total parameters are %d' % total_parameters
		Info(info)

	def RestoreModel(self):
		ckpt = tf.train.get_checkpoint_state(self.args.outputModel)
		if ckpt and ckpt.model_checkpoint_path:
			tf.train.Saver(var_list=tf.global_variables()).restore(self.sess, ckpt.model_checkpoint_path)
			Info('The model parameters are restored from ', ckpt.model_checkpoint_path)
		else:
			Warning('No checkpoint.')


	def Train(self, args, epoch, train_data, train_label, train_batch, batchsizeTrain):
		train_avg_acc = 0
		train_avg_loss = 0
		start_time = time.time()
		for step in range(train_batch):
			train_data_batch, train_label_batch = load_processed_data_batch(train_data, train_label, batchsizeTrain, step, is_flip=True, is_random_crop=True, is_whiten=True, is_shuffle=True)
			#print('train_data:',train_data_batch.shape)
			_, _, loss, acc = self.sess.run([self.train_step, self.res, self.loss, self.acc],
											feed_dict={self.x: train_data_batch, self.y: train_label_batch})

			train_avg_acc += acc
			train_avg_loss += loss

		duration = time.time() - start_time
		train_avg_acc = train_avg_acc / train_batch
		train_avg_loss = train_avg_loss / train_batch

		Info('%s: epoch %d, train acc = %.4f, train loss = %.4f, duration = %.1f sec' % (datetime.now(), epoch, train_avg_acc, train_avg_loss, duration))

	def Test(self, args, epoch, test_data, test_label, test_batch, batchsizeTest):
		test_avg_acc = 0
		test_avg_loss = 0
		start_time = time.time()
		for i in range(test_batch):
			batch_data, batch_label = load_test_batch(test_data, test_label, batchsizeTest, i)
			loss, acc = self.sess.run([self.loss, self.acc], feed_dict={self.x: batch_data, self.y: batch_label})
			test_avg_acc += acc
			test_avg_loss += loss

		duration = time.time() - start_time
		test_avg_acc = test_avg_acc / test_batch
		test_avg_loss = test_avg_loss / test_batch

		Info('%s: epoch %d, val acc = %.4f, val loss = %.4f, duration = %.1f sec' % (datetime.now(), epoch, test_avg_acc, test_avg_loss, duration))


	def NetRun(self):
		print('Start train work.')
		args = self.args
		if not args.pretrain:
			self.RestoreModel()

		train_data, train_label, train_batch, val_data, val_label, val_batch, test_data, test_label, test_batch = load_data(args.dataDir, args.batchsizeTrain, args.batchsizeTest)

		for epoch in range(args.epoch):
			self.Train(args, epoch, train_data, train_label, train_batch, args.batchsizeTrain)

			if (epoch+1) % 1 == 0:
				self.Test(args, epoch, test_data, test_label, test_batch, args.batchsizeTest)

			if (epoch % args.saveShot == 0) and (epoch != 0):
				self.saver.save(self.sess, self.checkpoint_path, global_step=epoch)
				Info('The model has been created')

#def Run(args):
	#args = parse_args()
#	train = TrainNetWork(args)
#	train.NetRun()




