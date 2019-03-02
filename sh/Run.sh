#!/bin/bash
echo $"Begin our new travel~"
CUDA_VISIBLE_DEVICES=1 nohup python ../Source/main.py \
	--gpuNumber 1 \
	--modelType AlexNet \
	--lossType MSE \
	--accType Acc_Mnist \
	--batchsizeTrain 50 \
	--batchsizeTest 50 \
	--epoch 2 \
	--baseLearningRate 0.01 \
	--delayShot 50 \
	--saveShot 50 \
	--printShot 1 \
	--outputResultInfo ../Result/ResultInfo/Test/ \
	--pretrain False > ../Result/ResultRun/test.log 2>&1 &
echo $"The program travel begins~"
echo $"You can get the result of this new travel from ./Result/ResultRun/"
echo $"If you still have problems with this program, you can email me with 962360190@qq.com" 
