# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.16
by Frank
main function
"""

from BasicTool.Define import *
from BasicTool.Argparser import *
from Network.ModelRun import *

# build the folder
def InitPro(args):
    # check the space
    Mkdir(args.outputResultInfo)
    InitLog(args.outputResultInfo + LOG_FILE, args.pretrain)
    Info("Finish init work")

def main(args):
	#args = parse_args()

	train = TrainNetWork(args)
	train.NetRun()

if __name__ == "__main__":
	args = parse_args()
	InitPro(args)
	main(args)