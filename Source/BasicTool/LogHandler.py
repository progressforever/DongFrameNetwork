# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.17
by Frank
Loghandler
"""

import logging
import os
from BasicTool.Define import *

LOG_FORMAT = '[%(levelname)s] %(asctime)s %(filename)s[line:%(lineno)d]: %(message)s'
LOG_DATE_FORMAT = '[%a] %Y-%m-%d %H:%M:%S'

def InitLog(path, renew):
	if renew and os.path.exists(path):
		os.remove(path)

	logging.basicConfig(level=logging.INFO,
						format=LOG_FORMAT,
						datefmt=LOG_DATE_FORMAT,
						filename=path,
						filemode='a')
	return

def Info(str):
	print("[INFO]" + str)
	logging.info(str)

def Debug(str):
	print("[DEBUG" + str)
	logging.debug(str)

def Warning(str):
	print("[WARNING]" + str)
	logging.warning(str)

def Error(str):
	print("[ERROR]" + str)
	logging.error(str)

def Mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    # check the file pat
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

    return