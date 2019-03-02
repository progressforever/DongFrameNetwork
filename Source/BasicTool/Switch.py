# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Copyright 2018.10.15
by Frank
Switch
"""

class Switch(object):
	def __init__(self, value):
		self.value = value
		self.fall = False

	def __iter__(self):
		yield self.match
		raise StopIteration

	def match(self, *args):
		if self.fall or not args:
			return True
		elif self.value in args:
			self.fall = True
			return True
		else:
			return False