import os
import time
from constants import *

def create_directory(dir):
	try:
		os.stat(dir)
	except:
		os.mkdir(dir)