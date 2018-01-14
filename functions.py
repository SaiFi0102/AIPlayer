import os

def create_directory(dir):
	try:
		os.stat(dir)
	except:
		os.mkdir(dir)