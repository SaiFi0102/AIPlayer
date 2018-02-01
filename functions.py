from constants import *
import os
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

class PlotLosses(keras.callbacks.Callback):
	def __init__(self):
		self.fig = plt.figure()
	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.losses = []
		self.val_losses = []
		self.accOnes = []
		self.val_accOnes = []
		self.logs = []
	def on_epoch_end(self, epoch, logs={}):
		self.logs.append(logs)
		self.x.append(self.i)
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.accOnes.append(logs.get('onesAccuracy'))
		self.val_accOnes.append(logs.get('val_onesAccuracy'))
		self.i += 1

		plt.clf()
		plt.plot(self.x, self.losses, label="Loss")
		plt.plot(self.x, self.val_losses, label="Validation loss")
		plt.plot(self.x, self.accOnes, label="Ones accuracy")
		plt.plot(self.x, self.val_accOnes, label="Validation ones accuracy")
		plt.legend()
		plt.draw()

# Metrics
def accuracy(noteStates, decoded):
	return 1 - K.mean(K.abs(noteStates - decoded))
def onesAccuracy(true, prediction):
	onesInTrue = K.sum(true)
	rOnesEqual = K.sum(true * prediction)
	acc = rOnesEqual / onesInTrue;
	acc = tf.where(tf.is_nan(acc), tf.ones_like(acc), acc)
	return acc;

def create_directory(dir):
	try:
		os.stat(dir)
	except:
		os.mkdir(dir)