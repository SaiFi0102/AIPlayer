from constants import *
import os
import data
import functions as utils
from functions import accuracy, onesAccuracy
import pygame
import random
from keras import callbacks
from keras.models import Sequential
from keras.layers import LSTM, CuDNNLSTM, Input, Dense, Dropout, LeakyReLU
import numpy as np
import tensorflow as tf
import threading
import matplotlib.pyplot as plt

# Initialize music mixer
pygame.mixer.init()

# Data
noteStateSeq = data.loadCacheData()
inputTrain, inputVal, outputTrain, outputVal = data.seqToDataset(noteStateSeq)

# Model
def buildModel():
	model = Sequential()
	model.add(CuDNNLSTM(300, batch_size=N_BATCH_SIZE, input_shape=(N_INPUT_UNITS, PITCH_COUNT),
				   return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LeakyReLU())
	model.add(CuDNNLSTM(300, return_sequences=True, stateful=True))
	model.add(Dropout(0.5))
	model.add(LeakyReLU())
	model.add(CuDNNLSTM(300, return_sequences=True, stateful=True))
	model.add(Dropout(0.5))
	model.add(LeakyReLU())
	model.add(Dense((PITCH_COUNT), activation="sigmoid"))
	model.compile(optimizer='adam',
				  loss='binary_crossentropy',
				  metrics=[accuracy, onesAccuracy])

	wPath = os.path.join(DATA_FOLDER, "weights.w")
	if os.path.isfile(wPath):
		print("Loading weights file")
		model.load_weights(wPath)
	else:
		print("No weights file found! Not loading weights")

	print(model.layers[0].input_shape)
	for layer in model.layers:
		print(layer.output_shape)

	return model

# Training
def trainModel(trainingPlot):
	model = buildModel()
	checkpoint = callbacks.ModelCheckpoint(os.path.join(DATA_FOLDER, "weights.w"),
		monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=10)
	model.fit(inputTrain, outputTrain, validation_data=(inputVal, outputVal), shuffle=False,
		epochs=TRAIN_EPOCHS, batch_size=N_BATCH_SIZE, callbacks=[trainingPlot, checkpoint])

# Training!
trainingPlot = utils.PlotLosses()
plt.ion()
plt.show()
pTrain = threading.Thread(target=trainModel, args=[trainingPlot])
pTrain.start()
while plt.get_fignums():
	plt.pause(0.2)
pTrain.join()

# Predicting!
'''
model = buildModel()
x = random.choice(inputTrain)
while(x.sum() == 0):
	x = random.choice(inputTrain)
x = x.reshape((1, x.shape[0], x.shape[1]))

output = []
output.extend(x.reshape(x.shape[1], x.shape[2]))
for i in range(OUTPUT_DURATION_SAMPLES-1):
	x = model.predict(x, batch_size=1, verbose=1) > N_PLAY_THRESHOLD
	output.extend(x.reshape(x.shape[1], x.shape[2]))

output = np.asarray(output)
ostream = data.noteStateSeqToMidiStream(output)
print("Playing!")
pygame.mixer.music.load(ostream)
pygame.mixer.music.play()
'''

# Wait for music to finish
while pygame.mixer.music.get_busy():
	pygame.time.wait(100)
