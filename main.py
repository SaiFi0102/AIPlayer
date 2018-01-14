from constants import *
import data
import pygame
from keras.models import Sequential
from keras.layers import LSTM, Input, Dense, Dropout, LeakyReLU
import keras.backend as K
import tensorflow as tf
from keras import metrics

# Initialize music mixer
pygame.mixer.init()

# Data
noteStateSeq = data.loadCacheData()
inputTrain, inputVal, outputTrain, outputVal = data.seqToDataset(noteStateSeq)

#inputTrain = inputTrain.reshape([-1, N_BATCH_SIZE, N_INPUT_UNITS, PITCH_COUNT])
#outputTrain = outputTrain.reshape([-1, N_BATCH_SIZE, N_OUTPUT_UNITS, PITCH_COUNT])

def accuracy(noteStates, decoded):
	return 100.0 * (1 - K.mean(K.abs(noteStates - decoded)))
def onesAccuracy(true, prediction):
	onesInTrue = K.sum(true)
	rOnesEqual = K.sum(true * prediction)
	acc = rOnesEqual / onesInTrue;
	acc = tf.where(tf.is_nan(acc), tf.ones_like(acc), acc)
	return acc*100;
def trueOnes(true, prediction):
	return K.sum(true)
def trueZeros(true, prediction):
	return tf.cast(tf.size(true), tf.float32) - K.sum(true)

model = Sequential()
model.add(LSTM(40, batch_size=N_BATCH_SIZE, input_shape=(N_INPUT_UNITS, PITCH_COUNT),
			   return_sequences=True, dropout=0.3))
model.add(LeakyReLU())
model.add(LSTM(80, return_sequences=True, dropout=0.5))
model.add(LeakyReLU())
model.add(LSTM(80, return_sequences=True, dropout=0.5))
model.add(LeakyReLU())
model.add(LSTM(80, return_sequences=True, dropout=0.5))
model.add(LeakyReLU())
model.add(LSTM(40, return_sequences=True, dropout=0.5))
model.add(LeakyReLU())
model.add(Dense(PITCH_COUNT, activation="sigmoid"))
model.compile(optimizer='adam', loss='mae', metrics=[accuracy, onesAccuracy, trueOnes, trueZeros])

print(model.layers[0].input_shape)
for layer in model.layers:
	print(layer.output_shape)

model.fit(inputTrain, outputTrain, epochs=N_TRAIN_EPOCHS, batch_size=N_BATCH_SIZE)

# Wait for music to finish
while pygame.mixer.music.get_busy():
	continue
