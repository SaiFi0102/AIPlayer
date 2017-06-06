import theano, keras, seq2seq, lasagne, collections, time
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from constants import *
from data import *
from keras.models import Sequential, model_from_json
from keras.layers import Dense, TimeDistributedDense, Activation
from seq2seq.models import AttentionSeq2seq
#from seq2seq.layers.encoders import LSTMEncoder
#from seq2seq.layers.decoders import AttentionDecoder

def buildNetwork(loadSerialized=False, loadParams=False):
	if type(loadSerialized) == str:
		with open(loadSerialized, 'r') as f:
			model = model_from_json(f.read())
	else:
		model = Sequential()
		seq2seq = AttentionSeq2seq(
			#input_length=N_INPUT_UNITS,
			#input_dim=PITCH_COUNT,
			batch_input_shape=(GPU_BATCH_SIZE, N_INPUT_UNITS, PITCH_COUNT,),
			hidden_dim=NSE_ENCODED_SIZE,
			output_length=N_OUTPUT_UNITS,
			output_dim=PITCH_COUNT,
			dropout=0.3,
			depth=2,
			bidirectional=False,
			stateful=True
		)

		#model.add(TimeDistributedDense(NSE_ENCODED_SIZE, input_shape=(N_INPUT_UNITS, PITCH_COUNT)))
		model.add(seq2seq)
		#model.add(TimeDistributedDense(PITCH_COUNT))
		model.add(Activation('sigmoid'))

		print("Compiling model...")
		model.compile('adadelta', 'binary_crossentropy')
		with open('data/serialized_latest.json', 'w+') as f:
			f.write(model.to_json())

	if type(loadParams) == str:
		print("Loading network params...")
		model.load_weights(loadParams)

	return model

def trainNetwork(model):
	print("Loading music dataset...")
	trainInput, valInput, trainOutput, valOutput = loadDataset()

	print("Starting network training...")
	checkpointer = keras.callbacks.ModelCheckpoint(filepath="data/best.w", verbose=0, save_best_only=True)

	model.fit(trainInput, trainOutput,
		batch_size=GPU_BATCH_SIZE,
		nb_epoch=TRAIN_EPOCHS,
		verbose=2,
		callbacks=[checkpointer],
		validation_data=(valInput, valOutput),
		shuffle=False)