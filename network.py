import theano, keras, seq2seq, lasagne, collections, time
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from constants import *
from data import *
from keras.models import Sequential
from keras.layers import Dense, TimeDistributedDense
from seq2seq.models import AttentionSeq2seq
#from seq2seq.layers.encoders import LSTMEncoder
#from seq2seq.layers.decoders import AttentionDecoder

def buildNetwork(loadParams=False):
	model = Sequential()
	seq2seq = AttentionSeq2seq(
		input_dim=NSE_ENCODED_SIZE,
		input_length=N_INPUT_UNITS,
		hidden_dim=100,
		output_length=N_OUTPUT_UNITS,
		output_dim=PITCH_COUNT,
		dropout=0.3,
		depth=2,
		bidirectional=False
	)

	model.add(TimeDistributedDense(NSE_ENCODED_SIZE, input_shape=(N_INPUT_UNITS, PITCH_COUNT)))
	model.add(seq2seq)

	if type(loadParams) == str:
		print("Loading network params...")
		model.load_weights(loadParams)

	model.compile(optimizer='adadelta', loss='mse')
	return model

def trainNetwork(model, nse):
	print("Loading music dataset...")
	trainInput, valInput, trainOutput, valOutput = loadDataset(nse)

	print("Starting network training...")
	checkpointer = keras.callbacks.ModelCheckpoint(filepath="data/best.w", verbose=1, save_best_only=True)

	model.fit(trainInput, trainOutput,
		batch_size=GPU_BATCH_SIZE,
		nb_epoch=TRAIN_EPOCHS,
		verbose=2,
		callbacks=[checkpointer],
		validation_data=(valInput, valOutput),
		shuffle=True)



'''def logTraining(file, str):
	print(str)
	file.write(str + '\n')

def loadLasagneParams(network, fileName):
	npFile = np.load("data/"+fileName)
	inParams = npFile['arr_0']

	for i, layer in enumerate(lasagne.layers.get_all_layers(network['output'])):
		lasagne.layers.set_all_param_values(layer, inParams[i])

def buildNoteStateEncoder():
	network = collections.OrderedDict()
	network['input'] = lasagne.layers.InputLayer(shape=(None, PITCH_COUNT))
	network['encoded'] = lasagne.layers.DenseLayer(network['input'], num_units=NSE_ENCODED_SIZE, nonlinearity=lasagne.nonlinearities.identity)
	network['output'] = lasagne.layers.DenseLayer(network['encoded'], num_units=PITCH_COUNT, nonlinearity=lasagne.nonlinearities.sigmoid)

	network['encodedInput'] = lasagne.layers.InputLayer(shape=(None, NSE_ENCODED_SIZE))
	network['decoderOutput'] = lasagne.layers.DenseLayer(
		network['encodedInput'],
		num_units=PITCH_COUNT,
		W=network['output'].W,
		b=network['output'].b,
		nonlinearity=lasagne.nonlinearities.sigmoid
	)

	for name, layer in network.items():
		print("Note state encoder layer {}'s output shape: {}".format(name, layer.output_shape))

	return network

def trainNoteStateEncoder(network):
	print("Loading music dataset...")
	inputTrain, inputVal = loadNseDataset()

	print("Starting training...")
	inputVar = network['input'].input_var

	logFileName = "nse_train_" + time.strftime("%Y%m%d-%H%M%S") + ".log"
	print("Opening training log file data/" + logFileName + "...")
	logFile = open("data/"+logFileName, "w", 0)

	prediction = lasagne.layers.get_output(network['output'])
	loss = lasagne.objectives.binary_crossentropy(prediction, inputVar)
	loss = loss.mean()

	params = lasagne.layers.get_all_params(network['output'], trainable=True)
	updates = lasagne.updates.adadelta(loss, params)

	test_prediction = lasagne.layers.get_output(network['output'], deterministic=True)
	test_loss = lasagne.objectives.binary_crossentropy(test_prediction, inputVar)
	test_loss = test_loss.mean()
	test_acc = T.mean(lasagne.objectives.binary_accuracy(test_prediction, inputVar, threshold=N_PLAY_THRESHOLD))
	test_prediction_binary = T.ge(test_prediction, N_PLAY_THRESHOLD)
	test_prediction_binary_sum = test_prediction_binary.sum()
	#Consider only 1's predicted by network
	test_playAcc = ifelse(test_prediction_binary_sum > 0, (test_prediction_binary * inputVar).sum() / test_prediction_binary_sum, np.float64(1))

	train_fn = theano.function([inputVar], loss, updates=updates)
	val_fn = theano.function([inputVar], [test_loss, test_acc, test_playAcc])

	# We iterate over epochs:
	lowest_err = 999999.
	for epoch in range(TRAIN_EPOCHS):
		logTraining(logFile, "Training epoch {}".format(epoch+1))
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterateBatches(inputTrain, inputTrain, shuffle=True):
			inputs, _ = batch
			err = train_fn(inputs)
			train_err += err
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_playAcc = 0
		val_batches = 0
		for batch in iterateBatches(inputVal, inputVal, shuffle=False):
			inputs, _ = batch
			err, acc, playAcc = val_fn(inputs)
			val_err += err
			val_acc += acc
			val_playAcc += playAcc
			val_batches += 1

		# Then we print the results for this epoch:
		logTraining(logFile, "Epoch {} of {} took {:.3f}s".format(epoch + 1, TRAIN_EPOCHS, time.time() - start_time))
		logTraining(logFile, "  training loss:\t\t{:.8f}".format(train_err / train_batches))
		logTraining(logFile, "  validation loss:\t\t{:.8f}".format(val_err / val_batches))
		logTraining(logFile, "  validation accuracy:\t\t{:.4f} %".format(val_acc / val_batches * 100))
		logTraining(logFile, "  validation play accuracy:\t\t{:.4f} %".format(val_playAcc / val_batches * 100))
		
		#Save training
		outParams = []
		for layer in lasagne.layers.get_all_layers(network['output']):
			outParams.append(lasagne.layers.get_all_param_values(layer))

		if train_err < lowest_err:
			np.savez(open('data/nse_best.npz','wb'), outParams)
			lowest_err = train_err

	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	test_playAcc = 0
	for batch in iterateBatches(inputTest, inputTest, shuffle=False):
		inputs, _ = batch
		err, acc, playAcc = val_fn(inputs)
		test_err += err
		test_acc += acc
		test_playAcc += playAcc
		test_batches += 1

	logTraining(logFile, "Final results:")
	logTraining(logFile, "  test loss:\t\t\t{:.8f}".format(test_err / test_batches))
	logTraining(logFile, "  test accuracy:\t\t{:.4f} %".format(test_acc / test_batches * 100))
	logTraining(logFile, "  test play accuracy:\t\t{:.4f} %".format(test_playAcc / test_batches * 100))
	logFile.close()'''





'''def buildNoteStateEncoder():
input = Input(shape=(PITCH_COUNT))
encodedInput = Input(shape=(NSE_ENCODED_SIZE))

encoded = Dense(20)
decoded = Dense(PITCH_COUNT, activation="sigmoid")

autoencoder = Model(input=input, output=decoded)
encoder = Model(input=input, output=encoded)

decoderLayer = autoencoder.layers[-1]
decoder = Model(input=encodedInput, output=decoderLayer(encodedInput))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
return autoencoder, encoder, decoder

def trainNoteStateEncoder(model):
print("Loading music dataset...")
inputTrain, inputVal = loadDataset(noteEncoderFormat=True)

print("Starting note state encoder training...")
checkpointer = keras.callbacks.ModelCheckpoint(filepath="data/best_nse.w", verbose=0, save_best_only=True)

model.fit(inputTrain, inputTrain,
	batch_size=NSE_BATCH_SIZE,
	nb_epoch=TRAIN_EPOCHS,
	verbose=2,
	callbacks=[checkpointer],
	validation_data=(inputVal, inputVal),
	shuffle=True
)'''