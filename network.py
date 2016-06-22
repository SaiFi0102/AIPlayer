import theano, lasagne, collections, time
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from constants import *
from data import *

class ConvToLSTMShapeLayer(lasagne.layers.ReshapeLayer):
	def get_output_for(self, input, **kwargs):
		return super(ConvToLSTMShapeLayer, self).get_output_for(T.swapaxes(input, 1, 2), **kwargs) #Swap channels with time units
	def get_output_shape_for(self, input_shape, **kwargs):
		newInputShape = list(input_shape)
		newInputShape[1] = input_shape[2]
		newInputShape[2] = input_shape[1]
		output_shape = super(ConvToLSTMShapeLayer, self).get_output_shape_for(tuple(newInputShape), **kwargs)
		return output_shape

def logTraining(file, str):
	print(str)
	file.write(str + '\n')

def buildNetwork():
	network = collections.OrderedDict()
	network['input'] = lasagne.layers.InputLayer(shape=(N_FIXED_BATCH_SIZE, N_CHANNELS, N_INPUT_UNITS, PITCH_OCTAVES, len(NOTES)))
	
	network['conv1'] = Conv3DDNNLayer(network['input'], num_filters=8, filter_size=(80, 1, 1), nonlinearity=lasagne.nonlinearities.identity)
	network['pool1'] = Pool3DDNNLayer(network['conv1'], pool_size=(3, 1, 1), mode='max')

	#Swaps N_CHANNELS axis with N_INPUT_UNITS then flattens layer output till N_INPUT_UNITS axis
	network['shape1'] = ConvToLSTMShapeLayer(network['pool1'], ([0], [1], -1)) 

	network['lstm1'] = lasagne.layers.LSTMLayer(network['shape1'], num_units=400, nonlinearity=lasagne.nonlinearities.identity)
	network['drop3'] = lasagne.layers.DropoutLayer(network['lstm1'], p=0.5)
	network['lstm2'] = lasagne.layers.LSTMLayer(network['drop3'], num_units=400, nonlinearity=lasagne.nonlinearities.identity)
	network['drop4'] = lasagne.layers.DropoutLayer(network['lstm2'], p=0.5)

	network['norm'] = lasagne.layers.BatchNormLayer(network['drop4'])
	network['shape2'] = lasagne.layers.ReshapeLayer(network['norm'], ([0], -1))

	network['output'] = lasagne.layers.DenseLayer(network['shape2'], num_units=PITCH_COUNT*N_OUTPUT_UNITS, nonlinearity=lasagne.nonlinearities.sigmoid)

	for name, layer in network.items():
		print("Layer {}'s output shape: {}".format(name, layer.output_shape))

	return network

def trainNetwork(network):
	print("Loading music dataset...")
	inputTrain, inputTest, inputVal, outputTrain, outputTest, outputVal = loadDataset(noteEncoderFormat=False)

	print("Starting training...")
	inputVar = network['input'].input_var
	targetVar = T.fmatrix('targets')

	logFileName = "train_" + time.strftime("%Y%m%d-%H%M%S") + ".log"
	print("Opening training log file data/" + logFileName + "...")
	logFile = open("data/"+logFileName, "w", 0)

	prediction = lasagne.layers.get_output(network['output'])
	loss = lasagne.objectives.binary_crossentropy(prediction, targetVar)
	loss = loss.mean()

	params = lasagne.layers.get_all_params(network['output'], trainable=True)
	updates = lasagne.updates.adadelta(loss, params, learning_rate=3)
	#updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

	test_prediction = lasagne.layers.get_output(network['output'], deterministic=True)
	test_prediction_binary = T.ge(test_prediction, N_PLAY_THRESHOLD)
	test_loss = lasagne.objectives.binary_crossentropy(test_prediction, targetVar)
	test_loss = test_loss.mean()
	test_prediction_binary_sum = test_prediction_binary.sum()
	#Consider only 1's predicted by network
	test_playAcc = ifelse(test_prediction_binary_sum > 0, (test_prediction_binary * inputVar).sum() / test_prediction_binary_sum, 1)

	test_max = test_prediction.max()
	test_acc = T.mean(lasagne.objectives.binary_accuracy(test_prediction, targetVar, threshold=N_PLAY_THRESHOLD))

	train_fn = theano.function([inputVar, targetVar], loss, updates=updates)
	val_fn = theano.function([inputVar, targetVar], [test_loss, test_acc, test_playAcc, test_max])

	# We iterate over epochs:
	for epoch in range(TRAIN_EPOCHS):
		logTraining(logFile, "Training epoch {}".format(epoch+1))
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterateBatches(inputTrain, outputTrain, shuffle=True):
			inputs, targets = batch
			err = train_fn(inputs, targets)
			train_err += err
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_playAcc = 0
		val_batches = 0
		val_max = 0.
		for batch in iterateBatches(inputVal, outputVal, shuffle=False):
			inputs, targets = batch
			err, acc, playAcc, max = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_playAcc += playAcc
			val_batches += 1
			max = np.float32(max)
			if val_max < max:
				val_max = max

		# Then we print the results for this epoch:
		logTraining(logFile, "Epoch {} of {} took {:.3f}s".format(epoch + 1, TRAIN_EPOCHS, time.time() - start_time))
		logTraining(logFile, "  training loss:\t\t{:.6f}".format(train_err / train_batches))
		logTraining(logFile, "  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		logTraining(logFile, "  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
		logTraining(logFile, "  validation play accuracy:\t\t{:.2f} %".format(val_playAcc / val_batches * 100))
		logTraining(logFile, "  max output activation:\t{:.4f}".format(val_max))
		
		#Save training
		outParams = []
		for layer in lasagne.layers.get_all_layers(network['output']):
			outParams.append(lasagne.layers.get_all_param_values(layer))

		np.savez(open('data/latest.npz','wb'), outParams)

	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_playAcc = 0
	test_batches = 0
	test_max = 0
	for batch in iterateBatches(inputTest, outputTest, shuffle=False):
		inputs, targets = batch
		err, acc, playAcc, max = val_fn(inputs, targets)
		test_err += err
		test_acc += acc
		test_playAcc += playAcc
		test_batches += 1
		test_max = np.float32(max)
		if test_max < max:
			test_max = max

	logTraining(logFile, "Final results:")
	logTraining(logFile, "  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	logTraining(logFile, "  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
	logTraining(logFile, "  test play accuracy:\t\t{:.2f} %".format(test_playAcc / test_batches * 100))
	logTraining(logFile, "  max output activation:\t{:.4f}".format(test_max))
	logFile.close()

def buildNoteStateEncoder():
	network = collections.OrderedDict()
	network['input'] = lasagne.layers.InputLayer(shape=(NSE_BATCH_SIZE, PITCH_COUNT))
	network['hidden'] = lasagne.layers.DenseLayer(network['input'], num_units=20, nonlinearity=lasagne.nonlinearities.identity)
	network['output'] = lasagne.layers.DenseLayer(network['hidden'], num_units=PITCH_COUNT, nonlinearity=lasagne.nonlinearities.sigmoid)

	for name, layer in network.items():
		print("Note state encoder layer {}'s output shape: {}".format(name, layer.output_shape))

	return network

def trainNoteStateEncoder(network):
	print("Loading music dataset...")
	inputTrain, inputTest, inputVal = loadDataset(noteEncoderFormat=True)

	print("Starting training...")
	inputVar = network['input'].input_var

	logFileName = "nse_train_" + time.strftime("%Y%m%d-%H%M%S") + ".log"
	print("Opening training log file data/" + logFileName + "...")
	logFile = open("data/"+logFileName, "w", 0)

	prediction = lasagne.layers.get_output(network['output'])
	loss = lasagne.objectives.binary_crossentropy(prediction, inputVar)
	loss = loss.mean()

	params = lasagne.layers.get_all_params(network['output'], trainable=True)
	updates = lasagne.updates.adam(loss, params)

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

		np.savez(open('data/nse_latest.npz','wb'), outParams)

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
	logFile.close()

def loadNetworkParams(network, fileName):
	npFile = np.load("data/"+fileName)
	inParams = npFile['arr_0']
	#for i in range(len(npFile.files)):
	#	inParams.append(npFile['arr_{}'.format(i)])

	for i, layer in enumerate(lasagne.layers.get_all_layers(network['output'])):
		lasagne.layers.set_all_param_values(layer, inParams[i])