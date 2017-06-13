import tensorflow as tf
import numpy as np
from constants import *
from data import *

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


'''def generateFromRandomSequence(network, outputFileName):
	inputVar = network['input'].input_var
	predictionExpr = lasagne.layers.get_output(network['output'])
	predict_fn = theano.function([inputVar], predictionExpr)

	midiFileNames = [name for name in os.listdir("input") if name[-4:] in ('.mid','.MID')]
	fileName = random.choice(midiFileNames)

	print("Loading input midi file {}".format(fileName))
	midiFile = mido.MidiFile(os.path.join("input", fileName))
	data = midiTracksToInputData(midiFile)
	start = random.randrange(0, len(data)-N_INPUT_UNITS)

	previousInput = data[start:start+N_INPUT_UNITS]
	previousInput = np.array(previousInput, dtype="float32")
	previousInput.shape = (1, 1, previousInput.shape[0], previousInput.shape[1], previousInput.shape[2])

	outputData = np.zeros((OUTPUT_DURATION_UNITS, PITCH_OCTAVES, len(NOTES)), dtype="float32")
	outputData[:N_INPUT_UNITS] = previousInput

	for timeUnitLapsed in range(N_INPUT_UNITS, OUTPUT_DURATION_UNITS - N_OUTPUT_UNITS, N_OUTPUT_UNITS):
		prediction = predict_fn(previousInput)
		prediction = np.greater_equal(prediction, N_PLAY_THRESHOLD)
		prediction = prediction.reshape(N_OUTPUT_UNITS, PITCH_OCTAVES, len(NOTES))

		timeUnitIncludingPrediction = timeUnitLapsed + N_OUTPUT_UNITS
		outputData[timeUnitLapsed:timeUnitIncludingPrediction] = prediction

		previousInput = outputData[timeUnitIncludingPrediction - N_INPUT_UNITS:timeUnitIncludingPrediction]
		previousInput.shape = (1, 1, N_INPUT_UNITS, PITCH_OCTAVES, len(NOTES))

	outputMidiFile = mido.MidiFile()
	midiTrack = outputDataToMidiTrack(outputData)
	print midiTrack
	outputMidiFile.tracks.append(midiTrack)
	outputMidiFile.save("output/"+outputFileName)'''