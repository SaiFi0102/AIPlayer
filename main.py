from constants import *
from functions import *
from data import *
from network import *

def generateFromRandomSequence(network, outputFileName):
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
	outputMidiFile.save("output/"+outputFileName)

# MAIN
if __name__ == '__main__':
	create_directory("data") #For trained weights and states
	create_directory("output") #For generated output

	#NSE
	print("Building note state encoder...")
	nse = buildNoteStateEncoder()

	print("Loading note state encoder params...")
	loadLasagneParams(nse, "nse_best.npz")

	#print("Training note state encoder...")
	#trainNoteStateEncoder(nse)

	#Network
	print("Building network...")
	network = buildNetwork(loadParams=False)

	#print("Loading network params...")
	#loadNetworkParams(network, "latest.npz")

	print("Training network...")
	trainNetwork(network, nse)

	#print("Generating midi file")
	#generateFromRandomSequence(network, "test.mid")

#OLD AND COMMENTED OUT
if 0:
	cf = mido.MidiFile("music/chromatic.mid")
	data = midiTracksToInputData(cf)
	track = outputDataToMidiTrack(data)

	nf = mido.MidiFile()
	nf.ticks_per_beat = OUTPUT_TICKS_PER_BEAT
	nf.tracks.append(track)
	nf.save("music/pass.mid")

	print nf.tracks[0]

	while True:
		print("Enter:")
		print("1: Train network")
		print("2: Generate midi file from a random midi segment")
		try:
			command = int(raw_input('Input:'))
		except ValueError:
			print "Not a number"