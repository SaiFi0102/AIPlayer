import mido, os, theano, lasagne
import numpy as np
from constants import *

def midiTracksToTempoMap(midiFile):
	tempoMap = {}
	for track in midiFile.tracks:
		ticks = 0

		for i, event in enumerate(track):
			ticks += event.time
			if event.type == 'set_tempo':
				if not tempoMap.has_key(ticks) or (tempoMap.has_key(ticks) and i == 0):
					tempoMap[ticks] = event.tempo
	return sorted(tempoMap.items(), key=lambda t: t[0])

def midiNoteToInputIndex(midiNote):
	return midiNote - PITCH_LOWERBOUND

def inputIndexToMidiNote(note):
	return PITCH_LOWERBOUND + note

def midiTracksToInputData(midiFile):
	tempoMap = midiTracksToTempoMap(midiFile)
	data = []
	for track in midiFile.tracks:
		#Initialize variables for each track
		ticksLapsed = 0
		timeLapsed = 0
		tempoIndex = 0
		isPercussion = False
		tickResolutionInUs = mido.bpm2tempo(120) / midiFile.ticks_per_beat #Intial tempo/tick resolution
		currentTrackNotesState = np.zeros((PITCH_COUNT), dtype="float32") #All notes off

		nextTempoEvent = None
		if len(tempoMap) != 0:
			nextTempoEvent = tempoMap[tempoIndex]

		for event in track:
			#Check if there is a tempo event, update tempo and the next tempo event var
			if nextTempoEvent is not None and ticksLapsed >= nextTempoEvent[0]:
				tickResolutionInUs = nextTempoEvent[1] / midiFile.ticks_per_beat
				tempoIndex = tempoIndex + 1
				if len(tempoMap) > tempoIndex:
					nextTempoEvent = tempoMap[tempoIndex]
				else:
					nextTempoEvent = None

			#Update ticks
			ticksLapsed += event.time
			deltaTime = event.time * tickResolutionInUs
			timeLapsed += deltaTime
			timeUnitsLapsed = float(timeLapsed) / RESOLUTION_TIME
			currentTimeUnit = int(round(timeUnitsLapsed))
						
			#Ignore percussion instrument segments
			if event.type == 'program_change':
				if event.program >= 113 and event.program <= 120:
					isPercussion = True
				else:
					isPercussion = False
			if isPercussion:
				continue

			#Insert values to data before updating notes state
			if currentTimeUnit > len(data):
				for i in range(currentTimeUnit - len(data)): #Number of units required to insert before currentTimeUnit
					data.append(np.copy(currentTrackNotesState))

			#Determine event
			noteEvent = False
			if event.type == 'note_on':
				noteEvent = True
				noteState = True
			if event.type == 'note_off':
				noteEvent = True
				noteState = False

			if noteEvent and event.note >= PITCH_LOWERBOUND and event.note <= PITCH_UPPERBOUND:
				#Determine all note events with 0 velocity as note_off
				if event.velocity == 0:
					noteState = False

				#Update current note state
				note = midiNoteToInputIndex(event.note)
				currentTrackNotesState[note] = np.float32(noteState)

				#Add the current note item data
				if currentTimeUnit >= len(data): 
					data.append(np.copy(currentTrackNotesState))
				elif noteState: #Only update if state is true
					data[currentTimeUnit][note] = np.float32(noteState)		
	return data

def outputDataToMidiTrack(data):
	track = mido.MidiTrack()
	track.append(mido.Message('program_change', program=1, time=0))
	#track.append(mido.MetaMessage('set_tempo', tempo=OUTPUT_TEMPO, time=0))

	previousState = np.zeros((PITCH_COUNT), dtype="float32") #All notes off
	previousEventTicks = 0

	for timeUnitsLapsed, currentState in enumerate(data):
		timeLapsed = timeUnitsLapsed * RESOLUTION_TIME #In micro seconds
		ticksLapsed = timeLapsed / OUTPUT_RESOLUTION_TIME

		for note, noteState in enumerate(currentState):
			if previousState[note] != noteState:
				midiNote = inputIndexToMidiNote(note)
				deltaTicks = ticksLapsed - previousEventTicks
				previousEventTicks = ticksLapsed

				if noteState:
					track.append(mido.Message('note_on', note=midiNote, velocity=80, time=deltaTicks))
				else:
					track.append(mido.Message('note_off', note=midiNote, velocity=80, time=deltaTicks))

				previousState[note] = noteState #Next state

	#track.append(mido.MetaMessage('end_of_track', time=0))
	return track

def loadNseDataset():
	midiFileNames = [name for name in os.listdir(TRAIN_MUSIC_FOLDER) if name[-4:] in ('.mid','.MID')]
	fileCount = 0
	all = []

	for fName in midiFileNames:
		with mido.MidiFile(os.path.join(TRAIN_MUSIC_FOLDER, fName)) as midiFile:
			piece = midiTracksToInputData(midiFile)
			all.extend(piece)
			fileCount += 1
	print("{} midi files loaded".format(fileCount))

	extra = len(all) % NSE_BATCH_SIZE
	for _ in range(extra):
		all.pop()

	all = np.array(all, dtype="float32")
	all = all.reshape(-1, NSE_BATCH_SIZE, PITCH_COUNT)

	numBatches = len(all)
	assert(numBatches >= 3)
	numTrain = int(round(numBatches * 0.7))
	numVal = numBatches - numTrain

	print("Number of batches: {}".format(numBatches))
	print("Number of training batches: {}".format(numTrain))
	print("Number of validation batches: {}".format(numVal))

	nseTrain, nseVal = all[:numTrain], all[numTrain:]
	return nseTrain, nseVal

def loadDataset(nse):
	pieceMinimumSegmentUnits = N_INPUT_UNITS + N_OUTPUT_UNITS
	midiFileNames = [name for name in os.listdir(TRAIN_MUSIC_FOLDER) if name[-4:] in ('.mid','.MID')]
	fileCount = 0
	input = []
	output = []

	encode_fn = theano.function([nse['input'].input_var], lasagne.layers.get_output(nse['encoded']))
	#decode_fn = theano.function([nse['encodedInput'].input_var], lasagne.layers.get_output(nse['decoderOutput']))

	for fName in midiFileNames:
		with mido.MidiFile(os.path.join(TRAIN_MUSIC_FOLDER, fName)) as midiFile:
			piece = midiTracksToInputData(midiFile)
			if len(piece) < pieceMinimumSegmentUnits:
				print("Piece {} not loaded because it's not long enough".format(fName))
				continue

			encodedPiece = encode_fn(piece)

			timeUnitLapsed = 0
			while timeUnitLapsed < len(encodedPiece):
				#If remianing data does not have enough for a whole minibatch then discard it
				if len(encodedPiece) - timeUnitLapsed < pieceMinimumSegmentUnits:
					break

				input.append(encodedPiece[timeUnitLapsed:timeUnitLapsed + N_INPUT_UNITS])
				output.append(encodedPiece[timeUnitLapsed + N_INPUT_UNITS : timeUnitLapsed + N_INPUT_UNITS + N_OUTPUT_UNITS])
				timeUnitLapsed += TRAIN_DATASET_STEP
			fileCount += 1
	print("{} midi files loaded".format(fileCount))

	extra = len(input) % GPU_BATCH_SIZE
	for _ in range(extra):
		input.pop()
		output.pop()

	assert(len(input) == len(output)) #Should have same number of batches
	input = np.array(input, dtype="float32")
	output = np.array(output, dtype="float32")
	
	numBatches = len(input)
	assert(numBatches >= 3)
	numTrain = int(round(numBatches * 0.7))
	numVal = numBatches - numTrain

	inputTrain, inputVal = input[:numTrain], input[numTrain:]
	outputTrain, outputVal = output[:numTrain], output[numTrain:]

	inputTimeUnits = int(input.shape[0] * input.shape[1])
	inputTrainTimeUnits = int(inputTrain.shape[0] * inputTrain.shape[1])
	outputTimeUnits = int(output.shape[0] * output.shape[1])
	outputTrainTimeUnits = int(outputTrain.shape[0] * outputTrain.shape[1])

	print("Total input duration: {}min ({} units)".format(inputTimeUnits*RESOLUTION_TIME/1000./1000/60, inputTimeUnits))
	print("Total output duration: {}min ({} units)".format(outputTimeUnits*RESOLUTION_TIME/1000./1000/60, outputTimeUnits))
	print("Training input duration: {}min ({} units)".format(inputTrainTimeUnits*RESOLUTION_TIME/1000./1000/60, inputTrainTimeUnits))
	print("Training output duration: {}min ({} units)".format(outputTrainTimeUnits*RESOLUTION_TIME/1000./1000/60, outputTrainTimeUnits))
	print("")
	print("Number of batches: {}".format(numBatches))
	print("Number of training batches: {}".format(numTrain))
	print("Number of validation batches: {}".format(numVal))

	return inputTrain, inputVal, outputTrain, outputVal

def iterateBatches(inputs, outputs, shuffle):
	assert len(inputs) > 0
	assert len(inputs) == len(outputs)

	indices = np.arange(len(inputs))
	if shuffle:
		np.random.shuffle(indices)

	for batch in indices:
		yield inputs[batch], outputs[batch]
