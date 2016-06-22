import mido, os
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
	note = midiNote - PITCH_LOWERBOUND
	#octave = 0
	octave = int(float(note) / len(NOTES))
	note = note % len(NOTES)
	return octave, note

def inputIndexToMidiNote(octave, note):
	return PITCH_LOWERBOUND + octave * len(NOTES) + note

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
		currentTrackNotesState = np.zeros((PITCH_OCTAVES, len(NOTES)), dtype="float32") #All notes off

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
			deltaTimeUnits = float(deltaTime) / RESOLUTION_TIME
			currentTimeUnit = int(round(timeUnitsLapsed))

			#Insert values to data before updating notes state
			if currentTimeUnit > len(data):
				for i in range(currentTimeUnit - len(data)): #Number of units required to insert before currentTimeUnit
					data.append(np.copy(currentTrackNotesState))
						
			#Ignore percussion instrument segments
			if event.type == 'program_change':
				if event.program >= 113 and event.program <= 120:
					isPercussion = True
				else:
					isPercussion = False
			if isPercussion:
				continue

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
				octave, note = midiNoteToInputIndex(event.note)
				currentTrackNotesState[octave, note] = np.float32(noteState)

				#Add the current note item data
				if currentTimeUnit >= len(data): 
					data.append(np.copy(currentTrackNotesState))
				elif noteState: #Only update if state is true
					data[currentTimeUnit][octave, note] = np.float32(noteState)		
	return data

def outputDataToMidiTrack(data):
	track = mido.MidiTrack()
	track.append(mido.Message('program_change', program=1, time=0))
	#track.append(mido.MetaMessage('set_tempo', tempo=OUTPUT_TEMPO, time=0))

	previousState = np.zeros((PITCH_OCTAVES, len(NOTES)), dtype="float32") #All notes off
	previousEventTicks = 0

	for timeUnitsLapsed, currentState in enumerate(data):
		timeLapsed = timeUnitsLapsed * RESOLUTION_TIME #In micro seconds
		ticksLapsed = timeLapsed / OUTPUT_RESOLUTION_TIME

		for octave, noteStates in enumerate(currentState):
			for note, noteState in enumerate(noteStates):
				if previousState[octave, note] != noteState:
					midiNote = inputIndexToMidiNote(octave, note)
					deltaTicks = ticksLapsed - previousEventTicks
					previousEventTicks = ticksLapsed

					if noteState:
						track.append(mido.Message('note_on', note=midiNote, velocity=80, time=deltaTicks))
					else:
						track.append(mido.Message('note_off', note=midiNote, velocity=80, time=deltaTicks))

					previousState[octave, note] = noteState

	#track.append(mido.MetaMessage('end_of_track', time=0))
	return track

def datasetToMusicVocabulary(inputData, outputData):
	inputData = inputData.reshape(-1, PITCH_COUNT)
	outputData = outputData.reshape(-1, PITCH_COUNT)

	bothData = np.concatenate((inputData, outputData))
	print bothData.shape

	temp = np.ascontiguousarray(bothData).view(np.dtype((np.void, bothData.dtype.itemsize * bothData.shape[1])))
	_, idx = np.unique(temp, return_index=True)
	unique = bothData[idx]
	print unique.shape
	print unique

	return vocab

def loadDataset(noteEncoderFormat = False):
	#Input shape
	#[batch, minibatch, channels,  timeunit,		 octave,	  note]
	#[ .., GPU_BATCH_SIZE,  1, N_INPUT_UNITS, PITCH_OCTAVES,   12 ]
	
	#Output shape
	#[batch,	minibatch,	  notes]
	#[ ..  , GPU_BATCH_SIZE,  PITCH_COUNT * N_OUTPUT_UNITS]

	midiFileNames = [name for name in os.listdir(TRAIN_MUSIC_FOLDER) if name[-4:] in ('.mid','.MID')]
	fileCount = 0
	all = []

	for fName in midiFileNames:
		with mido.MidiFile(os.path.join(TRAIN_MUSIC_FOLDER, fName)) as midiFile:
			piece = midiTracksToInputData(midiFile)
			if len(piece) < N_INPUT_UNITS + N_OUTPUT_UNITS:
				print("Piece {} not loaded because it's not long enough".format(fName))
				continue

			extra = len(piece) % N_INPUT_UNITS + N_OUTPUT_UNITS
			if extra > 0:
				all.extend(piece[:-extra])
			else:
				all.extend(piece)
			fileCount += 1

	print("{} midi files loaded".format(fileCount))
	all = np.array(all, dtype="float32")
	
	if not noteEncoderFormat:
		input = []
		output = []
		batch = 0
		minibatch = 0

		timeUnitLapsed = 0
		while timeUnitLapsed < len(all):
			#If remianing data does not have enough for a whole minibatch then discard it
			if len(all) - timeUnitLapsed < N_INPUT_UNITS + N_OUTPUT_UNITS:
				break

			input[batch][minibatch] = all[timeUnitLapsed:timeUnitLapsed + N_INPUT_UNITS]
			output[batch][minibatch] = all[timeUnitLapsed + N_INPUT_UNITS : timeUnitLapsed + N_INPUT_UNITS + N_OUTPUT_UNITS]
			timeUnitLapsed += TRAIN_DATASET_STEP

			minibatch += 1
			if minibatch >= GPU_BATCH_SIZE:
				minibatch = 0
				batch += 1

		#If the last batch doesn't have enough minibatches for a whole CPU batch, pop it 
		if len(input[-1]) != GPU_BATCH_SIZE:
			input.pop()
			output.pop()

		assert(len(input) == len(output)) #Should have same number of batches

		input = np.array(input, dtype="float32")
		output = np.array(output, dtype="float32")
		output = output.reshape(output.shape[0], GPU_BATCH_SIZE, -1)

	if noteEncoderFormat:
		extra = len(all) % NSE_BATCH_SIZE
		if extra > 0:
			all = all[:-extra]
		all = all.reshape(-1, NSE_BATCH_SIZE, PITCH_COUNT)
		numBatches = len(all)
	else:
		numBatches = len(input)

	assert(numBatches >= 3)

	numTrain = int(round(numBatches * 0.6))
	numTestAndVal = numBatches - numTrain
	numTest = int(round(numTestAndVal * 0.5))
	numVal = numTestAndVal - numTest

	if noteEncoderFormat:
		nseTrain, nseTest, nseVal = all[:numTrain], all[numTrain:numTrain + numTest], all[numTrain + numTest:]
	else:
		inputTrain, inputTest, inputVal = input[:numTrain], input[numTrain:numTrain + numTest], input[numTrain + numTest:]
		outputTrain, outputTest, outputVal = output[:numTrain], output[numTrain:numTrain + numTest], output[numTrain + numTest:]

		inputTimeUnits = int(input.shape[0] * input.shape[1] * input.shape[3])
		inputTrainTimeUnits = int(inputTrain.shape[0] * inputTrain.shape[1] * inputTrain.shape[3])
		outputTimeUnits = int(output.shape[0] * output.shape[1] * output.shape[2]/PITCH_COUNT)
		outputTrainTimeUnits = int(outputTrain.shape[0] * outputTrain.shape[1] * outputTrain.shape[2]/PITCH_COUNT)

		print("Total input duration: {}min ({} units)".format(inputTimeUnits*RESOLUTION_TIME/1000./1000/60, inputTimeUnits))
		print("Total output duration: {}min ({} units)".format(outputTimeUnits*RESOLUTION_TIME/1000./1000/60, outputTimeUnits))
		print("Training input duration: {}min ({} units)".format(inputTrainTimeUnits*RESOLUTION_TIME/1000./1000/60, inputTrainTimeUnits))
		print("Training output duration: {}min ({} units)".format(outputTrainTimeUnits*RESOLUTION_TIME/1000./1000/60, outputTrainTimeUnits))

	print("Number of CPU batches: {}".format(numBatches))
	print("Number of GPU batches: {}".format(GPU_BATCH_SIZE))
	print("Number of training batches: {}".format(numTrain))
	print("Number of validation batches: {}".format(numVal))
	print("Number of test batches: {}".format(numTest))

	if noteEncoderFormat:
		return nseTrain, nseTest, nseVal
	else:
		return inputTrain, inputTest, inputVal, outputTrain, outputTest, outputVal

def iterateBatches(inputs, outputs, shuffle):
	assert len(inputs) > 0
	assert len(inputs) == len(outputs)

	indices = np.arange(len(inputs))
	if shuffle:
		np.random.shuffle(indices)

	for batch in indices:
		yield inputs[batch], outputs[batch]
