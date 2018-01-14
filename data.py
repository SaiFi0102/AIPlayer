import numpy as np
import mido, collections, pickle, io, os
from hashable import hashable
from itertools import compress
from functions import create_directory
from constants import *

###################
# Utility functions
###################

def midiNoteToInputIndex(midiNote):
	return midiNote - PITCH_LOWERBOUND

def inputIndexToMidiNote(note):
	return PITCH_LOWERBOUND + note

def millisecondsToTimeUnits(time):
	return int(math.ceil(time / (RESOLUTION_TIME / 1000)))

def timeUnitsToMilliseconds(timeUnits):
	return timeUnits * RESOLUTION_TIME / 1000

#####################################
# MIDI file extraction and generation
#####################################

def midoFileToTempoMap(midoFile):
	tempoMap = {}
	for track in midoFile.tracks:
		ticks = 0
		for event in track:
			ticks += event.time
			if event.type == 'set_tempo':
				tempoMap[ticks] = event.tempo # if (ticks not in tempoMap) or (ticks in tempoMap and i == 0):
	return sorted(tempoMap.items(), key=lambda t: t[0])

def midoFileToNoteStateSeq(midoFile):
	assert(midoFile.type != 2)
	tempoMap = midoFileToTempoMap(midoFile)
	seq = []
	for track in midoFile.tracks:
		# Initialize variables for each track
		ticksLapsed = 0
		timeLapsed = 0
		currentTimeUnit = 0
		tempoIndex = 0
		isPercussion = False
		tickResolutionInUs = mido.bpm2tempo(120) / midoFile.ticks_per_beat  # Intial tempo/tick resolution
		currentTrackNoteState = np.zeros(PITCH_COUNT, dtype="float32")  # All notes off

		# Set the first nextTempoEvent
		nextTempoEvent = None
		if tempoMap:
			nextTempoEvent = tempoMap[tempoIndex]

		for event in track:
			# Update ticks
			# Check if there is a tempo event, update tempo and also
			# update delta time making sure to account for tempo events
			ticksLapsed += event.time
			deltaTime = 0
			if nextTempoEvent is not None and ticksLapsed >= nextTempoEvent[0]:
				prevTempoEventTicks = ticksLapsed - event.time
				while nextTempoEvent is not None and ticksLapsed >= nextTempoEvent[0]:
					deltaTime += (nextTempoEvent[0] - prevTempoEventTicks) * tickResolutionInUs #Update from ticks between tempo events

					tickResolutionInUs = nextTempoEvent[1] / midoFile.ticks_per_beat
					tempoIndex += 1

					prevTempoEventTicks = nextTempoEvent[0]
					nextTempoEvent = tempoMap[tempoIndex] if tempoIndex < len(tempoMap) else None
				deltaTime += (ticksLapsed - prevTempoEventTicks) * tickResolutionInUs #Update from ticks after the last tempo event
			else:
				deltaTime = event.time * tickResolutionInUs  # Update from event delta time
			timeLapsed += deltaTime

			# Ignore meta events
			if event.is_meta:
				continue

			# Update time units
			timeUnitsLapsed = timeLapsed / RESOLUTION_TIME
			previousTimeUnit = currentTimeUnit
			currentTimeUnit = int(round(timeUnitsLapsed))

			# Fill from previous notes state values to seq till currentTimeUnit
			if currentTimeUnit > len(seq) - 1:
				for _ in range(currentTimeUnit - len(seq) + 1):
					seq.append(np.copy(currentTrackNoteState))
			else: # Add notes to seq if modifying previously written notes from other tracks
				if not currentTrackNoteState.all(0):
					for i in range(previousTimeUnit+1, currentTimeUnit+1):
						seq[i] = np.add(seq[i], currentTrackNoteState)
						seq[i] = seq[i].clip(0, 1)

			# Ignore percussion instrument segments
			if event.type == 'program_change':
				if event.program >= 113 and event.program <= 120:
					isPercussion = True
				else:
					isPercussion = False
				continue
			if isPercussion:
				continue

			# Determine event
			noteEvent = False
			if event.type == 'note_on':
				noteEvent = True
				noteState = True
			elif event.type == 'note_off':
				noteEvent = True
				noteState = False
			if not noteEvent:
				continue

			if event.note >= PITCH_LOWERBOUND and event.note <= PITCH_UPPERBOUND:
				# Determine all note events with 0 velocity as note_off
				if event.velocity == 0:
					noteState = False

				# Update current note state
				note = midiNoteToInputIndex(event.note)
				currentTrackNoteState[note] = np.float32(noteState)

				# Create room between note off and note on if necessary so that they're evident in seq
				pass

				# Add the current note state to seq
				if currentTimeUnit == len(seq) - 1:
					seq[currentTimeUnit] = np.copy(currentTrackNoteState)
				else:
					seq[currentTimeUnit] = np.add(seq[currentTimeUnit], currentTrackNoteState)
					seq[currentTimeUnit] = seq[currentTimeUnit].clip(0, 1)

	return seq

def musicFolderToNoteStateSeq(path):
	print("Loading MIDI data from music directory: {}".format(path))

	midiFileNames = [name for name in os.listdir(path) if name[-4:] in ('.mid', '.MID')]
	seq = []
	fileCount = 0
	for fName in midiFileNames:
		with mido.MidiFile(os.path.join(path, fName)) as midiFile:
			fileSeq = midoFileToNoteStateSeq(midiFile)
			if len(fileSeq) < N_INPUT_UNITS + N_OUTPUT_UNITS:
				print("MIDI file {} not loaded because it's not long enough".format(fName))
				continue

			seq.extend(fileSeq)
			seq.extend(np.zeros((millisecondsToTimeUnits(FILE_GAP_TIME), PITCH_COUNT), dtype="float32"))
			fileCount += 1

	print("MIDI files loaded: {}".format(fileCount))
	print("Music sequence length: {} units".format(len(seq)))
	print("Music sequence duration: {} hours".format(timeUnitsToMilliseconds(len(seq)) / 1000 / 60 / 60))
	print("-------------------")
	return np.asarray(seq)

def noteStateSeqToMidiTrack(noteStateSeq):
	track = mido.MidiTrack()
	track.append(mido.Message('program_change', program=1, time=0))
	track.append(mido.MetaMessage('set_tempo', tempo=OUTPUT_TEMPO, time=0))

	previousState = np.zeros(PITCH_COUNT, dtype="float32") #All notes off
	previousEventTicks = 0

	for timeUnitsLapsed, currentState in enumerate(noteStateSeq):
		timeLapsed = timeUnitsLapsed * RESOLUTION_TIME #In micro seconds
		ticksLapsed = int(timeLapsed / OUTPUT_RESOLUTION_TIME)

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

	track.append(mido.MetaMessage('end_of_track', time=0))
	return track

def noteStateSeqToMidiStream(noteStateSeq):
	midiFile = mido.MidiFile()
	midiTrack = noteStateSeqToMidiTrack(noteStateSeq)
	midiFile.tracks.append(midiTrack)

	stream = io.BytesIO()
	midiFile.save(file=stream)
	stream.seek(0)
	return stream

def vocabularyToMidiStream(noteStateList, sustain=1000):
	sustainTimeUnits = millisecondsToTimeUnits(sustain)
	assert(sustainTimeUnits >= 2)
	noteStateSeq = []
	for noteStates in noteStateList:
		for _ in range(sustainTimeUnits-1):
			noteStateSeq.append(noteStates)
		noteStateSeq.append(np.zeros(PITCH_COUNT, dtype="float32"))
	return noteStateSeqToMidiStream(noteStateSeq)

######################################
# Vocabulary extraction and generation
######################################

def loadVocabularyData(noteStateSeq):
	print("Generating vocabulary data")
	noteStateToWordIdx = {}
	wordIdxToSortable = [] # 0=>noteState, 1=>count

	#Fill in data
	for currentState in noteStateSeq:
		stateWrapper = hashable(currentState)
		if stateWrapper in noteStateToWordIdx:
			wordIdx = noteStateToWordIdx[stateWrapper]
			wordIdxToSortable[wordIdx][1] += 1
		else:
			wordIdx = len(wordIdxToSortable)
			noteStateToWordIdx[stateWrapper] = wordIdx
			wordIdxToSortable.insert(wordIdx, [currentState, 1])

	#Sort in descending order of the word counts and split variables
	wordIdxToSortable = sorted(wordIdxToSortable, key=lambda k: k[1], reverse=True)
	wordIdxToNoteState = np.asarray([x[0] for x in wordIdxToSortable])
	wordIdxToCount = [x[1] for x in wordIdxToSortable]
	del wordIdxToSortable

	print("Vocabulary size: {} unique musical words".format(len(wordIdxToNoteState)))
	print("-------------------")

	return wordIdxToNoteState, wordIdxToCount

####################
# Dataset generation
####################

def splitInputOutputSeq(seq):
	print("Splitting input/output data")

	input = []
	output = []

	timeUnitLapsed = 0
	while timeUnitLapsed + N_INPUT_UNITS + N_OUTPUT_UNITS < len(seq):
		input.append(seq[timeUnitLapsed:timeUnitLapsed + N_INPUT_UNITS])
		output.append(seq[timeUnitLapsed + N_INPUT_UNITS: timeUnitLapsed + N_INPUT_UNITS + N_OUTPUT_UNITS])
		timeUnitLapsed += N_INPUT_UNITS

	extra = len(input) % N_BATCH_SIZE
	for _ in range(extra):
		input.pop()
		output.pop()

	assert(len(input) == len(output))  # Should have same number of batches
	input = np.asarray(input, dtype="float32")
	output = np.asarray(output, dtype="float32")

	return input, output

def seqToDataset(seq):
	input, output = splitInputOutputSeq(seq)
	print("Splitting train/validation data")

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

	print("Total input duration: {}min ({} units)".format(inputTimeUnits*RESOLUTION_TIME/1000/1000/60, inputTimeUnits))
	print("Total output duration: {}min ({} units)".format(outputTimeUnits*RESOLUTION_TIME/1000/1000/60, outputTimeUnits))
	print("Training input duration: {}min ({} units)".format(inputTrainTimeUnits*RESOLUTION_TIME/1000/1000/60, inputTrainTimeUnits))
	print("Training output duration: {}min ({} units)".format(outputTrainTimeUnits*RESOLUTION_TIME/1000/1000/60, outputTrainTimeUnits))
	print()
	print("Number of batches: {}".format(numBatches))
	print("Number of training batches: {}".format(numTrain))
	print("Number of validation batches: {}".format(numVal))
	print("-------------------")

	return inputTrain, inputVal, outputTrain, outputVal

###########################
# Cache dumping and loading
###########################

def createCacheData(outputFile="cachedData.npz"):
	if outputFile.find('.') == -1:
		outputFile += ".npz"
	outputFile = os.path.join(DATA_FOLDER, outputFile)

	#Prevent overwriting
	assert(not os.path.isfile(outputFile))

	create_directory(DATA_FOLDER)
	noteStateSeq = musicFolderToNoteStateSeq(TRAIN_MUSIC_FOLDER)
	#wordIdxToNoteState, wordIdxToCount = loadVocabularyData(noteStateSeq)

	print("Creating data cache")
	print("-------------------")
	np.savez(outputFile, noteStateSeq=noteStateSeq)
	print("Data cache created: {}".format(outputFile))

def loadCacheData(inputFile="cachedData.npz"):
	print("Loading from data cache")

	inputFile = os.path.join(DATA_FOLDER, inputFile)
	cache = np.load(inputFile)
	noteStateSeq = cache['noteStateSeq']

	print("Music sequence length: {} units".format(len(noteStateSeq)))
	print("Music sequence duration: {} hours".format(timeUnitsToMilliseconds(len(noteStateSeq)) / 1000 / 60 / 60))

	return noteStateSeq

#MAIN
if __name__ == '__main__':
	createCacheData()
