import math, mido

#############################################################################
# No need to change
NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
#############################################################################

#############################################################################
# Constants
TRAIN_MUSIC_FOLDER = "music_all"
DATA_FOLDER = "data"

PITCH_LOWERBOUND = 36 # 24 = C2, 36 = C3
PITCH_OCTAVES = 4
PITCH_COUNT = len(NOTES) * PITCH_OCTAVES
PITCH_UPPERBOUND = PITCH_LOWERBOUND + PITCH_COUNT - 1

RESOLUTION_TIME = 50 * 1000 # microseconds. Represents 1 time unit
FILE_GAP_TIME = 1000 # milliseconds

TRAIN_EPOCHS = 2000
OUTPUT_DURATION = 180 #seconds
VALIDATION_DATA_RATIO = 0.25

# Hyper-parameters
N_INPUT_TIME = 2 * 1000 * 1000 # microseconds
N_OUTPUT_TIME = 2 * 1000 * 1000 # microseconds
N_PLAY_THRESHOLD = 0.5
#N_CHANNELS = 1

N_BATCH_SIZE = 2000 # samples
W2V_EMBEDDING_SIZE = 50
W2V_BATCH_SIZE = 100
W2V_NUM_SKIPS = 2
W2V_SKIP_WINDOW = 1
W2V_TRAIN_EPOCHS = 100
#############################################################################

#############################################################################
# No need to change
N_INPUT_UNITS = int(math.ceil(N_INPUT_TIME / RESOLUTION_TIME))
N_OUTPUT_UNITS = int(math.ceil(N_OUTPUT_TIME / RESOLUTION_TIME))

OUTPUT_TEMPO = mido.bpm2tempo(120)
OUTPUT_TICKS_PER_BEAT = 500
OUTPUT_RESOLUTION_TIME = OUTPUT_TEMPO / OUTPUT_TICKS_PER_BEAT

OUTPUT_DURATION_US = OUTPUT_DURATION * 1000 * 1000
OUTPUT_DURATION_UNITS = int(OUTPUT_DURATION_US / RESOLUTION_TIME)
OUTPUT_DURATION_SAMPLES = int(OUTPUT_DURATION_UNITS / N_OUTPUT_UNITS)
#############################################################################

print("=========")
print("Constants")
print("=========")
print("Unit resolution time: " + str(RESOLUTION_TIME/1000) + "ms")
print("Pitch lower bound: " + str(PITCH_LOWERBOUND))
print("Pitch upper bound: " + str(PITCH_UPPERBOUND))
print("Pitch count: " + str(PITCH_COUNT))
print("Sample input size: " + str(N_INPUT_TIME/1000/1000) + "s (" + str(N_INPUT_UNITS) + " units)" )
print("Sample output size: " + str(N_OUTPUT_UNITS*RESOLUTION_TIME/1000/1000) + "s (" + str(N_OUTPUT_UNITS) + " units)" )
print("Output midi duration: " + str(OUTPUT_DURATION/60) + "min (" + str(OUTPUT_DURATION_UNITS) + " units)")
print("GPU batch size: " + str(N_BATCH_SIZE) + " samples (" + str(N_BATCH_SIZE*N_INPUT_UNITS) + " input units, "
      + str(N_BATCH_SIZE*N_OUTPUT_UNITS) + " output units)")
print("Validation split ratio: " + str(VALIDATION_DATA_RATIO))
print()

#Training is repeated TRAIN_EPOCHS times
#Training is in batches
#Each batch is trained with GPU_BATCH_SIZE input sequences
#Each input sequence is N_INPUT_UNITS time units long
#Each time unit has the notes state
#Each time unit is a time point with a time interval of RESOLUTION_TIME miscroseconds  

#Tempo = microseconds per quarter note/beat
#Input data: [time unit] => State
#State: [octave] => [Chromatic notes states]