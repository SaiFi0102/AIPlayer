import math, mido

#DEBUG = True
TRAIN_MUSIC_FOLDER = "music_dev"
NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
RESOLUTION_TIME = 10000 # In microseconds
OUTPUT_TICKS_PER_BEAT = 500
OUTPUT_TEMPO = mido.bpm2tempo(120)
OUTPUT_RESOLUTION_TIME = OUTPUT_TEMPO / OUTPUT_TICKS_PER_BEAT
PITCH_LOWERBOUND = 24 # C1 (Midi note value)
PITCH_OCTAVES = 5 # Upperbound: B6
PITCH_COUNT = len(NOTES) * PITCH_OCTAVES
PITCH_UPPERBOUND = PITCH_LOWERBOUND + PITCH_COUNT - 1

N_INPUT_TIME = 2 * 1000 * 1000 # seconds * (microseconds)
N_OUTPUT_TIME = 500 * 1000
N_OUTPUT_UNITS = int(math.ceil(float(N_OUTPUT_TIME) / RESOLUTION_TIME))
N_INPUT_UNITS = int(math.ceil(float(N_INPUT_TIME) / RESOLUTION_TIME))
N_FIXED_BATCH_SIZE = None
N_CHANNELS = 1
N_PLAY_THRESHOLD = 0.5

NSE_BATCH_SIZE = 25000
NSE_ENCODED_SIZE = 20

GPU_BATCH_SIZE = 500
TRAIN_EPOCHS = 500
TRAIN_DATASET_STEP_FACTOR = 1
TRAIN_DATASET_STEP = int(N_INPUT_UNITS * TRAIN_DATASET_STEP_FACTOR)

OUTPUT_DURATION = 180 #In seconds
OUTPUT_DURATION_US = OUTPUT_DURATION * 1000 * 1000
OUTPUT_DURATION_UNITS = OUTPUT_DURATION_US / RESOLUTION_TIME

print("Resolution Time: " + str(RESOLUTION_TIME/1000.) + "ms")
print("Pitch lowerbound: " + str(PITCH_LOWERBOUND))
print("Pitch upperbound: " + str(PITCH_UPPERBOUND))
print("Pitch count: " + str(PITCH_COUNT))
print("Network input timesteps: " + str(N_INPUT_TIME/1000./1000) + "s (" + str(N_INPUT_UNITS) + " units)" )
print("Network output timesteps: " + str(N_OUTPUT_UNITS*RESOLUTION_TIME/1000.) + "ms (" + str(N_OUTPUT_UNITS) + " units)" )
print("Output midi duration: " + str(OUTPUT_DURATION/60) + "min (" + str(OUTPUT_DURATION_UNITS) + " units)")
print("GPU batch size: " + str(GPU_BATCH_SIZE) + " units")
print("======================")

assert(TRAIN_DATASET_STEP >= N_OUTPUT_UNITS)
assert(N_OUTPUT_UNITS <= N_INPUT_UNITS)

#Training is repeated TRAIN_EPOCHS times
#Training is in batches
#Each batch is trained with GPU_BATCH_SIZE input sequences
#Each input sequence is N_INPUT_UNITS time units long
#Each time unit has the notes state
#Each time unit is a time point with a time interval of RESOLUTION_TIME miscroseconds  

#Tempo = microseconds per quarter note/beat
#Input data: [time unit] => State
#State: [octave] => [Chromatic notes states]