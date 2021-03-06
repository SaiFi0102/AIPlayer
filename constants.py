import math, mido

TRAIN_MUSIC_FOLDER = "music_all"
DATA_FOLDER = "data"
NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
RESOLUTION_TIME = 50 * 1000 # In microseconds represents 1 time unit
FILE_GAP_TIME = 2000 # In milliseconds
OUTPUT_TICKS_PER_BEAT = 500
OUTPUT_TEMPO = mido.bpm2tempo(120)
OUTPUT_RESOLUTION_TIME = OUTPUT_TEMPO / OUTPUT_TICKS_PER_BEAT
PITCH_LOWERBOUND = 24 # C2 (Midi note value)
PITCH_OCTAVES = 5 # Upperbound: B6
PITCH_COUNT = len(NOTES) * PITCH_OCTAVES
PITCH_UPPERBOUND = PITCH_LOWERBOUND + PITCH_COUNT - 1

N_INPUT_TIME = 0.5 * 1000 * 1000 #In microseconds
N_OUTPUT_TIME = 0.5 * 1000 * 1000
N_OUTPUT_UNITS = int(math.ceil(N_OUTPUT_TIME / RESOLUTION_TIME))
N_INPUT_UNITS = int(math.ceil(N_INPUT_TIME / RESOLUTION_TIME))
N_CHANNELS = 1
N_PLAY_THRESHOLD = 0.5

N_BATCH_SIZE = 100
N_TRAIN_EPOCHS = 500

W2V_EMBEDDING_SIZE = 50
W2V_BATCH_SIZE = 100
W2V_NUM_SKIPS = 2
W2V_SKIP_WINDOW = 1
W2V_TRAIN_EPOCHS = 100

OUTPUT_DURATION = 180 #In seconds
OUTPUT_DURATION_US = OUTPUT_DURATION * 1000 * 1000
OUTPUT_DURATION_UNITS = int(OUTPUT_DURATION_US / RESOLUTION_TIME)

print("Resolution Time: " + str(RESOLUTION_TIME/1000) + "ms")
print("Pitch lowerbound: " + str(PITCH_LOWERBOUND))
print("Pitch upperbound: " + str(PITCH_UPPERBOUND))
print("Pitch count: " + str(PITCH_COUNT))
print("Network input timesteps: " + str(N_INPUT_TIME/1000./1000) + "s (" + str(N_INPUT_UNITS) + " units)" )
print("Network output timesteps: " + str(N_OUTPUT_UNITS*RESOLUTION_TIME/1000.) + "ms (" + str(N_OUTPUT_UNITS) + " units)" )
print("Output midi duration: " + str(OUTPUT_DURATION/60) + "min (" + str(OUTPUT_DURATION_UNITS) + " units)")
print("GPU batch size: " + str(N_BATCH_SIZE) + " units")
print("======================")

#Training is repeated TRAIN_EPOCHS times
#Training is in batches
#Each batch is trained with GPU_BATCH_SIZE input sequences
#Each input sequence is N_INPUT_UNITS time units long
#Each time unit has the notes state
#Each time unit is a time point with a time interval of RESOLUTION_TIME miscroseconds  

#Tempo = microseconds per quarter note/beat
#Input data: [time unit] => State
#State: [octave] => [Chromatic notes states]