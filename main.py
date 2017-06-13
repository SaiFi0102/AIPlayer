from constants import *
from functions import *
from data import *
from network import *

# MAIN
if __name__ == '__main__':
	create_directory("data") #For trained weights and states
	create_directory("output") #For generated output

	#Data
	dataSeq = musicFolderToNoteStateSeq(TRAIN_MUSIC_FOLDER)
	noteStateToWordIdx, wordIdxToNoteState, wordIdxToCount = loadVocabularyData(dataSeq)
