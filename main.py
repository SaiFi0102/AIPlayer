from constants import *
from functions import *
from data import *
from network import *

# MAIN
if __name__ == '__main__':
	#Data
	noteStateSeq, wordSeq, noteStateToWordIdx, wordIdxToNoteState, wordIdxToCount = loadCacheData()