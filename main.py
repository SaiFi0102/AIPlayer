import pygame
from constants import *
from functions import *
from data import *
from network import *
import random
import tensorflow as tf
from matplotlib import pylab, cbook
# from sklearn.manifold import TSNE

#Initialize music mixer
pygame.mixer.init()

#Data
noteStateSeq, wordIdxToNoteState, wordIdxToCount = loadCacheData()
vocabulary = np.random.permutation(wordIdxToNoteState)
trainIdx = int(len(vocabulary) * 0.6)
valIdx = trainIdx + int(len(vocabulary) * 0.2)
vocabularyTrain = vocabulary[:trainIdx]
vocabularyVal = vocabulary[trainIdx:valIdx]
vocabularyTest = vocabulary[valIdx:]

def accuracy(decoded, noteStates):
	return 100.0 * (1 - np.sum(np.abs(noteStates - decoded))/decoded.shape[0]/decoded.shape[1])

def createModel(dataset):
	encoded = tf.matmul(dataset, weightsEncoded) + biasesEncoded
	encodedActivated = tf.nn.leaky_relu(encoded)

	decodedLogits = tf.matmul(encodedActivated, weightsDecoded) + biasesDecoded
	decodedProb = tf.nn.sigmoid_cross_entropy_with_logits(labels=dataset, logits=decodedLogits)
	decoded = tf.greater(decodedProb, 0.6)

	return decoded, decodedProb

graph = tf.Graph()
with graph.as_default():
	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_global_step = tf.Variable(0) # count the number of steps taken.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(W2V_BATCH_SIZE, PITCH_COUNT))
	tf_val_dataset = tf.constant(vocabularyVal)
	tf_test_dataset = tf.constant(vocabularyTest)
	tf_eval_dataset = tf.placeholder(tf.float32, shape=(1, PITCH_COUNT))

	# Variables.
	weightsEncoded = tf.Variable(tf.truncated_normal([PITCH_COUNT, 30]))
	weightsDecoded = tf.Variable(tf.truncated_normal([30, PITCH_COUNT]))

	biasesEncoded = tf.Variable(tf.zeros([30]))
	biasesDecoded = tf.Variable(tf.zeros([PITCH_COUNT]))

	saver = tf.train.Saver([weightsEncoded, weightsDecoded, biasesEncoded, biasesDecoded])

	tf_train_decoded, tf_train_decodedProb = createModel(tf_train_dataset)
	tf_val_decoded, _ = createModel(tf_val_dataset)
	tf_test_decoded, _ = createModel(tf_test_dataset)
	tf_eval_decoded, _ = createModel(tf_eval_dataset)

	# Loss
	#tf_train_loss = tf.reduce_mean(tf_train_decodedProb)
	tf_train_loss = tf.reduce_mean(tf.pow(tf_train_dataset - tf_train_decodedProb, 2))

	# Optimizer.
	tf_optimizer = tf.train.AdamOptimizer().minimize(tf_train_loss)

num_epochs = 100

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Initialized")

	try:
		saver.restore(session, tf.train.latest_checkpoint(DATA_FOLDER))
	except Exception:
		print("Can't restore model, no file found")

	'''
	vocabTest = [vocabularyTrain[10]]
	pygame.mixer.music.load(vocabularyToMidiStream(vocabTest));
	pygame.mixer.music.play();
	while pygame.mixer.music.get_busy():
		continue

	for input in vocabTest:
		input = input.reshape(1, input.shape[0])
		decoded = session.run(tf_eval_decoded, feed_dict={tf_eval_dataset: input})
		pygame.mixer.music.load(vocabularyToMidiStream(decoded));
		pygame.mixer.music.play();
		while pygame.mixer.music.get_busy():
			continue
	'''

	for epoch in range(num_epochs):
		idx = 0
		while idx <= len(vocabularyTrain) - 1:
			if idx + W2V_BATCH_SIZE > len(vocabularyTrain):
				batch_data = vocabularyTrain[-W2V_BATCH_SIZE:]
			else:
				batch_data = vocabularyTrain[idx:idx + W2V_BATCH_SIZE]

			_, loss, predictions = session.run([tf_optimizer, tf_train_loss, tf_train_decoded], feed_dict={tf_train_dataset: batch_data})
			idx += W2V_BATCH_SIZE
			# print(idx)

		saver.save(session, os.path.join(DATA_FOLDER, "my-model"), global_step=epoch)
		print("Minibatch loss at epoch %d: %f" % (epoch, loss))
		print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_data))
		print("Validation accuracy: %.1f%%" % accuracy(tf_val_decoded.eval(), vocabularyVal))

	print("Test accuracy: %.1f%%" % accuracy(tf_test_decoded.eval(), vocabularyTest))

# Wait for music to finish
while pygame.mixer.music.get_busy():
	continue