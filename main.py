import pygame
from constants import *
from functions import *
from data import *
from network import *
from sklearn.manifold import TSNE
from matplotlib import pylab, cbook
import random

pygame.mixer.init()

#Data
noteStateSeq, wordSeq, noteStateToWordIdx, wordIdxToNoteState, wordIdxToCount = loadCacheData()
vocabularySize = len(noteStateToWordIdx)

#PLOTTING
class DataCursor(object):
	# https://stackoverflow.com/a/4674445/190597
	"""A simple data cursor widget that displays the x,y location of a
	matplotlib artist when it is selected."""
	def __init__(self, artists, x = [], y = [], wordIdx = [], tolerance = 5, offsets = (-20, 20),
				 formatter = lambda x,y: 'x: {x:0.2f}\ny: {y:0.2f}'.format(x = x, y = y), display_all = False):
		"""Create the data cursor and connect it to the relevant figure.
		"artists" is the matplotlib artist or sequence of artists that will be
			selected.
		"tolerance" is the radius (in points) that the mouse click must be
			within to select the artist.
		"offsets" is a tuple of (x,y) offsets in points from the selected
			point to the displayed annotation box
		"formatter" is a callback function which takes 2 numeric arguments and
			returns a string
		"display_all" controls whether more than one annotation box will
			be shown if there are multiple axes.  Only one will be shown
			per-axis, regardless.
		"""
		self._points = np.column_stack((x,y))
		self._wordIdxs = wordIdx
		self.formatter = formatter
		self.offsets = offsets
		self.display_all = display_all
		if not cbook.iterable(artists):
			artists = [artists]
		self.artists = artists
		self.axes = tuple(set(art.axes for art in self.artists))
		self.figures = tuple(set(ax.figure for ax in self.axes))

		self.annotations = {}
		for ax in self.axes:
			self.annotations[ax] = self.annotate(ax)

		for artist in self.artists:
			artist.set_picker(tolerance)
		for fig in self.figures:
			fig.canvas.mpl_connect('pick_event', self)

	def annotate(self, ax):
		"""Draws and hides the annotation box for the given axis "ax"."""
		annotation = ax.annotate(self.formatter, xy = (0, 0), ha = 'right',
				xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
				bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
				arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
				)
		annotation.set_visible(False)
		return annotation

	def snap(self, x, y):
		"""Return the value in self._points closest to (x, y).
		"""
		idx = np.nanargmin(((self._points - (x,y))**2).sum(axis = -1))
		return idx
	def __call__(self, event):
		"""Intended to be called through "mpl_connect"."""
		# Rather than trying to interpolate, just display the clicked coords
		# This will only be called if it's within "tolerance", anyway.
		x, y = event.mouseevent.xdata, event.mouseevent.ydata
		annotation = self.annotations[event.artist.axes]
		if x is not None:
			if not self.display_all:
				# Hide any other annotation boxes...
				for ann in self.annotations.values():
					ann.set_visible(False)
			# Update the annotation in the current axis..
			pIdx = self.snap(x, y)
			x, y = self._points[pIdx]
			wordIdx = self._wordIdxs[pIdx]

			stream = wordsToDemonstrationMidiStream([wordIdx], wordIdxToNoteState)
			pygame.mixer.music.load(stream)
			pygame.mixer.music.play()

			annotation.xy = x, y
			annotation.set_text(self.formatter(x, y))
			#annotation.set_visible(True)
			event.canvas.draw()

def plot(embeddings, labels):
	assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
	fig = pylab.figure(figsize=(15,15))  # in inches
	ax = fig.add_subplot(1,1,1)
	xs = []
	ys = []
	wordIdxs = []
	for i, label in enumerate(labels):
		x, y = embeddings[i,:]
		xs.append(x)
		ys.append(y)
		wordIdxs.append(label)
	scat = ax.scatter(xs, ys, picker=True)
	DataCursor(scat, xs, ys, wordIdxs)
	pylab.show()

#MODEL
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
	# Input data.
	train_dataset = tf.placeholder(tf.int32, shape=[W2V_BATCH_SIZE, W2V_NUM_SKIPS])
	train_labels = tf.placeholder(tf.int32, shape=[W2V_BATCH_SIZE, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	# Variables.
	embeddings = tf.Variable(tf.random_uniform([vocabularySize, W2V_EMBEDDING_SIZE], -1.0, 1.0))
	softmax_weights = tf.Variable(tf.truncated_normal([vocabularySize, W2V_EMBEDDING_SIZE], stddev=1.0 / math.sqrt(W2V_EMBEDDING_SIZE)))
	softmax_biases = tf.Variable(tf.zeros([vocabularySize]))

	# Model.
	# Look up embeddings for inputs.
	embed = tf.zeros([W2V_BATCH_SIZE, W2V_EMBEDDING_SIZE])
	for i in range(W2V_NUM_SKIPS):
		embed += tf.nn.embedding_lookup(embeddings, train_dataset[:,i])
	embed /= W2V_NUM_SKIPS

	# Compute the softmax loss, using a sample of the negative labels each time.
	loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
													 labels=train_labels, num_sampled=num_sampled, num_classes=vocabularySize))

	# Optimizer.
	# Note: The optimizer will optimize the softmax_weights AND the embeddings.
	# This is because the embeddings are defined as a variable quantity and the
	# optimizer's `minimize` method will by default modify all variable quantities
	# that contribute to the tensor it is passed.
	# See docs on `tf.train.Optimizer.minimize()` for more details.
	optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

	# Compute the similarity between minibatch examples and all embeddings.
	# We use the cosine distance:
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

#TRAINING
num_steps = 200001

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')

	average_loss = 0
	for step in range(num_steps):
		batch_data, batch_labels = generateWord2VecBatch(wordSeq)
		feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
		_, l = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += l
		if step % 2000 == 0:
			if step > 0:
				average_loss = average_loss / 2000
			# The average loss is an estimate of the loss over the last 2000 batches.
			print('Average loss at step %d: %f' % (step, average_loss))
			average_loss = 0
		# note that this is expensive (~20% slowdown if computed every 500 steps)

	# Validation
	sim = similarity.eval()
	wordList = []
	for i in range(valid_size):
		wordIdx = valid_examples[i]
		top_k = 8  # number of nearest neighbors
		nearest = (-sim[i, :]).argsort()[1:top_k + 1]
		wordList.append(wordIdx)
		wordList.append(0)
		for k in range(top_k):
			close_wordIdx = nearest[k]
			wordList.append(close_wordIdx)
		wordList.append(0)
		wordList.append(0)

	stream = wordsToDemonstrationMidiStream(wordList, wordIdxToNoteState)
	pygame.mixer.music.load(stream)
	pygame.mixer.music.play()

	final_embeddings = normalized_embeddings.eval()
	num_points = 800
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])
	words = list(range(1, num_points + 1))
	plot(two_d_embeddings, words)


while pygame.mixer.music.get_busy():
	continue