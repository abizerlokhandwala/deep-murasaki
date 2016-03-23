#!/usr/bin/env python

import numpy
import theano
import theano.tensor as T
import os
from sklearn.cross_validation import train_test_split
import pickle
import random
import itertools
from theano.tensor.nnet import sigmoid
import scipy.sparse
import h5py
import math
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from numpy import array

import itertools

import os, sys, time, random

BATCH_SIZE = 2000

rng = numpy.random

DATA_FOLDER = 'data'

def floatX(x):
	return numpy.asarray(x, dtype=theano.config.floatX)

def load_data(dir = DATA_FOLDER):
	for fn in os.listdir(dir):
		if not fn.endswith('.hdf5'):
			continue

		fn = os.path.join(dir, fn)
		try:
			yield h5py.File(fn, 'r')
		except:
			print 'could not read', fn


def get_data(series=['x', 'm']):
	data = [[] for s in series]
	for f in load_data():
		try:
			for i, s in enumerate(series):
				data[i].append(f[s].value)
		except:
			raise
			print 'failed reading from', f

	def stack(vectors):
		if len(vectors[0].shape) > 1:
			return numpy.vstack(vectors)
		else:
			return numpy.hstack(vectors)

	data = [stack(d) for d in data]

	#test_size = 10000.0 / len(data[0])	# does not work for small data sets (<10k entries)
	test_size = 0.1		# let's make it fixed 10% instead
	print 'Splitting', len(data[0]), 'entries into train/test set'
	data = train_test_split(*data, test_size=test_size)

	print data[0].shape[0], 'train set', data[1].shape[0], 'test set'
	return data


def train():
#	MODEL_PICKLE = 'model.pickle'

	X_train, X_test, m_train, m_test = get_data(['x', 'm'])
#	for board in X_train[:2] :
#		for row in xrange(8):
#			print ' '.join('%2d' % x for x in board[(row*8):((row+1)*8)])
#		print

#	if os.path.isfile( MODEL_PICKLE ) :		# saved model exists, load it
#		with open(MODEL_PICKLE) as fin :
#			saved_ws, saved_bs = pickle.load(fin)
#		Ws_s, bs_s = get_parameters(n_in=n_in, n_hidden_units=[2048] * 3, Ws = saved_ws, bs = saved_bs)
#	else :
#		Ws_s, bs_s = get_parameters(n_in=n_in, n_hidden_units=[2048] * 3)

	model = Sequential()
	model.add(Dense(2048, input_dim = 64, init='uniform', activation='relu' ))
	model.add(Dropout(0.2))
	model.add(Dense(2048, init='uniform', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2048, init='uniform', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(4, init='uniform', activation='relu'))

	print 'compiling...'
	#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#	model.compile(loss='squared_hinge', optimizer='adadelta')
	model.compile(loss='mean_squared_error', optimizer='sgd')

	#print 'fitting...'
	model.fit( X_train, m_train, nb_epoch = 10, batch_size = BATCH_SIZE)	#, verbose=2)	#, show_accuracy = True )

	print 'evaluating...'
	score = model.evaluate(X_test, m_test, batch_size = BATCH_SIZE )

	print 'score:', score

#	model.save_weights( 'new.model', overwrite = True )

#	print X_test[:10]
#	print model.predict( X_test[:10], batch_size = 10 )


if __name__ == '__main__':
	train()
