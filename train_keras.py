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
from keras.layers import Convolution2D, Reshape, MaxPooling2D, Flatten
# new keras moved imports
#from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, Callback
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

#	#test_size = 10000.0 / len(data[0])	# does not work for small data sets (<10k entries)
#	test_size = 0.05		# let's make it fixed 5% instead
#	print 'Splitting', len(data[0]), 'entries into train/test set'
#	data = train_test_split(*data, test_size=test_size)
#
#	print data[0].shape[0], 'train set', data[1].shape[0], 'test set'
	return data

def show_board( board ) :
	for row in xrange(8):
		print ' '.join('%2d' % x for x in board[(row*8):((row+1)*8)])
	print

def train():
	MODEL_SIZE = [1024, 1024, 1024, 1024, 1024, 1024, 1024]
#	MODEL_SIZE = [8192, 8192, 4096, 2048, 2048, 1024, 1024]
#	MODEL_SIZE = [4096, 4096, 2048, 2048, 1024, 512, 256]
#	MODEL_SIZE = [512, 512, 512, 512, 512, 512, 512]
#	MODEL_SIZE = [256, 256, 256, 256, 256, 256, 256]

	CONVOLUTION = 64
	MODEL_SIZE = [4096, 2048, 1024, 1024]	# 45M @ AWS
	MODEL_SIZE = [4096, 4096, 2048, 1024]	# 45M (1999-2001)
	MODEL_DATA = 'new_%s.model' % ('_'.join(['%d' % i for i in MODEL_SIZE]))
	MODEL_DATA = 'conv%d_%s.model' % (CONVOLUTION, '_'.join(['%d' % i for i in MODEL_SIZE]))

	X, m = get_data(['x', 'm'])
#	X_train, X_test, m_train, m_test = get_data(['x', 'm'])
#	for board in X_train[:2] :
#		show_board( board )

	model = Sequential()
	model.add(Reshape( dims = (1, 8, 8), input_shape = (64,)))
	model.add(Convolution2D( CONVOLUTION, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
#	model.add(Convolution2D(8, 3, 3))
#	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	for i in MODEL_SIZE :
		model.add(Dense( i, init='uniform', activation='relu'))

	model.add(Dense( 4, init='uniform', activation='relu'))

#	model.add(Dense(MODEL_SIZE[0], input_dim = 64, init='uniform', activation='relu' ))
##	model.add(Dropout(0.2))
#	for i in MODEL_SIZE[1:] :
#		model.add(Dense( i, init='uniform', activation='relu'))
##		model.add(Dropout(0.2))
#	model.add(Dense(4, init='uniform', activation='relu'))

	if os.path.isfile( MODEL_DATA ) :		# saved model exists, load it
		model.load_weights( MODEL_DATA )

	print 'compiling...'
#	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#	model.compile(loss='squared_hinge', optimizer='adadelta')
	model.compile(loss='mean_squared_error', optimizer='adadelta')

	early_stopping = EarlyStopping( monitor = 'loss', patience = 50 )	# monitor='val_loss', verbose=0, mode='auto'
	#print 'fitting...'
	history = model.fit( X, m, nb_epoch = 100, batch_size = BATCH_SIZE)	#, callbacks = [early_stopping])	#, validation_split=0.05)	#, verbose=2)	#, show_accuracy = True )

#	print 'evaluating...'
#	score = model.evaluate(X_test, m_test, batch_size = BATCH_SIZE )
#	print 'score:', score

	model.save_weights( MODEL_DATA, overwrite = True )

	#print X_train[:10]
#	print m_train[:20]
#	print model.predict( X_train[:20], batch_size = 5 )
	print m[:20]
	print model.predict( X[:20], batch_size = 5 )

#	print m_test[:20]
#	print model.predict( X_test[:20], batch_size = 5 )

#	with open( MODEL_DATA + '.history', 'w') as fout :
#		print >>fout, history.losses


if __name__ == '__main__':
	train()
