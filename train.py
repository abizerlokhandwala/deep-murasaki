#!/usr/bin/env python

import numpy
import theano
import pickle
import itertools
import scipy.sparse
import h5py
import math
import theano.tensor as T

from keras.models import Sequential

try :
	# old imports (v0.3.1)
	from keras.layers import Dense, Dropout, Activation
	from keras.layers import Convolution2D, Reshape, MaxPooling2D, Flatten
except ImportError :
	# new keras imports (v0.3.3)
	from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import SGD

from numpy import array

import os, sys, time, random

BATCH_SIZE = 2000

rng = numpy.random

DATA_FOLDER = 'data'

def nesterov_updates(loss, all_params, learn_rate, momentum):
    updates = []
    all_grads = T.grad(loss, all_params)
    for param_i, grad_i in zip(all_params, all_grads):
        # generate a momentum parameter
        mparam_i = theano.shared(
            numpy.array(param_i.get_value()*0., dtype=theano.config.floatX))
        v = momentum * mparam_i - learn_rate * grad_i
        w = param_i + momentum * v - learn_rate * grad_i
        updates.append((param_i, w))
        updates.append((mparam_i, v))
    return updates

def get_model(Ws_s, bs_s, dropout=False):
    print 'building expression graph'
    x_s = T.matrix('x')

    if type(dropout) != list:
        dropout = [dropout] * len(Ws_s)

    # Convert input into a 12 * 64 list
    pieces = []
    for piece in [1,2,3,4,5,6, 8,9,10,11,12,13]:
        # pieces.append((x_s <= piece and x_s >= piece).astype(theano.config.floatX))
        pieces.append(T.eq(x_s, piece))

    binary_layer = T.concatenate(pieces, axis=1)

    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))

    last_layer = binary_layer
    n = len(Ws_s)
    for l in xrange(n - 1):
        # h = T.tanh(T.dot(last_layer, Ws[l]) + bs[l])
        h = T.dot(last_layer, Ws_s[l]) + bs_s[l]
        h = h * (h > 0)

        if dropout[l]:
            mask = srng.binomial(n=1, p=0.5, size=h.shape)
            h = h * T.cast(mask, theano.config.floatX) * 2

        last_layer = h

    p_s = T.dot(last_layer, Ws_s[-1]) + bs_s[-1]
    return x_s, p_s

def get_parameters(n_in=None, n_hidden_units=2048, n_hidden_layers=None, Ws=None, bs=None):
    if Ws is None or bs is None:
        print 'initializing Ws & bs'
        if type(n_hidden_units) != list:
            n_hidden_units = [n_hidden_units] * n_hidden_layers
        else:
            n_hidden_layers = len(n_hidden_units)

        Ws = []
        bs = []

        def W_values(n_in, n_out):
            return numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)


        for l in xrange(n_hidden_layers):
            if l == 0:
                n_in_2 = n_in
            else:
                n_in_2 = n_hidden_units[l-1]
            if l < n_hidden_layers - 1:
                n_out_2 = n_hidden_units[l]
                W = W_values(n_in_2, n_out_2)
                gamma = 0.1 # initialize it to slightly positive so the derivative exists
                b = numpy.ones(n_out_2, dtype=theano.config.floatX) * gamma
            else:
                W = numpy.zeros(n_in_2, dtype=theano.config.floatX)
                b = floatX(0.)
            Ws.append(W)
            bs.append(b)

    Ws_s = [theano.shared(W) for W in Ws]
    bs_s = [theano.shared(b) for b in bs]

    return Ws_s, bs_s

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

def make_model(data = None) :
	global MODEL_DATA
	MODEL_SIZE = [1024, 1024, 1024, 1024, 1024, 1024, 1024]
#	MODEL_SIZE = [8192, 8192, 4096, 2048, 2048, 1024, 1024]
#	MODEL_SIZE = [4096, 4096, 2048, 2048, 1024, 512, 256]
#	MODEL_SIZE = [512, 512, 512, 512, 512, 512, 512]
#	MODEL_SIZE = [256, 256, 256, 256, 256, 256, 256]

	MODEL_SIZE = [4096, 2048, 1024, 1024]	# 45M @ AWS
	MODEL_SIZE = [4096, 4096, 2048, 1024]	# 45M (1999-2001)
	MODEL_SIZE = [3072, 2048, 2048, 1024]	# 19M @ work (1999-2000)
	MODEL_SIZE = [2048, 1024, 1024, 1024]	# 5M, 1.011 @ E250
	MODEL_SIZE = [2048, 2048, 1024, 1024]	# 5M, 0.7122 @ E350
	MODEL_SIZE = [2048, 2048, 2048, 1024]	# 5M, 0.7673 @ E100, 0.6638 @ E150
	MODEL_SIZE = [3072, 2048, 2048, 1024]	# 5M, 0.6818 @ E100, 0.6153 @ E125
#	MODEL_SIZE = [8192, 4096, 2048, 1024]	# 19M @ work (1999-2000)
	MODEL_SIZE = [2048, 2048, 1024, 1024]	# 287k 10moves, 0.8181 @ E100, 0.8054 @ E200
	MODEL_SIZE = [3072, 2048, 2048, 1024]	# 287k 10moves, 0.8174 @ E100
	MODEL_SIZE = [1024, 1024, 1024, 1024]	# 287k 10moves

	CONVOLUTION = min( 64, MODEL_SIZE[0] / 64 )	# 64 for 4096 first layer, 32 for 2048 layer

	if data :
		MODEL_DATA = data
	else :
		MODEL_DATA = 'new_%s.model' % ('_'.join(['%d' % i for i in MODEL_SIZE]))
		MODEL_DATA = 'conv%d_%s.model' % (CONVOLUTION, '_'.join(['%d' % i for i in MODEL_SIZE]))

	model = Sequential()
#	model.add(Reshape( dims = (1, 8, 8), input_shape = (64,)))
	model.add(Reshape( (8, 8, 1), input_shape = (64,)))
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

	return model

def train():
	X, m = get_data(['x', 'm'])
#	X_train, X_test, m_train, m_test = get_data(['x', 'm'])
#	for board in X_train[:2] :
#		show_board( board )

	model = make_model()

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
