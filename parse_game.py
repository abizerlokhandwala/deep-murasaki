#!/usr/bin/env python

import chess, chess.pgn
import numpy
import sys
import os
import multiprocessing
import itertools
import random
import h5py

DATA_FOLDER = 'data'

if not os.path.isdir( DATA_FOLDER ) :
	sys.exit(DATA_FOLDER + ' is not accessible')

def read_games(fn):
	f = open(fn)

	while True:
		try:
			g = chess.pgn.read_game(f)
		except KeyboardInterrupt:
			raise
		except:
			continue

		if not g:
			break

		yield g


def bb2array(b, flip=False):
	x = numpy.zeros(64, dtype=numpy.int8)

#	for pos, piece in enumerate(b.pieces()):	# broken in pychess v0.13.2, hence the next two lines
	for pos in range(64) :
		piece = b.piece_type_at(pos)
		if piece :
			color = int(bool(b.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
			col = int(pos % 8)
			row = int(pos / 8)
			if flip:
				row = 7-row
				col = 7-col		# preserve the symmetry after flipping
				color = 1 - color

			piece = color*7 + piece

			x[row * 8 + col] = piece

	return x


def parse_game(g):
	rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
	r = g.headers['Result']
	if r not in rm:
		return None
	y = rm[r]
	# print >> sys.stderr, 'result:', y

	# Generate all boards
	gn = g.end()
	if not gn.board().is_game_over():
		return None

	gns = []
	moves_left = 0
	while gn:
		gns.append((moves_left, gn, gn.board().turn == 0))
		gn = gn.parent
		moves_left += 1

#	print len(gns)
#	if len(gns) < 10:
#		print g.end()

	if len(gns) > 10 :
		num = random.randint(0,5)
		for i in range(num) :
			gns.pop()		# remove first N positions to lessen repetitions
	gns.pop()

	#moves_left, gn, flip = random.choice(gns)

	result = []
	for moves_left, gn, flip in gns :

		b = gn.board()
		x = bb2array(b, flip=flip)
		b_parent = gn.parent.board()
		x_parent = bb2array(b_parent, flip=(not flip))
		if flip:
			y = - rm[r]

		# generate a random board
#		moves = list(b_parent.legal_moves)
#		move = random.choice(moves)
#		b_parent.push(move)
#		x_random = bb2array(b_parent, flip=flip)

		#if moves_left < 3:
		#	print moves_left, 'moves left'
		#	print 'winner:', y
		#	print g.headers
		#	print b
		#	print 'checkmate:', g.end().board().is_checkmate()

		# print x
		# print x_parent
		# print x_random

#		result.append( (x, x_parent, x_random, moves_left, y) )
		result.append( (x, x_parent) )

	return result

def read_all_games(fn_in, fn_out):
	g = h5py.File(fn_out, 'w')
#	X, Xr, Xp = [g.create_dataset(d, (0, 64), dtype='b', maxshape=(None, 64), chunks=True) for d in ['x', 'xr', 'xp']]
#	Y, M = [g.create_dataset(d, (0,), dtype='b', maxshape=(None,), chunks=True) for d in ['y', 'm']]
	X, Xp = [g.create_dataset(d, (0, 64), dtype='b', maxshape=(None, 64), chunks=True) for d in ['x', 'xp']]
	size = 0
	line = 0
	for game in read_games(fn_in):
		game = parse_game(game)
		if game is None:
			continue

#		for x, x_parent, x_random, moves_left, y in game :
		for x, x_parent in game :
			if line + 1 >= size:
				g.flush()
				size = 2 * size + 1
				print 'resizing to', size
#				[d.resize(size=size, axis=0) for d in (X, Xr, Xp, Y, M)]
				[d.resize(size=size, axis=0) for d in (X, Xp)]

			X[line] = x
#			Xr[line] = x_random
			Xp[line] = x_parent
#			Y[line] = y
#			M[line] = moves_left

			line += 1

#	[d.resize(size=line, axis=0) for d in (X, Xr, Xp, Y, M)]
	[d.resize(size=line, axis=0) for d in (X, Xp)]
	g.close()

def read_all_games_2(a):
	return read_all_games(*a)

def parse_dir():
	files = []

	for fn_in in os.listdir(DATA_FOLDER):
		if not fn_in.endswith('.pgn'):
			continue
		fn_in = os.path.join(DATA_FOLDER, fn_in)
		fn_out = fn_in.replace('.pgn', '.hdf5')
		if not os.path.exists(fn_out) :
			files.append((fn_in, fn_out))

	print files
	if len(files) :
		pool = multiprocessing.Pool()
		pool.map(read_all_games_2, files)
		pool.close()

if __name__ == '__main__':
	parse_dir()

