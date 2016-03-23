#!/usr/bin/env python

import chess, chess.pgn
import numpy
import sys
import os, time
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


letters = { 'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g' : 6, 'h' : 7 }
numbers = { '1' : 0, '2' : 1, '3' : 2, '4' : 3, '5' : 4, '6' : 5, '7' : 6, '8' : 7 }

def numeric_notation( move ) :
	m = numpy.zeros( 4, dtype=numpy.int8)
	m[0] = letters[move[0]]
	m[1] = numbers[move[1]]
	m[2] = letters[move[2]]
	m[3] = numbers[move[3]]
	return m

def parse_game(g):
	rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
	r = g.headers['Result']
	if r not in rm:
		return None
	y = rm[r]
	# print >> sys.stderr, 'result:', y

	# Generate all boards
	gn = g.end()
#	if not gn.board().is_game_over():
#		return None

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

#		print b_parent
#		print b_parent.parse_san(gn.san()).uci()
#		if len(result) > 6 :
#			return None

		x_parent = bb2array(b_parent, flip=(not flip))
		if flip:
			y = - rm[r]

		move = b_parent.parse_san(gn.san()).uci()

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
#		result.append( (x, x_parent, x_random) )
		result.append( (x_parent, numeric_notation(move)) )

	return result

def read_all_games(fn_in, fn_out):
	g = h5py.File(fn_out, 'w')
	X = g.create_dataset('x', (0, 64), dtype='b', maxshape=(None, 64), chunks=True)
	M = g.create_dataset('m', (0, 4), dtype='b', maxshape=(None, 4), chunks=True)
	size = 0
	line = 0
	for game in read_games(fn_in):
		game = parse_game(game)
		if game is None:
			continue

		for x, m in game :
			if line + 1 >= size:
				g.flush()
				size = 2 * size + 1
				print 'resizing to', size
				[d.resize(size=size, axis=0) for d in (X, M)]

			X[line] = x
			M[line] = m

			line += 1

	[d.resize(size=line, axis=0) for d in (X, M)]	# shrink to fit
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

def pretty_time( t ) :
	if t > 86400 :
		return '%.2fd' % (t / 86400)
	if t > 3600 :
		return '%.2fh' % (t / 3600)
	if t > 60 :
		return '%.2fm' % (t / 60)
	return '%.2fs' % t

if __name__ == '__main__':
	start = time.time()
	parse_dir()
	print 'done in', pretty_time(time.time() - start)

