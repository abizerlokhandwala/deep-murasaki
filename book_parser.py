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
	data = [i.strip() for i in open(fn).readlines() if not i.startswith('#')]

	for g in data :
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
				color = 1 - color

			#piece = color*7 + piece

			#x[row * 8 + col] = piece
			x[row * 8 + col] = -piece if color else piece

	return x


letters = { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5, 'f' : 6, 'g' : 7, 'h' : 8 }
numbers = { '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8 }

def numeric_notation( move ) :
	m = numpy.zeros( 4, dtype=numpy.int8)
	m[0] = letters[move[0]]
	m[1] = numbers[move[1]]
	m[2] = letters[move[2]]
	m[3] = numbers[move[3]]
	return m

def parse_game(g):

	fen, moves = g.split('{')

	board = chess.Board( fen )
	moves = [m.split(':') for m in moves[:-1].split(', ') if len(m) > 1]

	result = []
	for m in moves :
		board.push_san( m[0] )
		result.append( (bb2array(board), int(m[1])) )
		board.pop()

	return result

def read_all_games(fn_in, fn_out):
	g = h5py.File(fn_out, 'w')
	X = g.create_dataset('x', (0, 64), dtype='b', maxshape=(None, 64), chunks=True)
	M = g.create_dataset('m', (0, 1), dtype='float32', maxshape=(None, 1), chunks=True)
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

	for fn_in in os.listdir('.'):
		#print fn_in,
		if not fn_in.endswith('.txt'):
			continue
		if not fn_in.startswith('ficsgamesdb_'):
			continue

		fn_out = os.path.join(DATA_FOLDER, fn_in.replace('.txt', '.hdf5'))
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

