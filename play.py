#!/usr/bin/env python

import train
import pickle
import math
import chess, chess.pgn
from parse_game import numeric_notation
from parse_game import bb2array
import heapq
import time
import re
import string
import numpy
import sunfish
import pickle
import random
import traceback

def create_move(board, crdn):
	# workaround for pawn promotions
	move = chess.Move.from_uci(crdn)
	if board.piece_at(move.from_square).piece_type == chess.PAWN:
		if int(move.to_square/8) in [0, 7]:
			move.promotion = chess.QUEEN # always promote to queen
	return move

class Player(object):
	def move(self, gn_current):
		raise NotImplementedError()


class Murasaki(Player):
	def __init__(self):
		self._model = train.make_model()
		self._model.compile(loss='mean_squared_error', optimizer='adam')

	def move(self, gn_current):
		assert(gn_current.board().turn == True)

		color = 0

		# X = numpy.array([sf2array(self._pos, flip=(color==1)),])
		X = numpy.array([bb2array( gn_current.board(), flip=(color==1) )])
		# print X
		predicted = self._model.predict( X )
		print predicted

		best_move = ""
		best_value = 1e6
		print "Valid_moves"+"	"+"meansqr"+"	"+"bestmnsqr"+"	bestmv"
		for move in gn_current.board().generate_legal_moves() :
			notation = numeric_notation(str(move))
			# print notation
			value = sum([(i-j)*(i-j) for i,j in zip(predicted[0],notation)])
			print notation, value, best_value, best_move
			if best_value > value :
				best_value = value
				best_move = move
		#print

		print 'best:', best_value, str(best_move)

		move = create_move(gn_current.board(), str(best_move))	# consider promotions

		gn_new = chess.pgn.GameNode()
		gn_new.parent = gn_current
		gn_new.move = move

		return gn_new

class Murasaki2(Player):
	def __init__(self):
		self._model = train.make_model()
		self._model.compile(loss='mean_squared_error', optimizer='adadelta')

	def move(self, gn_current):
		assert(gn_current.board().turn == False)

		color = 0

		# X = numpy.array([sf2array(self._pos, flip=(color==1)),])
		X = numpy.array([bb2array( gn_current.board(), flip=(color==1) )])
		#print X
		predicted = self._model.predict( X )
		#print predicted

		best_move = ""
		best_value = 1e6
		for move in gn_current.board().generate_legal_moves() :
			notation = numeric_notation(str(move))
			#print notation
			value = sum([(i-j)*(i-j) for i,j in zip(predicted[0],notation)])
			print value, best_value
			if best_value > value :
				best_value = value
				best_move = move
		#print

		print 'best:', best_value, str(best_move)

		move = create_move(gn_current.board(), str(best_move))	# consider promotions

		gn_new = chess.pgn.GameNode()
		gn_new.parent = gn_current
		gn_new.move = move

		return gn_new


class Human(Player):
	def move(self, gn_current):
		bb = gn_current.board()

		print bb

		def get_move(move_str):
			try:
				move = chess.Move.from_uci(move_str)
			except:
				print 'cant parse'
				return False
			if move not in bb.legal_moves:
				print 'not a legal move'
				return False
			else:
				return move

		while True:
			print 'your turn:'
			move = get_move(raw_input())
			if move:
				break

		gn_new = chess.pgn.GameNode()
		gn_new.parent = gn_current
		gn_new.move = move

		return gn_new


class Sunfish(Player):
	def __init__(self, maxn=1e4):
		self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
		self._maxn = maxn

	def move(self, gn_current):
		import sunfish

		assert(gn_current.board().turn == False)

		# Apply last_move
		crdn = str(gn_current.move)
		move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
		self._pos = self._pos.move(move)

		t0 = time.time()
		move, score = sunfish.search(self._pos, maxn=self._maxn)
		print time.time() - t0, move, score
		self._pos = self._pos.move(move)

		crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
		move = create_move(gn_current.board(), crdn)

		gn_new = chess.pgn.GameNode()
		gn_new.parent = gn_current
		gn_new.move = move

		return gn_new

def game():
	gn_current = chess.pgn.Game()

	maxn = 10 ** (2.0 + random.random() * 1.0) # max nodes for sunfish

	print 'maxn %f' % maxn

	player_a = Murasaki()
	# player_b = Human()
	player_b = Sunfish(maxn=maxn)
	# player_b = Murasaki2()

	times = {'A': 0.0, 'B': 0.0}

	while True:
		for side, player in [('A', player_a), ('B', player_b)]:
			t0 = time.time()
			try:
				gn_current = player.move(gn_current)
			except KeyboardInterrupt:
				return
			except:
				traceback.print_exc()
				return side + '-exception', times

			times[side] += time.time() - t0
			print '=========== Player %s: %s' % (side, gn_current.move)
			s = str(gn_current.board())
			print s
			if gn_current.board().is_checkmate():
				return side, times
			elif gn_current.board().is_stalemate():
				return '-', times
			elif gn_current.board().can_claim_fifty_moves():
				return '-', times
			elif s.find('K') == -1 or s.find('k') == -1:
				# Both AI's suck at checkmating, so also detect capturing the king
				return side, times

def play():
	while True:
		side, times = game()
		f = open('stats.txt', 'a')
		f.write('%s %f %f\n' % (side, times['A'], times['B']))
		f.close()

if __name__ == '__main__':
	# play()
	fin=game()
	if(fin[0]=="-"):
		print "DRAW"
	else:
		print fin[0]+" WINS"
