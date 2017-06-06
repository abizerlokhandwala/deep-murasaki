#!/usr/bin/env python

import chess
import chess.uci

board = chess.Board()

engine = chess.uci.popen_engine('stockfish')
engine.uci()

info_handler = chess.uci.InfoHandler()
engine.info_handlers.append(info_handler)

def evaluation( board ) :
	engine.position( board )
	engine.go( movetime = 3000 )
	return  -info_handler.info["score"][1].cp

def generate_evaluation( board ) :
	moves = []
	for m in board.legal_moves :
		board.push( m )
		#print m, evaluation(board),
		moves.append( (evaluation(board), m) )
		board.pop()

	moves = sorted(moves, reverse = True)
	print board.fen(), '{',
	for m in moves :
		if moves[0][0] - m[0] > 30 : break	# 0.3 pawn limit
		print m[1], m[0], ',',
	print '}'

if __name__ == '__main__' :
	generate_evaluation( board )

