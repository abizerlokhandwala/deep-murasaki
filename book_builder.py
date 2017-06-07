#!/usr/bin/env python

import sys, os, random, json, time

import chess
import chess.uci

import chess.pgn

board = chess.Board()

engine = chess.uci.popen_engine('stockfish')
engine.uci()
engine.isready()
engine.setoption( {'Hash' : 8192, 'Threads' : 24 } )
#print engine.options
#sys.exit(1)

info_handler = chess.uci.InfoHandler()
engine.info_handlers.append(info_handler)

already_have = set()
if os.path.isfile( 'already_have.json' ) :
	with open( 'already_have.json' ) as fin :
		already_have = set(json.load( fin ))

def evaluation( board ) :
	engine.position( board )
	engine.go( movetime = 400 )

	try :
		if info_handler.info['score'][1].cp == None :
			return info_handler.info['score'][1].mate * 1000 - 60000
		else :
			return  -info_handler.info["score"][1].cp
	except :
		print info_handler.info

def generate_evaluation( board ) :
	moves = []
	for m in board.legal_moves :
		board.push( m )
		#print m, evaluation(board),
		moves.append( (evaluation(board), m) )
		board.pop()

	moves = sorted(moves, reverse = True)
	m_list = []
	for m in moves :
		#if moves[0][0] - m[0] > 30 : break	# 0.3 pawn limit
		m_list.append( '%s:%d' % (board.san(m[1]), m[0]) )
	return m_list

def normalize( fen, m_eval ) :
	parts = fen.split()

	parts[4], parts[5] = '0', '1'

	if parts[1] == 'b' :
		parts[0] = '/'.join(reversed(parts[0].split('/'))).swapcase()
		parts[1] = 'w'
		parts[2] = ''.join( sorted(parts[2].swapcase()) )	# castling
		if parts[3] != '-' :
			parts[3] = parts[3][0] + str(9-int(parts[3][1]))	# enpassant

		b_eval = []
		for m in m_eval :
			vals = m.split(':')
			vals[0] = ''.join([str(9-int(c)) if c.isdigit() else c for c in vals[0]])
			b_eval.append( ':'.join(vals) )
	else :
		b_eval = m_eval

	#print parts
	fen = ' '.join( parts )

	return fen, b_eval

if __name__ == '__main__' :

	if len(sys.argv) < 2 :
		print 'USAGE: book_builder file.pgn'
		sys.exit(1)

	counter = 0
	with open( sys.argv[1] ) as pgn :
		while True :
			game = chess.pgn.read_game( pgn )
			if game == None : break
			counter += 1

			engine.ucinewgame()
			print chess.Board().variation_san( game.main_line() )

			cnt = 0
			fen_data = [ '# ' + chess.Board().variation_san( game.main_line() ) ]
			board = chess.Board()
			for m in game.main_line() :
				print m,
				board.push(m)
				if board.is_game_over() :
					print 'game over'
					break

#				cnt += 1
#				if cnt < 50 : continue

				print board
				zobrist = board.zobrist_hash()
				if zobrist not in already_have :
					already_have.add( zobrist )

					fen, b_eval = normalize( board.fen(), generate_evaluation( board ) )
					print fen, '{' + ', '.join( b_eval ) + '}'
					fen_data.append( fen + ' {' + ', '.join( b_eval ) + '}' )

				else :
					print board.fen(), 'already have'

			#break

			with open( 'fen_data.txt', 'a' ) as fout :
				fout.write( '\n'.join( fen_data ) )
				fout.write( '\n' )

			with open( 'already_have.json', 'w' ) as fout :
				json.dump( list(already_have), fout )

	print 'read:', counter

#	for i in range(5) :
#		fen, b_eval = normalize( board.fen(), generate_evaluation( board ) )
#		print fen, '{' + ', '.join( b_eval ) + '}'
#		for m in board.legal_moves :
#			board.push(m)
#			break

