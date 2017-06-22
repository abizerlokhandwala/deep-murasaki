#!/usr/bin/env python

import sys, os, random, json, time, datetime

import hashlib

import chess
import chess.uci

import chess.pgn

board = chess.Board()

engine = chess.uci.popen_engine('stockfish')
engine.uci()
engine.isready()
engine.setoption( {'Threads' : 24 } )
engine.isready()
engine.setoption( {'Hash' : 8192, 'Threads' : 24 } )
#print engine.options
#sys.exit(1)

info_handler = chess.uci.InfoHandler()
engine.info_handlers.append(info_handler)

already_played = set()
if os.path.isfile( 'already_played.json' ) :
	with open( 'already_played.json' ) as fin :
		already_played = set( json.load(fin) )

already_have = set()
_, _, files = os.walk('.').next()
for f in files :
	if f.startswith( 'already_have' ) and f.endswith( '.json' ) :
		print 'loading', f
		with open( f ) as fin :
			already_have.update( json.load( fin ))

def evaluation( board ) :
	engine.position( board )
	engine.go( movetime = 400 )

	try :
		if info_handler.info['score'][1].cp == None :
			mate = info_handler.info['score'][1].mate
			return (-60000 if mate > 0 else 60000) + mate * 1000
		else :
			return -info_handler.info["score"][1].cp
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

#b = chess.Board('3R4/6RQ/1q3p2/4p3/1P2Pk2/6rP/p4P2/5K2 b - - 0 1')
#b = chess.Board('R7/8/8/8/8/1K6/8/6k1 w - - 0 1')
#print evaluation( b )
#sys.exit(1)

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

			principal = chess.Board().variation_san( game.main_line() )
			p_hash = hashlib.md5(principal).hexdigest()
			if p_hash in already_played : continue
			already_played.add( p_hash )

			engine.ucinewgame()
			print principal

			cnt = 0
			fen_data = [ '# ' + chess.Board().variation_san( game.main_line() ) ]
			board = chess.Board()
			for m in game.main_line() :
				print m,
				board.push(m)
				if board.is_game_over() :
					print 'game over'
					break

				#cnt += 1
				#if cnt < 50 : continue

				#print '\n', board
				zobrist = board.zobrist_hash()
				if zobrist not in already_have :
					already_have.add( zobrist )

					fen, b_eval = normalize( board.fen(), generate_evaluation( board ) )
					print fen, '{' + ', '.join( b_eval ) + '}'
					fen_data.append( fen + ' {' + ', '.join( b_eval ) + '}' )

				else :
					print board.fen(), 'already have'

			#break

			if len(fen_data) > 1 :
				with open( os.path.basename(sys.argv[1]) + '.txt', 'a' ) as fout :
					fout.write( '\n'.join( fen_data ) )
					fout.write( '\n' )

			now = datetime.datetime.now()
			suffix = str(now.strftime('%Y-%m-%d'))
			with open( 'already_have_%s.json' % suffix, 'w' ) as fout :
				json.dump( list(already_have), fout )

			with open( 'already_played.json', 'w' ) as fout :
				json.dump( list(already_played), fout )

	print 'read:', counter

#	for i in range(5) :
#		fen, b_eval = normalize( board.fen(), generate_evaluation( board ) )
#		print fen, '{' + ', '.join( b_eval ) + '}'
#		for m in board.legal_moves :
#			board.push(m)
#			break

