#!/usr/bin/env python

import h5py, os, sys

for r, d, files in os.walk('data') :
	for f in sorted(files) :
		if not f.endswith('.hdf5') : continue

		print f,
		data = h5py.File( os.path.join( r, f ), 'r')

		if sum(data['m'][0]) > 20 :
			print 'scaled'
		else :
			print 'normal'
