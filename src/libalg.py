'''
Optimised linear algebra routines.

S. Maraniello, 10 Jul 2018

@todo: most time in this routine is spent within numpy.core.multiarray.array.
Consider redefining output as list.
'''

import numpy as np


def cross3d(ra,rb):
	'''Faster than np.cross'''
	return np.array([ra[1]*rb[2]-ra[2]*rb[1],
					    ra[2]*rb[0]-ra[0]*rb[2],
					      	ra[0]*rb[1]-ra[1]*rb[0] ])

def norm3d(ra):
	'''Faster than np.linalg.norm'''
	return np.sqrt(ra[0]*ra[0]+ra[1]*ra[1]+ra[2]*ra[2])