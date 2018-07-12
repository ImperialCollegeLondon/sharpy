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

def norm3d(v):
	'''Faster than np.linalg.norm'''
	return np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def normsq3d(v):
	return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]

def skew3d(v):
	''' Produce skew matrix such that v x b = skew(v)*b	'''	

	v[0],v[1],v[2]=v[0],v[1],v[2]
	Vskew=np.array([[  0., -v[2], v[1]],
					[ v[2],   0.,-v[0]],
					[-v[1], v[0],   0.]])
	return Vskew




