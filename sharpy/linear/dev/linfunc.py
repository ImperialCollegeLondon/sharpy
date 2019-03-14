'''
author: S. Maraniello
date: 25 May 2018

Functions for symbolic manipulation
'''


import sympy as sm
import sympy.tensor.array as smarr
from IPython import embed


def scalar_product(Av,Bv):
	'''
	Scalar product for sympy.tensor.array
	'''

	N=len(Av)
	assert N==len(Bv), 'Array dimension not matching'

	P=Av[0]*0
	for ii in range(len(Av)):
		P=P+Av[ii]*Bv[ii]

	return P


def matrix_product(Asm,Bsm):
	'''
	Matrix product between 2D sympy.tensor.array
	'''

	assert len(Asm.shape)<3 or len(Bsm.shape)<3,\
	                  'Attempting matrix product between 3D (or higher) arrays!'
	assert Asm.shape[1]==Bsm.shape[0], 'Matrix dimensions not compatible!'

	P=smarr.tensorproduct(Asm,Bsm)
	Csm=smarr.tensorcontraction(P,(1,2))

	return Csm


def skew(Av):
	'''
	Compute skew matrix of vector Av, such that:
		skew(Av)*Bv = Av x Bv
	'''

	assert len(Av)==3, 'Av must be a 3 elements array'

	ax,ay,az=Av[0],Av[1],Av[2]

	Askew=smarr.MutableDenseNDimArray([ [  0,-az, ay],
										[ az,  0,-ax],
										[-ay, ax,  0] ])
	
	return Askew


def cross_product(Av,Bv):
	'''
	Cross-product Av x Bv
	'''
	return matrix_product(skew(Av),Bv)


def norm2(Av):
	'''
	Computes Euclidean norm of a vector
	'''
	return sm.sqrt(scalar_product(Av,Av))


def simplify(Av):
	'''
	Simplify each element of matrix/array
	'''

	Av_simple=[]

	if len(Av.shape)==1:
		for ii in range(len(Av)):
			Av_simple.append(Av[ii].simplify())

	elif len(Av.shape)==2:
		for ii in range(Av.shape[0]):
			row=[]
			for jj in range(Av.shape[1]):
				row.append(Av[ii,jj].simplify())
			Av_simple.append(row)

	else:
		raise NameError('Method not developed for 3D arrays!')

	return smarr.MutableDenseNDimArray(Av_simple)


def subs(Av,expr_old,expr_new):
	'''
	Iteratively apply the subs method to each element of tensor.
	'''

	Av_sub=[]

	if len(Av.shape)==1:
		for ii in range(len(Av)):
			Av_sub.append(Av[ii].subs(expr_old,expr_new))

	elif len(Av.shape)==2:
		for ii in range(Av.shape[0]):
			row=[]
			for jj in range(Av.shape[1]):
				row.append(Av[ii,jj].subs(expr_old,expr_new))
			Av_sub.append(row)

	else:
		raise NameError('Method not developed for 3D arrays!')

	return smarr.MutableDenseNDimArray(Av_sub)


def scalar_deriv(a,xList):
	'''Compute derivatives of a scalar w.r.t. a list of valirables'''

	Nx=len(xList)
	Der=[]
	for ii in range(Nx):
		Der.append(a.diff(xList[ii]))

	return smarr.MutableDenseNDimArray(Der)



if __name__=='__main__':

	import numpy as np


	print('Verification matrix product:')
	# numerical matrix product
	Alist=[[1,4],[2,5],[7,4],[1,9]]
	Blist=[[5,8,4],[1,9,7]]
	Cref=np.dot(np.array(Alist),np.array(Blist))
	Cref=smarr.MutableDenseNDimArray(list(Cref))
	# symbolic matrix product
	Asm=smarr.Array(Alist)
	Bsm=smarr.Array(Blist)
	Csm=matrix_product(Asm,Bsm)
	Csm_simple=simplify(Csm)
	print(Csm_simple-Cref)


	print('Verification cross product:')
	for ii in range(3):
		for jj in range(3):
			Av=smarr.MutableDenseNDimArray([0,0,0])
			Bv=smarr.MutableDenseNDimArray([0,0,0])
			Av[ii]=1
			Bv[jj]=1
			Cv=cross_product(Av,Bv)
			print('%d x %d ='%(ii,jj,))
			print(Cv)


	print('Verify Euclidean norm of a vector')
	av=[3,1,-4]
	norm_ref=np.linalg.norm(av)
	Av=smarr.MutableDenseNDimArray(av)
	norm_sm=norm2(Av)
	print('Error %.10f' %(norm_ref-norm_sm.evalf(n=10)) )







	