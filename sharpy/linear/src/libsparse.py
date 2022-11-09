'''
Collect tools to manipulate sparse and/or mixed dense/sparse matrices.

author: S. Maraniello
date: Dec 2018

Comment: manipulating large linear system may require using both dense and sparse
matrices. While numpy/scipy automatically handle most operations between mixed
dense/sparse arrays, some (e.g. dot product) require more attention. This
library collects methods to handle these situations.

Classes:
scipy.sparse matrices are wrapped so as to ensure compatibility with numpy arrays
upon conversion to dense.
- csc_matrix: this is a wrapper of scipy.csc_matrix.
- SupportedTypes: types supported for operations
- WarningTypes: due to some bugs in scipy (v.1.1.0), sum (+) operations between
np.ndarray and scipy.sparse matrices can result in numpy.matrixlib.defmatrix.matrix
types. This list contains such undesired types that can result from dense/sparse
operations and raises a warning if required.
(b) convert these types into numpy.ndarrays.

Methods:
- dot: handles matrix dot products across different types.
- solve: solves linear systems Ax=b with A and b dense, sparse or mixed.
- dense: convert matrix to numpy array

Warning:
- only sparse types into SupportedTypes are supported!

To Do:
- move these methods into an algebra module?
'''

import warnings
import numpy as np
import scipy.sparse as sparse
from scipy.sparse._sputils import upcast_char
import scipy.sparse.linalg as spalg

# --------------------------------------------------------------------- Classes

class csc_matrix(sparse.csc_matrix):
	'''
	Wrapper of scipy.csc_matrix that ensures best compatibility with numpy.ndarray.
	The following methods have been overwritten to ensure that numpy.ndarray are
	returned instead of numpy.matrixlib.defmatrix.matrix.
		- todense
		- _add_dense

	Warning: this format is memory inefficient to allocate new sparse matrices.
	Consider using:
	- scipy.sparse.lil_matrix, which supports slicing, or
	- scipy.sparse.coo_matrix, though slicing is not supported :(

	'''

	def __init__(self,arg1, shape=None, dtype=None, copy=False):
		super().__init__(arg1, shape=shape, dtype=dtype, copy=copy)

	def todense(self):
		''' As per scipy.spmatrix.todense but returns a numpy.ndarray. '''
		return super().toarray()

	def _add_dense(self, other):
		if other.shape != self.shape:
		    raise ValueError('Incompatible shapes.')
		dtype = upcast_char(self.dtype.char, other.dtype.char)
		order = self._swap('CF')[0]
		result = np.array(other, dtype=dtype, order=order, copy=True)
		M, N = self._swap(self.shape)
		y = result if result.flags.c_contiguous else result.T
		sparse._sparsetools.csr_todense(M, N, self.indptr, self.indices, self.data, y)
		return result #np.matrix(result, copy=False)


SupportedTypes=[np.ndarray,csc_matrix]
WarningTypes=[np.matrixlib.defmatrix.matrix]


# --------------------------------------------------------------------- Methods


def block_dot(A, B):
	'''
	dot product between block matrices.

	Inputs:
	A, B: are nested lists of dense/sparse matrices of compatible shape for
	block matrices product. Empty blocks can be defined with None. (see numpy.block)
	'''

	rA, cA = len(A), len(A[0])
	rB, cB = len(B), len(B[0])

	for arow,brow in zip(A,B):
		assert len(brow) == cB,\
						'B rows do not contain the same number of column blocks'
		assert len(arow) == cA,\
						'A rows do not contain the same number of column blocks'
	assert cA==rB, 'Columns of A not equal to rows of B!'

	P=[]
	for ii in range(rA):
		prow = cB * [None]
		for jj in range(cB):
			# check first that the result will not be None
			Continue = False
			for kk in range(cA):
				if A[ii][kk] is not None and B[kk][jj] is not None:
					Continue = True
					break
			if Continue:
				prow[jj] = 0.
				for kk in range(cA):
					if A[ii][kk] is not None and B[kk][jj] is not None:
						prow[jj] += dot( A[ii][kk], B[kk][jj] )
		P.append(prow)

	return P


def block_matrix_dot_vector(A, v):
        '''
        dot product between block matrix and block vector

        Inputs:
        A, v: are nested lists of dense/sparse matrices of compatible shape for
        block matrices product. Empty blocks can be defined with None. (see numpy.block)
        '''

        rA, cA = len(A), len(A[0])
        rv = len(B)

        for arow in A:
            assert len(arow) == cA,\
                'A rows do not contain the same number of column blocks'
        assert cA==rv, 'Columns of A not equal to rows of v!'

        P=[None]*rA
        for ii in range(rA):
            for jj in range(cA):
                # check first that the result will not be None
                if A[ii][jj] is not None and B[jj] is not None:
                    P[ii] += dot(A[ii][jj], B[jj])
        return P

def block_sum(A, B, factA = None, factB = None):
	'''
	dot product between block matrices.

	Inputs:
	A, B: are nested lists of dense/sparse matrices of compatible shape for
	block matrices product. Empty blocks can be defined with None. (see numpy.block)
	'''

	rA, cA = len(A), len(A[0])
	rB, cB = len(B), len(B[0])

	assert cA==cB and rA==rB, 'Block matrices do not have same size'

	for arow,brow in zip(A,B):
		assert len(brow) == cB,\
						'B rows do not contain the same number of column blocks'
		assert len(arow) == cA,\
						'A rows do not contain the same number of column blocks'

	P=[]
	for ii in range(rA):
		prow = cA * [None]

		for jj in range(cA):

			if A[ii][jj] is None:
				if B[ii][jj] is None:
					prow[jj] = None
				else:
					if factB is None:
						prow[jj] = B[ii][jj]
					else:
						prow[jj] = factB*B[ii][jj]
			else:
				if B[ii][jj] is None:
					if factA is None:
						prow[jj] = A[ii][jj]
					else:
						prow[jj] = factA*A[ii][jj]
				else:
					if factA is None and factA is None:
						prow[jj] = A[ii][jj] + B[ii][jj]
					elif factA is None:
						prow[jj] = A[ii][jj] + factB*B[ii][jj]
					elif factB is None:
						prow[jj] = factA*A[ii][jj] + B[ii][jj]
					else:
						prow[jj] = factA*A[ii][jj] + factB*B[ii][jj]

		P.append(prow)

	return P


def dot(A,B,type_out=None):
	'''
	Method to compute
		C = A*B ,
	where * is the matrix product, with dense/sparse/mixed matrices.

	The format (sparse or dense) of C is specified through 'type_out'. If
	type_out==None, the output format is sparse if both A and B are sparse, dense
	otherwise.

	The following formats are supported:
	- numpy.ndarray
	- scipy.csc_matrix
	'''

	# determine types:
	tA=type(A)
	tB=type(B)

	assert tA in SupportedTypes, 'Type of A matrix (%s) not supported'%tA
	assert tB in SupportedTypes, 'Type of B matrix (%s) not supported'%tB
	if type_out == None:
		type_out=tA
	else:
		assert type_out in SupportedTypes, 'type_out not supported'

	# multiply
	# if tA==float or tb==float:
	# 	C = A*B
	# else:
	if tA==np.ndarray and tB==csc_matrix:
		C=(B.transpose()).dot(A.transpose()).transpose()
		# C=A.dot(B.todense())
	else:
		C=A.dot(B)

	# format output
	if tA != type_out:
		if type_out==csc_matrix:
			return csc_matrix(C)
		else:
			return C.toarray()

	return C


def solve(A,b):
	'''
	Wrapper of
		numpy.linalg.solve and scipy.sparse.linalg.spsolve
	for solution of the linear system A x = b.
	- if A is a dense numpy array np.linalg.solve is called for solution. Note
	that if B is sparse, this requires convertion to dense. In this case,
	solution through LU factorisation of A should be considered to exploit the
	sparsity of B.
	- if A is sparse, scipy.sparse.linalg.spsolve is used.
	'''

	# determine types:
	tA=type(A)
	tB=type(b)

	assert tA in SupportedTypes, 'Type of A matrix (%s) not supported'%tA
	assert tB in SupportedTypes, 'Type of B matrix (%s) not supported'%tB
	# multiply
	if tA==np.ndarray:
		if tB==csc_matrix:
			x=np.linalg.solve(A,b.toarray())
		else:
			x=np.linalg.solve(A,b)
	else:
		x=spalg.spsolve(A,b)

	assert type(x) in SupportedTypes, 'Unexpected output type!'

	return x


def dense(M):
	''' If required, converts sparse array to dense. '''
	if type(M) == csc_matrix:
		return np.array(M.toarray())
	elif type(M) == csc_matrix:
		return M.toarray()
	return M


def eye_as(M):
	''' Produces an identity matrix as per M, in shape and type '''

	tM=type(M)
	assert tM in SupportedTypes, 'Type %s not supported!'%tM
	nrows=M.shape[0]
	assert nrows==M.shape[1], 'Not a square matrix!'

	if tM==csc_matrix:
		D=csc_matrix((nrows,nrows))
		D.setdiag(1.)
	elif tM==np.ndarray:
		D=np.eye(nrows)

	return D


def zeros_as(M):
	''' Produces an identity matrix as per M, in shape and type '''

	tM=type(M)
	assert tM in SupportedTypes, 'Type %s not supported!'%tM
	nrows,ncols=M.shape

	if tM==csc_matrix:
		D=csc_matrix((nrows,ncols))
	elif tM==np.ndarray:
		D=np.zeros_like(M)

	return D


# -----------------------------------------------------------------------------


if __name__=='__main__':
	import unittest

	class Test_module(unittest.TestCase):
		''' Test methods into this module '''

		def setUp(self):
			self.A=np.random.rand(3,4)
			self.B=np.random.rand(4,2)

		def test_dense_plus_csc_matrix_type(self):
			A=self.A
			Asp=csc_matrix(A)

			for aa in [A,Asp]:
				for bb in [A,Asp]:
					tsum=type(aa+bb)
					tdiff=type(aa-bb)
					for tout,strout in zip([tsum,tdiff],['(+)','(-)']):
						if tout not in SupportedTypes:
							if tout in WarningTypes:
								warnings.warn(
									'Undesired type (%s) resulting from %s operations between %s and %s types'\
									%(tout,strout,type(aa),type(bb)))
							else:
								raise NameError(
									'Unexpected type (%s) resulting from %s operations between %s and %s types'\
									%(tout,strout,type(aa),type(bb)))

		def test_zeros_as(self):
			A=np.zeros((4,2))
			A1=zeros_as(A)
			A2=zeros_as(csc_matrix(A))
			assert np.max(np.abs(A-A1))<1e-16, 'Error in libsparse.zeros_as'
			assert np.max(np.abs(A-A2))<1e-16, 'Error in libsparse.zeros_as'

		def test_eye_as(self):
			A=np.random.rand(4,4)
			D0=np.eye(4)
			D1=eye_as(A)
			D2=eye_as(csc_matrix(A))
			assert np.max(np.abs(D0-D1))<1e-12, 'Error in libsparse.eye_as'
			assert np.max(np.abs(D0-D2))<1e-12, 'Error in libsparse.eye_as'

		def test_dot(self):
			A,B=self.A,self.B
			C0=np.dot(A,B)		# reference
			C1=dot(A,B)
			C2=dot(A,csc_matrix(B))
			C3=dot(csc_matrix(A),B)
			C4=dot(csc_matrix(A),csc_matrix(B))
			assert np.max(np.abs(C0-C1))<1e-12, 'Error in libsparse.dot'
			assert np.max(np.abs(C0-C2))<1e-12, 'Error in libsparse.dot'
			assert np.max(np.abs(C0-C3))<1e-12, 'Error in libsparse.dot'

		def test_solve(self):
			A=np.random.rand(4,4)
			B=np.random.rand(4,2)
			Asp=csc_matrix(A)
			Bsp=csc_matrix(B)

			X0=np.linalg.solve(A,B)
			X1=solve(A,B)
			X2=solve(A,Bsp)
			X3=solve(Asp,B)
			X4=solve(Asp,Bsp)

			assert np.max(np.abs(X0-X1))<1e-12, 'Error in libsparse.solve'
			assert np.max(np.abs(X0-X2))<1e-12, 'Error in libsparse.solve'
			assert np.max(np.abs(X0-X3))<1e-12, 'Error in libsparse.solve'
			assert np.max(np.abs(X0-X4))<1e-12, 'Error in libsparse.solve'


	outprint='Testing libsparse'
	print('\n' + 70*'-')
	print((70-len(outprint))*' ' + outprint )
	print(70*'-')
	unittest.main()
