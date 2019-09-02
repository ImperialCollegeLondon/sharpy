Collect tools to manipulate sparse and/or mixed dense/sparse matrices.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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


.. toctree::
	:glob:

	./block_dot
	./block_sum
	./csc_matrix
	./dense
	./dot
	./eye_as
	./solve
	./zeros_as
