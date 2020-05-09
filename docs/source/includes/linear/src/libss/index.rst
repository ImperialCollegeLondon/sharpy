Linear Time Invariant systems
+++++++++++++++++++++++++++++

Linear Time Invariant systems
author: S. Maraniello
date: 15 Sep 2017 (still basement...)

Library of methods to build/manipulate state-space models. The module supports
the sparse arrays types defined in libsparse.

The module includes:

Classes:
- ss: provides a class to build DLTI/LTI systems with full and/or sparse
	matrices and wraps many of the methods in these library. Methods include:
	- freqresp: wraps the freqresp function
	- addGain: adds gains in input/output. This is not a wrapper of addGain, as
	the system matrices are overwritten

Methods for state-space manipulation:
- couple: feedback coupling. Does not support sparsity
- freqresp: calculate frequency response. Supports sparsity.
- series: series connection between systems
- parallel: parallel connection between systems
- SSconv: convert state-space model with predictions and delays
- addGain: add gains to state-space model.
- join2: merge two state-space models into one.
- join: merge a list of state-space models into one.
- sum state-space models and/or gains
- scale_SS: scale state-space model
- simulate: simulates discrete time solution
- Hnorm_from_freq_resp: compute H norm of a frequency response
- adjust_phase: remove discontinuities from a frequency response

Special Models:
- SSderivative: produces DLTI of a numerical derivative scheme
- SSintegr: produces DLTI of an integration scheme
- build_SS_poly: build state-space model with polynomial terms.

Filtering:
- butter

Utilities:
- get_freq_from_eigs: clculate frequency corresponding to eigenvalues

Comments:
- the module supports sparse matrices hence relies on libsparse.

to do:
	- remove unnecessary coupling routines
	- couple function can handle sparse matrices but only outputs dense matrices
		- verify if typical coupled systems are sparse
		- update routine
		- add method to automatically determine whether to use sparse or dense?


.. toctree::
	:glob:

	./Hnorm_from_freq_resp
	./SSconv
	./SSderivative
	./SSintegr
	./addGain
	./adjust_phase
	./build_SS_poly
	./butter
	./compare_ss
	./couple
	./disc2cont
	./eigvals
	./freqresp
	./get_freq_from_eigs
	./join
	./join2
	./parallel
	./project
	./random_ss
	./scale_SS
	./series
	./simulate
	./ss
	./ss_block
	./ss_to_scipy
	./sum_ss
