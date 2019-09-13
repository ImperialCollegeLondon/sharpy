Methods for interpolation of DLTI ROMs
++++++++++++++++++++++++++++++++++++++


author: S. Maraniello

date: Mar-Apr 2019


Overview:
    This is library for  state-space models interpolation. These routines are intended
    for small size state-space models (ROMs), hence some methods may not be optimised
    to exploit sparsity structures. For generality purposes, all methods require in
    input interpolatory weights


The module includes the methods:
    - transfer_function: returns an interpolatory state-space model based on the
    transfer function method [1]. This method is general and is, effectively, a
    wrapper of the libss.join method.
    - BT_transfer_function: evolution of transfer function methods. The growth of
    the interpolated system size is avoided through balancing.


References:
    [1] Benner, P., Gugercin, S. & Willcox, K., 2015. A Survey of Projection-Based
    Model Reduction Methods for Parametric Dynamical Systems. SIAM Review, 57(4),
    pp.483–531.



.. toctree::
	:glob:

	./FLB_transfer_function
	./InterpROM
	./transfer_function
