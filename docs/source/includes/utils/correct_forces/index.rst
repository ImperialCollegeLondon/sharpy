Force correction utilities
++++++++++++++++++++++++++

Force correction utilities

The aerodynamic forces can be corrected with these functions.
The correction is done once they are projected on the structural beam.

Args:
    data (:class:`sharpy.PreSharpy`): SHARPy data
    aero_kstep (:class:`sharpy.utils.datastructures.AeroTimeStepInfo`): Current aerodynamic substep
    structural_kstep (:class:`sharpy.utils.datastructures.StructTimeStepInfo`): Current structural substep
    struct_forces (np.array): Array with the aerodynamic forces mapped on the structure in the B frame of reference

Returns:
    new_struct_forces (np.array): Array with the corrected forces


.. toctree::
	:glob:

	./efficiency
	./polars
