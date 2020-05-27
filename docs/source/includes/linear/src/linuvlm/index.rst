Linear UVLM solver classes
++++++++++++++++++++++++++

Linear UVLM solver classes

Contains classes to assemble a linear UVLM system. The three main classes are:

* :class:`~sharpy.linear.src.linuvlm.Static`: : for static VLM solutions.

* :class:`~sharpy.linear.src.linuvlm.Dynamic`: for dynamic UVLM solutions.

* :class:`~sharpy.linear.src.linuvlm.DynamicBlock`: a more efficient representation of ``Dynamic`` using lists for the
  different blocks in the UVLM equations

References:

    Maraniello, S., & Palacios, R.. State-Space Realizations and Internal Balancing in Potential-Flow
    Aerodynamics with Arbitrary Kinematics. AIAA Journal, 57(6), 1–14. 2019. https://doi.org/10.2514/1.J058153



.. toctree::
	:glob:

	./Dynamic
	./DynamicBlock
	./Frequency
	./Static
