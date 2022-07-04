Templates to build flying wing models
+++++++++++++++++++++++++++++++++++++

Templates to build flying wing models
S. Maraniello, Jul 2018

classes:
- FlyingWing: generate a flying wing model from a reduced set of input. The
built in method 'update_mass_stiff' can be re-defined by the user to enter more
complex inertial/stiffness properties
- Smith(FlyingWing): generate HALE wing model
- Goland(FlyingWing): generate Goland wing model


.. toctree::
	:glob:

	./FlyingWing
	./Goland
	./Pazy
	./QuasiInfinite
	./Smith
