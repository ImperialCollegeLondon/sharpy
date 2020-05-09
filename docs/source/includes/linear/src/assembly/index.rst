Assembly of linearised UVLM system
++++++++++++++++++++++++++++++++++


S. Maraniello, 25 May 2018

Includes:
    - Boundary conditions methods:
        - AICs: allocate aero influence coefficient matrices of multi-surfaces
          configurations
        - ``nc_dqcdzeta_Sin_to_Sout``: derivative matrix of ``nc*dQ/dzeta``
          where Q is the induced velocity at the bound collocation points of one
          surface to another.
        - ``nc_dqcdzeta_coll``: assembles ``nc_dqcdzeta_coll_Sin_to_Sout`` matrices in
          multi-surfaces configurations
        - ``uc_dncdzeta``: assemble derivative matrix dnc/dzeta*Uc at bound collocation
          points


.. toctree::
	:glob:

	./AICs
	./dfqsdgamma_vrel0
	./dfqsduinput
	./dfqsdvind_gamma
	./dfqsdvind_zeta
	./dfqsdzeta_omega
	./dfqsdzeta_vrel0
	./dfunstdgamma_dot
	./dvinddzeta
	./dvinddzeta_cpp
	./eval_panel_cpp
	./nc_domegazetadzeta
	./nc_dqcdzeta
	./nc_dqcdzeta_Sin_to_Sout
	./test_wake_prop_term
	./uc_dncdzeta
	./wake_prop
	./wake_prop_from_dimensions
