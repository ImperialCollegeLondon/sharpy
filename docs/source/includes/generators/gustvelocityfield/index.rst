Gust Velocity Field Generators
++++++++++++++++++++++++++++++


These generators are used to create a gust velocity field. :class:`.GustVelocityField` is the main class that should be
parsed as the ``velocity_field_input`` to the desired aerodynamic solver.

The remaining classes are the specific gust profiles and parsed as ``gust_shape``.

Examples:
    The typical input to the aerodynamic solver settings would therefore read similar to:

    >>> aero_settings = {'<some_aero_settings>': '<some_aero_settings>',
    >>>                  'velocity_field_generator': 'GustVelocityField',
    >>>                  'velocity_field_input': {'u_inf': 1,
    >>>                                           'gust_shape': '<desired_gust>',
    >>>                                           'gust_parameters': '<gust_settings>'}}



.. toctree::
	:glob:

	./DARPA
	./GustVelocityField
	./continuous_sin
	./lateral_one_minus_cos
	./one_minus_cos
	./span_sine
	./time_varying
	./time_varying_global
