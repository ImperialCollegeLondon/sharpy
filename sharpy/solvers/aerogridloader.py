import h5py as h5
import numpy as np

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.aero.models.aerogrid as aerogrid
import sharpy.utils.settings as settings_utils
import sharpy.utils.h5utils as h5utils
import sharpy.utils.generator_interface as gen_interface
from sharpy.solvers.gridloader import GridLoader


@solver
class AerogridLoader(GridLoader):
    """
    ``AerogridLoader`` class, inherited from ``GridLoader``

    Generates aerodynamic grid based on the input data

    The initial wake shape is now defined in SHARPy (instead of UVLM) through a wake shape generator
    ``wake_shape_generator`` and the required inputs ``wake_shape_generator_input``. The supported wake generators are
    :class:`sharpy.generators.straightwake.StraighWake` and :class:`sharpy.generators.helicoidalwake.HelicoidalWake`.

    The ``control_surface_deflection`` setting allows the user to use a time specific control surface deflection,
    should the problem include them. This setting takes a list of strings, each for the required control
    surface generator.

    The ``control_surface_deflection_generator_settings`` setting is a list of dictionaries, one for each control
    surface. The dictionaries specify the settings for the generator ``DynamicControlSurface``. If the relevant control
    surface is simply static, an empty string should be parsed. See the documentation for ``DynamicControlSurface``
    generators for accepted key-value pairs as settings.

    The ``initial_align`` setting aligns the wing panel discretization with the freestream for the undeformed structure, 
    and applies this Z rotation at every timestep (panels become misaligned when the wing deforms). The ``aligned_grid`` 
    setting aligns the wing panel discretization with the flow at every time step and takes precedence.

    Args:
        data (PreSharpy): ``ProblemData`` class structure

    Attributes:
        settings (dict): Name-value pair of the settings employed by the aerodynamic solver
        settings_types (dict): Acceptable types for the values in ``settings``
        settings_default (dict): Name-value pair of default values for the aerodynamic settings
        data (ProblemData): class structure
        file_name (str): name of the ``.aero.h5`` HDF5 file
        aero: empty attribute
        data_dict (dict): key-value pairs of aerodynamic data
        wake_shape_generator (class): Wake shape generator

    """
    solver_id = 'AerogridLoader'
    solver_classification = 'loader'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['unsteady'] = 'bool'
    settings_default['unsteady'] = False
    settings_description['unsteady'] = 'Unsteady effects'

    settings_types['aligned_grid'] = 'bool'
    settings_default['aligned_grid'] = True
    settings_description['aligned_grid'] = 'Align grid'

    settings_types['initial_align'] = 'bool'
    settings_default['initial_align'] = True
    settings_description['initial_align'] = "Initially align grid"

    settings_types['freestream_dir'] = 'list(float)'
    settings_default['freestream_dir'] = [1.0, 0.0, 0.0]
    settings_description['freestream_dir'] = 'Free stream flow direction'

    settings_types['mstar'] = ['int', 'list(int)']
    settings_default['mstar'] = 10
    settings_description['mstar'] = 'Number of chordwise wake panels'

    settings_types['control_surface_deflection'] = 'list(str)'
    settings_default['control_surface_deflection'] = []
    settings_description['control_surface_deflection'] = 'List of control surface generators for each control surface'

    settings_types['control_surface_deflection_generator_settings'] = 'dict'
    settings_default['control_surface_deflection_generator_settings'] = dict()
    settings_description['control_surface_deflection_generator_settings'] = 'List of dictionaries with the settings ' \
                                                                            'for each generator'

    settings_types['wake_shape_generator'] = 'str'
    settings_default['wake_shape_generator'] = 'StraightWake'
    settings_description['wake_shape_generator'] = 'ID of the generator to define the initial wake shape'
    settings_options['wake_shape_generator'] = ['StraightWake', 'HelicoidalWake']

    settings_types['wake_shape_generator_input'] = 'dict'
    settings_default['wake_shape_generator_input'] = dict()
    settings_description['wake_shape_generator_input'] = 'Dictionary of inputs needed by the wake shape generator'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description,
                                       settings_options=settings_options)

    def __init__(self):
        super().__init__
        self.file_name = '.aero.h5'
        self.aero = None
        self.wake_shape_generator = None

    def initialise(self, data, restart=False):
        super().initialise(data)

        wake_shape_generator_type = gen_interface.generator_from_string(
            self.settings['wake_shape_generator'])
        self.wake_shape_generator = wake_shape_generator_type()
        self.wake_shape_generator.initialise(data,
                                             self.settings['wake_shape_generator_input'],
                                             restart=restart)

    def run(self, **kwargs):
        self.data.aero = aerogrid.Aerogrid()
        self.data.aero.generate(self.data_dict,
                                self.data.structure,
                                self.settings,
                                self.data.ts)
        aero_tstep = self.data.aero.timestep_info[self.data.ts]
        self.wake_shape_generator.generate({'zeta': aero_tstep.zeta,
                                            'zeta_star': aero_tstep.zeta_star,
                                            'gamma': aero_tstep.gamma,
                                            'gamma_star': aero_tstep.gamma_star,
                                            'dist_to_orig': aero_tstep.dist_to_orig})
        # keep the call to the wake generator
        # because it might be needed by other solvers
        self.data.aero.wake_shape_generator = self.wake_shape_generator
        return self.data
