import h5py as h5
import numpy as np

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.aero.models.aerogrid as aerogrid
import sharpy.utils.settings as su
import sharpy.utils.h5utils as h5utils
import sharpy.utils.generator_interface as gen_interface


@solver
class AerogridLoader(BaseSolver):
    """
    ``AerogridLoader`` class, inherited from ``BaseSolver``

    Generates aerodynamic grid based on the input data

    Args:
        data (PreSharpy): ``ProblemData`` class structure

    Attributes:
        settings (dict): Name-value pair of the settings employed by the aerodynamic solver
        settings_types (dict): Acceptable types for the values in ``settings``
        settings_default (dict): Name-value pair of default values for the aerodynamic settings
        data (ProblemData): class structure
        aero_file_name (str): name of the ``.aero.h5`` HDF5 file
        aero: empty attribute
        aero_data_dict (dict): key-value pairs of aerodynamic data
        wake_shape_generator (class): Wake shape generator

    Notes:
        The ``control_surface_deflection`` setting allows the user to use a time specific control surface deflection,
        should the problem include them. This setting takes a list of strings, each for the required control
        surface generator.

        The ``control_surface_deflection_generator_settings`` setting is a list of dictionaries, one for each control
        surface. The dictionaries specify the settings for the generator ``DynamicControlSurface``. If the relevant control
        surface is simply static, an empty string should be parsed. See the documentation for ``DynamicControlSurface``
        generators for accepted key-value pairs as settings.

        The initial wake shape is now defined in SHARPy (instead of UVLM) through a wake shape generator ``wake_shape_generator`` and the
        required inputs ``wake_shape_generator_input``

    """
    solver_id = 'AerogridLoader'
    solver_classification = 'loader'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['unsteady'] = 'bool'
    settings_default['unsteady'] = False
    settings_description['unsteady'] = 'Unsteady effects'

    settings_types['aligned_grid'] = 'bool'
    settings_default['aligned_grid'] = True
    settings_description['aligned_grid'] = 'Align grid'

    settings_types['freestream_dir'] = 'list(float)'
    settings_default['freestream_dir'] = [1.0, 0.0, 0.0]
    settings_description['freestream_dir'] = 'Free stream flow direction'

    settings_types['mstar'] = 'int'
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

    settings_types['wake_shape_generator_input'] = 'dict'
    settings_default['wake_shape_generator_input'] = dict()
    settings_description['wake_shape_generator_input'] = 'Dictionary of inputs needed by the wake shape generator'

    settings_table = su.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.aero_file_name = ''
        # storage of file contents
        self.aero_data_dict = dict()

        # aero storage
        self.aero = None

        self.wake_shape_generator = None

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]

        # init settings
        su.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default)

        # read input file (aero)
        self.read_files()

        wake_shape_generator_type = gen_interface.generator_from_string(
            self.settings['wake_shape_generator'])
        self.wake_shape_generator = wake_shape_generator_type()
        self.wake_shape_generator.initialise(data,
                                             self.settings['wake_shape_generator_input'])

    def read_files(self):
        # open aero file
        # first, file names
        self.aero_file_name = (self.data.case_route +
                               '/' +
                               self.data.case_name +
                               '.aero.h5')

        #  then check that the file exists
        h5utils.check_file_exists(self.aero_file_name)

        #  read and store the hdf5 file
        with h5.File(self.aero_file_name, 'r') as aero_file_handle:
            # store files in dictionary
            self.aero_data_dict = h5utils.load_h5_in_dict(aero_file_handle)

    def run(self, **kwargs):
        self.data.aero = aerogrid.Aerogrid()
        self.data.aero.generate(self.aero_data_dict,
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
