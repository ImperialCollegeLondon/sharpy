import h5py as h5
import numpy as np

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.aero.models.aerogrid as aerogrid
import sharpy.utils.settings as settings_utils
import sharpy.utils.h5utils as h5utils


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

    Notes:
        The ``control_surface_deflection`` setting allows the user to use a time specific control surface deflection,
        should the problem include them. This setting takes a list of strings, each for the required control
        surface generator.

        The ``control_surface_deflection_generator_settings`` setting is a list of dictionaries, one for each control
        surface. The dictionaries specify the settings for the generator ``DynamicControlSurface``. If the relevant control
        surface is simply static, an empty string should be parsed. See the documentation for ``DynamicControlSurface``
        generators for accepted key-value pairs as settings.

    """
    solver_id = 'AerogridLoader'
    solver_classification = 'aero'

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

    settings_types['control_surface_deflection_generator_settings'] = 'list(dict)'
    settings_default['control_surface_deflection_generator_settings'] = list(dict())
    settings_description['control_surface_deflection_generator_settings'] = 'List of dictionaries with the settings ' \
                                                                            'for each generator'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.aero_file_name = ''
        # storage of file contents
        self.aero_data_dict = dict()

        # aero storage
        self.aero = None

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]

        # init settings
        settings_utils.to_custom_types(self.settings,
                                       self.settings_types,
                                       self.settings_default)

        # read input file (aero)
        self.read_files()

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

    def run(self):
        self.data.aero = aerogrid.Aerogrid()
        self.data.aero.generate(self.aero_data_dict,
                                self.data.structure,
                                self.settings,
                                self.data.ts)
        return self.data
