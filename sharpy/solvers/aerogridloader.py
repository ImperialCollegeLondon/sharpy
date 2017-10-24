import h5py as h5
import numpy as np

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.aero.models.aerogrid as aerogrid
import sharpy.utils.settings as settings_utils
import sharpy.utils.h5utils as h5utils


@solver
class AerogridLoader(BaseSolver):
    solver_id = 'AerogridLoader'

    def __init__(self):
        # settings list
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['unsteady'] = 'bool'
        self.settings_default['unsteady'] = False

        self.settings_types['aligned_grid'] = 'bool'
        self.settings_default['aligned_grid'] = True

        self.settings_types['freestream_dir'] = 'list(float)'
        self.settings_default['freestream_dir'] = np.array([1.0, 0, 0])

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
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # read input file (aero)
        self.read_files()

    def read_files(self):
        # open aero file
        # first, file names
        self.aero_file_name = self.data.case_route + '/' + self.data.case_name + '.aero.h5'
        # then check that the file exists
        h5utils.check_file_exists(self.aero_file_name)
        # read and store the hdf5 file
        with h5.File(self.aero_file_name, 'r') as aero_file_handle:
            # store files in dictionary
            self.aero_data_dict = h5utils.load_h5_in_dict(aero_file_handle)
            # TODO implement aero file validation
            # self.validate_aero_file()

    def validate_aero_file(self):
        raise NotImplementedError('validation of the aerofile in beamloader is not yet implemented!')

    def run(self):
        self.data.aero = aerogrid.Aerogrid()
        self.data.aero.generate(self.aero_data_dict, self.data.structure, self.settings)
        return self.data
