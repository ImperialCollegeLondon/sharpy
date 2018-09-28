import h5py as h5
import numpy as np

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.structure.models.beam as beam
import sharpy.utils.settings as settings_utils
import sharpy.utils.h5utils as h5utils
from IPython import embed


@solver
class BeamLoader(BaseSolver):
    solver_id = 'BeamLoader'

    def __init__(self):
        # settings list
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['unsteady'] = 'bool'
        self.settings_default['unsteady'] = False

        self.settings_types['orientation'] = 'list(float)'
        self.settings_default['orientation'] = np.array([1., 0, 0, 0])

        self.data = None
        self.settings = None
        self.fem_file_name = ''
        self.dyn_file_name = ''
        # storage of file contents
        self.fem_data_dict = dict()
        self.dyn_data_dict = dict()
        self.mb_data_dict = dict()

        # structure storage
        self.structure = None

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]

        # init settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # read input files (fem and dyn)
        self.read_files()

    def read_files(self):
        # open fem file
        # first, file names
        self.fem_file_name = self.data.case_route + '/' + self.data.case_name + '.fem.h5'
        if self.settings['unsteady']:
            self.dyn_file_name = self.data.case_route + '/' + self.data.case_name + '.dyn.h5'
        # then check that the files exists
        h5utils.check_file_exists(self.fem_file_name)
        if self.settings['unsteady']:
            try:
                h5utils.check_file_exists(self.dyn_file_name)
            except FileNotFoundError:
                self.settings['unsteady'] = False
        # read and store the hdf5 files
        with h5.File(self.fem_file_name, 'r') as fem_file_handle:
            # store files in dictionary
            self.fem_data_dict = h5utils.load_h5_in_dict(fem_file_handle)
            # TODO implement fem file validation
            # self.validate_fem_file()
        if self.settings['unsteady']:
            with h5.File(self.dyn_file_name, 'r') as dyn_file_handle:
                # store files in dictionary
                self.dyn_data_dict = h5utils.load_h5_in_dict(dyn_file_handle)
                # TODO implement dyn file validation
                # self.validate_dyn_file()

        # Multibody information
        self.mb_file_name = self.data.case_route + '/' + self.data.case_name + '.mb.h5'
        h5utils.check_file_exists(self.mb_file_name)
        with h5.File(self.mb_file_name, 'r') as mb_file_handle:
            self.mb_data_dict = h5utils.load_h5_in_dict(mb_file_handle)

        # Need to redefine strings to remove the "b" at the beginning
        for iconstraint in range(self.mb_data_dict['num_constraints']):
            self.mb_data_dict["constraint_%02d" % iconstraint]['behaviour'] = self.mb_data_dict["constraint_%02d" % iconstraint]['behaviour'].decode()
        #for ibody in range(self.mb_data_dict['num_bodies']):
        #    self.mb_data_dict["body_%02d" % ibody]['FoR_movement'] = self.mb_data_dict["body_%02d" % ibody]['FoR_movement'].decode()


    def validate_fem_file(self):
        raise NotImplementedError('validation of the fem file in beamloader is not yet implemented!')

    def validate_dyn_file(self):
        raise NotImplementedError('validation of the dyn file in beamloader is not yet implemented!')

    def run(self):
        self.data.structure = beam.Beam()
        self.data.structure.mb_dict = self.mb_data_dict
        self.data.structure.generate(self.fem_data_dict, self.settings)
        self.data.structure.dyn_dict = self.dyn_data_dict
        return self.data
