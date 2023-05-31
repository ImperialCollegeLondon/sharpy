import h5py as h5

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.structure.models.beam as beam
import sharpy.utils.settings as settings_utils
import sharpy.utils.h5utils as h5utils
import os


@solver
class BeamLoader(BaseSolver):
    """
    ``BeamLoader`` class solver inherited from ``BaseSolver``

    Loads the structural beam solver with the specified user settings.

    Args:
        data (ProblemData): class containing the problem information

    Attributes:
        settings (dict): contains the specific settings for the solver

        settings_types (dict): Key  value pairs of the accepted types for the settings values
        settings_default (dict): Dictionary containing the default solver settings, should none be provided.
        data (ProblemData): class containing the data for the problem
        fem_file_name (str): name of the ``.fem.h5`` HDF5 file
        dyn_file_name (str): name of the ``.dyn.h5`` HDF5 file
        fem_data_dict (dict): key-value pairs of FEM data
        dyn_data_dict (dict): key-value pairs of data for dynamic problems
        structure (None): Empty attribute

    Notes:
        For further reference on Quaternions see:
        `https://en.wikipedia.org/wiki/Quaternion <https://en.wikipedia.org/wiki/Quaternion>`_

    See Also:
          .. py:class:: sharpy.utils.solver_interface.BaseSolver

          .. py:class:: sharpy.structure.models.beam.Beam

    """
    solver_id = 'BeamLoader'
    solver_classification = 'loader'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['unsteady'] = 'bool'
    settings_default['unsteady'] = True
    settings_description['unsteady'] = 'If ``True`` it will be a dynamic problem and the solver will look for the' \
                                       ' ``.dyn.h5`` file that contains the time varying input to the problem.'

    settings_types['orientation'] = 'list(float)'
    settings_default['orientation'] = [1., 0, 0, 0]
    settings_description['orientation'] = 'Initial attitude of the structure given as the quaternion that parametrises the rotation from G to A frames of reference.'

    settings_types['for_pos'] = 'list(float)'
    settings_default['for_pos'] = [0., 0, 0]
    settings_description['for_pos'] = 'Initial position of the A FoR.'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
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

    def initialise(self, data, restart=False):
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
        if os.path.isfile(self.mb_file_name):
            # h5utils.check_file_exists(self.mb_file_name)
            with h5.File(self.mb_file_name, 'r') as mb_file_handle:
                self.mb_data_dict = h5utils.load_h5_in_dict(mb_file_handle)

            # Need to redefine strings to remove the "b" at the beginning
            for iconstraint in range(self.mb_data_dict['num_constraints']):
                self.mb_data_dict["constraint_%02d" % iconstraint]['behaviour'] = self.mb_data_dict["constraint_%02d" % iconstraint]['behaviour'].decode()
            for ibody in range(self.mb_data_dict['num_bodies']):
                self.mb_data_dict["body_%02d" % ibody]['FoR_movement'] = self.mb_data_dict["body_%02d" % ibody]['FoR_movement'].decode()

    def validate_fem_file(self):
        raise NotImplementedError('validation of the fem file in beamloader is not yet implemented!')

    def validate_dyn_file(self):
        raise NotImplementedError('validation of the dyn file in beamloader is not yet implemented!')

    def run(self, **kwargs):
        self.data.structure = beam.Beam()
        self.data.structure.ini_mb_dict = self.mb_data_dict
        self.data.structure.generate(self.fem_data_dict, self.settings)
        self.data.structure.dyn_dict = self.dyn_data_dict

        # Change the beam description to the local FoR for multibody
        # if (self.data.structure.num_bodies > 1):
        #     self.data.structure.ini_info.whole_structure_to_local_AFoR(self.data.structure)
        #     self.data.structure.timestep_info[0].whole_structure_to_local_AFoR(self.data.structure)

        return self.data
