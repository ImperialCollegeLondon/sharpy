import numpy as np
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings


@generator_interface.generator
class ModifyStructure(generator_interface.BaseGenerator):
    """
    ``ModifyStructure`` generator.

    This generator allows the user to modify structural parameters at runtime. At the moment, changes to lumped
    masses are supported. For each lumped mass you want to change, set ``change_variable`` to ``lumped_mass``, and the
    ``variable_index`` and ``file_list`` as specified in :class:`~sharpy.generators.modifystructure.ChangeLumpedMass`.

    This generator is called at the start of each time step in ``DynamicCoupled``.
    """
    generator_id = 'ModifyStructure'
    generator_classification = 'runtime'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['change_variable'] = 'list(str)'
    settings_default['change_variable'] = None
    settings_description['change_variable'] = 'Structural variable to modify'
    settings_options['change_variable'] = ['lumped_mass']

    settings_types['variable_index'] = 'list(int)'
    settings_default['variable_index'] = None
    settings_description['variable_index'] = 'List of indices of variables to change. ' \
                                             'For instance the 1st lumped mass would be ``[0]``'

    settings_types['file_list'] = 'list(str)'
    settings_default['file_list'] = None
    settings_description['file_list'] = 'File path for each variable containing the changing info, in the ' \
                                        'appropriate format. See each of the allowed variables for the correct format.'

    def __init__(self):
        self.settings = None

        self.num_changes = None  # :int number of variables that are changed
        self.variables = []  # :list of changed variables objects
        self.control_objects = {}  #: dictionary of changed variable name and its control object as value

    def initialise(self, in_dict, **kwargs):
        structure = kwargs['data'].structure
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True,
                                 options=self.settings_options)

        self.num_changes = len(self.settings['change_variable'])

        if 'lumped_mass' in self.settings['change_variable']:
            self.control_objects['lumped_mass'] = LumpedMassControl()

        lumped_mass_variables = []
        for i in range(self.num_changes):
            var_type = self.settings['change_variable'][i]
            if var_type == 'lumped_mass':
                variable = ChangeLumpedMass(var_index=self.settings['variable_index'][i],
                                            file=self.settings['file_list'][i])
                variable.initialise(structure)

                self.variables.append(variable)
                lumped_mass_variables.append(i)
                self.control_objects['lumped_mass'].append(i)
            else:
                raise NotImplementedError('Variable {:s} not yet coded to be modified in runtime'.format(var_type))
        try:
            self.control_objects['lumped_mass'].set_unchanged_vars_to_zero(structure)
        except KeyError:
            pass

    def generate(self, params):
        data = params['data']
        ts = data.ts
        structure = data.structure

        for variable in self.variables:
            variable(structure, ts)

        # should only be called once per time step
        try:
            self.control_objects['lumped_mass'].execute_change(structure)
        except KeyError:
            pass

        # for future variables supported, have the control objects have the same signatures such that they may be
        # called in a loop


class ChangedVariable:
    """
    Base class of a changed variable

    Attributes:
        name (str): Name of the changed variable
        variable_index (int): Index of the variable to change
        file (str): Name of the file containing the input data (.txt)
        original (np.ndarray): Original value of the desired value in the appropriate format
        current_value (np.ndarray): Running track of the current value of the desired variable
    """
    def __init__(self, name, var_index, file):
        self.name = name
        self.variable_index = var_index
        self.file = file

        self.original = None
        self.target_value = None
        self.current_value = None  # initially

    def initialise(self, structure):

        self.get_original(structure)
        self.load_file()

        self.current_value = self.original  # initially

    def __call__(self, structure, ts):
        pass

    def get_original(self, structure):
        # should be overridden for the desired variable class. it should set self.original in the appropriate format
        pass

    def load_file(self):
        self.target_value = np.loadtxt(self.file)


class ChangeLumpedMass(ChangedVariable):
    """
    Lumped Mass to be modified

    The arguments are parsed as items of the list in the settings for ``variable_index`` and ``file_list``. For
    those variables marked where ``change_variables = 'lumped_mass'``.

    The file should contain a time varying series with the following 10 columns:

    * Lumped mass

    * Lumped mass position in the material frame ``B`` (3 columns for ``xb``, ``yb`` and ``zb``)

    * Lumped mass inertia in the material frame ``B`` (6 columns for ``ixx``, ``iyy``, ``izz``, ``ixy``, ``ixz`` and
      ``iyz``.

    Not all 10 columns are necessary in the input file, missing columns are ignored and left unchanged. There should be
    one row per time step. If there are not enough entries for the number of time steps in the simulation, the changed
    variable value remains unchanged after all rows have been processed.

    Args:
        var_index (int): Index of lumped mass. NOT the lumped mass node.
        file (str): Path to file containing time history of the lumped mass.
    """
    def __init__(self, var_index, file):
        super().__init__('lumped_mass', var_index=var_index, file=file)

    def __call__(self, structure, ts):
        try:
            # lumped masses get added (+=) at structure.lump_masses(), therefore the increment with respect to the
            # previous time step must be provided. This is such that this generator is backwards compatible with the
            # way lumped masses are assembled.
            delta = self.target_value[ts] - self.current_value
        except IndexError:  # input file has less entries than the simulation time steps
            structure.lumped_mass[self.variable_index] = 0
            structure.lumped_mass_position[self.variable_index, :] = np.zeros(3)
            structure.lumped_mass_inertia[self.variable_index, :, :] = np.zeros((3, 3))

        else:
            structure.lumped_mass[self.variable_index] = delta[0]
            structure.lumped_mass_position[self.variable_index, :] = delta[1:4]
            ixx, iyy, izz, ixy, ixz, iyz = delta[-6:]
            inertia = np.block([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            structure.lumped_mass_inertia[self.variable_index, :, :] = inertia

            self.current_value += delta

    def load_file(self):
        """Sets ``self.target_value`` by reading from the file.

        If the input does not have as many columns as needed (10), these get padded with the original value such that
        they are not changed at runtime.

        """

        super().load_file()

        n_values = len(self.original)

        try:
            n_target_values = self.target_value.shape[1]  # number of columns
        except IndexError:
            n_target_values = 1

        if n_target_values != n_values:
            # if not enough column entries pad with original values
            self.target_value = np.column_stack((self.target_value,
                                                 self.original[-(n_values - n_target_values):]
                                                 * np.ones((self.target_value.shape[0], n_values - n_target_values)
                                                           )
                                                 ))

    def get_original(self, structure):
        m = structure.lumped_mass[self.variable_index]
        pos = structure.lumped_mass_position[self.variable_index, :]
        inertia = structure.lumped_mass_inertia[self.variable_index, :, :]

        self.original = np.hstack((m, pos, np.diag(inertia), inertia[0, 1], inertia[0, 2], inertia[1, 2]))


class LumpedMassControl:
    """Lumped Mass Control Class

    This class is instantiated when at least one lumped mass is modified.

    It allows control over unchanged lumped masses and calls the method to execute the change.

    Attributes:
        lumped_mass_variables (list): List of integers containing the indices of the variables to change. These indices
        refer to the order in which they are provided in the general settings for the generator.
    """
    def __init__(self):
        self.lumped_mass_variables = []

    def set_unchanged_vars_to_zero(self, structure):
        """
        Sets the lumped masses variables of unchanged lumped masses to zero.

        This is to avoid the lumped mass changing during execution

        Args:
            structure (sharpy.structure.models.beam.Beam): SHARPy structure object

        """
        for i_lumped_mass in range(len(structure.lumped_mass)):
            if i_lumped_mass not in self.lumped_mass_variables:
                structure.lumped_mass[i_lumped_mass] *= 0
                structure.lumped_mass_position[i_lumped_mass] *= 0
                structure.lumped_mass_inertia[i_lumped_mass] *= 0

    @staticmethod
    def execute_change(structure):
        """Executes the change in the lumped masses.

        Called only once per time step when all the changed lumped mass variables have been processed.
        """
        # called once all variables changed
        structure.lump_masses()
        structure.generate_fortran()

    def append(self, i):
        self.lumped_mass_variables.append(i)
