import numpy as np
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings


@generator_interface.generator
class ModifyStructure(generator_interface.BaseGenerator):
    generator_id = 'ModifyStructure'

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
    settings_description['variable_index'] = 'Index of variable to change. For instance the 1st lumped mass.'

    settings_types['file_list'] = 'list(str)'
    settings_default['file_list'] = None
    settings_description['file_list'] = 'File path for each variable containing the changing info, in the appropriate ' \
                                        'format'

    def __init__(self):
        self.settings = None

        self.num_changes = None
        self.variables = []

    def initialise(self, in_dict, structure):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True,
                                 options=self.settings_options)

        self.num_changes = len(self.settings['change_variable'])

        # lumped_mass_ready = False # unchanged variables are not yet set to zero
        lumped_mass_variables = []
        # run once - since it gets added it prevents unchanged variables to change with original value...
        for i in range(self.num_changes):
            var_type = self.settings['change_variable'][i]
            if var_type == 'lumped_mass':
                variable = ChangeLumpedMass(var_index=self.settings['variable_index'][i],
                                            file=self.settings['file_list'][i])
                variable.initialise(structure)

                self.variables.append(variable)
                lumped_mass_variables.append(i)
                # if not lumped_mass_ready:
                #     for i_mass in range(len(structure.lumped_mass)):
                #         # TODO: change such that it checks which variables are lumped mass
                #         if i_mass not in self.settings['variable_index']:
                #             ChangeLumpedMass.set_to_zero(structure, i_mass)
                #     lumped_mass_ready = True

        for i_mass in range(len(structure.lumped_mass)):
            if i_mass not in lumped_mass_variables:
                # set to zero unchanged lumped masses (that way they won't change)
                ChangeLumpedMass.set_to_zero(structure, i_mass)

    def generate(self, params):
        data = params['data']
        ts = len(data.structure.timestep_info) - 1
        structure = data.structure
        # print('Time step: {:g}'.format(ts))

        for variable in self.variables:
            variable(structure, ts)

        # should only be called once per time step
        ChangeLumpedMass.execute_change(structure)


class ChangedVariable:

    def __init__(self, name, var_index, file):
        self.name = name
        self.variable_index = var_index
        self.file = file

        self.original = None
        self.target_value = None
        self.delta = None
        self.current_value = None  # initially

    def initialise(self, structure):

        self.load_file()
        self.get_original(structure)

        self.delta = self.target_value
        self.current_value = self.original  # initially

    def __call__(self, structure, ts):
        pass

    def get_original(self, structure):
        pass

    def load_file(self):
        self.target_value = np.loadtxt(self.file)


class ChangeLumpedMass(ChangedVariable):

    def __init__(self, var_index, file):
        super().__init__('lumped_mass', var_index=var_index, file=file)

    def __call__(self, structure, ts):
        try:
            delta = self.target_value[ts] - self.current_value
            structure.lumped_mass[self.variable_index] = delta[0]
            structure.lumped_mass_position[self.variable_index, :] = delta[1:4]
            ixx, iyy, izz, ixy, ixz, iyz = delta[-6:]
            inertia = np.block([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            structure.lumped_mass_inertia[self.variable_index, :, :] = inertia

            self.current_value += delta
        except IndexError:
            structure.lumped_mass[self.variable_index] = 0
            structure.lumped_mass_position[self.variable_index, :] = np.zeros(3)
            structure.lumped_mass_inertia[self.variable_index, :, :] = np.zeros((3, 3))

    @staticmethod
    def execute_change(structure):
        # called once all variables changes
        structure.lump_masses()
        structure.generate_fortran()

    @staticmethod
    def set_to_zero(structure, i_unchanged_var):
        # called once during initialisation
        # it changes all the parameters to zero for lumped masses that are not being changed
        structure.lumped_mass[i_unchanged_var] *= 0
        structure.lumped_mass_position[i_unchanged_var] *= 0
        structure.lumped_mass_inertia[i_unchanged_var] *= 0

    def load_file(self):

        super().load_file()

        # if not enough column entries # TODO: pad with original!
        if self.target_value.shape[1] != 10:
            self.target_value = np.column_stack((self.target_value,
                                                 np.zeros((self.target_value.shape[0], 10 - self.target_value.shape[1]))))

    def get_original(self, structure):
        m = structure.lumped_mass[self.variable_index]
        pos = structure.lumped_mass_position[self.variable_index, :]
        inertia = structure.lumped_mass_inertia[self.variable_index, :, :]

        self.original = np.hstack((m, pos, np.diag(inertia), inertia[0, 1], inertia[0, 2], inertia[1, 2]))
