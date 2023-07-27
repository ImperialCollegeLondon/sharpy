"""
Linear State Space Assembler
"""
from sharpy.utils.datastructures import Linear
from sharpy.utils.solver_interface import solver, BaseSolver

import sharpy.linear.utils.ss_interface as ss_interface
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout


@solver
class LinearAssembler(BaseSolver):
    r"""
    Warnings:
        Under development - please advise of new features and bugs!

    Creates a workspace containing the different linear elements of the state-space.

    The user specifies which elements to build sequentially via the ``linear_system`` setting.

    The most common uses will be:

        * Aerodynamic: :class:`sharpy.linear.assembler.LinearUVLM` solver

        * Structural: :class:`sharpy.linear.assembler.LinearBeam` solver

        * Aeroelastic: :class:`sharpy.linear.assembler.LinearAeroelastic` solver

    The solver enables to load a user specific assembly of a state-space by means of the ``LinearCustom`` block.

    See :class:`sharpy.sharpy.linear.assembler.LinearAssembler` for a detailed description of each of the state-space assemblies.

    Upon assembly of the linear system, the data structure ``data.linear`` will be created. The :class:`.Linear`
    contains the state-space as an attribute. This state space will be the one employed by postprocessors.

    Important: running the linear routines requires information on the tangent mass, stiffness and gyroscopic
    structural matrices therefore the solver :class:`solvers.modal.Modal` must have been run prior to linearisation.
    In addition, if the problem includes rigid body velocities, at least one
    timestep of :class:`solvers.DynamicCoupled` must have run such that the rigid body velocity is included.

    Example:

    The typical ``flow`` setting used prior to using this solver for an aeroelastic simulation with rigid body dynamics
    will be similar to:

    >>> flow = ['BeamLoader',
    >>>        'AerogridLoader',
    >>>        'StaticTrim',
    >>>        'DynamicCoupled',  # a single time step will suffice
    >>>        'Modal',
    >>>        'LinearAssembler']

    """
    solver_id = 'LinearAssembler'
    solver_classification = 'Linear'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['linear_system'] = 'str'
    settings_default['linear_system'] = None
    settings_description['linear_system'] = 'Name of chosen state space assembly type'

    settings_types['linear_system_settings'] = 'dict'
    settings_default['linear_system_settings'] = dict()
    settings_description['linear_system_settings'] = 'Settings for the desired state space assembler'

    settings_types['linearisation_tstep'] = 'int'
    settings_default['linearisation_tstep'] = -1
    settings_description['linearisation_tstep'] = 'Chosen linearisation time step number from available time steps'

    settings_types['modal_tstep'] = 'int'
    settings_default['modal_tstep'] = -1
    settings_description['modal_tstep'] = 'Timestep in which modal information is stored. Useful if the ``Modal`` solver' \
                                          ' is run at the start of the SHARPy flow.'

    settings_types['inout_coordinates'] = 'str'
    settings_default['inout_coordinates'] = ''
    settings_description['inout_coordinates'] = 'Input/output coordinates of the system. Nodal or modal space.'
    settings_options['inout_coordinates'] = ['', 'nodes', 'modes']

    settings_types['retain_inputs'] = 'list(int)'
    settings_default['retain_inputs'] = []
    settings_description['retain_inputs'] = 'List of input channels to retain in the chosen ``inout_coordinates``.'

    settings_types['retain_outputs'] = 'list(int)'
    settings_default['retain_outputs'] = []
    settings_description['retain_outputs'] = 'List of output channels to retain in the chosen ``inout_coordinates``.'

    settings_types['retain_input_variables'] = 'list(str)'
    settings_default['retain_input_variables'] = []
    settings_description['retain_input_variables'] = 'List of input channels to retain in the chosen ' \
                                                     '``inout_coordinates``.'

    settings_types['retain_output_variables'] = 'list(str)'
    settings_default['retain_output_variables'] = []
    settings_description['retain_output_variables'] = 'List of output channels to retain in the chosen ' \
                                                      '``inout_coordinates``.'

    settings_types['recover_accelerations'] = 'bool'
    settings_default['recover_accelerations'] = False
    settings_description['recover_accelerations'] = 'Recover structural system accelerations as additional outputs.'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):

        self.settings = dict()
        self.data = None

    def initialise(self, data, custom_settings=None, restart=False):

        self.data = data
        if custom_settings:
            self.data.settings[self.solver_id] = custom_settings
            self.settings = self.data.settings[self.solver_id]
        # else:custom_settings

        else:
            self.settings = data.settings[self.solver_id]
        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           options=self.settings_options,
                           no_ctype=True)

        # Get consistent linearisation timestep
        ii_step = self.settings['linearisation_tstep']

        tsstruct0 = data.structure.timestep_info[ii_step]
        tsaero0 = data.aero.timestep_info[ii_step]

        try:
            tsstruct0.modal = data.structure.timestep_info[self.settings['modal_tstep']].modal
        except AttributeError:
            raise AttributeError('Unable to find modal information at desired '
                                 'timestep {:g}'.format(self.settings['modal_tstep']))

        # Create data.linear
        self.data.linear = Linear(tsaero0, tsstruct0)

        # Load available systems
        import sharpy.linear.assembler

        # Load roms
        import sharpy.rom

        lsys = ss_interface.initialise_system(self.settings['linear_system'])
        lsys.initialise(data)
        self.data.linear.linear_system = lsys

    def run(self, **kwargs):

        self.data.linear.ss = self.data.linear.linear_system.assemble()

        if self.settings['recover_accelerations']:
            gain = self.data.linear.linear_system.beam.recover_accelerations(self.data.linear.ss)
            self.data.linear.ss.addGain(gain, where='out')

        # modify inout coordinates
        if self.settings['inout_coordinates'] == 'nodes':
            try:
                self.data.linear.linear_system.to_nodal_coordinates()
            except AttributeError:
                pass

        # retain only selected inputs and outputs
        if len(self.settings['retain_inputs']) != 0:
            self.data.linear.ss.retain_inout_channels(self.settings['retain_inputs'], where='in')
        if len(self.settings['retain_outputs']) != 0:
            self.data.linear.ss.retain_inout_channels(self.settings['retain_outputs'], where='out')

        if len(self.settings['retain_input_variables']) != 0:
            ss = self.data.linear.ss
            input_vars = ss.input_variables
            removed_variables = []
            for variable in input_vars:
                if variable.name not in self.settings['retain_input_variables']:
                    removed_variables.append(variable.name)
            ss.remove_inputs(*removed_variables)

        if len(self.settings['retain_output_variables']) != 0:
            ss = self.data.linear.ss
            output_vars = ss.output_variables
            removed_variables = []
            for variable in output_vars:
                if variable.name not in self.settings['retain_output_variables']:
                    removed_variables.append(variable.name)
            ss.remove_outputs(*removed_variables)

        cout.cout_wrap('Final system is:', 1)
        cout.cout_wrap(str(self.data.linear.ss), 2)

        return self.data


