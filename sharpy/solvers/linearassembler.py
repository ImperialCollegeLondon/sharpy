"""
Linear State Space Assembler
"""

from sharpy.utils.solver_interface import solver, BaseSolver

import sharpy.linear.utils.ss_interface as ss_interface
import sharpy.utils.settings as settings
import sharpy.utils.h5utils as h5
import warnings


@solver
class LinearAssembler(BaseSolver):
    """
    Warnings:
        Under development - please advise of new features and bugs!

    Creates a workspace containing the different linear elements of the state-space.

    The user specifies which elements to build sequentially via the ``linear_system`` setting. If building
    more than one system, they can be joined in series.

    The most common uses will be:

        * Aerodynamic: ``LinearUVLM`` solver

        * Structural: ``LinearBeam`` solver

        * Aeroelastic: ``LinearAeroelastic`` solver

    The solver enables to load a user specific assembly of a state-space by means of the ``LinearCustom`` block.

    See ``sharpy.linear.assembler`` for a detailed description of each of the state-space assemblies.

    """
    solver_id = 'LinearAssembler'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['linear_system'] = 'str'
    settings_default['linear_system'] = None
    settings_description['linear_system'] = 'Name of chosen state space assembly type'

    settings_types['linear_system_settings'] = 'dict'
    settings_default['linear_system_settings'] = dict()
    settings_description['linear_system_settings'] = 'Settings for the desired state space assembler'

    settings_types['linearisation_tstep'] = 'int'
    settings_default['linearisation_tstep'] = -1
    settings_description['linearisation_tstep'] = 'Chosen linearisation time step from ran time steps'

    def __init__(self):

        warnings.warn('LinearAssembler solver under development')
        self.settings = dict()
        self.data = None

    def initialise(self, data, custom_settings=None):

        self.data = data
        if custom_settings:
            self.data.settings[self.solver_id] = custom_settings
            self.settings = self.data.settings[self.solver_id]
        # else:custom_settings

        else:
            self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # Get consistent linearisation timestep
        ii_step = self.settings['linearisation_tstep']
        if type(ii_step) != int:
            ii_step = self.settings['linearisation_tstep'].value

        tsstruct0 = data.structure.timestep_info[ii_step]
        tsaero0 = data.aero.timestep_info[ii_step]

        # Create data.linear
        self.data.linear = Linear(tsaero0, tsstruct0)

        # Load available systems
        import sharpy.linear.assembler

        # Load roms
        import sharpy.rom

        lsys = ss_interface.initialise_system(self.settings['linear_system'])
        lsys.initialise(data)
        self.data.linear.linear_system = lsys

    def run(self):

        self.data.linear.ss = self.data.linear.linear_system.assemble()

        return self.data


class Linear(object):
    """
    This is the class responsible for the transfer of information and can be accessed as data.linear
    """
    def __init__(self, tsaero0, tsstruct0):
        self.linear_system = None
        self.ss = None
        self.tsaero0 = tsaero0
        self.tsstruct0 = tsstruct0
        self.timestep_info = []
        self.uvlm = None
        self.beam = None


if __name__ == "__main__":
    print('Testing the assembly of the pendulum system')
    test = 'aeroelastic'
    if test == 'beam':
        data = h5.readh5('/home/ng213/sharpy_cases/CC_DevTests/01_LinearAssembly/flexible_beam_static.data.h5').data

        beam_settings = {'modal_projection': False,
                         'inout_coords': 'nodes',
                         'discrete_time': True,
                         'newmark_damp': 0.15*1,
                         'discr_method': 'newmark',
                         'dt': 0.001,
                         'proj_modes': 'undamped',
                         'use_euler': True,
                         'num_modes': 13,
                         'remove_dofs': ['V'],
                         'gravity': 'on'}
        custom_settings = {'linearisation_tstep': -1,
                           'flow': ['LinearBeam'],
                           'LinearBeam': beam_settings}

        linear_space = LinearAssembler()
        linear_space.initialise(data, custom_settings)
        data = linear_space.run()

        # import sharpy.solvers.lindynamicsim as lindynsim
        # linear_sim = lindynsim.LinearDynamicSimulation()
        # linear_sim.initialise(data)

        import numpy as np

        eigs = np.linalg.eig(data.linear.ss.A)
        eigs_ct = np.log(eigs[0]) / data.linear.ss.dt
        order = np.argsort(eigs_ct.real)[::-1]
        eigs_ct = eigs_ct[order]
        print('End')

    elif test == 'uvlm':
        data = h5.readh5('/home/ng213/sharpy_cases/CC_DevTests/01_LinearAssembly/sears_uinf0050_AR100_M8N12Ms10_KR15_sp0.data.h5').data

        uvlm_settings = {'dt': 0.001,
                           'integr_order': 2,
                           'density': 1.225,
                           'remove_predictor': False,
                           'use_sparse': False,
                           'ScalingDict': {'length': 1.,
                                           'speed': 1.,
                                           'density': 1.},
                         'remove_inputs': ['u_gust']}
        custom_settings = {'linearisation_tstep': -1,
                           'flow': ['LinearUVLM'],
                           'LinearUVLM': uvlm_settings}
        linear_space = LinearAssembler()
        linear_space.initialise(data, custom_settings)
        data = linear_space.run()

    elif test=='aeroelastic':
        data = h5.readh5('/home/ng213/sharpy_cases/ToSORT_FlyingWings/01_RichardsBFF/cases/horten/horten.data.h5').data

        custom_settings = {'flow': ['LinearAeroelastic'],
                                        'LinearAeroelastic': {
                                            'beam_settings': {'modal_projection': False,
                                                              'inout_coords': 'nodes',
                                                              'discrete_time': True,
                                                              'newmark_damp': 0.5,
                                                              'discr_method': 'newmark',
                                                              'dt': 0.001,
                                                              'proj_modes': 'undamped',
                                                              'use_euler': 'off',
                                                              'num_modes': 40,
                                                              'print_info': 'on',
                                                              'gravity': 'on',
                                                              'remove_dofs': []},
                                            'aero_settings': {'dt': 0.001,
                                                              'integr_order': 2,
                                                              'density': 1.225*0.0000000001,
                                                              'remove_predictor': False,
                                                              'use_sparse': True,
                                                              'rigid_body_motion': True,
                                                              'use_euler': False,
                                                              'remove_inputs': ['u_gust']},
                                            'rigid_body_motion': True}}
        linear_space = LinearAssembler()
        linear_space.initialise(data, custom_settings)
        data = linear_space.run()
        print('End')
