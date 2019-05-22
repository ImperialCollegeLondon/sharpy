"""
Linear State Space Assembler
"""

from sharpy.utils.solver_interface import solver, BaseSolver

import sharpy.linear.utils.ss_interface as ss_interface
import sharpy.utils.settings as settings
import sharpy.utils.h5utils as h5
import sharpy.linear.src.libss as libss


@solver
class LinearAssembler(BaseSolver):
    solver_id = 'LinearSpace'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['linearisation_tstep'] = 'int'
        self.settings_default['linearisation_tstep'] = -1

        self.settings = dict()
        self.data = None

    def initialise(self, data, custom_settings=None):

        self.data = data
        if custom_settings:
            self.data.settings[self.solver_id] = custom_settings
            self.settings = self.data.settings[self.solver_id]
        # else:custom_settings
        #     pass
        #     self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_default, self.settings_types)

        # Some ideas - NG 6/5/19
        #
        # Create a state space element class that contains the individual subsystems with the following common methods:
        #   - Assembly
        #   - Connection
        # Attributes:
        #   - Specific class (linuvlm / lingebm)
        #   - State Space Model
        #   - Inputs / Outputs (which can be modified)
        #
        # Use this solver to assemble the wanted elements and connect them


        # Get consistent linearisation timestep
        ii_step = self.settings['linearisation_tstep']
        tsstruct0 = data.structure.timestep_info[ii_step]
        tsaero0 = data.aero.timestep_info[ii_step]

        # Create data.linear
        self.data.linear = Linear(tsaero0, tsstruct0)  # TODO HOW TO DO THIS PROPERLY?

        # 22/5/19
        # Load available systems
        import sharpy.linear.assembler
        flow = {'LinearBeam'}

        lsys = dict()
        for lin_sys in flow:
            lsys[lin_sys] = ss_interface.initialise_system(lin_sys)
            lsys[lin_sys].initialise(data)
            lsys[lin_sys].assemble()
            print('Hi!')

        self.data.linear.lsys = lsys

    def run(self):

        # series connection
        sys_worked = 0
        for system in self.data.linear.lsys:
            if sys_worked == 0:
                self.data.linear.ss = self.data.linear.lsys[system].ss
            else:
                self.data.linear.ss = libss.series(self.ss, self.linear.lsys[system].ss)
            sys_worked += 1

        return self.data


class Linear(object):
    """
    This is the class responsible for the transfer of information and can be accessed as data.linear
    """
    def __init__(self, tsaero0, tsstruct0):
        self.lsys = None
        self.ss = None
        self.tsaero0 = tsaero0
        self.tsstruct0 = tsstruct0
        self.timestep_info = []


if __name__ == "__main__":
    print('Testing the assembly of the pendulum system')
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
                       'LinearBeam': beam_settings}

    linear_space = LinearAssembler()
    linear_space.initialise(data, custom_settings)
    data = linear_space.run()

    import sharpy.solvers.lindynamicsim as lindynsim
    linear_sim = lindynsim.LinearDynamicSimulation()
    linear_sim.initialise(data)

    import numpy as np

    eigs = np.linalg.eig(data.linear.ss.A)
    eigs_ct = np.log(eigs[0]) / data.linear.ss.dt
    order = np.argsort(eigs_ct.real)[::-1]
    eigs_ct = eigs_ct[order]
    print('End')


