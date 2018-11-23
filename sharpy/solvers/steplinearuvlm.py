"""
Temporary solver to integrate the linear UVLM aerodynamic solver
N Goizueta
Nov 18
"""

from sharpy.utils.solver_interface import BaseSolver, solver

os.environ["DIRuvlm3d"] = "/home/ng213/linuvlm/uvlm3d/src/"
sys.path.append(os.environ["DIRuvlm3d"])
import save, linuvlm, lin_aeroelastic, libss, librom, lin_utils

class StepLinearUVLM(BaseSolver):
    """
    Temporary solver to integrate the linear UVLM aerodynamics in SHARPy

    """
    solver_id = 'StepLinearUVLM'

    def __init__(self):
        """
        Create default settings
        """

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.1

        self.settings_types['intgr_order'] = 'int'
        self.settings_default['intgr_order'] = 2

        self.settings_types['density'] = 'float'
        self.settings_default['density'] = 1.225

        self.settings_types['ScalingDict'] = 'dict'
        self.settings_default['ScalingDict'] = {'length': 1.0,
                                                'speed': 1.0,
                                                'density': 1.0}

        self.data = None
        self.settings = None
        self.lin_uvlm_system = None

    def initialise(self, data, custom_settings=None):
        """
        Set up solver

        Args:
            data (PreSharpy): class containing the problem information
            custom_settings: custom settings dictionary

        """

        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # Check whether linear UVLM has been initialised
        try:
            self.data.aero.linear
        except AttributeError:
            self.data.aero.linear = dict()

            # Generate instance of linuvlm.Dynamic()
            lin_uvlm_system = linuvlm.Dynamic(self.data.aero.timestep_info[-1],
                dt=self.settings['dt'],
                integr_order=self.settings['integr_order'],
                ScalingDict=self.settings['ScalingDict'])

            # Assemble the state space system
            lin_uvlm_system.assemble_ss()
            self.data.aero.linear['SS'] = self.lin_uvlm_system.SS



    def run(self,
            aero_tstep,
            structure_tstep,
            convect_wake=False,
            dt=None,
            t=None,
            unsteady_contribution=False):

        if aero_tstep is None:
            aero_tstep = self.data.aero.timestep_info[-1]
        if structure_tstep is None:
            structure_tstep = self.data.structure.timestep_info[-1]
        if dt is None:
            dt = self.settings['dt'].value
        if t is None:
            t = self.data.ts*dt


        # Solve system given inputs. inputs to the linear UVLM is a column of zeta, zeta_dot and u_ext
        # Reshape zeta, zeta_dot and u_ext into a column vector
        # usta = [zeta, zeta_dot, u_ext]

        # Solve system
        #y_sta = self.data.aero.linear['SS'].solve(u_sta)




    def add_step(self):
        self.data.aero.add_timestep()

    def update_grid(self, beam):
        self.data.aero.generate_zeta(beam, self.data.aero.aero_settings, -1, beam_ts=-1)

    def update_custom_grid(self, structure_tstep, aero_tstep):
        self.data.aero.generate_zeta_timestep_info(structure_tstep, aero_tstep, self.data.structure, self.data.aero.aero_settings)
