"""
Temporary solver to integrate the linear UVLM aerodynamic solver
N Goizueta
Nov 18
"""
import os
import sys
from sharpy.utils.solver_interface import BaseSolver, solver
import numpy as np
import sharpy.utils.settings as settings

os.environ["DIRuvlm3d"] = "/home/ng213/linuvlm/uvlm3d/src/"
sys.path.append(os.environ["DIRuvlm3d"])
import save, linuvlm, lin_aeroelastic, libss, librom, lin_utils


@solver
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

        self.settings_types['integr_order'] = 'int'
        self.settings_default['integr_order'] = 2

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

            # TODO: verify of a better way to implement rho
            self.data.aero.timestep_info[-1].rho = self.settings['density'].value

            # Generate instance of linuvlm.Dynamic()
            lin_uvlm_system = linuvlm.Dynamic(self.data.aero.timestep_info[-1],
                dt=self.settings['dt'].value,
                integr_order=self.settings['integr_order'].value,
                ScalingDict=self.settings['ScalingDict'])

            # Assemble the state space system
            lin_uvlm_system.assemble_ss()
            self.data.aero.linear['System'] = lin_uvlm_system
            self.data.aero.linear['SS'] = lin_uvlm_system.SS



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
        # zeta, zeta_dot and u_ext are originally (3, M + 1, N + 1) matrices and are reshaped into a
        # (K,1) column vector following C ordering i.e. the last index changes the fastest
        zeta = np.concatenate([aero_tstep.zeta[ss].reshape(-1, order='C')
                               for ss in range(aero_tstep.n_surf)])
        zeta_dot = np.concatenate([aero_tstep.zeta_dot[ss].reshape(-1, order='C')
                                   for ss in range(aero_tstep.n_surf)])
        u_ext = np.concatenate([aero_tstep.u_ext[ss].reshape(-1, order='C')
                               for ss in range(aero_tstep.n_surf)])

        # Column vector that will be the input to the linearised UVLM system
        u_sta = np.concatenate((zeta, zeta_dot, u_ext))

        # Solve system
        x_sta, y_sta = self.data.aero.linear['System'].solve_steady(u_sta, method='direct')

        forces = []

        # Reshape output into forces[i_surface] where forces[i_surface] is a (6,M+1,N+1) matrix
        worked_points = 0
        for i_surf in range(aero_tstep.n_surf):
            # Tuple with dimensions of the aerogrid zeta, which is the same for forces
            dimensions = aero_tstep.zeta[i_surf].shape
            points_in_surface = aero_tstep.zeta[i_surf].size
            forces.append(y_sta[worked_points:worked_points+points_in_surface].reshape(dimensions, order='C'))
            forces[i_surf] = np.concatenate((forces[i_surf], np.zeros(dimensions)))
            worked_points += points_in_surface

        aero_tstep.forces = forces

        return self.data

    def add_step(self):
        self.data.aero.add_timestep()

    def update_grid(self, beam):
        self.data.aero.generate_zeta(beam, self.data.aero.aero_settings, -1, beam_ts=-1)

    def update_custom_grid(self, structure_tstep, aero_tstep):
        self.data.aero.generate_zeta_timestep_info(structure_tstep, aero_tstep, self.data.structure, self.data.aero.aero_settings)
