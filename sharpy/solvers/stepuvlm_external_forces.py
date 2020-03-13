import ctypes as ct
import numpy as np
import scipy.optimize
import scipy.signal

import sharpy.utils.algebra as algebra
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver, solver_from_string
import sharpy.utils.generator_interface as gen_interface
import sharpy.utils.cout_utils as cout

StepUvlm = solver_from_string('StepUvlm')


@solver
class StepUvlm_external_forces(StepUvlm):
    """
    StepUVLM_external_forces is a solver similar to :class:`~.solvers.stepuvlm.StepUVLM`.

    It solves the UVLM system just as the original solver. However, the computation of forces is done through
    tables interpolation based on the angle of attack obtained in the UVLM solver.

    """
    solver_id = 'StepUvlm_external_forces'
    solver_classification = 'aero'

    settings_types = StepUvlm.settings_types.copy()
    settings_default = StepUvlm.settings_default.copy()
    settings_description = StepUvlm.settings_description.copy()
    settings_options = StepUvlm.settings_options.copy()

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)


    def run(self,
            aero_tstep=None,
            structure_tstep=None,
            convect_wake=True,
            dt=None,
            t=None,
            unsteady_contribution=False):
        """
        Runs a step of the aerodynamics as implemented in UVLM.
        """

        if aero_tstep is None:
            aero_tstep = self.data.aero.timestep_info[-1]
        if structure_tstep is None:
            structure_tstep = self.data.structure.timestep_info[-1]
        if dt is None:
            dt = self.settings['dt'].value
        if t is None:
            t = self.data.ts*dt

        if not aero_tstep.zeta:
            return self.data

        # generate uext
        self.velocity_generator.generate({'zeta': aero_tstep.zeta,
                                          'override': True,
                                          't': t,
                                          'ts': self.data.ts,
                                          'dt': dt,
                                          'for_pos': structure_tstep.for_pos},
                                         aero_tstep.u_ext)
        if self.settings['convection_scheme'].value > 1 and convect_wake:
            # generate uext_star
            self.velocity_generator.generate({'zeta': aero_tstep.zeta_star,
                                              'override': True,
                                              'ts': self.data.ts,
                                              'dt': dt,
                                              't': t,
                                              'for_pos': structure_tstep.for_pos},
                                             aero_tstep.u_ext_star)

        uvlmlib.uvlm_solver_uindw(self.data.ts,
                            aero_tstep,
                            structure_tstep,
                            self.settings,
                            convect_wake=convect_wake,
                            dt=dt)

        # Computation the relative velocity on the nodes
        urel = []
        for_lin_vel = np.dot(structure_tstep.cga(), structure_tstep.for_vel[0:3])
        for_omega = np.dot(structure_tstep.cga(), structure_tstep.for_vel[3:6])
        for isurf in range(aero_tstep.n_surf):
            M = aero_tstep.dimensions[isurf, 0]
            N = aero_tstep.dimensions[isurf, 1]
            urel.append(np.zeros((3, M + 1, N + 1)))
            for i in range(M + 1):
                for j in range(N + 1):
                    urel[isurf][:, i, j] = aero_tstep.u_ext[isurf][:, i, j]
                    urel[isurf][:, i, j] -= aero_tstep.zeta_dot[isurf][:, i, j]
                    urel[isurf][:, i, j] -= for_lin_vel[0:3]
                    urel[isurf][:, i, j] -= np.cross(for_omega,
                                               aero_tstep.zeta[isurf][:, i, j])

        for isurf in range(aero_tstep.n_surf):
            M = aero_tstep.dimensions[isurf, 0]
            N = aero_tstep.dimensions[isurf, 1]
            for j in range(N + 1):
                if j == 0:
                    uindw_node = aero_tstep.uindw_col[isurf][:, 0, j]
                    span =  0.5*np.linalg.norm(aero_tstep.zeta[isurf][:, 0, 0] -
                                      aero_tstep.zeta[isurf][:, 0, 1])
                    dir_span = algebra.unit_vector(structure_tstep.pos[j + 1, :] - structure_tstep.pos[j, :])
                elif j == N:
                    uindw_node = aero_tstep.uindw_col[isurf][:, 0, N - 1]
                    span =  0.5*np.linalg.norm(aero_tstep.zeta[isurf][:, 0, N - 1] -
                                      aero_tstep.zeta[isurf][:,0, N])
                    dir_span = algebra.unit_vector(structure_tstep.pos[j, :] - structure_tstep.pos[j - 1, :])
                else:
                    uindw_node = 0.5*(aero_tstep.uindw_col[isurf][:, 0, j - 1] +
                                      aero_tstep.uindw_col[isurf][:, 0, j])
                    span =  0.5*np.linalg.norm(aero_tstep.zeta[isurf][:, 0, j - 1] -
                                      aero_tstep.zeta[isurf][:, 0, j + 1])
                    dir_span = algebra.unit_vector(structure_tstep.pos[j + 1, :] - structure_tstep.pos[j - 1, :])

                dir_chord = aero_tstep.zeta[isurf][:, -1, j] - aero_tstep.zeta[isurf][:, 0, j]
                chord = np.linalg.norm(dir_chord)
                dir_chord = algebra.unit_vector(dir_chord)
                dyn_pressure = 0.5*self.settings['rho'].value*np.linalg.norm(urel[isurf][:, 0, j])**2*chord*span
                urel_node = urel[isurf][:, 0, j] - uindw_node
                # urel_node = urel[isurf][:, 0, j]

                aoa = np.arccos(np.dot(urel_node, dir_chord)/np.linalg.norm(dir_chord)/np.linalg.norm(urel_node))

                cl = 2*np.pi*aoa
                cd = 0.0

                dir_cd = algebra.unit_vector(urel_node)
                dir_span = algebra.unit_vector(np.dot(structure_tstep.cag(), dir_span))
                dir_cl = algebra.unit_vector(np.cross(dir_cd, dir_span))

                aero_tstep.forces[isurf][0:3, 0, j] = (cl*dir_cl + cd*dir_cd)*dyn_pressure
                print(aoa, cl, cd, dyn_pressure, aero_tstep.forces[isurf][0:3, 0, j])

        # if unsteady_contribution:
        if False:
            # calculate unsteady (added mass) forces:
            self.data.aero.compute_gamma_dot(dt,
                                             aero_tstep,
                                             self.data.aero.timestep_info[-3:])
            if self.settings['gamma_dot_filtering'] is None:
                self.filter_gamma_dot(aero_tstep,
                                      self.data.aero.timestep_info,
                                      None)
            elif self.settings['gamma_dot_filtering'].value > 0:
                self.filter_gamma_dot(
                    aero_tstep,
                    self.data.aero.timestep_info,
                    self.settings['gamma_dot_filtering'].value)
            uvlmlib.uvlm_calculate_unsteady_forces(aero_tstep,
                                                   structure_tstep,
                                                   self.settings,
                                                   convect_wake=convect_wake,
                                                   dt=dt)
        else:
            for i_surf in range(len(aero_tstep.gamma)):
                aero_tstep.gamma_dot[i_surf][:] = 0.0

        return self.data
