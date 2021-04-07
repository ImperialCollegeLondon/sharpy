import numpy as np
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.correct_forces as cf
import sharpy.utils.algebra as algebra
import sharpy.aero.utils.uvlmlib as uvlmlib
import ctypes as ct


@generator_interface.generator
class PolarAerodynamicForces(generator_interface.BaseGenerator):
    generator_id = 'PolarCorrection'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['correct_lift'] = 'bool'
    settings_default['correct_lift'] = False

    # ... etc settings
    settings_types['cd_from_cl'] = 'bool'
    settings_default['cd_from_cl'] = False

    settings_types['compute_uind'] = 'bool'
    settings_default['compute_uind'] = False

    settings_types['compute_actual_aoa'] = 'bool'
    settings_default['compute_actual_aoa'] = False

    def __init__(self):
        self.settings = None

        self.aero = None
        self.beam = None
        self.rho = None
        self.vortex_radius = None

    def initialise(self, in_dict, **kwargs):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.aero = kwargs.get('aero')
        self.beam = kwargs.get('structure')
        self.rho = kwargs.get('rho')
        self.vortex_radius = kwargs.get('vortex_radius', 1e-6)

    def generate(self, **params):

        aero_kstep = params['aero_kstep']
        structural_kstep = params['structural_kstep']
        struct_forces = params['struct_forces']

        aerogrid = self.aero
        beam = self.beam
        rho = self.rho
        correct_lift = self.settings['correct_lift']
        cd_from_cl = self.settings['cd_from_cl']
        compute_induced_velocity = self.settings['compute_uind']
        compute_actual_aoa = self.settings['compute_actual_aoa']

        aero_dict = aerogrid.aero_dict
        if aerogrid.polars is None:
            return struct_forces
        new_struct_forces = np.zeros_like(struct_forces)

        nnode = struct_forces.shape[0]
        for inode in range(nnode):
            new_struct_forces[inode, :] = struct_forces[inode, :].copy()
            if aero_dict['aero_node'][inode]:
                ielem, inode_in_elem = beam.node_master_elem[inode]
                iairfoil = aero_dict['airfoil_distribution'][ielem, inode_in_elem]
                isurf = aerogrid.struct2aero_mapping[inode][0]['i_surf']
                i_n = aerogrid.struct2aero_mapping[inode][0]['i_n']
                N = aerogrid.aero_dimensions[isurf, 1]
                polar = aerogrid.polars[iairfoil]
                cab = algebra.crv2rotation(structural_kstep.psi[ielem, inode_in_elem, :])
                cga = algebra.quat2rotation(structural_kstep.quat)
                cgb = np.dot(cga, cab)

                # Deal with the extremes
                if i_n == 0:
                    node1 = 0
                    node2 = 1
                elif i_n == N:
                    node1 = nnode - 1
                    node2 = nnode - 2
                else:
                    node1 = inode + 1
                    node2 = inode - 1

                # Define the span and the span direction
                dir_span = 0.5 * np.dot(cga,
                                        structural_kstep.pos[node1, :] - structural_kstep.pos[node2, :])
                span = np.linalg.norm(dir_span)
                dir_span = algebra.unit_vector(dir_span)

                # Define the chord and the chord direction
                dir_chord = aero_kstep.zeta[isurf][:, -1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n]
                chord = np.linalg.norm(dir_chord)
                dir_chord = algebra.unit_vector(dir_chord)

                # Define the relative velocity and its direction
                urel = (structural_kstep.pos_dot[inode, :] +
                        structural_kstep.for_vel[0:3] +
                        np.cross(structural_kstep.for_vel[3:6],
                                 structural_kstep.pos[inode, :]))
                urel = -np.dot(cga, urel)
                urel += np.average(aero_kstep.u_ext[isurf][:, :, i_n], axis=1)

                dir_urel = algebra.unit_vector(urel)
                if compute_induced_velocity:
                    # TODO - is it worth saving as part of time step?
                    uind = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(aero_kstep,
                                                                                   target_triads=np.vstack((structural_kstep.pos[inode, :], structural_kstep.pos[inode, :])),
                                                                                   vortex_radius=self.vortex_radius,
                                                                                   for_pos=structural_kstep.for_pos,
                                                                                   ncores=8)[0]
                    urel += uind
                dir_urel = algebra.unit_vector(urel)

                # Force in the G frame of reference
                force = np.dot(cgb,
                               struct_forces[inode, 0:3])
                dir_force = algebra.unit_vector(force)

                # Coefficient to change from aerodynamic coefficients to forces (and viceversa)
                coef = 0.5 * rho * np.linalg.norm(urel) ** 2 * chord * span

                # Divide the force in drag and lift
                drag_force = np.dot(force, dir_urel) * dir_urel
                lift_force = force - drag_force

                # Compute the associated lift
                cl = np.linalg.norm(lift_force) / coef
                cd_sharpy = np.linalg.norm(drag_force) / coef

                if cd_from_cl:
                    # Compute the drag from the UVLM computed lift
                    cd, cm = polar.get_cdcm_from_cl(cl)

                else:
                    # Compute L, D, M from polar depending on:
                    if compute_actual_aoa:
                        # i) Compute the actual aoa given the induced velocity
                        aoa = np.arccos(dir_chord.dot(dir_urel) / np.linalg.norm(dir_urel) / np.linalg.norm(dir_chord))
                        cl_polar, cd, cm = polar.get_coefs(aoa)
                    else:
                        # ii) Compute the angle of attack assuming that UVLM gives a 2pi polar and using the CL calculated
                        # from the UVLM
                        aoa_deg_2pi = polar.get_aoa_deg_from_cl_2pi(cl)

                        # Compute the coefficients assocaited to that angle of attack
                        cl_polar, cd, cm = polar.get_coefs(aoa_deg_2pi)
                        # print(cl, cl_new)

                    if correct_lift:
                        # Use polar generated CL rather than UVLM computed CL
                        cl = cl_polar

                # Recompute the forces based on the coefficients
                lift_force = cl * algebra.unit_vector(lift_force) * coef
                drag_force += cd * dir_urel * coef
                force = lift_force + drag_force
                new_struct_forces[inode, 0:3] = np.dot(cgb.T,
                                                       force)

        return new_struct_forces
