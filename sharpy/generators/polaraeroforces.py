import numpy as np
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.aero.utils.uvlmlib as uvlmlib


@generator_interface.generator
class PolarAerodynamicForces(generator_interface.BaseGenerator):
    """
    This generator corrects the aerodynamic forces from UVLM based on the airfoil polars provided by the user in the
    ``aero.h5`` file. Polars are entered for each airfoil, in a table comprising ``AoA, CL, CD, CM``.

    These are the steps needed to correct the forces:

        * The force coming from UVLM is divided into induced drag (parallel to the incoming flow velocity) and lift
          (the remaining force).
        * The angle of attack is computed based on that lift force and the angle of zero lift computed form the
          airfoil polar and assuming a slope of :math:`2 \pi`
        * The drag force is computed based on the angle of attack and the polars provided by the user


    """
    generator_id = 'PolarCorrection'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['correct_lift'] = 'bool'
    settings_default['correct_lift'] = False
    settings_description['correct_lift'] = 'Correct lift according to the polars'

    settings_types['cd_from_cl'] = 'bool'
    settings_default['cd_from_cl'] = False
    settings_description['cd_from_cl'] = 'Interpolate the C_D for the given C_L, as opposed to getting the C_D from ' \
                                         'the section AoA.'

    settings_types['compute_uind'] = 'bool'
    settings_default['compute_uind'] = False
    settings_description['compute_uind'] = 'Compute (and include) vortex induced velocities in the angle of attack ' \
                                           'calculation.'


    settings_types['compute_actual_aoa'] = 'bool'
    settings_default['compute_actual_aoa'] = False
    settings_description['compute_actual_aoa'] = 'Compute the coefficients using the actual angle of attack of the ' \
                                                 'section, as opposed to the angle necessary to provide the UVLM ' \
                                                 'calculated C_L with an assumed lift curve slope of 2pi'

    settings_types['drag_from_polar'] = 'bool'
    settings_default['drag_from_polar'] = False
    settings_description['drag_from_polar'] = 'Take Cd directly from polar. Else, to the Cd computed by SHARPy, the ' \
                                              'difference with the Cd from the polar (mostly profile drag) is added.'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options,
                                       header_line='This generator takes in the following settings.')

    def __init__(self):
        self.settings = None

        self.aero = None
        self.structure = None
        self.rho = None
        self.vortex_radius = None

    def initialise(self, in_dict, **kwargs):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.aero = kwargs.get('aero')
        self.structure = kwargs.get('structure')
        self.rho = kwargs.get('rho')
        self.vortex_radius = kwargs.get('vortex_radius', 1e-6)

    def generate(self, **params):
        """
        Keyword Args:
            aero_kstep (:class:`sharpy.utils.datastructures.AeroTimeStepInfo`): Current aerodynamic substep
            structural_kstep (:class:`sharpy.utils.datastructures.StructTimeStepInfo`): Current structural substep
            struct_forces (np.array): Array with the aerodynamic forces mapped on the structure in the B frame of
              reference

        Returns:
            np.array: New corrected structural forces
        """
        aero_kstep = params['aero_kstep']
        structural_kstep = params['structural_kstep']
        struct_forces = params['struct_forces']

        aerogrid = self.aero
        structure = self.structure
        rho = self.rho
        correct_lift = self.settings['correct_lift']
        cd_from_cl = self.settings['cd_from_cl']
        compute_induced_velocity = self.settings['compute_uind']
        compute_actual_aoa = self.settings['compute_actual_aoa']

        aero_dict = aerogrid.aero_dict
        if aerogrid.polars is None:
            return struct_forces
        new_struct_forces = np.zeros_like(struct_forces)

        # Compute induced velocities at the structural points
        if compute_induced_velocity:
            uind = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(aero_kstep,
                                                                           target_triads=structural_kstep.pos,
                                                                           vortex_radius=self.vortex_radius,
                                                                           for_pos=structural_kstep.for_pos,
                                                                           ncores=8)
        else:
            uind = np.zeros_like(structural_kstep.pos)

        nnode = struct_forces.shape[0]
        for inode in range(nnode):
            new_struct_forces[inode, :] = struct_forces[inode, :].copy()
            if aero_dict['aero_node'][inode]:
                ielem, inode_in_elem = structure.node_master_elem[inode]
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

                urel += uind[inode, :]  # u_ind will be zero if compute_uind == False
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
                if self.settings['drag_from_polar']:
                    drag_force = cd * dir_urel * coef
                else:
                    drag_force += (cd - cd_sharpy) * dir_urel * coef
                force = lift_force + drag_force
                new_struct_forces[inode, 0:3] = np.dot(cgb.T,
                                                       force)

        return new_struct_forces


@generator_interface.generator
class EfficiencyAerodynamicForces(generator_interface.BaseGenerator):
    """
    The efficiency and constant terms are introduced by means of the array ``airfoil_efficiency`` in the ``aero.h5``

    .. math::
        \mathbf{f}_{struct}^B &= \varepsilon^f_0 \mathbf{f}_{i,struct}^B + \varepsilon^f_1\\
        \mathbf{m}_{struct}^B &= \varepsilon^m_0 \mathbf{m}_{i,struct}^B + \varepsilon^m_1

    Notice that the moment correction is applied on top of the force correction. As a consequence, the aerodynamic
    moments generated by the forces on the vertices are corrected sequentially by both efficiencies.

    See Also:
        The SHARPy case files documentation for a detailed overview on how to include the airfoil efficiencies.

    Returns:
         np.ndarray: corresponding aerodynamic force at the structural node from the force and moment at a grid vertex
    """
    generator_id = 'EfficiencyCorrection'

    settings_types = dict()
    settings_default = dict()

    def __init__(self):
        self.aero = None
        self.structure = None

    def initialise(self, in_dict, **kwargs):
        self.aero = kwargs.get('aero')
        self.structure = kwargs.get('structure')

    def generate(self, **params):
        """
        Keyword Args:
            aero_kstep (:class:`sharpy.utils.datastructures.AeroTimeStepInfo`): Current aerodynamic substep
            structural_kstep (:class:`sharpy.utils.datastructures.StructTimeStepInfo`): Current structural substep
            struct_forces (np.array): Array with the aerodynamic forces mapped on the structure in the B frame of
              reference

        Returns:
            np.array: New corrected structural forces
        """
        struct_forces = params['struct_forces']

        n_node = self.structure.num_node
        n_elem = self.structure.num_elem
        aero_dict = self.aero.aero_dict
        new_struct_forces = np.zeros_like(struct_forces)

        # load airfoil efficiency (if it exists); else set to one (to avoid multiple ifs in the loops)
        airfoil_efficiency = aero_dict['airfoil_efficiency']
        # force efficiency dimensions [n_elem, n_node_elem, 2, [fx, fy, fz]] - all defined in B frame
        force_efficiency = np.zeros((n_elem, 3, 2, 3))
        force_efficiency[:, :, 0, :] = 1.
        force_efficiency[:, :, :, 1] = airfoil_efficiency[:, :, :, 0]
        force_efficiency[:, :, :, 2] = airfoil_efficiency[:, :, :, 1]

        # moment efficiency dimensions [n_elem, n_node_elem, 2, [mx, my, mz]] - all defined in B frame
        moment_efficiency = np.zeros((n_elem, 3, 2, 3))
        moment_efficiency[:, :, 0, :] = 1.
        moment_efficiency[:, :, :, 0] = airfoil_efficiency[:, :, :, 2]

        for inode in range(n_node):
            i_elem, i_local_node = self.structure.node_master_elem[inode]
            new_struct_forces[inode, :] = struct_forces[inode, :].copy()
            new_struct_forces[inode, 0:3] *= force_efficiency[i_elem, i_local_node, 0, :] # element wise multiplication
            new_struct_forces[inode, 0:3] += force_efficiency[i_elem, i_local_node, 1, :]
            new_struct_forces[inode, 3:6] *= moment_efficiency[i_elem, i_local_node, 0, :]
            new_struct_forces[inode, 3:6] += moment_efficiency[i_elem, i_local_node, 1, :]
        return new_struct_forces
