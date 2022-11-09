import numpy as np
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
from sharpy.aero.utils.utils import magnitude_and_direction_of_relative_velocity, local_stability_axes, span_chord
from sharpy.utils.generate_cases import get_aoacl0_from_camber


@generator_interface.generator
class PolarCorrection(generator_interface.BaseGenerator):
    r"""
    This generator corrects the aerodynamic forces from UVLM based on the airfoil polars provided by the user in the
    ``aero.h5`` file. Polars are entered for each airfoil, in a table comprising ``AoA (rad), CL, CD, CM``.

    This ``generator_id = 'PolarCorrection'`` and can be used in the coupled solvers through the
    ``correct_forces_method`` setting as:

    .. python::
        settings = dict()  # SHARPy settings
        settings['StaticCoupled']['correct_forces_method'] = 'PolarCorrection'
        settings['StaticCoupled']['correct_forces_settings'] = {'cd_from_cl': 'off',  # recommended settings (default)
                                                                'correct_lift': 'off',
                                                                'moment_from_polar': 'off'}

    These are the steps needed to correct the forces:

        1. The force coming from UVLM is divided into induced drag (parallel to the incoming flow velocity) and lift
          (the remaining force).

    If ``cd_from_cl == 'on'``.
        2. The viscous drag and pitching moment are found at the computed lift coefficient. Then forces and
           moments are updated

    Else, the angle of attack is computed:

            2. The angle of attack is computed based on that lift force and the angle of zero lift computed from the
               airfoil polar and assuming the potential flow lift curve slope of :math:`2 \pi`

        3. The drag force is computed based on the angle of attack and the polars provided by the user

        4. If ``correct_lift == 'on'``, the lift coefficient is also corrected with the polar data. Else, only the
           UVLM results are used.

    The pitching moment is added in a similar manner as the viscous drag. However, if ``moment_from_polar == 'on'``
    and ``correct_lift == 'on'``, the total moment (the one used for the FSI) is computed just from polar data,
    overriding any moment computed in SHARPy. That is, the moment will include the polar pitching moment, and moments
    due to lift and drag computed from the polar data.

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

    settings_types['moment_from_polar'] = 'bool'
    settings_default['moment_from_polar'] = False
    settings_description['moment_from_polar'] = 'If ``correct_lift`` is selected, it will compute the pitching moment ' \
                                                'simply from polar derived data, i.e. the polars Cm and the moments' \
                                                'arising from the lift and drag (derived from the polar) contribution. ' \
                                                'Else, it will add the polar Cm to the moment already computed by ' \
                                                'SHARPy.'

    settings_types['add_rotation'] = 'bool'
    settings_default['add_rotation'] = False
    settings_description['add_rotation'] = 'Add rotation velocity. Probably needed in steady computations'

    settings_types['rot_vel_g'] = 'list(float)'
    settings_default['rot_vel_g'] = [0., 0., 0.] 
    settings_description['rot_vel_g'] = 'Rotation velocity in G FoR. Only used if add_rotation = True'

    settings_types['centre_rot_g'] = 'list(float)'
    settings_default['centre_rot_g'] = [0., 0., 0.] 
    settings_description['centre_rot_g'] = 'Centre of rotation in G FoR. Only used if add_rotation = True'

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
        r"""
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
        moment_from_polar = self.settings['moment_from_polar']

        aero_dict = aerogrid.aero_dict
        if aerogrid.polars is None:
            return struct_forces
        new_struct_forces = np.zeros_like(struct_forces)

        nnode = struct_forces.shape[0]

        # Compute induced velocities at the structural points
        cga = algebra.quat2rotation(structural_kstep.quat)
        pos_g = np.array([cga.dot(structural_kstep.pos[inode]) + np.array([0, 0, 0]) for inode in range(nnode)])

        structural_kstep.postproc_node['aoa'] = np.zeros(nnode)

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
                cgb = np.dot(cga, cab)

                if not cd_from_cl:
                    airfoil_coords = aerogrid.aero_dict['airfoils'][str(aero_dict['airfoil_distribution'][ielem, inode_in_elem])]

                dir_span, span, dir_chord, chord = span_chord(i_n, aero_kstep.zeta[isurf])

                # Define the relative velocity and its direction
                urel, dir_urel = magnitude_and_direction_of_relative_velocity(structural_kstep.pos[inode, :],
                                                                              structural_kstep.pos_dot[inode, :],
                                                                              structural_kstep.for_vel[:],
                                                                              cga,
                                                                              aero_kstep.u_ext[isurf][:, :, i_n],
                                                                              self.settings['add_rotation'],
                                                                              self.settings['rot_vel_g'],
                                                                              self.settings['centre_rot_g'],)

                # Coefficient to change from aerodynamic coefficients to forces (and viceversa)
                coef = 0.5 * rho * np.linalg.norm(urel) ** 2 * chord * span

                # Stability axes - projects forces in B onto S
                c_bs = local_stability_axes(cgb.T.dot(dir_urel), cgb.T.dot(dir_chord))
                forces_s = c_bs.T.dot(struct_forces[inode, :3])
                moment_s = c_bs.T.dot(struct_forces[inode, 3:])
                drag_force = forces_s[0]
                lift_force = forces_s[2]

                # Compute the associated lift
                cl = np.sign(lift_force) * np.linalg.norm(lift_force) / coef
                cd_sharpy = np.linalg.norm(drag_force) / coef

                # ii) Compute the effective angle of attack from potential flow theory. The local lift curve
                # slope is 2pi and the zero-lift angle of attack is given by thin airfoil theory. From this,
                # the effective angle of attack is computed for the section and includes 3D effects.
                aoa_0cl = get_aoacl0_from_camber(airfoil_coords[:, 0], airfoil_coords[:, 1])
                aoa = cl / 2 / np.pi + aoa_0cl
                structural_kstep.postproc_node['aoa'][inode] = aoa

                if cd_from_cl:
                    # Compute the drag from the UVLM computed lift
                    cd, cm = polar.get_cdcm_from_cl(cl)

                else:
                    # Compute L, D, M from polar depending on the coefficients associated to that angle of attack
                    cl_polar, cd, cm = polar.get_coefs(aoa)

                    if correct_lift:
                        # Use polar generated CL rather than UVLM computed CL
                        cl = cl_polar

                # Recompute the forces based on the coefficients (side force is uncorrected)
                forces_s[0] += cd * coef  # add viscous drag to induced drag from UVLM
                forces_s[2] = cl * coef

                new_struct_forces[inode, 0:3] = c_bs.dot(forces_s)

                # Pitching moment
                if moment_from_polar:
                    # The panels are shifted by 0.25 of a panel aft from the leading edge
                    panel_shift = 0.25 * (aero_kstep.zeta[isurf][:, 1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n])
                    ref_point = aero_kstep.zeta[isurf][:, 0, i_n] + 0.25 * chord * dir_chord - panel_shift

                    # viscous contribution (pure moment)
                    moment_s[1] += cm * coef * chord

                    # moment due to drag
                    arm = cgb.T.dot(ref_point - pos_g[inode])  # in B frame
                    moment_polar_drag = algebra.cross3(c_bs.T.dot(arm), cd * dir_urel * coef)  # in S frame
                    moment_s += moment_polar_drag

                    # moment due to lift (if corrected)
                    if correct_lift and moment_from_polar:
                        # add moment from scratch: cm_polar + cm_drag_polar + cl_lift_polar
                        moment_s = np.zeros(3)
                        moment_s[1] = cm * coef * chord
                        moment_s += moment_polar_drag
                        moment_polar_lift = algebra.cross3(c_bs.T.dot(arm), forces_s[2] * np.array([0, 0, 1]))
                        moment_s += moment_polar_lift

                    new_struct_forces[inode, 3:6] = c_bs.dot(moment_s)

        return new_struct_forces

    def generate_linear(self, **params):
        r"""
        Generate Linear Gain matrix with correction factors (lift only at the moment). These corrections are implemented
        by means of a Gain that goes from forces and moments in A frame to corrected forces and moments in A frame.

        The nodal corrections are implemented as

        .. math:: K_f^i \leftarrow w_f^i

        where

        .. math:: w_f^i = C^{AS} W_f^i C^{SA}

        and :math:`W_f` is a 3x3 correction matrix expressed in stability axes and is given for each node.

        For the moments, the correction is applied in a similar way, noting that the moments are expressed as
        :math:`T^\top \delta m_B` and thus need pre- and post- multiplication

        .. math::  K_m^i \leftarrow T^\top C^{BA} w_m^i C^{AB} T^{-\top}

        where

        .. math:: w_m^i = C^{AS} W_m^i C^{SA}

        and :math:`W_m^i` is a 3x3 correction matrix expressed in stability axes for the moment correction at each node.

        The total forces and moments are then computed as a summation of the above, where the total forces

        are computed simply as

        .. math:: K_F \leftarrow \Sum w_f^i

        and the total moments

        .. math:: K_M \leftarrow \Sum \tilde{R}_{A,i} w_f^i + \Sum w_m^i C^{AB} T^{-\top}

        which leaves the end result expressed in the A frame.

        An additional gain, referred to as ``forces_at_ref_gain`` is needed to obtain the nodal force at the
        :math:`A` frame in order to correct the forces at this point with the local polar. Thus, this gain
        subtracts from the total forces and moments the sum of those from the remaining nodes. Then the ``polar_gain``
        corrects the local force at the A frame and sums the corrected forces and moments at the other nodes.

        Keyword Args:
            beam (sharpy.linear.assembler.linearbeam.LinearBeam): Beam object
            tsstruct0 (sharpy.utils.datastructures.StructTimestepInfo): Ref structural time step
            tsaero0 (sharpy.utils.datastructures.AeroTimestepInfo): Ref aero time step info

        Returns:
            np.array: Gain matrix
        """
        beam = params['beam']
        tsstruct0 = params['tsstruct0']
        tsaero0 = params['tsaero0']

        aerogrid = self.aero
        structure = self.structure

        structural_kstep = tsstruct0
        aero_kstep = tsaero0
        nnode = tsstruct0.pos.shape[0]

        aero_dict = aerogrid.aero_dict
        if aerogrid.polars is None:
            return None

        cga = algebra.quat2rotation(structural_kstep.quat)

        # Matrix allocation
        num_dof_str = beam.sys.num_dof_str
        num_dof_rig = beam.sys.num_dof_rig
        use_euler = beam.sys.use_euler

        # GEBM degrees of freedom
        jj_for_tra = range(num_dof_str - num_dof_rig,
                           num_dof_str - num_dof_rig + 3)
        jj_for_rot = range(num_dof_str - num_dof_rig + 3,
                           num_dof_str - num_dof_rig + 6)

        if use_euler:
            jj_euler = range(num_dof_str - 3, num_dof_str)
            euler = algebra.quat2euler(tsstruct0.quat)
            tsstruct0.euler = euler
        else:
            jj_quat = range(num_dof_str - 4, num_dof_str)

        polar_gain = np.zeros((num_dof_str, num_dof_str))

        # this gain computes the aerodynamic forces and moments at the node at the A frame, which else
        # would be included in the total force and total moment. This is needed to correct these nodal forces too
        forces_at_ref_gain = np.zeros_like(polar_gain)

        jj = 0  # Global DOF index
        for inode in range(nnode):
            if aero_dict['aero_node'][inode]:

                ### detect bc at node (and no. of dofs)
                bc_here = structure.boundary_conditions[inode]

                if bc_here == 1:  # clamp (only rigid-body)
                    dofs_here = 0
                    jj_tra, jj_rot = [], []

                elif bc_here == -1 or bc_here == 0:  # (rigid+flex body)
                    dofs_here = 6
                    jj_tra = 6 * structure.vdof[inode] + np.array([0, 1, 2], dtype=int)
                    jj_rot = 6 * structure.vdof[inode] + np.array([3, 4, 5], dtype=int)
                else:
                    raise NameError('Invalid boundary condition (%d) at node %d!' \
                                    % (bc_here, inode))

                jj += dofs_here

                ielem, inode_in_elem = structure.node_master_elem[inode]
                iairfoil = aero_dict['airfoil_distribution'][ielem, inode_in_elem]
                isurf = aerogrid.struct2aero_mapping[inode][0]['i_surf']
                i_n = aerogrid.struct2aero_mapping[inode][0]['i_n']
                N = aerogrid.aero_dimensions[isurf, 1]
                polar = aerogrid.polars[iairfoil]
                psi = structural_kstep.psi[ielem, inode_in_elem, :]
                cab = algebra.crv2rotation(psi)
                cgb = np.dot(cga, cab)
                Tan = algebra.crv2tan(psi)
                r_a = tsstruct0.pos[inode]

                dir_span, span, dir_chord, chord = span_chord(i_n, aero_kstep.zeta[isurf])

                # Define the relative velocity and its direction
                urel, dir_urel = magnitude_and_direction_of_relative_velocity(structural_kstep.pos[inode, :],
                                                                              structural_kstep.pos_dot[inode, :],
                                                                              structural_kstep.for_vel[:],
                                                                              cga,
                                                                              aero_kstep.u_ext[isurf][:, :, i_n])

                # Stability axes - projects forces in B onto S
                c_bs = local_stability_axes(cgb.T.dot(dir_urel), cgb.T.dot(dir_chord))
                cas = cab.dot(c_bs)

                local_correction = np.eye(6)

                cla, cda, cma = polar.get_derivatives_at_aoa(tsstruct0.postproc_node['aoa'][inode])
                local_correction[2, 2] = cla / 2 / np.pi

                mom_b2a = cab.dot(np.linalg.inv(Tan.T))
                mom_a2b = Tan.T.dot(cab.T)

                wfi = cas.dot(local_correction[:3, :3]).dot(cas.T)
                wmi = cas.dot(local_correction[3:, 3:]).dot(cas.T)

                if bc_here != 1:
                    polar_gain[np.ix_(jj_tra, jj_tra)] += wfi
                    polar_gain[np.ix_(jj_rot, jj_rot)] += mom_a2b.dot(wmi.dot(mom_b2a))

                    # Total forces and moments
                    # total forces
                    polar_gain[np.ix_(jj_for_tra, jj_tra)] += wfi

                    # forces contribution to total moments
                    polar_gain[np.ix_(jj_for_rot, jj_tra)] += algebra.skew(r_a).dot(wfi)
                    # moments contribution to total moments
                    polar_gain[np.ix_(jj_for_rot, jj_rot)] += wmi.dot(mom_b2a)

                    # keep nodal forces and moments
                    forces_at_ref_gain[np.ix_(jj_tra, jj_tra)] += np.eye(3)
                    forces_at_ref_gain[np.ix_(jj_rot, jj_rot)] += np.eye(3)

                    # obtain the nodal force at the A frame
                    # total force minus the sum of all other nodes
                    # f_node_at_A = F_A - sum(f)
                    forces_at_ref_gain[np.ix_(jj_for_tra, jj_tra)] -= np.eye(3)

                    # obtain the nodal moment in A frame at the A frame
                    # m_node_at_A = M_A - sum(C^AB T^-\top m_b) - sum(r_a.cross(f_a))
                    forces_at_ref_gain[np.ix_(jj_for_rot, jj_tra)] -= algebra.skew(r_a)
                    forces_at_ref_gain[np.ix_(jj_for_rot, jj_rot)] -= mom_b2a

                else:
                    # local force and moment at the A frame
                    forces_at_ref_gain[np.ix_(jj_for_tra, jj_for_tra)] = np.eye(3)
                    forces_at_ref_gain[np.ix_(jj_for_rot, jj_for_rot)] = np.eye(3)

                    # scale nodal force at A frame node
                    polar_gain[np.ix_(jj_for_tra, jj_for_tra)] = wfi
                    polar_gain[np.ix_(jj_for_rot, jj_for_rot)] = wmi

        return polar_gain.dot(forces_at_ref_gain)


@generator_interface.generator
class EfficiencyCorrection(generator_interface.BaseGenerator):
    r"""
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

    def generate_linear(self, **params):
        beam = params['beam']
        tsstruct0 = params['tsstruct0']
        tsaero0 = params['tsaero0']

        aerogrid = self.aero
        structure = self.structure

        structural_kstep = tsstruct0
        aero_kstep = tsaero0
        nnode = tsstruct0.pos.shape[0]

        aero_dict = aerogrid.aero_dict

        cga = algebra.quat2rotation(structural_kstep.quat)

        # Matrix allocation
        num_dof_str = beam.sys.num_dof_str
        num_dof_rig = beam.sys.num_dof_rig
        use_euler = beam.sys.use_euler

        # GEBM degrees of freedom
        jj_for_tra = range(num_dof_str - num_dof_rig,
                           num_dof_str - num_dof_rig + 3)
        jj_for_rot = range(num_dof_str - num_dof_rig + 3,
                           num_dof_str - num_dof_rig + 6)

        if use_euler:
            jj_euler = range(num_dof_str - 3, num_dof_str)
            euler = algebra.quat2euler(tsstruct0.quat)
            tsstruct0.euler = euler
        else:
            jj_quat = range(num_dof_str - 4, num_dof_str)

        polar_gain = np.zeros((num_dof_str, num_dof_str))

        jj = 0  # Global DOF index
        for inode in range(nnode):
            if aero_dict['aero_node'][inode]:

                ### detect bc at node (and no. of dofs)
                bc_here = structure.boundary_conditions[inode]

                if bc_here == 1:  # clamp (only rigid-body)
                    dofs_here = 0
                    jj_tra, jj_rot = [], []

                elif bc_here == -1 or bc_here == 0:  # (rigid+flex body)
                    dofs_here = 6
                    jj_tra = 6 * structure.vdof[inode] + np.array([0, 1, 2], dtype=int)
                    jj_rot = 6 * structure.vdof[inode] + np.array([3, 4, 5], dtype=int)
                else:
                    raise NameError('Invalid boundary condition (%d) at node %d!' \
                                    % (bc_here, inode))

                jj += dofs_here

                ielem, inode_in_elem = structure.node_master_elem[inode]
                iairfoil = aero_dict['airfoil_distribution'][ielem, inode_in_elem]
                isurf = aerogrid.struct2aero_mapping[inode][0]['i_surf']
                i_n = aerogrid.struct2aero_mapping[inode][0]['i_n']
                N = aerogrid.aero_dimensions[isurf, 1]
                psi = structural_kstep.psi[ielem, inode_in_elem, :]
                cab = algebra.crv2rotation(psi)
                Tan = algebra.crv2tan(psi)
                r_a = tsstruct0.pos[inode]

                airfoil_efficiency = aero_dict['airfoil_efficiency']

                cgb = np.dot(cga, cab)

                dir_span, span, dir_chord, chord = span_chord(i_n, aero_kstep.zeta[isurf])

                # Define the relative velocity and its direction
                urel, dir_urel = magnitude_and_direction_of_relative_velocity(structural_kstep.pos[inode, :],
                                                                              structural_kstep.pos_dot[inode, :],
                                                                              structural_kstep.for_vel[:],
                                                                              cga,
                                                                              aero_kstep.u_ext[isurf][:, :, i_n])

                # Stability axes - projects forces in B onto S
                c_bs = local_stability_axes(cgb.T.dot(dir_urel), cgb.T.dot(dir_chord))
                cas = cab.dot(c_bs)

                local_correction = np.diag(airfoil_efficiency[ielem, inode_in_elem, 0, :])

                if bc_here != 1:
                    polar_gain[np.ix_(jj_tra, jj_tra)] += cas.dot(local_correction[:3, :3]).dot(cas.T)
                    polar_gain[np.ix_(jj_rot, jj_rot)] += cas.dot(local_correction[3:, 3:]).dot(cas.T)

                # Total forces and moments
                # total forces
                polar_gain[np.ix_(jj_for_tra, jj_tra)] += cas.dot(local_correction[:3, :3]).dot(cas.T)

                # forces contribution to total moments
                polar_gain[np.ix_(jj_for_rot, jj_tra)] += \
                    algebra.skew(r_a) * cas.dot(local_correction[:3, :3]).dot(cas.T)
                # moments contribution to total moments
                polar_gain[np.ix_(jj_for_rot, jj_rot)] += \
                    cas.dot(local_correction[3:, 3:]).dot(cas.T) * cab * np.linalg.inv(Tan.T)
        return polar_gain
