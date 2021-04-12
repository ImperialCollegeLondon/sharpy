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

        nnode = struct_forces.shape[0]

        # Compute induced velocities at the structural points
        cga = algebra.quat2rotation(structural_kstep.quat)
        pos_g = np.array([cga.dot(structural_kstep.pos[inode]) + np.array([0, 0, 0]) for inode in range(nnode)])
        target_test = np.zeros((100, 3))
        target_test[:, 0] = np.linspace(-50, 50, 100)
        # uind_test = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(aero_kstep,
        #                                                                target_triads=target_test,
        #                                                                vortex_radius=self.vortex_radius,
        #                                                                for_pos=structural_kstep.for_pos,
        #                                                                ncores=8)
        # np.savetxt('./uind_midspan_finite.txt', np.column_stack((target_test, uind_test)))
        # if compute_induced_velocity:
        #     pos_pitot_upstream = np.zeros((nnode, 3))
        #     pos_pitot_downstream = np.zeros((nnode, 3))
        #     pos_pitot_upstream += (pos_g + np.array([-25, 0, 0]))
        #     pos_pitot_downstream += (pos_g + np.array([45, 0, 0]))
        #     uind_pitot = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(aero_kstep,
        #                                                                    target_triads=np.vstack((pos_pitot_upstream,
        #                                                                                             pos_pitot_downstream)),
        #                                                                    vortex_radius=self.vortex_radius,
        #                                                                    for_pos=structural_kstep.for_pos,
        #                                                                    ncores=8)
        #     uind_pitot_upstream = uind_pitot[:nnode]
        #     uind_pitot_downstream = uind_pitot[nnode:]
        #
        #     uind_2 = uind_pitot_downstream - uind_pitot_upstream
        #     uind_2[:, 1] *= 0  # or where bocos = -1
        #
        # else:
        #     uind = np.zeros_like(structural_kstep.pos)

        for inode in range(nnode):
            new_struct_forces[inode, :] = struct_forces[inode, :].copy()
            print('i_node', inode)
            if aero_dict['aero_node'][inode]:
                ielem, inode_in_elem = structure.node_master_elem[inode]
                iairfoil = aero_dict['airfoil_distribution'][ielem, inode_in_elem]
                isurf = aerogrid.struct2aero_mapping[inode][0]['i_surf']
                i_n = aerogrid.struct2aero_mapping[inode][0]['i_n']
                N = aerogrid.aero_dimensions[isurf, 1]
                polar = aerogrid.polars[iairfoil]
                cab = algebra.crv2rotation(structural_kstep.psi[ielem, inode_in_elem, :])
                cgb = np.dot(cga, cab)

                dir_span, span, dir_chord, chord = span_chord(i_n, aero_kstep.zeta[isurf])
                print('Chord', dir_chord)

                # Define the relative velocity and its direction
                urel = (structural_kstep.pos_dot[inode, :] +
                        structural_kstep.for_vel[0:3] +
                        np.cross(structural_kstep.for_vel[3:6],
                                 structural_kstep.pos[inode, :]))
                urel = -np.dot(cga, urel)
                urel += np.average(aero_kstep.u_ext[isurf][:, :, i_n], axis=1)

                freestream = urel.copy()
                dir_freestream = algebra.unit_vector(freestream)
                print('Freestream', freestream)
                print('Node location, A', structural_kstep.pos[inode, :])
                print('Node location, G', pos_g[inode, :])
                dir_urel = algebra.unit_vector(urel)
                aoa = np.arccos(dir_chord.dot(dir_urel) / np.linalg.norm(dir_urel) / np.linalg.norm(dir_chord))
                print('Geo AoA', aoa * 180 / np.pi)

                if compute_induced_velocity:
                    # TODO: for performance improvement take this out of loop
                    chords_upstream = 20
                    chords_downstream = 20
                    pitot_upstream = pos_g[inode] - chords_upstream * chord * dir_urel
                    pitot_downstream = pos_g[inode] + chords_downstream * chord * dir_urel
                    uind_pitot = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(aero_kstep,
                                                                                         target_triads=np.vstack((pitot_upstream,
                                                                                                                  pitot_downstream)),
                                                                                         vortex_radius=self.vortex_radius,
                                                                                         for_pos=structural_kstep.for_pos,
                                                                                         ncores=8)
                    uind_pitot_upstream = uind_pitot[0]
                    uind_pitot_downstream = uind_pitot[1]

                    uind_2 = uind_pitot_downstream - uind_pitot_upstream
                    if structure.boundary_conditions[inode] == -1:
                        uind_2[1] *= 0  # or where bocos = -1
                    print('Node Induced', uind_2)
                    urel += uind_2
                dir_urel = algebra.unit_vector(urel)

                print('Induced AoA', algebra.angle_between_vectors(dir_urel, algebra.unit_vector(freestream)) * 180 / np.pi)

                # Force in the G frame of reference
                force = np.dot(cgb,
                               struct_forces[inode, 0:3])
                moment = cgb.dot(struct_forces[inode, 3:6])
                dir_force = algebra.unit_vector(force)

                # Coefficient to change from aerodynamic coefficients to forces (and viceversa)
                coef = 0.5 * rho * np.linalg.norm(urel) ** 2 * chord * span

                proj_sg = change_stability_axes(dir_urel, dir_freestream).T
                # proj_sg = np.eye(3)
                print('aoa as diff z', algebra.angle_between_vectors(proj_sg.T[:, 2], [0, 0, 1]) * 180 / np.pi)
                print('dir_urel in inertial', dir_urel)
                # dir_urel = proj_sg.dot(dir_urel)  # S

                print('dir_urel in stab', dir_urel)

                # Divide the force in drag and lift
                # drag_force = np.dot(force, dir_urel) * dir_urel
                drag_force = np.dot(force, dir_freestream) * dir_freestream
                print('force_g', force / coef)
                print('drag', drag_force / coef)
                print('u_rel', urel)
                lift_force = force - drag_force

                # Compute the associated lift
                cl = np.linalg.norm(lift_force) / coef
                cd_sharpy = np.linalg.norm(drag_force) / coef
                print('cl sharpy', cl)
                print('cd sharpy', cd_sharpy)

                if cd_from_cl:
                    # Compute the drag from the UVLM computed lift
                    cd, cm = polar.get_cdcm_from_cl(cl)

                else:
                    # Compute L, D, M from polar depending on:
                    if compute_actual_aoa:
                        # i) Compute the actual aoa given the induced velocity
                        aoa = np.arccos(dir_chord.dot(dir_urel) / np.linalg.norm(dir_urel) / np.linalg.norm(dir_chord))
                        print('Actual AoA', aoa * 180 / np.pi)
                        # print('AoA via Induced', aoa_section * 180 / np.pi)
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
                print('lift_vec_dir', algebra.unit_vector(lift_force))
                print('drag_vec_dir', algebra.unit_vector(dir_urel))
                print('new_drag', drag_force / coef)
                print('cd_polar only', cd)
                force = lift_force + drag_force
                print('force_stab', force / coef)
                force_g = proj_sg.T.dot(force)
                print('force_inertial', force_g / coef)
                new_struct_forces[inode, 0:3] = np.dot(cgb.T,
                                                       force)

                # Compute new moments
                # pitching_moment
                print('Leading Edge (G)', aero_kstep.zeta[isurf][:, 0, i_n])
                panel_shift = 0.25 * (aero_kstep.zeta[isurf][:, 1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n])
                ref_point = aero_kstep.zeta[isurf][:, 0, i_n] + 0.25 * chord * dir_chord - panel_shift
                print('Panel shift', panel_shift)
                print('QC', 0.25 * chord * dir_chord)
                print('Quarter chord (G)', ref_point)
                print('LEN', np.linalg.norm(ref_point - aero_kstep.zeta[isurf][:, 0, i_n]))
                dir_moment = algebra.unit_vector(algebra.cross3(algebra.unit_vector(lift_force), dir_urel))
                pitching_moment_sharpy_ref = moment.dot(dir_moment) * algebra.unit_vector(dir_moment)
                print('Pitching moment at SHARPy ref', pitching_moment_sharpy_ref / coef / chord)

                # modify moment:
                # add viscous moment or full moment
                # add moment at sharpy reference - need diference between ref and ea
                # option to define ref point
                moment_polar = cm * dir_moment * coef * chord
                moment = moment_polar

                # moment due to drag
                moment_polar_drag = algebra.cross3(
                    ref_point - pos_g[inode], cd * dir_urel * coef
                )
                moment += moment_polar_drag
                print('Moment due to drag', moment_polar_drag / coef / chord)

                # moment due to lift (if corrected)
                if correct_lift:
                    moment = moment_polar
                    moment += moment_polar_drag
                    moment_polar_lift = algebra.cross3(
                        ref_point - pos_g[inode], lift_force
                    )
                    moment += moment_polar_lift
                    print('Moment due to lift', moment_polar_lift / coef / chord)

                new_struct_forces[inode, 3:6] = cgb.T.dot(moment)


        return new_struct_forces

def change_stability_axes(u_new, u_freestream):
    # CHECK THIS
    for_g = np.eye(3)

    xs = algebra.unit_vector(u_new)
    # ys = algebra.unit_vector(algebra.cross3(for_g[:, 2], xs))
    ys = for_g[:, 1]
    zs = algebra.unit_vector(algebra.cross3(xs, ys))

    return algebra.triad2rotation(xs, ys, zs)


# def sectional_coefficients(aero,
#                            structure,
#                            aero_kstep,
#                            structural_kstep,
#                            struct_forces,
#                            rho,
#                            compute_induced_velocity=False,
#                            vortex_radius=1e-6):
#
#     aero_dict = aero.aero_dict
#     new_struct_forces = np.zeros_like(struct_forces)
#
#     nnode = struct_forces.shape[0]
#
#     coefficients = np.zeros((nnode, 3))
#     factors = np.zeros((nnode, 3))
#
#     # Compute induced velocities at the structural points
#     cga = algebra.quat2rotation(structural_kstep.quat)
#     pos_g = np.array([cga.dot(structural_kstep.pos[inode]) + np.array([0, 0, 0]) for inode in range(nnode)])
#     if compute_induced_velocity:
#         print(pos_g)
#         uind = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(aero_kstep,
#                                                                        target_triads=pos_g,
#                                                                        vortex_radius=vortex_radius,
#                                                                        for_pos=structural_kstep.for_pos,
#                                                                        ncores=8)
#
#         for surf in aero_kstep.zeta:
#             points = np.zeros((np.prod(surf.shape[1:]), 3))
#             print(points.shape)
#             print(surf.shape)
#             ip = 0
#             for i_m in range(surf.shape[2]):
#                 for i_n in range(surf.shape[1]):
#                     # print('points', i_m, i_n)
#                     # print('Coords', surf[:, i_m, i_n])
#                     points[ip, :] = surf[:, i_n, i_m]
#                     ip += 1
#             uind_zeta = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(aero_kstep,
#                                                                                 target_triads=points,
#                                                                                 vortex_radius=1e-6,
#                                                                                 for_pos=structural_kstep.for_pos,
#                                                                                 ncores=8)
#             print('Induced Velocity')
#             for i in range(points.shape[0]):
#                 print(f'Coord (x, y, z)G {points[i]} \t u_ind (x, y, z)G {uind_zeta[i]}')
#
#         print('AoA')
#         aero_kstep.postproc_cell['incidence_angle'] = []
#         for isurf in range(aero_kstep.n_surf):
#             aero_kstep.postproc_cell['incidence_angle'].append(
#                 np.zeros_like(aero_kstep.gamma[isurf]))
#         uvlmlib.uvlm_calculate_incidence_angle(aero_kstep, structural_kstep)
#
#
#     else:
#         uind = np.zeros_like(structural_kstep.pos)
#
#     for inode in range(nnode):
#         new_struct_forces[inode, :] = struct_forces[inode, :].copy()
#         print('i_node', inode)
#         if aero_dict['aero_node'][inode]:
#             ielem, inode_in_elem = structure.node_master_elem[inode]
#             iairfoil = aero_dict['airfoil_distribution'][ielem, inode_in_elem]
#             isurf = aero.struct2aero_mapping[inode][0]['i_surf']
#             i_n = aero.struct2aero_mapping[inode][0]['i_n']
#             N = aero.aero_dimensions[isurf, 1]
#             cab = algebra.crv2rotation(structural_kstep.psi[ielem, inode_in_elem, :])
#             cgb = np.dot(cga, cab)
#
#             print('i_n', i_n)
#             print(aero_kstep.postproc_cell['incidence_angle'][isurf].shape)
#             aoa_section = np.average(aero_kstep.postproc_cell['incidence_angle'][isurf][:, 0])
#
#             # Deal with the extremes
#             if i_n == 0:
#                 node1 = 0
#                 node2 = 1
#                 aoa_section = np.average(aero_kstep.postproc_cell['incidence_angle'][isurf][:, 0])
#             elif i_n == N:
#                 node1 = nnode - 1
#                 node2 = nnode - 2
#                 aoa_section = np.average(aero_kstep.postproc_cell['incidence_angle'][isurf][:, N - 1])
#             else:
#                 node1 = inode + 1
#                 node2 = inode - 1
#                 aoa_section = np.average(
#                     np.concatenate((aero_kstep.postproc_cell['incidence_angle'][isurf][:, i_n],
#                                     aero_kstep.postproc_cell['incidence_angle'][isurf][:, i_n - 1])))
#
#             # Define the span and the span direction
#             dir_span = 0.5 * np.dot(cga,
#                                     structural_kstep.pos[node1, :] - structural_kstep.pos[node2, :])
#             span = np.linalg.norm(dir_span)
#             dir_span = algebra.unit_vector(dir_span)
#
#             # Define the chord and the chord direction
#             dir_chord = aero_kstep.zeta[isurf][:, -1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n]
#             chord = np.linalg.norm(dir_chord)
#             dir_chord = algebra.unit_vector(dir_chord)
#
#             dir_span, span, dir_chord, chord = span_chord(inode, i_n, cga, aero_kstep.zeta[isurf], structural_kstep.pos)
#             # Define the relative velocity and its direction
#             urel = (structural_kstep.pos_dot[inode, :] +
#                     structural_kstep.for_vel[0:3] +
#                     np.cross(structural_kstep.for_vel[3:6],
#                              structural_kstep.pos[inode, :]))
#             urel = -np.dot(cga, urel)
#             urel += np.average(aero_kstep.u_ext[isurf][:, :, i_n], axis=1)
#
#             freestream = urel.copy()
#             print('Freestream', freestream)
#             print('Node location, A', structural_kstep.pos[inode, :])
#             print('Node location, G', pos_g[inode, :])
#             dir_urel = algebra.unit_vector(urel)
#             aoa = np.arccos(dir_chord.dot(dir_urel) / np.linalg.norm(dir_urel) / np.linalg.norm(dir_chord))
#             print('Geo AoA', aoa * 180 / np.pi)
#             urel += uind[inode, :]  # u_ind will be zero if compute_uind == False
#             dir_urel = algebra.unit_vector(urel)
#
#             print('Induced AoA', algebra.angle_between_vectors(dir_urel, algebra.unit_vector(freestream)) * 180 / np.pi)
#
#             # Force in the G frame of reference
#             force = np.dot(cgb,
#                            struct_forces[inode, 0:3])
#             moment = cgb.dot(struct_forces[inode, 3:6])
#             dir_force = algebra.unit_vector(force)
#
#             # Coefficient to change from aerodynamic coefficients to forces (and viceversa)
#             coef = 0.5 * rho * np.linalg.norm(urel) ** 2 * chord * span
#
#             # proj_sg = change_stability_axes(dir_urel, [1, 0, 0]).T
#             proj_sg = np.eye(3)
#             print('aoa as diff z', algebra.angle_between_vectors(proj_sg.T[:, 2], [0, 0, 1]) * 180 / np.pi)
#             print('dir_urel in inertial', dir_urel)
#             # dir_urel = proj_sg.dot(dir_urel)  # S
#
#             print('dir_urel in stab', dir_urel)
#
#             # Divide the force in drag and lift
#             drag_force = np.dot(force, dir_urel) * dir_urel
#             print('force_g', force / coef)
#             print('drag', drag_force / coef)
#             print('u_ind', uind[inode])
#             print('u_rel', urel)
#             lift_force = force - drag_force
#
#             # Compute the associated lift
#             cl_sharpy = np.linalg.norm(lift_force) / coef
#             cd_sharpy = np.linalg.norm(drag_force) / coef
#             print('cl sharpy', cl_sharpy)
#             print('cd sharpy', cd_sharpy)
#
#
#             # Compute new moments
#             # pitching_moment
#             print('Leading Edge (G)', aero_kstep.zeta[isurf][:, 0, i_n])
#             panel_shift = 0.25 * (aero_kstep.zeta[isurf][:, 1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n])
#             ref_point = aero_kstep.zeta[isurf][:, 0, i_n] + 0.25 * chord * dir_chord - panel_shift
#             print('Quarter chord (G)', ref_point)
#             dir_moment = algebra.unit_vector(algebra.cross3(algebra.unit_vector(lift_force), dir_urel))
#             pitching_moment_sharpy_ref = moment.dot(dir_moment) * algebra.unit_vector(dir_moment)
#             print('Pitching moment at SHARPy ref', pitching_moment_sharpy_ref / coef / chord)
#             cm_sharpy = pitching_moment_sharpy_ref / coef / chord
#             coefficients[inode] = np.array([cl_sharpy, cd_sharpy, cm_sharpy])
#             factors[inode] = np.array([coef, coef, coef * chord])
#
#     return coefficients
#

def span_chord(i_node_surf, zeta):
    """
    Retrieve the local span and local chord

    Args:
        i_node_surf (int): Node index in aerodynamic surface
        zeta (np.array): Aerodynamic surface coordinates ``(3 x n_chord x m_span)``

    Returns:
        tuple: ``dir_span``, ``span``, ``dir_chord``, ``chord``
    """
    N = zeta.shape[2] - 1 # spanwise vertices in surface (-1 for index)

    # Deal with the extremes
    if i_node_surf == 0:
        node_p = 1
        node_m = 0
    elif i_node_surf == N:
        node_p = N
        node_m = N - 1
    else:
        node_p = i_node_surf + 1
        node_m = i_node_surf - 1

    # Define the span and the span direction
    dir_span = 0.5 * (zeta[:, 0, node_p] - zeta[:, 0, node_m])

    span = np.linalg.norm(dir_span)
    dir_span = algebra.unit_vector(dir_span)

    # Define the chord and the chord direction
    dir_chord = zeta[:, -1, i_node_surf] - zeta[:, 0, i_node_surf]
    chord = np.linalg.norm(dir_chord)
    dir_chord = algebra.unit_vector(dir_chord)

    return dir_span, span, dir_chord, chord


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

if __name__ == '__main__':
    import unittest

    class TestStabilityAxes(unittest.TestCase):

        def test_transform(self):

            u_infty = np.array([10, 0. , 0])

            # beta = 0.1 * np.pi / 180
            # alpha = -1 * np.pi / 180
            # induced_velocity = np.array([-0.5, 0.5, -0.5]) * 1e-3  # make small on purpose so modulus is mostly constant
            induced_velocity = np.array([0, -0.5, -1])  # make small on purpose so modulus is mostly constant

            rot_sg = change_stability_axes(u_infty + induced_velocity, u_infty)

            print(rot_sg)

            for_g = np.eye(3)
            for i in range(3):
                vec = rot_sg.dot(for_g[:, i])
                print(f'vec {i} = ', vec)


            cgs = rot_sg.T


    unittest.main()