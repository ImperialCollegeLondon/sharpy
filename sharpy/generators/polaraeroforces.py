import numpy as np
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.aero.utils.uvlmlib as uvlmlib
from sharpy.utils.generate_cases import get_aoacl0_from_camber


@generator_interface.generator
class PolarCorrection(generator_interface.BaseGenerator):
    """
    This generator corrects the aerodynamic forces from UVLM based on the airfoil polars provided by the user in the
    ``aero.h5`` file. Polars are entered for each airfoil, in a table comprising ``AoA (rad), CL, CD, CM``.

    This ``generator_id = 'PolarCorrection'`` and can be used in the coupled solvers through the
    ``correct_forces_method`` setting as:

    .. python::
        settings = dict()  # SHARPy settings
        settings['StaticCoupled']['correct_forces_method'] = 'PolarCorrection'
        settings['StaticCoupled']['correct_forces_settings'] = {'cd_from_cl': 'off',  # recommended settings (default)
                                                                'correct_lift': 'off',
                                                                'compute_actual_aoa': 'off',
                                                                'compute_uind': 'off'
                                                                'moment_from_polar': 'off'}

    These are the steps needed to correct the forces:

        1. The force coming from UVLM is divided into induced drag (parallel to the incoming flow velocity) and lift
          (the remaining force).

    If ``cd_from_cl == 'on'``.
        2. The viscous drag and pitching moment is found at the computed lift coefficient and the forces and
           moments updated

    Else, the angle of attack is computed based on a few options:

        If ``compute_actual_aoa == 'on'``

            2. The angle of attack is computed between the free stream velocity and the
               chord of the section. If the setting ``compute_uind == 'on'`` the vortex induced velocity (downwash) is
               found by taking the difference between the induced velocity far upstream and downstream (20 chords).

        Else, if ``compute_actual_aoa == 'off'`` (recommended):
            2. The angle of attack is computed based on that lift force and the angle of zero lift computed from the
               airfoil polar and taking the potential flow lift curve slope of :math:`2 \pi`

        3. The drag force is computed based on the angle of attack and the polars provided by the user

        4. If ``correct_lift == 'on'``, the lift coefficient is also corrected with the polar data. Else only the
           UVLM results are used.

    The pitching moment is added in a similar manner to the viscous drag. However, if ``moment_from_polar == 'on'``
    and ``correct_lift == 'on'``, the total moment (the one used for the FSI) is computed just from polar data,
    overriding any moment computed in SHARPy. That is, the moment will include the polar pitching moment, and moments
    due to lift and drag computed from the polar data.

    Note:
        Computing the induced velocity far upstream and far downstream is performance intensive and may not be a
        robust method when multiple surfaces are present downstream or upstream, as they will impact the result.
        It is recommended that the user chooses to use the Cd given by the aoa required to give the Cl with a 2pi lift
        curve slope (i.e. ``compute_actual_aoa = 'off'``)
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
                                           'calculation. (use for comparison. Not recommended).'

    settings_types['compute_actual_aoa'] = 'bool'
    settings_default['compute_actual_aoa'] = False
    settings_description['compute_actual_aoa'] = 'Compute the coefficients using the angle of attack of the ' \
                                                 'section (between the chord and the flow velocity, ' \
                                                 'as opposed to the angle necessary to provide the UVLM ' \
                                                 'calculated C_L using potential flow theory lift curve slope of 2pi '

    settings_types['moment_from_polar'] = 'bool'
    settings_default['moment_from_polar'] = False
    settings_description['moment_from_polar'] = 'If ``correct_lift`` is selected, it will compute the pitching moment ' \
                                                'simply from poalr derived data, i.e. the polars Cm and the moments' \
                                                'arising from the lift and drag (derived from the polar) contribution. ' \
                                                'Else, it will add the polar Cm to the moment already computed by ' \
                                                'SHARPy.'

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
        moment_from_polar = self.settings['moment_from_polar']

        aero_dict = aerogrid.aero_dict
        if aerogrid.polars is None:
            return struct_forces
        new_struct_forces = np.zeros_like(struct_forces)

        nnode = struct_forces.shape[0]

        # Compute induced velocities at the structural points
        cga = algebra.quat2rotation(structural_kstep.quat)
        pos_g = np.array([cga.dot(structural_kstep.pos[inode]) + np.array([0, 0, 0]) for inode in range(nnode)])

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

                airfoil_coords = aerogrid.aero_dict['airfoils'][str(aero_dict['airfoil_distribution'][ielem, inode_in_elem])]

                dir_span, span, dir_chord, chord = span_chord(i_n, aero_kstep.zeta[isurf])

                # Define the relative velocity and its direction
                urel = (structural_kstep.pos_dot[inode, :] +
                        structural_kstep.for_vel[0:3] +
                        algebra.cross3(structural_kstep.for_vel[3:6], structural_kstep.pos[inode, :]))
                urel = -np.dot(cga, urel)
                urel += np.average(aero_kstep.u_ext[isurf][:, :, i_n], axis=1)

                freestream = urel.copy()
                dir_freestream = algebra.unit_vector(freestream)
                dir_urel = algebra.unit_vector(urel)

                if compute_induced_velocity and compute_actual_aoa:
                    # TODO: for performance improvement take this out of loop
                    chords_upstream = 20
                    chords_downstream = 20
                    pitot_upstream = pos_g[inode] - chords_upstream * chord * dir_urel
                    pitot_downstream = pos_g[inode] + chords_downstream * chord * dir_urel
                    uind_pitot = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(
                        aero_kstep,
                        target_triads=np.vstack((pitot_upstream, pitot_downstream)),
                        vortex_radius=self.vortex_radius,
                        for_pos=structural_kstep.for_pos,
                        ncores=8)
                    uind_pitot_upstream = uind_pitot[0]
                    uind_pitot_downstream = uind_pitot[1]

                    uind_2 = uind_pitot_downstream - uind_pitot_upstream
                    if structure.boundary_conditions[inode] == -1:
                        uind_2[1] *= 0
                    urel += uind_2
                dir_urel = algebra.unit_vector(urel)

                # Force in the G frame of reference
                force = np.dot(cgb,
                               struct_forces[inode, 0:3])
                moment = cgb.dot(struct_forces[inode, 3:6])
                dir_force = algebra.unit_vector(force)

                # Coefficient to change from aerodynamic coefficients to forces (and viceversa)
                coef = 0.5 * rho * np.linalg.norm(urel) ** 2 * chord * span

                # Divide the force in drag and lift
                # drag_force = np.dot(force, dir_urel) * dir_urel
                drag_force = np.dot(force, dir_freestream) * dir_freestream
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
                        # ii) Compute the effective angle of attack from potential flow theory. The local lift curve
                        # slope is 2pi and the zero-lift angle of attack is given by thin airfoil theory. From this,
                        # the effective angle of attack is computed for the section and includes 3D effects.
                        aoa_0cl = get_aoacl0_from_camber(airfoil_coords[:, 0], airfoil_coords[:, 1])
                        aoa = cl / 2 / np.pi + aoa_0cl

                        # Compute the coefficients associated to that angle of attack
                        cl_polar, cd, cm = polar.get_coefs(aoa)

                    if correct_lift:
                        # Use polar generated CL rather than UVLM computed CL
                        cl = cl_polar

                # Recompute the forces based on the coefficients
                lift_force = cl * algebra.unit_vector(lift_force) * coef
                drag_force += cd * dir_urel * coef
                force = lift_force + drag_force
                new_struct_forces[inode, 0:3] = np.dot(cgb.T,
                                                       force)

                # Compute new moments
                # pitching_moment
                panel_shift = 0.25 * (aero_kstep.zeta[isurf][:, 1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n])
                ref_point = aero_kstep.zeta[isurf][:, 0, i_n] + 0.25 * chord * dir_chord - panel_shift
                dir_moment = algebra.unit_vector(algebra.cross3(algebra.unit_vector(lift_force), dir_urel))
                pitching_moment_sharpy_ref = moment.dot(dir_moment) * algebra.unit_vector(dir_moment)

                # modify moment:
                # add viscous moment or full moment
                moment_polar = cm * dir_moment * coef * chord
                moment += moment_polar

                # moment due to drag
                moment_polar_drag = algebra.cross3(
                    ref_point - pos_g[inode], cd * dir_urel * coef
                )
                moment += moment_polar_drag

                # moment due to lift (if corrected)
                if correct_lift and moment_from_polar:
                    # add moment from scratch: cm_polar + cm_drag_polar + cl_lift_polar
                    moment = moment_polar
                    moment += moment_polar_drag
                    moment_polar_lift = algebra.cross3(
                        ref_point - pos_g[inode], lift_force
                    )
                    moment += moment_polar_lift

                new_struct_forces[inode, 3:6] = cgb.T.dot(moment)

        return new_struct_forces


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
class EfficiencyCorrection(generator_interface.BaseGenerator):
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

