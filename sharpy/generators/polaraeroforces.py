import numpy as np
import os
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
from sharpy.aero.utils.utils import magnitude_and_direction_of_relative_velocity, local_stability_axes, span_chord
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

    settings_types['skip_surfaces'] = 'list(int)'
    settings_default['skip_surfaces'] = []
    settings_description['skip_surfaces'] = 'Surfaces on which force correction is skipped.'


    settings_types['aoa_cl0'] = 'list(float)'
    settings_default['aoa_cl0'] = []
    settings_description['aoa_cl0'] = 'Angle of attack for which zero lift is achieved specified in deg for each airfoil.'
    
    settings_types['write_induced_aoa'] = 'bool'
    settings_default['write_induced_aoa'] = False
    settings_description['write_induced_aoa'] = 'Write induced aoa of each node to txt file.'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options,
                                       header_line='This generator takes in the following settings.')

    def __init__(self):
        self.settings = None

        self.aero = None
        self.structure = None
        self.rho = None
        self.vortex_radius = None
        self.n_node = None
        self.flag_node_shared_by_multiple_surfaces = None

    def initialise(self, in_dict, **kwargs):
        self.settings = in_dict
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.aero = kwargs.get('aero')
        self.structure = kwargs.get('structure')
        self.n_node = self.structure.num_node
        self.rho = kwargs.get('rho')
        self.vortex_radius = kwargs.get('vortex_radius', 1e-6)
        self.list_aoa_cl0 = self.settings['aoa_cl0']
        self.cd_from_cl = self.settings['cd_from_cl']
        self.folder = kwargs.get('output_folder') + '/aoa_induced/'

        if not self.cd_from_cl and len(self.list_aoa_cl0) == 0:
            # compute aoa for cl0 if not specified in settings
            self.compute_aoa_cl0_from_airfoil_data(self.aero)

        self.check_for_special_cases(self.aero)


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
        ts = params['ts']

        aerogrid = self.aero
        structure = self.structure
        rho = self.rho
        correct_lift = self.settings['correct_lift']
        moment_from_polar = self.settings['moment_from_polar']
        
        list_aoa_induced = []

        data_dict = aerogrid.data_dict
        if aerogrid.polars is None:
            return struct_forces
        new_struct_forces = np.zeros_like(struct_forces)

        nnode = struct_forces.shape[0]

        # Compute induced velocities at the structural points
        cga = algebra.quat2rotation(structural_kstep.quat)
        pos_g = np.array([cga.dot(structural_kstep.pos[inode]) + np.array([0, 0, 0]) for inode in range(nnode)])

        for inode in range(nnode):
            new_struct_forces[inode, :] = struct_forces[inode, :].copy()
            if data_dict['aero_node'][inode]:
                ielem, inode_in_elem = structure.node_master_elem[inode]
                iairfoil = data_dict['airfoil_distribution'][ielem, inode_in_elem]
                isurf = aerogrid.struct2aero_mapping[inode][0]['i_surf']
                if isurf not in self.settings['skip_surfaces']:
                    i_n = aerogrid.struct2aero_mapping[inode][0]['i_n']
                    polar = aerogrid.polars[iairfoil]
                    cab = algebra.crv2rotation(structural_kstep.psi[ielem, inode_in_elem, :])
                    cgb = np.dot(cga, cab)

                    if not self.cd_from_cl:
                        airfoil = str(data_dict['airfoil_distribution'][ielem, inode_in_elem])
                        aoa_0cl = self.list_aoa_cl0[int(airfoil)]

                    # computing surface area of panels contributing to force
                    dir_span, span, dir_chord, chord = span_chord(i_n, aero_kstep.zeta[isurf])
                    area = span * chord
                    area = self.correct_surface_area(inode, aerogrid.struct2aero_mapping, aero_kstep.zeta, area)
               
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
                    coef = 0.5 * rho * np.linalg.norm(urel) ** 2 * area
                    # Stability axes - projects forces in B onto S
                    c_bs = local_stability_axes(cgb.T.dot(dir_urel), cgb.T.dot(dir_chord))
                    forces_s = c_bs.T.dot(struct_forces[inode, :3])
                    moment_s = c_bs.T.dot(struct_forces[inode, 3:])
                    drag_force = forces_s[0]
                    lift_force = forces_s[2]
                    # Compute the associated lift
                    cl = np.sign(lift_force) * np.linalg.norm(lift_force) / coef
                    cd_sharpy = np.linalg.norm(drag_force) / coef

                    if self.cd_from_cl:
                        # Compute the drag from the UVLM computed lift
                        cd, cm = polar.get_cdcm_from_cl(cl)

                    else:
                        """
                        Compute L, D, M from polar depending on:
                        ii) Compute the effective angle of attack from potential flow theory or specified it as setting
                        input. The local lift curve slope is 2pi and the zero-lift angle of attack is given by thin 
                        airfoil theory or specified it as setting input. From this, the effective angle of attack is 
                        computed for the section and includes 3D effects.
                        """
                        aoa = cl / 2 / np.pi + aoa_0cl 
                        list_aoa_induced.append(aoa)
                        # Compute the coefficients associated to that angle of attack
                        cl_polar, cd, cm = polar.get_coefs(aoa)
                        
                        if correct_lift:
                            # Use polar generated CL rather than UVLM computed CL
                            cl = cl_polar

                    # Recompute the forces based on the coefficients (side force is uncorrected)
                    forces_s[0] += cd * coef  # add viscous drag to induced drag from UVLM
                    forces_s[2] = cl * coef

                    new_struct_forces[inode, 0:3] = c_bs.dot(forces_s)

                    # Pitching moment
                    # The panels are shifted by 0.25 of a panel aft from the leading edge
                    panel_shift = 0.25 * (aero_kstep.zeta[isurf][:, 1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n])
                    ref_point = aero_kstep.zeta[isurf][:, 0, i_n] + 0.25 * chord * dir_chord - panel_shift

                    # viscous contribution (pure moment)
                    moment_s[1] += cm * coef * chord

                    # moment due to drag
                    arm = cgb.T.dot(ref_point - pos_g[inode])  # in B frame
                    moment_polar_drag = algebra.cross3(c_bs.T.dot(arm), cd * dir_urel * coef)  # in S frame
                    moment_s += moment_polar_drag

                    # Pitching moment
                    if moment_from_polar:
                        # The panels are shifted by 0.25 of a panel aft from the leading edge
                        panel_shift = 0.25 * (aero_kstep.zeta[isurf][:, 1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n])
                        ref_point = aero_kstep.zeta[isurf][:, 0, i_n] + 0.25 * chord * dir_chord - panel_shift
                        new_struct_forces[inode, 3:6] = c_bs.dot(moment_s)

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

        if self.settings['write_induced_aoa']:
            self.write_induced_aoa_of_each_node(ts, list_aoa_induced)

        return new_struct_forces
    

    def correct_surface_area(self, inode, struct2aero_mapping, zeta_ts, area):
        '''
        Corrects the surface area if the structural node is shared  by multiple surfaces. 

        For example, when the wing is split into right and left wing both surfaces share the center node.
        Necessary for cl calculation as the force on the node is already the sum of the forces generated 
        at the adjacent panels of each surface.
        
        Args:
            inode (int): global node id
            struct2aero_mapping (list of dicts): maps the structural (global) nodes to aero surfaces and nodes
            zeta_ts (array): zeta of current aero timestep

        Returns:
            float: corrected surface area of other surfaces
        '''
        if self.flag_shared_node_by_surfaces[inode]:
            n_surfaces_shared_by_node = len(struct2aero_mapping[inode])
            # add area for all other surfaces connected to this node
            for isurf in range(1,n_surfaces_shared_by_node):
                shared_surf = struct2aero_mapping[inode][isurf]['i_surf']
                i_n_shared_surf = struct2aero_mapping[inode][isurf]['i_n']
                _, span_shared_surf, _, chord_shared_surf = span_chord(i_n_shared_surf, zeta_ts[shared_surf])
                area += span_shared_surf * chord_shared_surf
        return area
        
    def check_for_special_cases(self, aerogrid):
        '''
        Checks if the outboard node is shared by multiple surfaces. 

        Args:
            aerogrid :class:`~sharpy.aero.models.AerogridLoader
        '''
        # check if outboard node of aerosurface
        self.flag_shared_node_by_surfaces = np.zeros((self.n_node,1))
        for inode in range(self.n_node):
            if aerogrid.data_dict['aero_node'][inode]:
                i_n = aerogrid.struct2aero_mapping[inode][0]['i_n']                
                isurf = aerogrid.struct2aero_mapping[inode][0]['i_surf']
                N = aerogrid.dimensions[isurf, 1]
                if i_n in [0, N]:
                    if len(aerogrid.struct2aero_mapping[inode]) > 1:
                        self.flag_shared_node_by_surfaces[inode] = 1
      


    def write_induced_aoa_of_each_node(self,ts, list_aoa_induced):
        '''
        Writes induced aoa of each node to txt file for each timestep. 

        Args:
            ts (int): simulation timestep 
            list_aoa_induced (list(float)): list with induced aoa of each node
        '''
        if ts == 0 and not os.path.exists(self.folder):
            os.makedirs(self.folder)
            
        np.savetxt(self.folder + '/aoa_induced_ts_{}.txt'.format(ts),
                   np.transpose(np.transpose(np.array(list_aoa_induced))),
                   fmt='%10e',
                   delimiter=',',
                   header='aoa_induced',
                   comments='#')
                   

    def compute_aoa_cl0_from_airfoil_data(self, aerogrid):
        """
         Computes the angle of attack for which zero lift is achieved for every airfoil
        """
        self.list_aoa_cl0 = np.zeros((len(aerogrid.data_dict['airfoils']),1))
        for i, airfoil in enumerate(aerogrid.data_dict['airfoils']):
            airfoil_coords = aerogrid.data_dict['airfoils'][airfoil]
            self.list_aoa_cl0[i] = get_aoacl0_from_camber(airfoil_coords[:, 0], airfoil_coords[:, 1])

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
        data_dict = self.aero.data_dict
        new_struct_forces = np.zeros_like(struct_forces)

        # load airfoil efficiency (if it exists); else set to one (to avoid multiple ifs in the loops)
        airfoil_efficiency = data_dict['airfoil_efficiency']
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
