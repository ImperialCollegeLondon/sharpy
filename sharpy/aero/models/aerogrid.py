"""Aerogrid

Aerogrid contains all the necessary routines to generate an aerodynamic
grid based on the input dictionaries.
"""
import ctypes as ct
import warnings

import numpy as np
import scipy.interpolate

import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout
from sharpy.utils.datastructures import AeroTimeStepInfo
import sharpy.utils.generator_interface as gen_interface

from sharpy.aero.models.grid import Grid


class Aerogrid(Grid):
    """
    ``Aerogrid`` is the main object containing information of the grid of panels

    It is created by the solver :class:`sharpy.solvers.aerogridloader.AerogridLoader`

    """
    def __init__(self):
        super().__init__()
        self.dimensions_star = None
        self.airfoil_db = dict()

        self.grid_type = "aero"
        self.n_control_surfaces = 0

        self.cs_generators = []

        self.initial_strip_z_rot = None

    def generate(self, data_dict, beam, settings, ts):
        super().generate(data_dict, beam, settings, ts)

        # write grid info to screen
        self.output_info()

        # allocating initial grid storage
        self.ini_info = AeroTimeStepInfo(self.dimensions,
                                         self.dimensions_star)

        # Initial panel orientation, used when aligned grid is off
        self.initial_strip_z_rot = np.zeros([self.n_elem, 3]) 
        if not settings['aligned_grid'] and settings['initial_align']:
            for i_elem in range(self.n_elem):
                for i_local_node in range(3):
                    Cab = algebra.crv2rotation(beam.ini_info.psi[i_elem, i_local_node, :])
                    self.initial_strip_z_rot[i_elem, i_local_node] = \
                        algebra.angle_between_vectors_sign(settings['freestream_dir'], Cab[:, 1], Cab[:, 2])

        # load airfoils db
        # for i_node in range(self.n_node):
        for i_elem in range(self.n_elem):
            for i_local_node in range(self.beam.num_node_elem):
                try:
                    self.airfoil_db[self.data_dict['airfoil_distribution'][i_elem, i_local_node]]
                except KeyError:
                    airfoil_coords = self.data_dict['airfoils'][str(self.data_dict['airfoil_distribution'][i_elem, i_local_node])]
                    self.airfoil_db[self.data_dict['airfoil_distribution'][i_elem, i_local_node]] = (
                        scipy.interpolate.interp1d(airfoil_coords[:, 0],
                                                   airfoil_coords[:, 1],
                                                   kind='quadratic',
                                                   copy=False,
                                                   fill_value='extrapolate',
                                                   assume_sorted=True))
        try:
            self.n_control_surfaces = np.sum(np.unique(self.data_dict['control_surface']) >= 0)
        except KeyError:
            pass

        # Backward compatibility: check whether control surface deflection aero_settings have been specified. If not, create
        # section with empty list, such that no cs generator is appended
        try:
            settings['control_surface_deflection']
        except KeyError:
            settings.update({'control_surface_deflection': ['']*self.n_control_surfaces})

        # pad ctrl surfaces dict with empty strings if not defined
        if len(settings['control_surface_deflection']) != self.n_control_surfaces:
            undef_ctrl_sfcs = ['']*(self.n_control_surfaces - len(settings['control_surface_deflection']))
            settings['control_surface_deflection'].extend(undef_ctrl_sfcs)


        # initialise generators
        with_error_initialising_cs = False
        for i_cs in range(self.n_control_surfaces):
            if settings['control_surface_deflection'][i_cs] == '':
                self.cs_generators.append(None)
            else:
                cout.cout_wrap('Initialising Control Surface {:g} generator'.format(i_cs), 1)
                # check that the control surface is not static
                if self.data_dict['control_surface_type'][i_cs] == 0:
                    raise TypeError('Control surface {:g} is defined as static but there is a control surface generator'
                                    'associated with it'.format(i_cs))
                generator_type = gen_interface.generator_from_string(
                    settings['control_surface_deflection'][i_cs])
                self.cs_generators.append(generator_type())
                try:
                    self.cs_generators[i_cs].initialise(
                        settings['control_surface_deflection_generator_settings'][str(i_cs)])
                except KeyError:
                    with_error_initialising_cs = True
                    cout.cout_wrap('Error, unable to locate a settings dictionary for control surface '
                                   '{:g}'.format(i_cs), 4)

        if with_error_initialising_cs:
            raise KeyError('Unable to locate settings for at least one control surface.')


        self.add_timestep()
        self.generate_mapping()
        self.generate_zeta(self.beam, self.aero_settings, ts)

        if 'polars' in self.data_dict:
            import sharpy.aero.utils.airfoilpolars as ap
            self.polars = []
            nairfoils = np.amax(self.data_dict['airfoil_distribution']) + 1
            for iairfoil in range(nairfoils):
                new_polar = ap.Polar()
                new_polar.initialise(data_dict['polars'][str(iairfoil)])
                self.polars.append(new_polar)

    def output_info(self):
        cout.cout_wrap('The aerodynamic grid contains %u surfaces' % self.n_surf, 1)
        for i_surf in range(self.n_surf):
            cout.cout_wrap('  Surface %u, M=%u, N=%u' % (i_surf,
                                                         self.dimensions[i_surf, 0],
                                                         self.dimensions[i_surf, 1]), 1)
            cout.cout_wrap('     Wake %u, M=%u, N=%u' % (i_surf,
                                                         self.dimensions_star[i_surf, 0],
                                                         self.dimensions_star[i_surf, 1]))
        cout.cout_wrap('  In total: %u bound panels' % (sum(self.dimensions[:, 0]*
                                                            self.dimensions[:, 1])))
        cout.cout_wrap('  In total: %u wake panels' % (sum(self.dimensions_star[:, 0]*
                                                           self.dimensions_star[:, 1])))
        cout.cout_wrap('  Total number of panels = %u' % (sum(self.dimensions[:, 0]*
                                                              self.dimensions[:, 1]) +
                                                          sum(self.dimensions_star[:, 0]*
                                                              self.dimensions_star[:, 1])))

    def calculate_dimensions(self):
        super().calculate_dimensions()

        self.dimensions_star = self.dimensions.copy()
        self.dimensions_star[:, 0] = self.aero_settings['mstar']

    def generate_zeta_timestep_info(self, structure_tstep, aero_tstep, beam, settings, it=None, dt=None):
        if it is None:
            it = len(beam.timestep_info) - 1
        global_node_in_surface = []
        for i_surf in range(self.n_surf):
            global_node_in_surface.append([])

        # check that we have control surface information
        try:
            self.data_dict['control_surface']
            with_control_surfaces = True
        except KeyError:
            with_control_surfaces = False

        # check that we have sweep information
        try:
            self.data_dict['sweep']
        except KeyError:
            self.data_dict['sweep'] = np.zeros_like(self.data_dict['twist'])

        # Define first_twist for backwards compatibility
        if 'first_twist' not in self.data_dict:     
            self.data_dict['first_twist'] = [True]*self.data_dict['surface_m'].shape[0]
        
        # one surface per element
        for i_elem in range(self.n_elem):
            i_surf = self.data_dict['surface_distribution'][i_elem]
            # check if we have to generate a surface here
            if i_surf == -1:
                continue

            for i_local_node in range(len(self.beam.elements[i_elem].global_connectivities)):
                i_global_node = self.beam.elements[i_elem].global_connectivities[i_local_node]
                # i_global_node = self.beam.elements[i_elem].global_connectivities[
                #     self.beam.elements[i_elem].ordering[i_local_node]]
                if not self.data_dict['aero_node'][i_global_node]:
                    continue
                if i_global_node in global_node_in_surface[i_surf]:
                    continue
                else:
                    global_node_in_surface[i_surf].append(i_global_node)

                # master_elem, master_elem_node = beam.master[i_elem, i_local_node, :]
                # if master_elem < 0:
                    # master_elem = i_elem
                    # master_elem_node = i_local_node

                # find the i_surf and i_n data from the mapping
                i_n = -1
                ii_surf = -1
                for i in range(len(self.struct2aero_mapping[i_global_node])):
                    i_n = self.struct2aero_mapping[i_global_node][i]['i_n']
                    ii_surf = self.struct2aero_mapping[i_global_node][i]['i_surf']
                    if ii_surf == i_surf:
                        break
                # make sure it found it
                if i_n == -1 or ii_surf == -1:
                    raise AssertionError('Error 12958: Something failed with the mapping in aerogrid.py. Check/report!')

                # control surface implementation
                control_surface_info = None
                if with_control_surfaces:
                # 1) check that this node and elem have a control surface
                    if self.data_dict['control_surface'][i_elem, i_local_node] >= 0:
                        i_control_surface = self.data_dict['control_surface'][i_elem, i_local_node]
                # 2) type of control surface + write info
                        control_surface_info = dict()
                        if self.data_dict['control_surface_type'][i_control_surface] == 0:
                            control_surface_info['type'] = 'static'
                            control_surface_info['deflection'] = self.data_dict['control_surface_deflection'][i_control_surface]
                            control_surface_info['chord'] = self.data_dict['control_surface_chord'][i_control_surface]
                            try:
                                control_surface_info['hinge_coords'] = self.data_dict['control_surface_hinge_coords'][i_control_surface]
                            except KeyError:
                                control_surface_info['hinge_coords'] = None
                        elif self.data_dict['control_surface_type'][i_control_surface] == 1:
                            control_surface_info['type'] = 'dynamic'
                            control_surface_info['chord'] = self.data_dict['control_surface_chord'][i_control_surface]
                            try:
                                control_surface_info['hinge_coords'] = self.data_dict['control_surface_hinge_coords'][i_control_surface]
                            except KeyError:
                                control_surface_info['hinge_coords'] = None

                            params = {'it': it}
                            control_surface_info['deflection'], control_surface_info['deflection_dot'] = \
                                self.cs_generators[i_control_surface](params)

                        elif self.data_dict['control_surface_type'][i_control_surface] == 2:
                            control_surface_info['type'] = 'controlled'

                            try:
                                old_deflection = self.data.aero.timestep_info[-1].control_surface_deflection[i_control_surface]
                            except AttributeError:
                                try:
                                    old_deflection = aero_tstep.control_surface_deflection[i_control_surface]
                                except IndexError:
                                    old_deflection = self.data_dict['control_surface_deflection'][i_control_surface]

                            try:
                                control_surface_info['deflection'] = aero_tstep.control_surface_deflection[i_control_surface]
                            except IndexError:
                                control_surface_info['deflection'] = self.data_dict['control_surface_deflection'][i_control_surface]

                            if dt is not None:
                                control_surface_info['deflection_dot'] = (
                                        (control_surface_info['deflection'] - old_deflection)/dt)
                            else:
                                control_surface_info['deflection_dot'] = 0.0

                            control_surface_info['chord'] = self.data_dict['control_surface_chord'][i_control_surface]

                            try:
                                control_surface_info['hinge_coords'] = self.data_dict['control_surface_hinge_coords'][i_control_surface]
                            except KeyError:
                                control_surface_info['hinge_coords'] = None
                        else:
                            raise NotImplementedError(str(self.data_dict['control_surface_type'][i_control_surface]) +
                                ' control surfaces are not yet implemented')



                node_info = dict()
                node_info['i_node'] = i_global_node
                node_info['i_local_node'] = i_local_node
                node_info['chord'] = self.data_dict['chord'][i_elem, i_local_node]
                node_info['eaxis'] = self.data_dict['elastic_axis'][i_elem, i_local_node]
                node_info['twist'] = self.data_dict['twist'][i_elem, i_local_node]
                node_info['sweep'] = self.data_dict['sweep'][i_elem, i_local_node]
                node_info['M'] = self.dimensions[i_surf, 0]
                node_info['M_distribution'] = self.data_dict['m_distribution'].decode('ascii')
                node_info['airfoil'] = self.data_dict['airfoil_distribution'][i_elem, i_local_node]
                node_info['control_surface'] = control_surface_info
                node_info['beam_coord'] = structure_tstep.pos[i_global_node, :]
                node_info['pos_dot'] = structure_tstep.pos_dot[i_global_node, :]
                node_info['beam_psi'] = structure_tstep.psi[i_elem, i_local_node, :]
                node_info['psi_dot'] = structure_tstep.psi_dot[i_elem, i_local_node, :]
                node_info['for_delta'] = beam.frame_of_reference_delta[i_elem, i_local_node, :]
                node_info['elem'] = beam.elements[i_elem]
                node_info['for_pos'] = structure_tstep.for_pos
                node_info['cga'] = structure_tstep.cga()
                if node_info['M_distribution'].lower() == 'user_defined':
                    ielem_in_surf = i_elem - np.sum(self.surface_distribution < i_surf)
                    node_info['user_defined_m_distribution'] = self.data_dict['user_defined_m_distribution'][str(i_surf)][:, ielem_in_surf, i_local_node]
                (aero_tstep.zeta[i_surf][:, :, i_n],
                 aero_tstep.zeta_dot[i_surf][:, :, i_n]) = (
                    generate_strip(node_info,
                                   self.airfoil_db,
                                   self.aero_settings['aligned_grid'],
                                   initial_strip_z_rot=self.initial_strip_z_rot[i_elem, i_local_node],
                                   orientation_in=self.aero_settings['freestream_dir'],
                                   calculate_zeta_dot=True))
        # set junction boundary conditions for later phantom cell creation in UVLM
        if "junction_boundary_condition" in self.data_dict:
            if np.any(self.data_dict["junction_boundary_condition"] >= 0):
                self.generate_phantom_panels_at_junction(aero_tstep)


    def generate_phantom_panels_at_junction(self, aero_tstep):
        for i_surf in range(self.n_surf):
                aero_tstep.flag_zeta_phantom[0, i_surf] = self.data_dict["junction_boundary_condition"][0,i_surf] 

    @staticmethod
    def compute_gamma_dot(dt, tstep, previous_tsteps):
        r"""
        Computes the temporal derivative of circulation (gamma) using finite differences.

        It will use a first order approximation for the first evaluation
        (when ``len(previous_tsteps) == 1``), and then second order ones.

        .. math:: \left.\frac{d\Gamma}{dt}\right|^n \approx \lim_{\Delta t \rightarrow 0}\frac{\Gamma^n-\Gamma^{n-1}}{\Delta t}

        For the second time step and onwards, the following second order approximation is used:

        .. math:: \left.\frac{d\Gamma}{dt}\right|^n \approx \lim_{\Delta t \rightarrow 0}\frac{3\Gamma^n -4\Gamma^{n-1}+\Gamma^{n-2}}{2\Delta t}

        Args:
            dt (float): delta time for the finite differences
            tstep (AeroTimeStepInfo): tstep at time n (current)
            previous_tsteps (list(AeroTimeStepInfo)): previous tstep structure in order: ``[n-N,..., n-2, n-1]``

        Returns:
            float: first derivative of circulation with respect to time

        See Also:
            .. py:class:: sharpy.utils.datastructures.AeroTimeStepInfo
        """
        # Check whether the iteration is part of FSI (ie the input is a k-step) or whether it is an only aerodynamic
        # simulation
        part_of_fsi = True
        try:
            if tstep is previous_tsteps[-1]:
                part_of_fsi = False
        except IndexError:
            for i_surf in range(tstep.n_surf):
                tstep.gamma_dot[i_surf].fill(0.0)
            return

        if len(previous_tsteps) == 0:
            for i_surf in range(tstep.n_surf):
                tstep.gamma_dot[i_surf].fill(0.0)
        # elif len(previous_tsteps) == 1:
            # # first order
            # # f'(n) = (f(n) - f(n - 1))/dx
            # for i_surf in range(tstep.n_surf):
                # tstep.gamma_dot[i_surf] = (tstep.gamma[i_surf] - previous_tsteps[-1].gamma[i_surf])/dt
        # else:
            # # second order
            # for i_surf in range(tstep.n_surf):
                # if (not np.isfinite(tstep.gamma[i_surf]).any()) or \
                    # (not np.isfinite(previous_tsteps[-1].gamma[i_surf]).any()) or \
                        # (not np.isfinite(previous_tsteps[-2].gamma[i_surf]).any()):
                    # raise ArithmeticError('NaN found in gamma')

                # if part_of_fsi:
                    # tstep.gamma_dot[i_surf] = (3.0*tstep.gamma[i_surf]
                                               # - 4.0*previous_tsteps[-1].gamma[i_surf]
                                               # + previous_tsteps[-2].gamma[i_surf])/(2.0*dt)
                # else:
                    # tstep.gamma_dot[i_surf] = (3.0*tstep.gamma[i_surf]
                                               # - 4.0*previous_tsteps[-2].gamma[i_surf]
                                               # + previous_tsteps[-3].gamma[i_surf])/(2.0*dt)
        if part_of_fsi:
            for i_surf in range(tstep.n_surf):
                tstep.gamma_dot[i_surf] = (tstep.gamma[i_surf] - previous_tsteps[-1].gamma[i_surf])/dt
        else:
            for i_surf in range(tstep.n_surf):
                tstep.gamma_dot[i_surf] = (tstep.gamma[i_surf] - previous_tsteps[-2].gamma[i_surf])/dt



def generate_strip(node_info, airfoil_db, aligned_grid,
                   initial_strip_z_rot,
                   orientation_in=np.array([1, 0, 0]),
                   calculate_zeta_dot = False,
                   first_twist=True):
    """
    Returns a strip of panels in ``A`` frame of reference, it has to be then rotated to
    simulate angles of attack, etc
    """
    strip_coordinates_a_frame = np.zeros((3, node_info['M'] + 1), dtype=ct.c_double)
    strip_coordinates_b_frame = np.zeros((3, node_info['M'] + 1), dtype=ct.c_double)
    zeta_dot_a_frame = np.zeros((3, node_info['M'] + 1), dtype=ct.c_double)

    # airfoil coordinates
    # we are going to store everything in the x-z plane of the b
    # FoR, so that the transformation Cab rotates everything in place.
    if node_info['M_distribution'] == 'uniform':
        strip_coordinates_b_frame[1, :] = np.linspace(0.0, 1.0, node_info['M'] + 1)
    elif node_info['M_distribution'] == '1-cos':
        domain = np.linspace(0, 1.0, node_info['M'] + 1)
        strip_coordinates_b_frame[1, :] = 0.5*(1.0 - np.cos(domain*np.pi))
    elif node_info['M_distribution'].lower() == 'user_defined':
        strip_coordinates_b_frame[1,:] = node_info['user_defined_m_distribution']
    else:
        raise NotImplemented('M_distribution is ' + node_info['M_distribution'] +
                             ' and it is not yet supported')
    strip_coordinates_b_frame[2, :] = airfoil_db[node_info['airfoil']](
                                            strip_coordinates_b_frame[1, :])

    # elastic axis correction
    for i_M in range(node_info['M'] + 1):
        strip_coordinates_b_frame[1, i_M] -= node_info['eaxis']

    # chord_line_b_frame = strip_coordinates_b_frame[:, -1] - strip_coordinates_b_frame[:, 0]
    cs_velocity = np.zeros_like(strip_coordinates_b_frame)

    # control surface deflection
    if node_info['control_surface'] is not None:
        b_frame_hinge_coords = strip_coordinates_b_frame[:, node_info['M'] - node_info['control_surface']['chord']]
        # support for different hinge location for fully articulated control surfaces
        if node_info['control_surface']['hinge_coords'] is not None:
            # make sure the hinge coordinates are only applied when M == cs_chord
            if not node_info['M'] - node_info['control_surface']['chord'] == 0:
                node_info['control_surface']['hinge_coords'] = None
            else:
                b_frame_hinge_coords =  node_info['control_surface']['hinge_coords']

        for i_M in range(node_info['M'] - node_info['control_surface']['chord'], node_info['M'] + 1):
            relative_coords = strip_coordinates_b_frame[:, i_M] - b_frame_hinge_coords
            # rotate the control surface
            relative_coords = np.dot(algebra.rotation3d_x(-node_info['control_surface']['deflection']),
                                     relative_coords)
            # deflection velocity
            try:
                cs_velocity[:, i_M] += np.cross(np.array([-node_info['control_surface']['deflection_dot'], 0.0, 0.0]),
                                            relative_coords)
            except KeyError:
                pass

            # restore coordinates
            relative_coords += b_frame_hinge_coords

            # substitute with new coordinates
            strip_coordinates_b_frame[:, i_M] = relative_coords

    # chord scaling
    strip_coordinates_b_frame *= node_info['chord']

    # twist transformation (rotation around x_b axis)
    if np.abs(node_info['twist']) > 1e-6:
        Ctwist = algebra.rotation3d_x(node_info['twist'])
    else:
        Ctwist = np.eye(3)

    # Cab transformation
    Cab = algebra.crv2rotation(node_info['beam_psi'])

    if aligned_grid:
        rot_angle = algebra.angle_between_vectors_sign(orientation_in, Cab[:, 1], Cab[:, 2])
    else:
        rot_angle = initial_strip_z_rot
    Crot = algebra.rotation3d_z(-rot_angle)

    c_sweep = np.eye(3)
    if np.abs(node_info['sweep']) > 1e-6:
        c_sweep = algebra.rotation3d_z(node_info['sweep'])

    # transformation from beam to beam prime (with sweep and twist)
    for i_M in range(node_info['M'] + 1):
        if first_twist:
            strip_coordinates_b_frame[:, i_M] = np.dot(c_sweep, np.dot(Crot,
                                                   np.dot(Ctwist, strip_coordinates_b_frame[:, i_M])))
        else:
            strip_coordinates_b_frame[:, i_M] = np.dot(Ctwist, np.dot(Crot,
                                                   np.dot(c_sweep, strip_coordinates_b_frame[:, i_M])))
        strip_coordinates_a_frame[:, i_M] = np.dot(Cab, strip_coordinates_b_frame[:, i_M])

        cs_velocity[:, i_M] = np.dot(Cab, cs_velocity[:, i_M])

    # zeta_dot
    if calculate_zeta_dot:
        # velocity due to pos_dot
        for i_M in range(node_info['M'] + 1):
            zeta_dot_a_frame[:, i_M] += node_info['pos_dot']

        # velocity due to psi_dot
        omega_a = algebra.crv_dot2omega(node_info['beam_psi'], node_info['psi_dot'])
        for i_M in range(node_info['M'] + 1):
            zeta_dot_a_frame[:, i_M] += (
                np.dot(algebra.skew(omega_a), strip_coordinates_a_frame[:, i_M]))

        # control surface deflection velocity contribution
        try:
            if node_info['control_surface'] is not None:
                node_info['control_surface']['deflection_dot']
                for i_M in range(node_info['M'] + 1):
                    zeta_dot_a_frame[:, i_M] += cs_velocity[:, i_M]
        except KeyError:
            pass

    else:
        zeta_dot_a_frame = np.zeros((3, node_info['M'] + 1), dtype=ct.c_double)

    # add node coords
    for i_M in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_M] += node_info['beam_coord']

    # add quarter-chord disp
    delta_c = (strip_coordinates_a_frame[:, -1] - strip_coordinates_a_frame[:, 0])/node_info['M']
    if node_info['M_distribution'] == 'uniform':
        for i_M in range(node_info['M'] + 1):
                strip_coordinates_a_frame[:, i_M] += 0.25*delta_c
    else:
        warnings.warn("No quarter chord disp of grid for non-uniform grid distributions implemented", UserWarning)

    # rotation from a to g
    for i_M in range(node_info['M'] + 1):
        strip_coordinates_a_frame[:, i_M] = np.dot(node_info['cga'],
                                                   strip_coordinates_a_frame[:, i_M])
        zeta_dot_a_frame[:, i_M] = np.dot(node_info['cga'],
                                          zeta_dot_a_frame[:, i_M])

    return strip_coordinates_a_frame, zeta_dot_a_frame
