'''
Templates to build flying wing models
S. Maraniello, Jul 2018

classes:
- FlyingWing: generate a flying wing model from a reduced set of input. The
built in method 'update_mass_stiff' can be re-defined by the user to enter more
complex inertial/stiffness properties
- Smith(FlyingWing): generate HALE wing model
- Goland(FlyingWing): generate Goland wing model
'''

import warnings
import h5py as h5
import numpy as np
import configobj
import os
# from IPython import embed
import sharpy.utils.algebra as algebra
import sharpy.utils.geo_utils as geo_utils

np.set_printoptions(linewidth=120)


class FlyingWing():
    '''
    Flying wing template.
    - discretisation, and basic geometry/flight conditions can be passed in input
    - stiffness/mass properties must defined within the "update_mass_stiffness"
    method.
    - other aeroelastic params are attached as attributes to the class.
    - the method update_config_dict provides default setting for solver. These
    can be modified after an instance of the FlyingWing class is created.

    Args:
        M,N:            chord/span-wise discretisations
        Mstar_fact:     wake
        u_inf:          flow speed
        alpha:          FoR A pitch angle [deg]
        rho:            density
        b_ref:          geometry
        main_chord:     main chord
        aspect_ratio:
        roll=0.:        FoR A roll  angle [deg] (see RollNodes flag)
        beta=0@         FoR A side  angle [deg]
        sweep=0:        sweep angle [deg]
        n_surfaces=2:
        physical_time=2:
        route='.':       saving route
        case_name='flying_wing':
        RollNodes=False : If true, the wing nodes are rolled insted of the FoR A

    Usage:
        ws=flying_wings.FlyingWing(*args)
        ws.clean_test_files()
        ws.update_derived_params()
        ws.generate_fem_file()
        ws.generate_aero_file()
        ws.set_default_config_dict()
        ws.config['SHARPy']= bla bla bla
        ws.config.write()
    '''

    def __init__(self,
                 M, N,
                 Mstar_fact,
                 u_inf,
                 alpha,
                 rho,
                 b_ref,
                 main_chord,
                 aspect_ratio,
                 roll=0.,
                 yaw=0.,
                 beta=0,
                 sweep=0,
                 n_surfaces=2,
                 physical_time=2,
                 route='.',
                 case_name='flying_wing',
                 RollNodes=False,
                 polar_file=None):

        ### parametrisation
        assert n_surfaces < 3, "use 1 or 2 surfaces only!"
        assert N % 2 != 1, \
            'UVLM spanwise panels must be even when using 3-noded FEs!'
        self.M = M  # chord-wise panels
        self.N = N  # total spanwise panels (over all surfaces)
        self.Mstar_fact = Mstar_fact  # wake chord-wise panel factor
        self.n_surfaces = n_surfaces
        self.num_node_elem = 3

        ### store input
        self.u_inf = u_inf  # flight cond
        self.rho = rho
        self.alpha = alpha  # angles
        self.beta = beta
        self.roll = roll
        self.yaw = yaw
        self.sweep = sweep
        self.b_ref = b_ref  # geometry
        self.main_chord = main_chord
        self.aspect_ratio = aspect_ratio
        self.route = route + '/'
        self.case_name = case_name
        self.RollNodes = RollNodes

        # Verify that the route exists and create directory if necessary
        try:
            os.makedirs(self.route)
        except FileExistsError:
            pass

        ### other params
        self.u_inf_direction = np.array([1., np.sin(beta * np.pi / 180), 0.])
        self.gravity_on = True

        # aeroelasticity
        self.sigma = 1
        self.main_ea = 0.2
        self.main_cg = 0.5
        self.c_ref = 1.  # ref. chord

        # Aerofoil shape: root and tip
        self.root_airfoil_P = 0
        self.root_airfoil_M = 0
        self.tip_airfoil_P = 0
        self.tip_airfoil_M = 0

        self.polars = None # list of polar for each airfoil
        if polar_file is not None:
            self.load_polar(polar_file)

        # Numerics for dynamic simulations
        self.dt_factor = 1
        self.n_tstep = None
        self.physical_time = physical_time
        self.horseshoe = False
        self.fsi_tolerance = 1e-10
        self.relaxation_factor = 0.2
        self.gust_intensity = 0.01
        self.gust_length = 5
        self.tolerance = 1e-6

        n_lumped_mass = 1
        self.lumped_mass = np.zeros((n_lumped_mass))
        self.lumped_mass_position = np.zeros((n_lumped_mass, 3))
        self.lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
        self.lumped_mass_nodes = np.zeros((n_lumped_mass), dtype=int)

        # Control surface initialisation
        self.n_control_surfaces = 0
        self.control_surface = np.zeros((N + 1, 3), dtype=int) - 1
        self.control_surface_type = np.zeros((self.n_control_surfaces), dtype=int)
        self.control_surface_deflection = np.zeros((self.n_control_surfaces,))
        self.control_surface_chord = np.array([M//2], dtype=int)
        self.control_surface_hinge_coord = np.zeros_like(self.control_surface_type, dtype=float)

    def settings_to_config(self, settings):
        file_name = self.route + '/' + self.case_name + '.sharpy'
        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()

        return file_name

    def update_mass_stiff(self):
        '''This method can be substituted to produce different wing configs'''
        # uniform mass/stiffness

        ea, ga = 1e7, 1e7
        gj, eiy, eiz = 1e6, 2e5, 5e6
        base_stiffness = np.diag([ea, ga, ga, gj, eiy, eiz])

        self.stiffness = np.zeros((1, 6, 6))
        self.stiffness[0] = self.sigma * base_stiffness

        self.mass = np.zeros((1, 6, 6))
        self.mass[0, :, :] = np.diag([1., 1., 1., .1, .1, .1])

        self.elem_stiffness = np.zeros((self.num_elem_tot,), dtype=int)
        self.elem_mass = np.zeros((self.num_elem_tot,), dtype=int)

        n_lumped_mass = 1
        self.lumped_mass = np.zeros((n_lumped_mass))
        self.lumped_mass_position = np.zeros((n_lumped_mass, 3))
        self.lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
        self.lumped_mass_nodes = np.zeros((n_lumped_mass), dtype=int)

        self.lumped_mass[0] = 5.
        self.lumped_mass_position[0] = np.array([0, 0.25, 0])

    def load_polar(self, file):
        """
        Read polar from Airfoil Tools polar txt file

        Args:
            file (str):
        """
        polar_raw_data = np.loadtxt(file, skiprows=12)
        self.polars = np.column_stack((polar_raw_data[:, 0] * np.pi / 180, # aoa
                                       polar_raw_data[:, 1], # cl
                                       polar_raw_data[:, 2], # cd
                                       polar_raw_data[:, 4])) #cm

    def update_derived_params(self):
        ### Derived

        # time-step
        self.dt = self.main_chord / self.M / self.u_inf * self.dt_factor
        self.n_tstep = int(np.round(self.physical_time / self.dt))

        # angles
        self.alpha_rad = self.alpha * np.pi / 180.
        self.roll_rad = self.roll * np.pi / 180.
        self.beta_rad = self.beta * np.pi / 180.
        self.sweep_rad = self.sweep * np.pi / 180.
        self.yaw_rad = self.yaw * np.pi / 180.

        # FoR A orientation
        # if self.RollNodes:
        self.quat = algebra.euler2quat(np.array([self.roll_rad, self.alpha_rad, self.yaw_rad]))
        # else:
        #     if np.abs(self.roll)>1e-3:
        #         warnings.warn(
        #             'FoR A quaternion will be built with inverted '+\
        #             'roll angle sign to compensate bug in algebra.euler2quat')
        #     self.quat=algebra.euler2quat(np.array([-self.roll_rad,self.alpha_rad,self.beta_rad]))

        # geometry
        self.wing_span = self.aspect_ratio * self.main_chord  # /np.cos(self.sweep_rad)
        self.S = self.wing_span * self.main_chord

        # discretisation
        assert self.n_surfaces < 3, "use 1 or 2 surfaces only!"
        assert self.N % 2 != 1, \
            'UVLM spanwise panels must be even when using 3-noded FEs!'
        self.num_elem_tot = self.N // 2

        assert self.num_elem_tot % self.n_surfaces != 1, \
            "Can't distribute equally FEM elements over surfaces"
        self.num_elem_surf = self.num_elem_tot // self.n_surfaces
        self.num_node_surf = self.N // self.n_surfaces + 1
        self.num_node_tot = self.N + 1

        self.control_surface = np.zeros((self.num_elem_tot, self.num_node_elem), dtype=int) - 1

        # FEM connectivity, coords definition and mapping
        self.update_fem_prop()

        # Mass/stiffness properties
        self.update_mass_stiff()

        # Aero props
        self.update_aero_prop()

    def update_fem_prop(self):
        ''' Produce FEM connectivity, coordinates, mapping and BCs'''

        n_surfaces = self.n_surfaces
        num_node_elem = self.num_node_elem
        num_node_surf = self.num_node_surf
        num_node_tot = self.num_node_tot
        num_elem_surf = self.num_elem_surf
        num_elem_tot = self.num_elem_tot
        sweep_rad = self.sweep_rad
        half_span = 0.5 * self.wing_span

        #### Connectivity and nodal coordinates
        # Warning: the elements direction determines the xB axis direction.
        # Hence, to avoid accidentally rotating aerofoil profiles, if a wing is
        # defined through  multiple surfaces, nodes should be oriented in the
        # same direction.

        # generate connectivity. Mid node at the end of array.
        conn_loc = np.array([0, 2, 1], dtype=int)
        conn_surf = np.zeros((num_elem_surf, num_node_elem), dtype=int)
        conn_glob = np.zeros((num_elem_tot, num_node_elem), dtype=int)

        # connectivity surface 01
        for ielem in range(num_elem_surf):
            conn_surf[ielem, :] = conn_loc + ielem * (num_node_elem - 1)
        # global connectivity. Multiple surfaces merge at node 0.

        conn_glob[:num_elem_surf, :] = conn_surf
        for ss in range(1, n_surfaces):
            conn_glob[ss * num_elem_surf:(ss + 1) * num_elem_surf, :] = \
                conn_surf + ss * (num_node_surf - 1) + 1
            conn_glob[(ss + 1) * num_elem_surf - 1, 1] = 0

        ### Nodal coordinates
        z = np.zeros((num_node_tot,))
        if n_surfaces == 1:
            ### Local coord from half surface
            x01 = np.sin(sweep_rad) * np.linspace(0., half_span, (num_node_surf + 1) // 2)
            y01 = np.cos(sweep_rad) * np.linspace(0., half_span, (num_node_surf + 1) // 2)
            # and mirrow
            x = np.concatenate([x01[-1:0:-1], x01])
            y = np.concatenate([-y01[-1:0:-1], y01])
        if n_surfaces == 2:
            ### Local coord for surface 00
            x01 = np.sin(sweep_rad) * np.linspace(0., half_span, num_node_surf)
            y01 = np.cos(sweep_rad) * np.linspace(0., half_span, num_node_surf)
            # and mirrow
            x = np.concatenate([x01, x01[-1:0:-1]])
            y = np.concatenate([y01, -y01[-1:0:-1]])
        if n_surfaces > 2:
            raise NameError(
                'Geometry not implemented for multiple surfaces! Rotate them.')

        if self.RollNodes:
            sr, cr = np.sin(self.roll_rad), np.cos(self.roll_rad)
            yold = y.copy()
            y = cr * yold
            z = sr * yold

            ### surface/beam to element mapping
        # beam_number and surface_distribution in fem/aero files
        surface_number = np.zeros((num_elem_tot,), dtype=int)
        for ss in range(n_surfaces):
            surface_number[ss * num_elem_surf:(ss + 1) * num_elem_surf] = ss

        ##### boundary conditions
        boundary_conditions = np.zeros((num_node_tot,), dtype=int)
        if n_surfaces == 1:
            boundary_conditions[[0, -1]] = -1  # free-ends
            boundary_conditions[(num_node_surf - 1) // 2] = 1  # mid-clamp
        if n_surfaces == 2:
            boundary_conditions[0] = 1  # clamp at root (node 0)
            boundary_conditions[num_node_surf - 1] = -1  # free surf 00
            boundary_conditions[num_node_surf] = -1  # free surf 01
        if n_surfaces > 2:
            raise NameError('BCs not implemented for more than 2 surfaces')

        ### Define yB, where yB points to the LE.
        frame_of_reference_delta = np.zeros((num_elem_tot, num_node_elem, 3))
        for ielem in range(num_elem_tot):
            for inode in range(num_node_elem):
                frame_of_reference_delta[ielem, inode, :] = [-1, 0, 0]

        self.frame_of_reference_delta = frame_of_reference_delta
        self.boundary_conditions = boundary_conditions
        self.conn_glob = conn_glob
        self.conn_surf = conn_surf
        self.surface_number = surface_number
        self.x = x
        self.y = y
        self.z = z


    def update_aero_prop(self):
        assert hasattr(self, 'conn_glob'), \
            'Run "update_derived_params" before generating files'

        n_surfaces = self.n_surfaces
        num_node_surf = self.num_node_surf
        num_node_tot = self.num_node_tot
        num_elem_surf = self.num_elem_surf
        num_elem_tot = self.num_elem_tot

        ### Generate aerofoil profiles. Only on surf 0.
        Airfoils_surf = []
        if n_surfaces == 2:
            for inode in range(num_node_surf):
                eta = inode / num_node_surf
                Airfoils_surf.append(
                    np.column_stack(
                        geo_utils.interpolate_naca_camber(
                            eta,
                            self.root_airfoil_M, self.root_airfoil_P,
                            self.tip_airfoil_M, self.tip_airfoil_P)))
            airfoil_distribution_surf = self.conn_surf
            airfoil_distribution = np.concatenate([airfoil_distribution_surf,
                                                   airfoil_distribution_surf[::-1, [1, 0, 2]]])
        if n_surfaces == 1:
            num_node_half = (num_node_surf + 1) // 2
            for inode in range(num_node_half):
                eta = inode / num_node_half
                Airfoils_surf.append(
                    np.column_stack(
                        geo_utils.interpolate_naca_camber(
                            eta,
                            self.root_airfoil_M, self.root_airfoil_P,
                            self.tip_airfoil_M, self.tip_airfoil_P)))
            airfoil_distribution_surf = self.conn_surf[:num_elem_surf // 2, :]
            airfoil_distribution = np.concatenate([
                airfoil_distribution_surf[::-1, [1, 0, 2]],
                airfoil_distribution_surf])

        self.Airfoils_surf = Airfoils_surf
        self.airfoil_distribution = airfoil_distribution

        ### others
        self.aero_node = np.ones((num_node_tot,), dtype=bool)
        self.surface_m = self.M * np.ones((n_surfaces,), dtype=int)

        self.twist = np.zeros((num_elem_tot, 3))
        self.chord = self.main_chord * np.ones((num_elem_tot, 3))
        self.elastic_axis = self.main_ea * np.ones((num_elem_tot, 3,))

    def set_default_config_dict(self):

        str_u_inf_direction = [str(self.u_inf_direction[cc]) for cc in range(3)]

        config = configobj.ConfigObj()
        config.filename = self.route + '/' + self.case_name + '.sharpy'
        settings = dict()

        config['SHARPy'] = {
            'flow': ['BeamLoader', 'AerogridLoader',
                     # 'StaticUvlm',
                     'StaticCoupled',
                     'AerogridPlot', 'BeamPlot', 'SaveData'],
            'case': self.case_name, 'route': self.route,
            'write_screen': 'on', 'write_log': 'on',
            'log_folder': './output/' + self.case_name + '/',
            'log_file': self.case_name + '.log'}

        config['BeamLoader'] = {
            'unsteady': 'off',
            'orientation': self.quat}

        config['AerogridLoader'] = {
            'unsteady': 'off',
            'aligned_grid': 'on',
            'mstar': self.Mstar_fact * self.M,
            'freestream_dir': str_u_inf_direction
        }
        config['NonLinearStatic'] = {'print_info': 'off',
                                     'max_iterations': 150,
                                     'num_load_steps': 0,
                                     'delta_curved': 1e-5,
                                     'min_delta': 1e-5,
                                     'gravity_on': self.gravity_on,
                                     'gravity': 9.754,
                                     'orientation': self.quat}
        config['StaticUvlm'] = {
            'rho': self.rho,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': self.u_inf,
                'u_inf_direction': self.u_inf_direction},
            'rollup_dt': self.dt,
            'print_info': 'on',
            'horseshoe': 'off',
            'num_cores': 4,
            'n_rollup': 0,
            'rollup_aic_refresh': 0,
            'rollup_tolerance': 1e-4}

        config['StaticCoupled'] = {
            'print_info': 'on',
            'max_iter': 200,
            'n_load_steps': 1,
            'tolerance': 1e-10,
            'relaxation_factor': 0.,
            'aero_solver': 'StaticUvlm',
            'aero_solver_settings': {
                'rho': self.rho,
                'print_info': 'off',
                'horseshoe': 'off',
                'num_cores': 4,
                'n_rollup': 0,
                'rollup_dt': self.dt,
                'rollup_aic_refresh': 1,
                'rollup_tolerance': 1e-4,
                'velocity_field_generator': 'SteadyVelocityField',
                'velocity_field_input': {
                    'u_inf': self.u_inf,
                    'u_inf_direction': str_u_inf_direction}},
            #
            'structural_solver': 'NonLinearStatic',
            'structural_solver_settings': {'print_info': 'off',
                                           'max_iterations': 150,
                                           'num_load_steps': 0,
                                           'delta_curved': 1e-1,
                                           'min_delta': 1e-10,
                                           'gravity_on': self.gravity_on,
                                           'gravity': 9.81}}

        config['LinearUvlm'] = {'dt': self.dt,
                                'integr_order': 2,
                                'density': self.rho,
                                'remove_predictor': True,
                                'use_sparse': True,
                                'ScalingDict': {'length': 1.,
                                                'speed': 1.,
                                                'density': 1.}}

        settings['StepLinearUVLM'] = {'dt': self.dt,
                                      'solution_method': 'direct',
                                      'velocity_field_generator': 'SteadyVelocityField',
                                      'velocity_field_input': {
                                          'u_inf': self.u_inf,
                                          'u_inf_direction': self.u_inf_direction}}

        settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'off',
                                                      'max_iterations': 950,
                                                      'delta_curved': 1e-1,
                                                      'min_delta': self.tolerance*1e3,
                                                      'newmark_damp': 5e-3,
                                                      'gravity_on': self.gravity_on,
                                                      'gravity': 9.81,
                                                      'num_steps': self.n_tstep,
                                                      'dt': self.dt}

        settings['StepUvlm'] = {'print_info': 'on',
                                'horseshoe': self.horseshoe,
                                'num_cores': 4,
                                'n_rollup': 100,
                                'convection_scheme': 0,
                                'rollup_dt': self.dt,
                                'rollup_aic_refresh': 1,
                                'rollup_tolerance': 1e-4,
                                # 'velocity_field_generator': 'TurbSimVelocityField',
                                # 'velocity_field_input': {'turbulent_field': '/2TB/turbsim_fields/TurbSim_wide_long_A_low.h5',
                                #                          'offset': [30., 0., -10],
                                #                          'u_inf': 0.},
                                # 'velocity_field_generator': 'GustVelocityField',
                                # 'velocity_field_input': {'u_inf': self.u_inf,
                                #                          'u_inf_direction': self.u_inf_direction,
                                #                          'gust_shape': 'continuous_sin',
                                #                          'gust_length': self.gust_length,
                                #                          'gust_intensity': self.gust_intensity * self.u_inf,
                                #                          'offset': 15.0,
                                #                          'span': self.main_chord * self.aspect_ratio},
                                'velocity_field_generator': 'SteadyVelocityField',
                                'velocity_field_input': {'u_inf': self.u_inf*1,
                                                            'u_inf_direction': [1., 0., 0.]},
                                'rho': self.rho,
                                'n_time_steps': self.n_tstep,
                                'dt': self.dt,
                                'gamma_dot_filtering': 3}

        config['DynamicCoupled'] = {'print_info': 'on',
                                    'structural_substeps': 0,
                                    'dynamic_relaxation': 'on',
                                    'cleanup_previous_solution': 'on',
                                    'structural_solver': 'NonLinearDynamicPrescribedStep',
                                    'structural_solver_settings': settings['NonLinearDynamicPrescribedStep'],
                                    'aero_solver': 'StepUvlm',
                                    'aero_solver_settings': settings['StepUvlm'],
                                    'fsi_substeps': 200,
                                    'fsi_tolerance': self.fsi_tolerance,
                                    'relaxation_factor': self.relaxation_factor,
                                    'minimum_steps': 1,
                                    'relaxation_steps': 150,
                                    'final_relaxation_factor': 0.0,
                                    'n_time_steps': self.n_tstep,
                                    'dt': self.dt,
                                    'include_unsteady_force_contribution': 'off',
                                    'postprocessors': ['BeamLoads', 'StallCheck', 'BeamPlot', 'AerogridPlot'],
                                    'postprocessors_settings': {'BeamLoads': {'csv_output': 'off'},
                                                                'StallCheck': {'output_degrees': True,
                                                                               'stall_angles': {
                                                                                   '0': [-12 * np.pi / 180,
                                                                                         6 * np.pi / 180],
                                                                                   '1': [-12 * np.pi / 180,
                                                                                         6 * np.pi / 180],
                                                                                   '2': [-12 * np.pi / 180,
                                                                                         6 * np.pi / 180]}},
                                                                'BeamPlot': {'include_rbm': 'on',
                                                                             'include_applied_forces': 'on'},
                                                                'AerogridPlot': {
                                                                    'u_inf': self.u_inf,
                                                                    'include_rbm': 'on',
                                                                    'include_applied_forces': 'on',
                                                                    'minus_m_star': 0}}}

        config['DynamicUVLM'] = {'print_info': 'on',
                                 'aero_solver': 'StepUvlm',
                                 'aero_solver_settings': settings['StepUvlm'],
                                 'n_time_steps': self.n_tstep,
                                 'dt': self.dt,
                                 'include_unsteady_force_contribution': 'on',
                                 'postprocessors': ['AerogridPlot'],
                                 'postprocessors_settings': {'AerogridPlot': {'u_inf': self.u_inf,
                                                                              'include_rbm': 'off',
                                                                              'include_applied_forces': 'on',
                                                                              'minus_m_star': 0}}
                                 }

        config['AerogridPlot'] = {'include_rbm': 'off',
                                  'include_applied_forces': 'on',
                                  'minus_m_star': 0}

        config['AeroForcesCalculator'] = {'write_text_file': 'on',
                                          'text_file_name': self.case_name + '_aeroforces.csv',
                                          'screen_output': 'on',
                                          'unsteady': 'off'}

        config['BeamPlot'] = {'include_rbm': 'off',
                              'include_applied_forces': 'on'}

        config['SaveData'] = {}

        config['Modal'] = {'NumLambda': 20,
                           'rigid_body_modes': 'off',
                           'print_matrices': 'off',
                           'save_data': 'off',
                           'continuous_eigenvalues': 'off',
                           'dt': 0,
                           'plot_eigenvalues': False,
                           'max_rotation_deg': 15.,
                           'max_displacement': 0.15,
                           'write_modes_vtk': True,
                           'use_undamped_modes': True}

        config['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                     'linear_system_settings': {
                                         'beam_settings': {'modal_projection': False,
                                                           'inout_coords': 'nodes',
                                                           'discrete_time': True,
                                                           'newmark_damp': 0.5,
                                                           'discr_method': 'newmark',
                                                           'dt': self.dt,
                                                           'proj_modes': 'undamped',
                                                           'use_euler': 'off',
                                                           'num_modes': 40,
                                                           'print_info': 'on',
                                                           'gravity': 'on',
                                                           'remove_dofs': []},
                                         'aero_settings': {'dt': self.dt,
                                                           'integr_order': 2,
                                                           'density': self.rho,
                                                           'remove_predictor': False,
                                                           'use_sparse': True,
                                                           'rigid_body_motion': False,
                                                           'use_euler': False,
                                                           'remove_inputs': ['u_gust']},
                                         'rigid_body_motion': False}}

        config['AsymptoticStability'] = {'print_info': True,
                                        'velocity_analysis': [30, 180, 151]}

        config['LinDynamicSim'] = {'dt': self.dt,
                                     'n_tsteps': self.n_tstep,
                                     'sys_id': 'LinearAeroelastic',
                                     'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                     'postprocessors_settings': {'AerogridPlot': {
                                         'u_inf': self.u_inf,
                                         'include_rbm': 'on',
                                         'include_applied_forces': 'on',
                                         'minus_m_star': 0},
                                         'BeamPlot': {'include_rbm': 'on',
                                                      'include_applied_forces': 'on'}}}

        config['FrequencyResponse'] = {'compute_fom': 'on',
                                       'frequency_unit': 'k',
                                       'frequency_bounds': [0.0001, 1.0],
                                       'quick_plot': 'on'}


        config.write()
        self.config = config
        # print('config dictionary set-up with flow:')
        # print(config['SHARPy']['flow'])

    def generate_aero_file(self, airfoil_efficiency=None):

        with h5.File(self.route + '/' + self.case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            for aa in range(len(self.Airfoils_surf)):
                airfoils_group.create_dataset('%d' % aa, data=self.Airfoils_surf[aa])

            chord_input = h5file.create_dataset('chord', data=self.chord)
            dim_attr = chord_input.attrs['units'] = 'm'

            twist_input = h5file.create_dataset('twist', data=self.twist)
            dim_attr = twist_input.attrs['units'] = 'rad'

            # airfoil distribution
            airfoil_distribution_input = h5file.create_dataset(
                'airfoil_distribution', data=self.airfoil_distribution)
            surface_distribution_input = h5file.create_dataset(
                'surface_distribution', data=self.surface_number)
            surface_m_input = h5file.create_dataset(
                'surface_m', data=self.surface_m)
            m_distribution_input = h5file.create_dataset(
                'm_distribution', data='uniform'.encode('ascii', 'ignore'))
            aero_node_input = h5file.create_dataset(
                'aero_node', data=self.aero_node)
            elastic_axis_input = h5file.create_dataset(
                'elastic_axis', data=self.elastic_axis)
            control_surface_input = h5file.create_dataset(
                'control_surface', data=self.control_surface)
            control_surface_type_input = h5file.create_dataset(
                'control_surface_type', data=self.control_surface_type)
            control_surface_deflection_input = h5file.create_dataset(
                'control_surface_deflection', data=self.control_surface_deflection)
            control_surface_chord_input = h5file.create_dataset(
                'control_surface_chord', data=self.control_surface_chord)
            if airfoil_efficiency is not None:
                a_eff_handle = h5file.create_dataset(
                    'airfoil_efficiency', data=airfoil_efficiency)

            if self.control_surface_hinge_coord is not None:
                cs_hinge = h5file.create_dataset(
                    'control_surface_hinge_coord', data=self.control_surface_hinge_coord
                )

            if self.polars is not None:
                polars_group = h5file.create_group('polars')
                for i_airfoil in range(len(self.Airfoils_surf)):
                    polars_group.create_dataset('{:g}'.format(i_airfoil), data=self.polars)

    def generate_fem_file(self):

        assert hasattr(self, 'conn_glob'), \
            'Run "update_derived_params" before generating files'

        with h5.File(self.route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            coordinates = h5file.create_dataset(
                'coordinates', data=np.column_stack((self.x, self.y, self.z)))
            conectivities = h5file.create_dataset(
                'connectivities', data=self.conn_glob)
            num_nodes_elem_handle = h5file.create_dataset(
                'num_node_elem', data=self.num_node_elem)
            num_nodes_handle = h5file.create_dataset(
                'num_node', data=self.num_node_tot)
            num_elem_handle = h5file.create_dataset(
                'num_elem', data=self.num_elem_tot)
            stiffness_db_handle = h5file.create_dataset(
                'stiffness_db', data=self.stiffness)
            stiffness_handle = h5file.create_dataset(
                'elem_stiffness', data=self.elem_stiffness)
            mass_db_handle = h5file.create_dataset(
                'mass_db', data=self.mass)
            mass_handle = h5file.create_dataset(
                'elem_mass', data=self.elem_mass)
            frame_of_reference_delta_handle = h5file.create_dataset(
                'frame_of_reference_delta', data=self.frame_of_reference_delta)
            structural_twist_handle = h5file.create_dataset(
                'structural_twist', data=np.zeros((self.num_elem_tot, 3)))
            bocos_handle = h5file.create_dataset(
                'boundary_conditions', data=self.boundary_conditions)
            beam_handle = h5file.create_dataset(
                'beam_number', data=self.surface_number)
            app_forces_handle = h5file.create_dataset(
                'app_forces', data=np.zeros((self.num_node_tot, 6)))
            lumped_mass_handle = h5file.create_dataset(
                'lumped_mass', data=self.lumped_mass)
            lumped_mass_inertia_handle = h5file.create_dataset(
                'lumped_mass_inertia', data=self.lumped_mass_inertia)
            lumped_mass_position_handle = h5file.create_dataset(
                'lumped_mass_position', data=self.lumped_mass_position)
            lumped_mass__nodes_handle = h5file.create_dataset(
                'lumped_mass_nodes', data=self.lumped_mass_nodes)

    def generate_rom_files(self, left_tangent, right_tangent, ro, rc, fo, fc):
        with h5.File(self.route + '/' + self.case_name + '.rom.h5', 'a') as h5file:
            lt_handle = h5file.create_dataset('left_tangent',
                                              data=left_tangent)
            rt_handle = h5file.create_dataset('right_tangent',
                                              data=right_tangent)
            ro_h = h5file.create_dataset('ro', data=ro)
            fo_h = h5file.create_dataset('fo', data=fo)
            fc_h = h5file.create_dataset('fc', data=fc)
            rc_h = h5file.create_dataset('rc', data=rc)

    def clean_test_files(self):
        fem_file_name = self.route + '/' + self.case_name + '.fem.h5'
        if os.path.isfile(fem_file_name):
            os.remove(fem_file_name)

        aero_file_name = self.route + '/' + self.case_name + '.aero.h5'
        if os.path.isfile(aero_file_name):
            os.remove(aero_file_name)

        solver_file_name = self.route + '/' + self.case_name + '.sharpy'
        if os.path.isfile(solver_file_name):
            os.remove(solver_file_name)

        flightcon_file_name = self.route + '/' + self.case_name + '.flightcon.txt'
        if os.path.isfile(flightcon_file_name):
            os.remove(flightcon_file_name)

        lininput_file_name = self.route + '/' + self.case_name + '.lininput.h5'
        if os.path.isfile(lininput_file_name):
            os.remove(lininput_file_name)

        rom_file = self.route + '/' + self.case_name + '.rom.h5'
        if os.path.isfile(rom_file):
            os.remove(rom_file)

class Smith(FlyingWing):
    '''
    Build Smith HALE wing.
    This class is nothing but a FlyingWing with pre-defined geometry properties
    and mass/stiffness data ("update_mass_stiffness" method)
     '''

    def __init__(self,
                 M, N,  # chord/span-wise discretisations
                 Mstar_fact,
                 u_inf,  # flight cond
                 alpha,
                 rho=0.08891,
                 b_ref=32.,  # geometry
                 main_chord=1.,
                 aspect_ratio=32,
                 roll=0.,
                 beta=0.,
                 sweep=0.,
                 n_surfaces=2,
                 route='.',
                 case_name='smith',
                 RollNodes=False):
        super().__init__(M=M, N=N,
                         Mstar_fact=Mstar_fact,
                         u_inf=u_inf,
                         alpha=alpha,
                         rho=rho,
                         b_ref=b_ref,
                         main_chord=main_chord,
                         aspect_ratio=aspect_ratio,
                         roll=roll,
                         beta=beta,
                         sweep=sweep,
                         n_surfaces=n_surfaces,
                         route=route,
                         case_name=case_name,
                         RollNodes=RollNodes)
        self.c_ref = 1.

    def update_mass_stiff(self):
        '''This method can be substituted to produce different wing configs'''
        # uniform mass/stiffness

        ea, ga = 1e5, 1e5
        gj, eiy, eiz = 1e4, 2e4, 5e6
        base_stiffness = np.diag([ea, ga, ga, gj, eiy, eiz])

        self.stiffness = np.zeros((1, 6, 6))
        self.stiffness[0] = self.sigma * base_stiffness

        self.mass = np.zeros((1, 6, 6))
        self.mass[0, :, :] = np.diag([0.75, 0.75, 0.75, .1, .1, .1])

        self.elem_stiffness = np.zeros((self.num_elem_tot,), dtype=int)
        self.elem_mass = np.zeros((self.num_elem_tot,), dtype=int)


class Goland(FlyingWing):
    '''
    Build a Goland wing.
    This class is nothing but a FlyingWing with pre-defined geometry properties
    and mass/stiffness data ("update_mass_stiffness" method)
    '''

    def __init__(self,
                 M, N,  # chord/span-wise discretisations
                 Mstar_fact,
                 u_inf,  # flight cond
                 alpha,
                 rho=1.02,
                 b_ref=2. * 6.096,  # geometry
                 main_chord=1.8288,
                 aspect_ratio=(2. * 6.096) / 1.8288,
                 roll=0.,
                 yaw=0.,
                 beta=0.,
                 sweep=0.,
                 n_surfaces=1,
                 physical_time=2,
                 route='.',
                 case_name='goland',
                 RollNodes=False):
        super().__init__(M=M, N=N,
                         Mstar_fact=Mstar_fact,
                         u_inf=u_inf,
                         alpha=alpha,
                         rho=rho,
                         b_ref=b_ref,
                         main_chord=main_chord,
                         aspect_ratio=aspect_ratio,
                         roll=roll,
                         beta=beta,
                         yaw=yaw,
                         sweep=sweep,
                         physical_time=physical_time,
                         n_surfaces=n_surfaces,
                         route=route,
                         case_name=case_name,
                         RollNodes=RollNodes)

        # aeroelasticity parameters
        self.main_ea = 0.33
        self.main_cg = 0.43
        self.sigma = 1

        # other
        self.c_ref = 1.8288

    def update_mass_stiff(self):
        '''
        This method can be substituted to produce different wing configs.

        Forthis model, remind that the delta_frame_of_reference is chosen such
        that the B FoR axis are:
        - xb: along the wing span
        - yb: pointing towards the leading edge (i.e. roughly opposite than xa)
        - zb: upward as za
        '''
        # uniform mass/stiffness

        # ea,ga=1e7,1e6
        ea, ga = 1e9, 1e9
        gj = 0.987581e6
        eiy = 9.77221e6
        eiz = 1e2 * eiy
        base_stiffness = np.diag([ea, ga, ga, self.sigma * gj, self.sigma * eiy, eiz])

        self.stiffness = np.zeros((1, 6, 6))
        self.stiffness[0] = base_stiffness

        m_unit = 35.71
        j_tors = 8.64
        pos_cg_b = np.array([0., self.c_ref * (self.main_cg - self.main_ea), 0.])
        m_chi_cg = algebra.skew(m_unit * pos_cg_b)
        self.mass = np.zeros((1, 6, 6))
        self.mass[0, :, :] = np.diag([m_unit, m_unit, m_unit,
                                      j_tors, .1 * j_tors, .9 * j_tors])

        self.mass[0, :3, 3:] = m_chi_cg
        self.mass[0, 3:, :3] = -m_chi_cg

        self.elem_stiffness = np.zeros((self.num_elem_tot,), dtype=int)
        self.elem_mass = np.zeros((self.num_elem_tot,), dtype=int)


class GolandControlSurface(Goland):

    def __init__(self,
                 M, N,  # chord/span-wise discretisations
                 Mstar_fact,
                 u_inf,  # flight cond
                 alpha,
                 cs_deflection=[0,0],
                 n_control_surfaces=2,
                 rho=1.02,
                 b_ref=2. * 6.096,  # geometry
                 main_chord=1.8288,
                 pct_flap=0.2,
                 aspect_ratio=(2. * 6.096) / 1.8288,
                 roll=0.,
                 yaw=0.,
                 beta=0.,
                 sweep=0.,
                 n_surfaces=1,
                 physical_time=2,
                 cs_type=0,
                 route='.',
                 case_name='goland',
                 RollNodes=False):

        super().__init__(M=M, N=N,
                         Mstar_fact=Mstar_fact,
                         u_inf=u_inf,
                         alpha=alpha,
                         rho=rho,
                         b_ref=b_ref,
                         main_chord=main_chord,
                         aspect_ratio=aspect_ratio,
                         roll=roll,
                         beta=beta,
                         yaw=yaw,
                         sweep=sweep,
                         physical_time=physical_time,
                         n_surfaces=n_surfaces,
                         route=route,
                         case_name=case_name,
                         RollNodes=RollNodes)

        # aeroelasticity parameters
        self.main_ea = 0.33
        self.main_cg = 0.43
        self.sigma = 1

        self.n_control_surfaces = n_control_surfaces
        self.control_surface_deflection = np.zeros(self.n_control_surfaces, dtype=float)
        self.control_surface_deflection[:] = np.deg2rad(cs_deflection)
        self.control_surface_chord = M // 2 * np.ones(self.n_control_surfaces, dtype=int)
        self.control_surface_type = np.zeros(self.n_control_surfaces, dtype=int) + cs_type
        self.control_surface_hinge_coord = np.zeros_like(self.control_surface_type, dtype=int)
        # other
        self.c_ref = 1.8288
        self.pct_flap = pct_flap

    def update_aero_prop(self):
        assert hasattr(self, 'conn_glob'), \
            'Run "update_derived_params" before generating files'

        n_surfaces = self.n_surfaces
        num_node_surf = self.num_node_surf
        num_node_tot = self.num_node_tot
        num_elem_surf = self.num_elem_surf
        num_elem_tot = self.num_elem_tot
        pct_flap = self.pct_flap

        control_surface = self.control_surface

        ### Generate aerofoil profiles. Only on surf 0.
        Airfoils_surf = []
        if n_surfaces == 2:
            for inode in range(num_node_surf):
                eta = inode / num_node_surf
                Airfoils_surf.append(
                    np.column_stack(
                        geo_utils.interpolate_naca_camber(
                            eta,
                            self.root_airfoil_M, self.root_airfoil_P,
                            self.tip_airfoil_M, self.tip_airfoil_P)))
                # if inode >= num_node_surf // 2:
            ws_elem = 0
            for i_surf in range(2):
                # print('Surface' + str(i_surf))
                for i_elem in range(num_elem_surf):
                    for i_local_node in range(self.num_node_elem):
                        if i_elem >= int(num_elem_surf *(1- pct_flap)):
                            if i_surf == 0:
                                control_surface[ws_elem + i_elem, i_local_node] = 0  # Right flap
                            else:
                                control_surface[ws_elem + i_elem, i_local_node] = 1  # Left flap
                ws_elem += num_elem_surf
                        # control_surface[i_elem, i_local_node] = 0

            airfoil_distribution_surf = self.conn_surf
            airfoil_distribution = np.concatenate([airfoil_distribution_surf,
                                                   airfoil_distribution_surf[::-1, [1, 0, 2]]])
            control_surface[-num_elem_surf:] = control_surface[-num_elem_surf:, :][::-1]

        if n_surfaces == 1:
            num_node_half = (num_node_surf + 1) // 2
            for inode in range(num_node_half):
                eta = inode / num_node_half
                Airfoils_surf.append(
                    np.column_stack(
                        geo_utils.interpolate_naca_camber(
                            eta,
                            self.root_airfoil_M, self.root_airfoil_P,
                            self.tip_airfoil_M, self.tip_airfoil_P)))
            airfoil_distribution_surf = self.conn_surf[:num_elem_surf // 2, :]
            airfoil_distribution = np.concatenate([
                airfoil_distribution_surf[::-1, [1, 0, 2]],
                airfoil_distribution_surf])

        self.Airfoils_surf = Airfoils_surf
        self.airfoil_distribution = airfoil_distribution

        ### others
        self.aero_node = np.ones((num_node_tot,), dtype=bool)
        self.surface_m = self.M * np.ones((n_surfaces,), dtype=int)

        self.twist = np.zeros((num_elem_tot, 3))
        self.chord = self.main_chord * np.ones((num_elem_tot, 3))
        self.elastic_axis = self.main_ea * np.ones((num_elem_tot, 3,))
        self.control_surface = control_surface

    def create_linear_files(self, x0, input_vec):
        with h5.File(self.route + '/' + self.case_name + '.lininput.h5', 'a') as h5file:
            x0 = h5file.create_dataset(
                'x0', data=x0)
            u = h5file.create_dataset(
                'u', data=input_vec)

class QuasiInfinite(FlyingWing):
    '''
    Builds a very high aspect ratio wing, for simulating 2D aerodynamics
    This class is nothing but a FlyingWing with pre-defined geometry properties
    and mass/stiffness data ("update_mass_stiffness" method)
     '''

    def __init__(self,
                 M, N,
                 Mstar_fact,
                 u_inf,
                 alpha,
                 aspect_ratio,
                 rho=0.08891,
                 b_ref=32.,
                 main_chord=3.,
                 roll=0.,
                 beta=0.,
                 sweep=0.,
                 n_surfaces=1,
                 route='.',
                 case_name='qsinf',
                 RollNodes=False,
                 polar_file=None):
        super().__init__(M=M, N=N,
                         Mstar_fact=Mstar_fact,
                         u_inf=u_inf,
                         alpha=alpha,
                         rho=rho,
                         b_ref=b_ref,
                         main_chord=main_chord,
                         aspect_ratio=aspect_ratio,
                         roll=roll,
                         beta=beta,
                         sweep=sweep,
                         n_surfaces=n_surfaces,
                         route=route,
                         case_name=case_name,
                         RollNodes=RollNodes,
                         polar_file=polar_file)
        self.c_ref = main_chord
        self.main_ea = 0.5
        self.main_cg = 0.5

    def update_mass_stiff(self):
        '''This method can be substituted to produce different wing configs'''
        # uniform mass/stiffness

        ea, ga = 1e9, 1e9
        gj = 2e8
        eiy = 1e8
        eiz = 1e8
        base_stiffness = np.diag([ea, ga, ga, gj, eiy, eiz])
        self.stiffness = np.zeros((1, 6, 6))
        self.stiffness[0] = self.sigma * base_stiffness

        m_unit = 1.
        self.mass = np.zeros((1, 6, 6))
        self.mass[0, :, :] = np.diag([m_unit, m_unit, m_unit, 1., .5, .5])
        self.elem_stiffness = np.zeros((self.num_elem_tot,), dtype=int)
        self.elem_mass = np.zeros((self.num_elem_tot,), dtype=int)

class Pazy(FlyingWing):
    '''
    Build a Pazy wing.

    The Pazy wing is a highly flexible wing designed and developed at Technion University as an aeroelastic
    test case.

    '''

    def __init__(self,
                 M, N,  # chord/span-wise discretisations
                 Mstar_fact,
                 u_inf,  # flight cond
                 alpha,
                 rho=1.225,
                 tip_rod=True,
                 b_ref=2. * 0.55,  # geometry
                 main_chord=0.1,
                 aspect_ratio=(2. * 0.55) / 0.1,
                 roll=0.,
                 yaw=0.,
                 beta=0.,
                 sweep=0.,
                 n_surfaces=1,
                 physical_time=2,
                 route='.',
                 case_name='pazy',
                 RollNodes=False):

        super().__init__(M=M, N=N,
                         Mstar_fact=Mstar_fact,
                         u_inf=u_inf,
                         alpha=alpha,
                         rho=rho,
                         b_ref=b_ref,
                         main_chord=main_chord,
                         aspect_ratio=aspect_ratio,
                         roll=roll,
                         beta=beta,
                         yaw=yaw,
                         sweep=sweep,
                         physical_time=physical_time,
                         n_surfaces=n_surfaces,
                         route=route,
                         case_name=case_name,
                         RollNodes=RollNodes)

        # aeroelasticity parameters
#         self.main_ea = 0.3
        self.main_ea = 0.4475
        self.main_cg = 0.4510
        self.sigma = 1

        # other
        self.c_ref = main_chord

        self.tip_rod = tip_rod

    def update_mass_stiff(self):
        '''
        This method can be substituted to produce different wing configs.

        For this model, remind that the delta_frame_of_reference is chosen such
        that the B FoR axis are:
        - xb: along the wing span
        - yb: pointing towards the leading edge (i.e. roughly opposite than xa)
        - zb: upward as za
        '''
        # uniform mass/stiffness

        # Scaling factors (for debugging):
        sigma_scale_stiff = 1
        sigma_scale_I = 1

        # Pulling:
        ea = 7.12E+06

        # In-plane bending:
        ga_inp = 3.31E+06
        ei_inp = 3.11E+03

        # Out-of-plane bending:
        # Original:
        # ga_oup = -4.16E+03
        ei_oup = 4.67E+00

        # Adjusted:
        # Adjusted to become non-negative
        ga_oup = 1E+06

        # Torsion:
        gj = 7.20E+00

        base_stiffness = np.diag([ea, ga_oup, ga_inp, gj, ei_oup, ei_inp]) * sigma_scale_stiff

        self.stiffness = np.zeros((1, 6, 6))
        self.stiffness[0] = base_stiffness

        m_unit = 5.50E-01 # kg/m

        # Test cases to confirm properties:
        # Tests in vacuum, AoA=0 deg, analysed with NonLinearStatic beam solver. Verification displacements taken from
        # full 3D non-linear analysis in Abaqus.

        # Bending:
        # 5 N load at the tip. Expected deflection of about 62.6 mm at the tip (no gravity)
        # Tip coordinate in beam-fitted coordinates: X = 0.55m, Y = 0.m, Z = 0.m
        # m_unit = 0.001 # negligible mass distribution to mimic no-gravity FEA analysis
        # self.lumped_mass[0] = 5.2866 / 9.81
        # self.lumped_mass_position[0] = np.array([0, 0, 0])
        # self.lumped_mass_nodes[0] = self.N // 2
        # self.lumped_mass_inertia[0, :, :] = np.diag([1e-1, 1e-1, 1e-1])

        # Alternative bending:
        # self.app_forces[self.N//2, :] = [0, 0, 5.2866, 0, 0, 0]

        # Torsion:
        # Torsion of 0.3275 Nm, difference in tip LE/TE height: 1.130498767 + 1.361232758 = 2.49 mm
        # self.app_forces[self.N // 2, :] = [0, 0, 0, 0.3275, 0, 0]

        pos_cg_b = np.array([0., self.c_ref * (self.main_cg - self.main_ea), 0.])*sigma_scale_I
        m_chi_cg = algebra.skew(m_unit * pos_cg_b)
        self.mass = np.zeros((1, 6, 6))

        Js = 3.03E-04  # kg.m2/m

        # Mass matrix J components: torsion, out of plane bending, in-plane bending
        # Torsion: increasing J decreases natural frequency
        # Bending: increasing J increases natural frequency
        # Chosen experimentally to match natural frequencies from Abaqus analysis
        self.mass[0, :, :] = np.diag([m_unit, m_unit, m_unit, Js, 0.5*Js, 12*Js])*sigma_scale_I
        self.mass[0, :3, 3:] = m_chi_cg
        self.mass[0, 3:, :3] = -m_chi_cg

        self.elem_stiffness = np.zeros((self.num_elem_tot,), dtype=int)
        self.elem_mass = np.zeros((self.num_elem_tot,), dtype=int)

        if self.tip_rod:
            n_lumped_mass = 2
            self.lumped_mass = np.zeros((n_lumped_mass))
            self.lumped_mass_position = np.zeros((n_lumped_mass, 3))
            self.lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
            self.lumped_mass_nodes = np.zeros((n_lumped_mass), dtype=int)

            # Lumped mass for approximating the wingtip weight (1):
            self.lumped_mass[0] = 19.95 / 1E3  # mass in kg
            # self.lumped_mass[0] = 1  # mass in kg - just to visually check
            self.lumped_mass_position[0] = np.array([0.005, -0.005, 0])
            self.lumped_mass_nodes[0] = self.N // 2
            self.lumped_mass_inertia[0, :, :] = np.diag([1.2815E-04, 2.87E-07, 1.17E-04])

            # Lumped mass for approximating the wingtip weight (2):
            self.lumped_mass[1] = 19.95 / 1E3  # mass in kg
            # self.lumped_mass[1] = 1  # mass in kg - just to visually check
            self.lumped_mass_position[1] = np.array([-0.005, -0.005, 0])  # positive x now ascending towards root
            self.lumped_mass_nodes[1] = self.N // 2 + 1
            self.lumped_mass_inertia[1, :, :] = np.diag([1.2815E-04, 2.87E-07, 1.17E-04])



class PazyControlSurface(Pazy):

    def __init__(self,
                 M, N,  # chord/span-wise discretisations
                 Mstar_fact,
                 u_inf,  # flight cond
                 alpha,
                 cs_deflection=0,
                 n_control_surfaces=2,
                 rho=1.225,
                 tip_rod=True,
                 b_ref= 2. * 0.55,  # geometry
                 main_chord= 0.1,
                 pct_flap= 0.2,
                 aspect_ratio= (2. * 0.55) / 0.1,
                 roll=0.,
                 yaw=0.,
                 beta=0.,
                 sweep=0.,
                 n_surfaces=1,
                 physical_time=2,
                 route='.',
                 case_name='pazy',
                 RollNodes=False,
                 cs_type=0):

        super().__init__(M=M, N=N,
                         Mstar_fact=Mstar_fact,
                         u_inf=u_inf,
                         alpha=alpha,
                         rho=rho,
                         tip_rod=tip_rod,
                         b_ref=b_ref,
                         main_chord=main_chord,
                         aspect_ratio=aspect_ratio,
                         roll=roll,
                         beta=beta,
                         yaw=yaw,
                         sweep=sweep,
                         physical_time=physical_time,
                         n_surfaces=n_surfaces,
                         route=route,
                         case_name=case_name,
                         RollNodes=RollNodes)

        # aeroelasticity parameters
#         self.main_ea = 0.3
        self.main_ea = 0.4475
        self.main_cg = 0.4510
        self.sigma = 1

        self.n_control_surfaces = self.n_surfaces
        self.control_surface_deflection = np.zeros(self.n_control_surfaces, dtype=float)
        self.control_surface_deflection[:] = np.deg2rad(cs_deflection)
        self.control_surface_chord = M // 2 * np.ones(self.n_control_surfaces, dtype=int)
        self.control_surface_type = np.zeros(self.n_control_surfaces, dtype=int) + cs_type
        self.control_surface_hinge_coord = np.zeros_like(self.control_surface_type, dtype=int)
        # other
        self.c_ref = main_chord
        self.pct_flap = pct_flap

    def update_aero_prop(self):
        assert hasattr(self, 'conn_glob'), \
            'Run "update_derived_params" before generating files'

        n_surfaces = self.n_surfaces
        num_node_surf = self.num_node_surf
        num_node_tot = self.num_node_tot
        num_elem_surf = self.num_elem_surf
        num_elem_tot = self.num_elem_tot
        pct_flap = self.pct_flap

        control_surface = self.control_surface

        ### Generate aerofoil profiles. Only on surf 0.
        Airfoils_surf = []
        if n_surfaces == 2:
            for inode in range(num_node_surf):
                eta = inode / num_node_surf
                Airfoils_surf.append(
                    np.column_stack(
                        geo_utils.interpolate_naca_camber(
                            eta,
                            self.root_airfoil_M, self.root_airfoil_P,
                            self.tip_airfoil_M, self.tip_airfoil_P)))
                # if inode >= num_node_surf // 2:
            ws_elem = 0
            for i_surf in range(2):
                for i_elem in range(num_elem_surf):
                    for i_local_node in range(self.num_node_elem):
                        if i_elem >= int(num_elem_surf *(1- pct_flap)):
                            if i_surf == 0:
                                control_surface[ws_elem + i_elem, i_local_node] = 0  # Right flap
                            else:
                                control_surface[ws_elem + i_elem, i_local_node] = 1  # Left flap
                ws_elem += num_elem_surf
                        # control_surface[i_elem, i_local_node] = 0

            airfoil_distribution_surf = self.conn_surf
            airfoil_distribution = np.concatenate([airfoil_distribution_surf,
                                                   airfoil_distribution_surf[::-1, [1, 0, 2]]])
            control_surface[-num_elem_surf:] = control_surface[-num_elem_surf:, :][::-1]

        if n_surfaces == 1:
            num_node_half = (num_node_surf + 1) // 2
            for inode in range(num_node_half):
                eta = inode / num_node_half
                Airfoils_surf.append(
                    np.column_stack(
                        geo_utils.interpolate_naca_camber(
                            eta,
                            self.root_airfoil_M, self.root_airfoil_P,
                            self.tip_airfoil_M, self.tip_airfoil_P)))
            airfoil_distribution_surf = self.conn_surf[:num_elem_surf // 2, :]
            airfoil_distribution = np.concatenate([
                airfoil_distribution_surf[::-1, [1, 0, 2]],
                airfoil_distribution_surf])

        self.Airfoils_surf = Airfoils_surf
        self.airfoil_distribution = airfoil_distribution

        ### others
        self.aero_node = np.ones((num_node_tot,), dtype=bool)
        self.surface_m = self.M * np.ones((n_surfaces,), dtype=int)

        self.twist = np.zeros((num_elem_tot, 3))
        self.chord = self.main_chord * np.ones((num_elem_tot, 3))
        self.elastic_axis = self.main_ea * np.ones((num_elem_tot, 3,))
        self.control_surface = control_surface

    def create_linear_files(self, x0, input_vec):
        with h5.File(self.route + '/' + self.case_name + '.lininput.h5', 'a') as h5file:
            x0 = h5file.create_dataset(
                'x0', data=x0)
            u = h5file.create_dataset(
                'u', data=input_vec)



if __name__ == '__main__':
    import os

    os.system('mkdir -p %s' % './test')

    ws = Goland(M=4, N=20, Mstar_fact=12, u_inf=25., alpha=2.0, route='./test')
    ws.clean_test_files()
    ws.update_derived_params()
    ws.generate_fem_file()
    ws.generate_aero_file()
