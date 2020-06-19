import numpy as np
import os
import unittest
import cases.templates.flying_wings as wings
import sharpy.sharpy_main

    # Problem Set up
def generate_pazy(u_inf, case_name, output_folder='/output/', cases_folder='', **kwargs):
    # u_inf = 60
    alpha_deg = kwargs.get('alpha', 0.)
    rho = 1.225
    num_modes = 16
    gravity_on = kwargs.get('gravity_on', True)

    # Lattice Discretisation
    M = kwargs.get('M', 4)
    N = kwargs.get('N', 32)
    M_star_fact = kwargs.get('Ms', 10)

    # SHARPy nonlinear reference solution
    ws = wings.PazyControlSurface(M=M,
                                    N=N,
                                    Mstar_fact=M_star_fact,
                                    u_inf=u_inf,
                                    alpha=alpha_deg,
                                    cs_deflection=[0, 0],
                                    rho=rho,
                                    sweep=0,
                                    physical_time=2,
                                    n_surfaces=2,
                                    route=cases_folder + '/' + case_name,
                                    case_name=case_name)

    ws.gust_intensity = 0.01
    ws.sigma = 1

    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()

    ws.generate_aero_file()
    ws.generate_fem_file()

    ws.config['SHARPy'] = {
        'flow':
            ['BeamLoader',
            'AerogridLoader',
             'StaticCoupled',
             'AerogridPlot',
             'BeamPlot',
             'WriteVariablesTime',
             ],
        'case': ws.case_name, 'route': ws.route,
        'write_screen': 'off', 'write_log': 'on',
        'save_settings': 'on',
        'log_folder': output_folder + '/' + ws.case_name + '/',
        'log_file': ws.case_name + '.log'}

    ws.config['BeamLoader'] = {
        'unsteady': 'off',
        'orientation': ws.quat}

    ws.config['AerogridLoader'] = {
        'unsteady': 'off',
        'aligned_grid': 'on',
        'mstar': ws.Mstar_fact * ws.M,
        'freestream_dir': ws.u_inf_direction,
        'wake_shape_generator': 'StraightWake',
        'wake_shape_generator_input': {'u_inf': ws.u_inf,
                                       'u_inf_direction': ws.u_inf_direction,
                                       'dt': ws.dt}}

    ws.config['StaticUvlm'] = {
        'rho': ws.rho,
        'velocity_field_generator': 'SteadyVelocityField',
        'velocity_field_input': {
            'u_inf': ws.u_inf,
            'u_inf_direction': ws.u_inf_direction},
        'rollup_dt': ws.dt,
        'print_info': 'on',
        'horseshoe': 'on',
        'num_cores': 4,
        'n_rollup': 0,
        'rollup_aic_refresh': 0,
        'rollup_tolerance': 1e-4}

    settings = dict()
    settings['NonLinearStatic'] = {'print_info': 'off',
                                   'max_iterations': 200,
                                   'num_load_steps': 5,
                                   'delta_curved': 1e-6,
                                   'min_delta': 1e-8,
                                   'gravity_on': gravity_on,
                                   'gravity': 9.81}

    ws.config['StaticCoupled'] = {
        'print_info': 'on',
        'max_iter': 200,
        'n_load_steps': 4,
        'tolerance': 1e-5,
        'relaxation_factor': 0.1,
        'aero_solver': 'StaticUvlm',
        'aero_solver_settings': {
            'rho': ws.rho,
            'print_info': 'off',
            'horseshoe': 'on',
            'num_cores': 4,
            'n_rollup': 0,
            'rollup_dt': ws.dt,
            'rollup_aic_refresh': 1,
            'rollup_tolerance': 1e-4,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': ws.u_inf,
                'u_inf_direction': ws.u_inf_direction},
            'vortex_radius': 1e-9},
        'structural_solver': 'NonLinearStatic',
        'structural_solver_settings': settings['NonLinearStatic']}

    ws.config['AerogridPlot'] = {'folder': output_folder,
                                 'include_rbm': 'off',
                                 'include_applied_forces': 'on',
                                 'minus_m_star': 0}

    ws.config['BeamPlot'] = {'folder': output_folder,
                             'include_rbm': 'off',
                             'include_applied_forces': 'on'}

    ws.config['WriteVariablesTime'] = {'folder': output_folder,
                                        'structure_variables': ['pos'],
                                        'structure_nodes': list(range(0, ws.num_node_surf)),
                                        'cleanup_old_solution': 'on'}

    ws.config.write()

    sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.sharpy'])

