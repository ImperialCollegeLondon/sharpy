"""
Assembling the state space for a full aircraft test
"""

import sharpy.utils.h5utils as h5
from sharpy.linear.src.lingebm import FlexDynamic
from sharpy.linear.src.lin_aeroelastic import LinAeroEla
import sharpy.sharpy_main
import numpy as np
import matplotlib.pyplot as plt
import sharpy.postproc.asymptotic_stability as asym_stability

import cases.templates.flying_wings as wings

# data = h5.readh5('horten_u_inf0018_nlinall_modes_rbm.data.h5').data
data = h5.readh5('horten_u_inf0018_nlinnlin_modes_small_sys.data.h5').data
# horten_u_inf0018_nlinnlin_modes_small_sys

# ws = wings.Goland(M=12,
#                   N=4,
#                   Mstar_fact=50,
#                   u_inf=50,
#                   alpha=1.,
#                   rho=1.225,
#                   sweep=0,
#                   physical_time=0.1,
#                   n_surfaces=2,
#                   route='test_cases',
#                   case_name='goland')
#
# # Other test parameters
# ws.gust_intensity = 0.01
#
# ws.sigma = 1
# ws.dt_factor = 1
#
# ws.clean_test_files()
# ws.update_derived_params()
# ws.update_aero_prop()
#
# ws.n_tstep = 2
#
# ws.update_fem_prop()
# ws.set_default_config_dict()
# ws.generate_aero_file()
# ws.generate_fem_file()
#
#
# ws.config['DynamicCoupled']['aero_solver'] = 'StepUvlm'
# ws.config['DynamicCoupled']['aero_solver_settings'] = {
#     'print_info': 'off',
#     'horseshoe': True,
#     'num_cores': 4,
#     'n_rollup': 100,
#     'convection_scheme': 0,
#     'rollup_dt': ws.dt,
#     'rollup_aic_refresh': 1,
#     'rollup_tolerance': 1e-4,
#     'velocity_field_generator': 'GustVelocityField',
#     'velocity_field_input': {'u_inf': ws.u_inf,
#                              'u_inf_direction': [1., 0, 0],
#                              'gust_shape': 'continuous_sin',
#                              'gust_length': ws.gust_length,
#                              'gust_intensity': ws.gust_intensity * ws.u_inf,
#                              'offset': 2.0,
#                              'span': ws.main_chord * ws.aspect_ratio},
#     'rho': ws.rho,
#     'n_time_steps': ws.n_tstep,
#     'dt': ws.dt,
#     'gamma_dot_filtering': 0,
#     'part_of_fsi': True}
# ws.config['DynamicCoupled']['include_unsteady_force_contribution'] = 'on'
# # Update settings file
#
# ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
#                                'StaticCoupled',
#                                'DynamicCoupled',
#                                'Modal']
# ws.config['SHARPy']['write_screen'] = 'on'
#
# ws.config['Modal'] = {'folder': ws.route+'/output/',
#                    # 'NumLambda': 20,
#                    'rigid_body_modes': True,
#                    'print_matrices': 'off',
#                    'keep_linear_matrices': 'on',
#                    'write_modes_vtk': True,
#                    'use_undamped_modes': False}
# ws.config.write()
#
# data = sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.solver.txt'])
# beam = FlexDynamic(data.structure.timestep_info[-1])

aeroelastic_settings = {'LinearUvlm':{
    'dt': data.settings['DynamicCoupled']['dt'],
    'integr_order': 2,
    'density': 1.225,
    'remove_predictor': False,
    'use_sparse': False,
    'ScalingDict': {'length': 1,
                    'speed': 1,
                    'density': 1},
    'rigid_body_motion': True
}}


analysis = asym_stability.AsymptoticStabilityAnalysis()

analysis.initialise(data, aeroelastic_settings)

eigenvalues, eigenvectors = analysis.run()

analysis.display_root_locus()


# aeroelastic = LinAeroEla(data, aeroelastic_settings)
# beam = aeroelastic.lingebm_str
# aero = aeroelastic.linuvlm
#
# aeroelastic.assemble_ss()


# eigenvalues_dt, eigenvectors = np.linalg.eig(aeroelastic.SS.A)
#
# eigenvalues_ct = np.log(eigenvalues_dt) / aeroelastic.dt
#
# order = np.argsort(-np.real(eigenvalues_ct))
#
# num_modes = 20
# order = order[:num_modes]
# eigenvalues_ct = eigenvalues_ct[order]
# modes = eigenvectors[:,order]





# plt.scatter(np.real(beam.eigs), np.imag(beam.eigs))
# plt.show()
