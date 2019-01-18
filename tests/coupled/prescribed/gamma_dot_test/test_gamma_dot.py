import cases.templates.flying_wings as wings
import os
import sharpy.sharpy_main
import numpy as np
import unittest


def x_dot(x, dt, integration_order=2):
    x_dot_r = np.zeros(len(x))
    if integration_order == 1:
        x_n = x.copy()[1:]
        x_m1 = x.copy()[:-1]
        x_dot_r[1:] = (x_n - x_m1) / dt
    else:
        x_n = x.copy()[2:]
        x_m1 = x.copy()[1:-1]
        x_m2 = x.copy()[:-2]
        x_dot_r[2:] = (3 * x_n - 4 * x_m1 + x_m2) / 2 / dt

    return x_dot_r


class Test_gamma_dot(unittest.TestCase):

    def set_up_test_case(self, aero_type, predictor, sparse, integration_order):

        # aero_type = 'lin'
        ws = wings.Goland(M=12,
                          N=4,
                          Mstar_fact=50,
                          u_inf=50,
                          alpha=1.,
                          rho=1.225,
                          sweep=0,
                          physical_time=0.1,
                          n_surfaces=2,
                          route='cases',
                          case_name='goland_' + aero_type + '_'+'P%g_S%g_I%g' %(predictor, sparse, integration_order))

        # Other test parameters
        ws.gust_intensity = 0.01
        ws.sigma = 1
        ws.dt_factor = 1

        ws.clean_test_files()
        ws.update_derived_params()
        ws.update_aero_prop()
        ws.update_fem_prop()
        ws.set_default_config_dict()

        ws.generate_aero_file()
        ws.generate_fem_file()

        ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
                                       'StaticCoupled',
                                       'DynamicCoupled']
        ws.config['SHARPy']['write_screen'] = 'off'

        # Remove newmark damping from structural solver settings
        ws.config['DynamicCoupled']['structural_solver_settings']['newmark_damp'] = 0

        if aero_type == 'lin':
            ws.config['DynamicCoupled']['aero_solver'] = 'StepLinearUVLM'
            ws.config['DynamicCoupled']['aero_solver_settings'] = {'dt': ws.dt,
                                                                   'remove_predictor': predictor,
                                                                   'use_sparse': sparse,
                                                                   'integr_order': integration_order,
                                                                   'velocity_field_generator': 'GustVelocityField',
                                                                   'velocity_field_input': {'u_inf': ws.u_inf,
                                                                                            'u_inf_direction': [1., 0.,
                                                                                                                0.],
                                                                                            'gust_shape': 'continuous_sin',
                                                                                            'gust_length': 2.,
                                                                                            'gust_intensity': ws.gust_intensity
                                                                                                              * ws.u_inf,
                                                                                            'offset': 2.,
                                                                                            'span': ws.main_chord * ws.aspect_ratio}}
        else:
            ws.config['DynamicCoupled']['aero_solver'] = 'StepUvlm'
            ws.config['DynamicCoupled']['aero_solver_settings'] = {
                'print_info': 'off',
                'horseshoe': True,
                'num_cores': 4,
                'n_rollup': 100,
                'convection_scheme': 0,
                'rollup_dt': ws.dt,
                'rollup_aic_refresh': 1,
                'rollup_tolerance': 1e-4,
                'velocity_field_generator': 'GustVelocityField',
                'velocity_field_input': {'u_inf': ws.u_inf,
                                         'u_inf_direction': [1., 0, 0],
                                         'gust_shape': 'continuous_sin',
                                         'gust_length': ws.gust_length,
                                         'gust_intensity': ws.gust_intensity * ws.u_inf,
                                         'offset': 2.0,
                                         'span': ws.main_chord * ws.aspect_ratio},
                'rho': ws.rho,
                'n_time_steps': ws.n_tstep,
                'dt': ws.dt,
                'gamma_dot_filtering': 0,
                'part_of_fsi': True}
            ws.config['DynamicCoupled']['include_unsteady_force_contribution'] = 'on'
        # Update settings file
        ws.config.write()

        self.case_name = ws.case_name
        self.case_route = ws.route
        self.ws = ws
        self.dt = ws.dt

    def run_test(self, aero_type, predictor, sparse, integration_order):

        self.set_up_test_case(aero_type, predictor, sparse, integration_order)
        ws = self.ws
        data = sharpy.sharpy_main.main(['', self.case_route + self.case_name + '.solver.txt'])

        # Obtain gamma
        gamma = np.zeros((ws.n_tstep,))
        gamma_dot = np.zeros((ws.n_tstep))

        for N in range(ws.n_tstep):
            gamma[N] = data.aero.timestep_info[N].gamma[0][0, 0]
            gamma_dot[N] = data.aero.timestep_info[N].gamma_dot[0][0, 0]

        gamma_dot_fd = x_dot(gamma, self.dt, integration_order)

        error_derivative = np.max(np.abs(gamma_dot - gamma_dot_fd))
        gamma_dot_at_max = gamma_dot_fd[np.argmax(np.abs(gamma_dot - gamma_dot_fd))]

        # The signal is close to zero
        if np.abs(gamma_dot_at_max) < 0.05:
            passed_test = error_derivative < 0.05
        else:
            passed_test = error_derivative < 1e-2 * np.abs(gamma_dot_at_max)

        if not passed_test:
            import matplotlib.pyplot as plt
            plt.plot(gamma_dot)
            plt.plot(gamma_dot_fd, color='k')
            plt.show()

            plt.plot(gamma_dot - gamma_dot_fd)
            plt.show()

        assert passed_test == True, \
            'Discrepancy between gamma_dot and that calculated using FD, relative difference is %.2f' % (
                    error_derivative / np.abs(gamma_dot_at_max))

    def test_case(self):
        for aero_type in ['lin', 'nlin']:
            if aero_type == 'lin':
                for predictor in [True, False]:
                    for sparse in [True, False]:
                        for integration_order in [1, 2]:
                            with self.subTest(
                                    aero_type=aero_type,
                                    predictor=predictor,
                                    sparse=sparse,
                                    integration_order=integration_order):
                                self.run_test(aero_type, predictor, sparse, integration_order)
            else:
                with self.subTest(
                        aero_type=aero_type,
                        predictor=False,
                        sparse=False,
                        integration_order=2):

                    self.run_test(aero_type, predictor, sparse, integration_order)


if __name__ == '__main__':
    unittest.main()
