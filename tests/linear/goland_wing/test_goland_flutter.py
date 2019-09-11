import numpy as np
import os
import unittest
import cases.templates.flying_wings as wings
import sharpy.sharpy_main
import sharpy.utils.sharpydir as sharpydir


class TestGolandFlutter(unittest.TestCase):

    def setup(self):
        # Problem Set up
        u_inf = 1.
        alpha_deg = 0.
        rho = 1.02
        num_modes = 8

        # Lattice Discretisation
        M = 14
        N = 32
        M_star_fact = 10

        # Linear UVLM settings
        integration_order = 2
        remove_predictor = False
        use_sparse = True

        # ROM Properties
        rom_settings = dict()
        rom_settings['algorithm'] = 'mimo_rational_arnoldi'
        rom_settings['r'] = 2
        frequency_continuous_k = np.array([0.1])

        # Case Admin - Create results folders
        case_name = 'goland_cs'
        case_nlin_info = 'M%dN%dMs%d_nmodes%d' % (M, N, M_star_fact, num_modes)
        case_rom_info = 'rom_MIMORA_r%d_sig%04d_%04dj' % (rom_settings['r'], frequency_continuous_k[-1].real * 100,
                                                          frequency_continuous_k[-1].imag * 100)

        case_name += case_nlin_info + case_rom_info

        fig_folder = sharpydir.SharpyDir + '/tests/linear/goland_wing/figures/'
        os.system('mkdir -p %s' % fig_folder)

        # SHARPy nonlinear reference solution
        ws = wings.GolandControlSurface(M=M,
                                        N=N,
                                        Mstar_fact=M_star_fact,
                                        u_inf=u_inf,
                                        alpha=alpha_deg,
                                        cs_deflection=[0, 0],
                                        rho=rho,
                                        sweep=0,
                                        physical_time=2,
                                        n_surfaces=2,
                                        route=sharpydir.SharpyDir + '/tests/linear/goland_wing/cases',
                                        case_name=case_name)

        ws.gust_intensity = 0.01
        ws.sigma = 1

        ws.clean_test_files()
        ws.update_derived_params()
        ws.set_default_config_dict()

        ws.generate_aero_file()
        ws.generate_fem_file()

        ws.config['SHARPy']['flow'] = ['BeamLoader',
                                       'AerogridLoader',
                                       'StaticCoupled',
                                       'AerogridPlot',
                                       'BeamPlot',
                                       'Modal',
                                       'LinearAssembler',
                                       'AsymptoticStability',
                                       ]
        ws.config['LinearAssembler']['linear_system_settings']['aero_settings']['use_sparse'] = use_sparse
        ws.config['LinearAssembler']['linear_system_settings']['aero_settings']['ScalingDict'] = {
            'length': 0.5 * ws.c_ref,
            'speed': u_inf,
            'density': rho}
        ws.config['LinearAssembler']['linear_system_settings']['aero_settings']['remove_inputs'] = ['u_gust']
        ws.config['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method'] = 'Krylov'
        frequency_continuous_w = 2 * u_inf * frequency_continuous_k / ws.c_ref
        rom_settings['frequency'] = frequency_continuous_w
        rom_settings['tangent_input_file'] = ws.route + '/' + ws.case_name + '.rom.h5'
        ws.config['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method_settings'] = rom_settings

        ws.config['LinearAssembler']['linear_system_settings']['beam_settings']['gravity'] = 'on'
        ws.config['LinearAssembler']['linear_system_settings']['beam_settings']['modal_projection'] = 'on'
        ws.config['LinearAssembler']['linear_system_settings']['beam_settings']['inout_coords'] = 'modes'
        ws.config['LinearAssembler']['linear_system_settings']['beam_settings']['num_modes'] = num_modes
        ws.config['LinearAssembler']['linear_system_settings']['beam_settings']['remove_sym_modes'] = 'on'
        ws.config['LinearAssembler']['linear_system_settings']['beam_settings']['newmark_damp'] = 0.5e-4
        ws.config['SHARPy']['write_screen'] = 'off'
        ws.config['Modal']['NumLambda'] = 20
        ws.config['Modal']['rigid_body_modes'] = False
        ws.config['Modal']['write_dat'] = True
        ws.config['Modal']['print_matrices'] = True
        ws.config['Modal']['rigid_modes_cg'] = False
        ws.config['NonLinearStatic']['gravity_on'] = 'off'
        ws.config['StaticCoupled']['structural_solver_settings']['gravity_on'] = 'on'
        ws.config['AsymptoticStability']['velocity_analysis'] = [165, 175, 20]
        ws.config.write()

        self.data = sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.solver.txt'])

    def run_rom_stable(self):
        ssrom = self.data.linear.linear_system.uvlm.rom.ssrom
        eigs_rom = np.linalg.eigvals(ssrom.A)

        assert all(np.abs(eigs_rom) <= 1.), 'UVLM Krylov ROM is unstable - flutter speed may not be correct. Change' \
                                            'ROM settings to achieve stability'
        print('ROM is stable')

    def run_flutter(self):
        flutter_ref_speed = 169

        u_inf = self.data.linear.stability['velocity_results']['u_inf']
        eval_real = self.data.linear.stability['velocity_results']['evals_real']
        eval_imag = self.data.linear.stability['velocity_results']['evals_imag']

        # Flutter onset
        ind_zero_real = np.where(eval_real >= 0)[0][0]
        flutter_speed = 0.5 * (u_inf[ind_zero_real] + u_inf[ind_zero_real - 1])
        flutter_frequency = np.sqrt(eval_real[ind_zero_real] ** 2 + eval_imag[ind_zero_real] ** 2)

        print('Flutter speed = %.1f m/s' % flutter_speed)
        print('Flutter frequency = %.2f rad/s' % flutter_frequency)
        assert np.abs(
            flutter_speed - flutter_ref_speed) / flutter_ref_speed < 1e-2, ' Flutter speed error greater than ' \
                                                                           '1 percent'
        print('Test Complete')

    def test_flutter(self):
        self.setup()
        self.run_rom_stable()
        self.run_flutter()


if __name__ == '__main__':
    unittest.main()
