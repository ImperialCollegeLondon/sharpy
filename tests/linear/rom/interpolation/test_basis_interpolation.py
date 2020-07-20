import numpy as np
import unittest
import os
from tests.linear.rom.interpolation.generate_goland import generate_goland
from tests.linear.rom.interpolation.generate_pmor import generate_pmor
import h5py as h5
import sharpy.utils.h5utils as h5utils
import sharpy.utils.frequencyutils as frequencyutils
import shutil
import configobj


class TestBasisInterpolation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        cls.u_inf_test_cases = [120]

        cls.interpolated_cases = dict()  #: dict where key is u_inf. Value is the case
        cls.actual_cases = dict()  #: dict where key is u_inf. Value is the case

        u_inf_source_cases = [100, 140]

        # Krylov ROM settings:
        rom_settings = dict()
        rom_settings['algorithm'] = 'mimo_rational_arnoldi'
        rom_settings['r'] = 6
        rom_settings['single_side'] = 'observability'
        rom_settings['frequency'] = 0

        for u_inf in u_inf_source_cases:
            generate_goland(u_inf,
                            problem_type='source',
                            rom_method_settings={'Krylov': rom_settings},
                            write_screen='off')

        for u_inf in cls.u_inf_test_cases:
            case = {u_inf: generate_goland(u_inf,
                                           problem_type='actual',
                                           rom_method_settings={'Krylov': rom_settings}),
                    'u_inf': u_inf}
            cls.actual_cases[u_inf] = case

    def test_basis_interpolation(self):

        interpolation_settings = dict()
        interpolation_settings['cases_folder'] = self.route_test_dir + '/source/output/'
        interpolation_settings['reference_case'] = 0
        interpolation_settings['interpolation_degree'] = 1

        for u_inf in self.u_inf_test_cases:
            interpolation_settings['interpolation_parameter'] = {'u_inf': u_inf}
            case = {u_inf: generate_goland(u_inf,
                                           problem_type='interpolation',
                                           rom_method_settings={'BasisInterpolation': interpolation_settings}),
                    'u_inf': u_inf}
            self.interpolated_cases[u_inf] = case

        for case in self.interpolated_cases.values():
            case['freqresp'] = self.load_freqresp(self.route_test_dir + '/interpolation/', case[case['u_inf']])

        for case in self.actual_cases.values():
            case['freqresp'] = self.load_freqresp(self.route_test_dir + '/actual/', case[case['u_inf']])

            i_case = self.interpolated_cases[case['u_inf']]
            y_error_system = case['freqresp']['response'] - i_case['freqresp']['response']
            l2_norm = frequencyutils.l2norm(y_error_system, case['freqresp']['frequency'])
            try:
                np.testing.assert_array_less([l2_norm], [1e-4],
                                             verbose=True,
                                             err_msg='L2 norm of error system too large')

            except AssertionError:
                # >>>>>>>>>>>>>>>>>>>>>>>>useful debug
                import matplotlib.pyplot as plt
                plt.semilogx(case['freqresp']['frequency'], case['freqresp']['response'][0, 0, :])
                plt.semilogx(i_case['freqresp']['frequency'], i_case['freqresp']['response'][0, 0, :], marker='+')
                plt.title(l2_norm)
                plt.show()
                raise AssertionError

    def test_pmor_interpolation(self):

        if not os.path.isdir(self.route_test_dir + '/pmor'):
            os.makedirs(self.route_test_dir + '/pmor')

        # create input yaml file
        with open(self.route_test_dir + '/pmor/pmor_input_file.yaml', 'w') as in_file:
            for u_inf in self.u_inf_test_cases:
                in_file.write('- u_inf: %f\n' % u_inf)
            in_file.close()

        pmor_case_name = generate_pmor(self.route_test_dir + '/source/output',
                                       pmor_route=self.route_test_dir + '/pmor',
                                       input_file=self.route_test_dir + '/pmor/pmor_input_file.yaml',
                                       pmor_output=self.route_test_dir + '/pmor')
        pmor_output_root = self.route_test_dir + '/output/' + pmor_case_name + '/pmor_summary.txt'
        pmor_results = configobj.ConfigObj(pmor_output_root)

        # import pdb; pdb.set_trace()

        for case_number, case_name in enumerate(pmor_results):
            pmor_results[case_name]['freqresp'] = self.load_pmor_freqresp(self.route_test_dir, pmor_case_name,
                                                                          case_number)

            a_case = self.actual_cases[float(pmor_results[case_name]['u_inf'])]
            y_error_system = pmor_results[case_name]['freqresp']['response'] - a_case['freqresp']['response']
            l2_norm = frequencyutils.l2norm(y_error_system, a_case['freqresp']['frequency'])
            try:
                np.testing.assert_array_less([l2_norm], [1e-3],
                                             verbose=True,
                                             err_msg='L2 norm of PMOR error system too large')
            except AssertionError:

            # >>>>>>>>>>>>>>>>>>>>>>>>useful debug
                import matplotlib.pyplot as plt
                plt.semilogx(a_case['freqresp']['frequency'], a_case['freqresp']['response'][0, 0, :])
                plt.semilogx(a_case['freqresp']['frequency'], pmor_results[case_name]['freqresp']['response'][0, 0, :], marker='+')
                plt.title(l2_norm)
                plt.show()
                raise AssertionError
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    @classmethod
    def tearDownClass(cls):
        folders = ['interpolation', 'actual', 'source', 'pmor']
        for folder in folders:
            if os.path.isdir(cls.route_test_dir + '/' + folder):
                shutil.rmtree(cls.route_test_dir + '/' + folder)
        # os.remove(cls.route_test_dir + '/pmor_input_file.yaml')

    @staticmethod
    def load_freqresp(folder, case_name):
        filename = folder + '/output/' + case_name + '/frequencyresponse/aeroelastic.freqresp.h5'

        with h5.File(filename, 'r') as freq_file_handle:
            # store files in dictionary
            freq_dict = h5utils.load_h5_in_dict(freq_file_handle)

        return freq_dict

    @staticmethod
    def load_pmor_freqresp(folder, pmor_case_name, case_number):
        filename = folder + '/output/' + pmor_case_name + '/frequencyresponse/param_case%02g' % case_number + '/freqresp.h5'

        with h5.File(filename, 'r') as freq_file_handle:
            # store files in dictionary
            freq_dict = h5utils.load_h5_in_dict(freq_file_handle)

        return freq_dict


if __name__ == '__main__':
    unittest.main()
