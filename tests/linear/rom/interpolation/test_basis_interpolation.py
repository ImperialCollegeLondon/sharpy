import numpy as np
import unittest
import os
from tests.linear.rom.interpolation.generate_goland import generate_goland
import h5py as h5
import sharpy.utils.h5utils as h5utils
import sharpy.utils.frequencyutils as frequencyutils
import shutil


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
                            rom_method_settings={'Krylov': rom_settings})

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

        for u_inf in self.u_inf_test_cases:
            interpolation_settings['interpolation_parameter'] = {'u_inf': u_inf}
            case = {u_inf: generate_goland(u_inf,
                                           problem_type='interpolation',
                                           rom_method_settings={'BasisInterpolation': interpolation_settings}),
                    'u_inf': u_inf}
            self.interpolated_cases[u_inf] = case

    def test_frequency_response(self):

        for case in self.interpolated_cases.values():
            case['freqresp'] = self.load_freqresp(self.route_test_dir + '/interpolation/', case[case['u_inf']])

        for case in self.actual_cases.values():
            case['freqresp'] = self.load_freqresp(self.route_test_dir + '/actual/', case[case['u_inf']])

            i_case = self.interpolated_cases[case['u_inf']]
            y_error_system = case['freqresp']['response'] - i_case['freqresp']['response']
            l2_norm = frequencyutils.l2norm(y_error_system, case['freqresp']['frequency'])
            np.testing.assert_array_less([l2_norm], [1e-4],
                                         verbose=True,
                                         err_msg='L2 norm of error system too large')

    @classmethod
    def tearDownClass(cls):
        folders = ['interpolation', 'actual', 'source']
        for folder in folders:
            if os.path.isdir(cls.route_test_dir + '/' + folder):
                shutil.rmtree(cls.route_test_dir + '/' + folder)

    @staticmethod
    def load_freqresp(folder, case_name):
        filename = folder + '/output/' + case_name + '/frequencyresponse/aeroelastic.freqresp.h5'

        with h5.File(filename, 'r') as freq_file_handle:
            # store files in dictionary
            freq_dict = h5utils.load_h5_in_dict(freq_file_handle)

        return freq_dict


if __name__ == '__main__':
    unittest.main()
