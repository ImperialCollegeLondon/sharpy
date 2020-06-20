import unittest
import numpy as np
import tests.io.generate_pazy_dynamic as gp
import os
import shutil

class TestPazyCoupledStatic(unittest.TestCase):
    """
    Test Pazy wing static coupled case and compare against a benchmark result.
    As of the time of writing, benchmark result has not been verified but it
    serves as a backward compatibility check for code improvements.
    """

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    def test_dynamic_aoa(self):
        u_inf = 50
        alpha = 0
        case_name = 'pazy_uinf{:04g}_alpha{:04g}'.format(u_inf * 10, alpha * 10)

        M = 4
        N = 16
        Msf = 1

        cases_folder = self.route_test_dir + '/cases/'
        output_folder = self.route_test_dir + '/cases/'
        # run case
        gp.generate_pazy(u_inf, case_name, output_folder, cases_folder,
                         alpha=alpha,
                         M=M,
                         N=N,
                         Msf=Msf)

        node_number = N / 2 # wing tip node

        # Get results in A frame
        tip_displacement = np.loadtxt(output_folder + '/' + case_name + '/WriteVariablesTime/struct_pos_node{:g}.dat'.format(node_number))

        # current reference from Technion abstract
        # ref_displacement = 2.033291e-1  # m
        # np.testing.assert_almost_equal(tip_displacement[-1], ref_displacement,
        #                                decimal=3,
        #                                err_msg='Wing tip displacement not within 3 decimal points of reference.',
        #                                verbose=True)

    def tearDown(self):
        cases_folder = self.route_test_dir + '/cases/'

        if os.path.isdir(cases_folder):
            import shutil
            shutil.rmtree(cases_folder)


if __name__ == '__main__':
    unittest.main()