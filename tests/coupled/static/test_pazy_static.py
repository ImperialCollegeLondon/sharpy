import unittest
import numpy as np
import tests.coupled.static.pazy.generate_pazy as gp
import os

class TestPazyCoupled(unittest.TestCase):
    """
    Base class for Pazy wing tests. Serves as parent class for ``TestPazyCoupledStatic`` and ``TestPazyCoupledDynamic``.
    """

    def setUp(self):
        self.u_inf = 50
        self.alpha = 7
        self.M = 16
        self.N= 64
        self.Msf = 1

        self.n_tsteps = 20

        self.route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        self.cases_folder = self.route_test_dir + '/pazy/cases/'
        self.output_folder = self.route_test_dir + '/pazy/cases/'

    def run_test(self, symmetry_condition, dynamic=False):
        self.case_name = 'pazy_uinf{:04g}_alpha{:04g}_symmetry_{}'.format(self.u_inf * 10, self.alpha * 10, str(int(symmetry_condition)))

        gp.generate_pazy(self.u_inf, self.case_name, self.output_folder, self.cases_folder,
                         alpha=self.alpha,
                         M=self.M,
                         N=self.N,
                         Msf=self.Msf,
                         symmetry_condition=symmetry_condition,
                         dynamic=dynamic,
                         n_tsteps=self.n_tsteps)

        self.evaluate_output()

    def evaluate_output(self):   
        pass     
    #     node_number = self.N / 2 # wing tip node

    #     # Get results in A frame
    #     tip_displacement = np.loadtxt(self.output_folder + '/' + self.case_name + '/WriteVariablesTime/struct_pos_node{:g}.dat'.format(node_number))
    #     # current reference from Technion abstract
    #     ref_displacement = 2.033291e-1  # m
    #     print("delta z = ", tip_displacement[-1])
    #     np.testing.assert_almost_equal(tip_displacement[-1], ref_displacement,
    #                                    decimal=3,
    #                                    err_msg='Wing tip displacement not within 3 decimal points of reference.',
    #                                    verbose=True)

    def tearDown(self):
        cases_folder = self.route_test_dir + '/pazy/cases/'

        if os.path.isdir(cases_folder):
            import shutil
            shutil.rmtree(cases_folder)

class TestPazyCoupledStatic(TestPazyCoupled):
    """
    Test Pazy wing static coupled case and compare against a benchmark result.

    As of the time of writing, benchmark result has not been verified but it
    serves as a backward compatibility check for code improvements.
    """

    def test_static_aoa(self):
        self.run_test(False)

    def test_static_aoa_symmetry(self):
        self.run_test(True)

    def evaluate_output(self):     
        node_number = self.N / 2 # wing tip node
        # Get results in A frame
        tip_displacement = np.loadtxt(self.output_folder + '/' + self.case_name + '/WriteVariablesTime/struct_pos_node{:g}.dat'.format(node_number))
        # current reference from Technion abstract
        ref_displacement = 2.033291e-1  # m
        np.testing.assert_almost_equal(tip_displacement[-1], ref_displacement,
                                       decimal=3,
                                       err_msg='Wing tip displacement not within 3 decimal points of reference.',
                                       verbose=True)

if __name__ == '__main__':
    unittest.main()
