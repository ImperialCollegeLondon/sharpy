import unittest
import numpy as np
import tests.coupled.static.pazy.generate_pazy as gp
import os
import shutil

class TestPazyCoupledStatic(unittest.TestCase):
    """
    Test Pazy wing static coupled case and compare against a benchmark result.

    As of the time of writing, benchmark result has not been verified but it
    serves as a backward compatibility check for code improvements.
    """

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    def init_simulation_parameters(self, symmetry_condition = False):
        self.u_inf = 50
        self.alpha = 7
        self.M = 16
        self.N= 64
        self.Msf = 1

        self.cases_folder = self.route_test_dir + '/pazy/cases/'
        self.output_folder = self.route_test_dir + '/pazy/cases/'

        self.symmetry_condition = symmetry_condition

    def test_static_aoa(self):
        self.init_simulation_parameters()
        self.case_name = 'pazy_uinf{:04g}_alpha{:04g}_symmetry_{}'.format(self.u_inf * 10, self.alpha * 10, str(int(self.symmetry_condition)))


        # run case
        gp.generate_pazy(self.u_inf, self.case_name, self.output_folder, self.cases_folder,
                         alpha=self.alpha,
                         M=self.M,
                         N=self.N,
                         Msf=self.Msf)     

        self.evaluate_output()

        self.tearDown()
        
    def test_static_aoa_symmetry(self):
        self.init_simulation_parameters(symmetry_condition=True)
        self.case_name = 'pazy_uinf{:04g}_alpha{:04g}_symmetry_{}'.format(self.u_inf * 10, self.alpha * 10, str(int(self.symmetry_condition)))

        # run case
        gp.generate_pazy(self.u_inf, self.case_name, self.output_folder, self.cases_folder,
                         alpha=self.alpha,
                         M=self.M,
                         N=self.N,
                         Msf=self.Msf,
                         symmetry_condition=self.symmetry_condition)

        self.evaluate_output()

        self.tearDown()




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

    def tearDown(self):
        cases_folder = self.route_test_dir + '/pazy/cases/'

        if os.path.isdir(cases_folder):
            import shutil
            shutil.rmtree(cases_folder)


if __name__ == '__main__':
    unittest.main()
