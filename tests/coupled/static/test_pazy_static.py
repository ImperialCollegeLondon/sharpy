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
        self.num_chordwise_panels = 16
        self.num_spanwise_nodes= 64
        self.Msf = 1
        self.n_tsteps = 2
        
        self.setup_test_folders('pazy')
       
    def run_test(self, symmetry_condition, dynamic=False, gust_vanes=False, cs_deflection_file=None):
        self.case_name = 'pazy_uinf{:04g}_alpha{:04g}_symmetry_{}_gustvanes_{}'.format(self.u_inf * 10, self.alpha * 10, str(int(symmetry_condition)), int(gust_vanes))

        gp.generate_pazy(self.u_inf, self.case_name, self.output_folder, self.cases_folder,
                         alpha=self.alpha,
                         M=self.num_chordwise_panels,
                         N=self.num_spanwise_nodes,
                         Msf=self.Msf,
                         symmetry_condition=symmetry_condition,
                         dynamic=dynamic,
                         n_tsteps=self.n_tsteps,
                         gust_vanes=gust_vanes,
                         cs_deflection_file=cs_deflection_file)
     
        self.evaluate_output()

    def evaluate_output(self):   
        pass

    def setup_test_folders(self, subfolder):
        """
        Set up the case and output directories for the current test,
        based on the location of the test file that calls this method.

        Args:
            subfolder (str): Subfolder name like "pazy" or "gust_vanes"
        """
        import inspect 

        caller_file = inspect.stack()[1].filename
        self.route_test_dir = os.path.abspath(os.path.dirname(caller_file))
        self.cases_folder = os.path.join(self.route_test_dir, subfolder, 'cases')
        self.output_folder = self.cases_folder

    def tearDown(self):
        cases_folder = self.route_test_dir + '/pazy/cases/'

        if os.path.isdir(cases_folder):
            import shutil
            shutil.rmtree(cases_folder)

class TestPazyCoupledStatic(TestPazyCoupled, unittest.TestCase):
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
        self.assert_tip_displacement_matches_reference(
            node_index=self.num_spanwise_nodes / 2,
            ref_displacement=2.033291e-1
        )
        
    def assert_tip_displacement_matches_reference(self, node_index: int, ref_displacement: float):
        """
        Check if the tip node displacement matches the reference value.

        Args:
            node_index (int): Node index along the span (e.g., N/2 for tip)
            ref_displacement (float): Reference displacement [m]
        """
        file_path = os.path.join(self.output_folder, self.case_name,
                                f'WriteVariablesTime/struct_pos_node{int(node_index)}.dat')
        tip_displacement = np.loadtxt(file_path)

        np.testing.assert_almost_equal(
            tip_displacement[-1], ref_displacement, decimal=3,
            err_msg='Wing tip displacement not within 0.001 m of reference.',
            verbose=True
        )   

if __name__ == '__main__':
    unittest.main()
