import unittest
import numpy as np
import os
from tests.coupled.static.test_pazy_static import TestPazyCoupled

class TestPazyCoupledDynamic(TestPazyCoupled, unittest.TestCase):
    """
    Test Pazy wing dynamic coupled case with a free wake convection scheme by compparing wing root loads and 
    moments after 2 timesteps with the reference case produced with SHARPy v2.0 for backward compability.
    Further, symmetry condition is checked for a dynamic free wake as well.
    """

    def test_dynamic_aoa_symmetry(self):
        self.setup_test_folders('pazy')
        self.num_chordwise_panels //= 4
        self.num_spanwise_nodes //= 4  
        self.run_test(True, dynamic = True)

    def evaluate_output(self):                                                      
        self.assert_root_forces_match_reference(ref_Fz=-2.5274941e+04, 
                                                ref_My=-1.0703502e+01)

    def assert_root_forces_match_reference(self, ref_Fz, ref_My):
        """
        Check if the vertical force and pitching moment at the wing root match reference values.

        Args:
            ref_Fz (float): Reference vertical force at wing root [N]
            ref_My (float): Reference pitching moment at wing root [Nm]
        """
        file_path = os.path.join(self.output_folder, self.case_name, 'beam', f'beam_loads_{self.n_tsteps}.csv')
        beam_loads_ts = np.loadtxt(file_path, delimiter=',')

        actual_Fz = float(beam_loads_ts[0, 6])
        actual_My = float(beam_loads_ts[0, 8])

        error_Fz = (actual_Fz - ref_Fz) / ref_Fz
        error_My = (actual_My - ref_My) / ref_My

        np.testing.assert_almost_equal(
            error_Fz, 0.0, decimal=3,
            err_msg='Vertical load on wing root differs more than 0.1% from reference value.',
            verbose=True
        )
        np.testing.assert_almost_equal(
            error_My, 0.0, decimal=3,
            err_msg='Pitching moment on wing root differs more than 0.1% from reference value.',
            verbose=True
        )    
    

if __name__ == '__main__':
    unittest.main()
