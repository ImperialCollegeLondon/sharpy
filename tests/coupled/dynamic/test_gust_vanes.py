import unittest
import numpy as np
import os
from tests.coupled.static.test_pazy_static import TestPazyCoupled

class TestPazyCoupledDynamicWithGustVanes(TestPazyCoupled, unittest.TestCase):
    """
    Test gust vanes with pazy wing in a dynamic coupled simulation with a free wake convection scheme. The induced velocity field
    and wing root loads and moments are compared after 20 timesteps with the reference case produced with SHARPy v2.0 for backward compability.
    """

    def test_dynamic_aoa_symmetry_with_gust_vanes(self):
        self.setup_test_folders('gust_vanes')
        cs_deflection_file = self.route_test_dir + '/gust_vanes/gust_vane_deflections.csv'
         
        self.num_chordwise_panels //= 2
        self.num_spanwise_nodes //= 2   
        self.n_tsteps = 6
        
        self.run_test(True, dynamic=True, gust_vanes=True, cs_deflection_file=cs_deflection_file)

    def evaluate_output(self):  
        self.assert_induced_velocity_matches_reference(
            time_step=6,
            ref_u_ind_z=-0.4026807
        )

    def assert_induced_velocity_matches_reference(self, time_step, ref_u_ind_z):
        """
        Check if the vertical induced velocity at a given time step and node matches the reference.

        Args:
            time_step (int): Index of the time step to check (e.g., 6)
             ref_u_ind_z (float): Reference vertical induced velocity [m/s]
        """
        file_path = os.path.join(self.output_folder, self.case_name, 'WriteVariablesTime', 'vel_field_uind_point0.dat')
        induced_velocities = np.loadtxt(file_path)

        actual_value = induced_velocities[time_step, -1]

        np.testing.assert_almost_equal(
            actual_value, ref_u_ind_z, decimal=3,
            err_msg='Induced vertical velocity differs more than 0.1% from reference value.',
            verbose=True
        )
        
if __name__ == '__main__':
    unittest.main()

