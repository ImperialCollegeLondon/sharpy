import unittest
import numpy as np
import os
from tests.coupled.static.test_pazy_static import TestPazyCoupled

class TestPazyCoupledDynamic(TestPazyCoupled):
    """
    Test Pazy wing dynamic coupled case with a free wake convection scheme by compparing wing root loads and 
    moments after 20 timesteps with the reference case produced with SHARPy v2.0 for backward compability.
    Further, symmetry condition is checked for a dynamic free wake as well.
    """

    def test_dynamic_aoa(self):        
        self.run_test(False, dynamic = True)

    def test_dynamic_aoa_symmetry(self):
        self.run_test(True, dynamic = True)

    def evaluate_output(self):       
        ref_Fz = -1663.6376358639793
        ref_My = -12.656888844293645

        file = os.path.join(self.output_folder, self.case_name, 'beam/beam_loads_%i.csv' % (self.n_tsteps))
        beam_loads_ts = np.loadtxt(file, delimiter=',')
        error_Fz = (float(beam_loads_ts[0, 6])-ref_Fz)/ref_Fz
        error_My = (float(beam_loads_ts[0, 8])-ref_My)/ref_My
        
        np.testing.assert_almost_equal(error_Fz, 0.,
                                       decimal=3,
                                       err_msg='Vertical load on wing root differs more than 0.1 %% from reference value.',
                                       verbose=True)
        np.testing.assert_almost_equal(error_My, 0.,
                                       decimal=3,
                                       err_msg='Pitching moment on wing root differs more than 0.1 %% from reference value.',
                                       verbose=True)

if __name__ == '__main__':
    unittest.main()
