import numpy as np
import unittest
import os


class TestCoupledDynamic(unittest.TestCase):
    """
    Tests for dynamic coupled problems to identify errors in the unsteady solvers.
    Implemented tests:
    - Gust response of the hale aircraft
    """

    def test_hale_dynamic(self):
        """
        Case and results from:
        tests/coupled/dynamic/hale 
        reference results produced with SHARPy version 2.0
        :return:
        """
        import sharpy.sharpy_main
        try:
            import hale.generate_hale
        except:
            import tests.coupled.dynamic.hale.generate_hale
        
        case_name = 'hale'
        self.route_file_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        cases_folder = os.path.join(self.route_file_dir, case_name)
        output_folder = cases_folder + '/output/'

        sharpy.sharpy_main.main(['', cases_folder + '/hale.sharpy'])
        n_tstep = 5
        # compare results with reference values 
        ref_Fz = -526.712530589119
        ref_My =-1513.64676181859

        file = os.path.join(output_folder, case_name, 'beam/beam_loads_%i.csv' % (n_tstep))
        beam_loads_ts = np.loadtxt(file, delimiter=',')
        np.testing.assert_almost_equal(float(beam_loads_ts[0, 6]), ref_Fz,
                                       decimal=3,
                                       err_msg='Vertical load on wing root not within 3 decimal points of reference.',
                                       verbose=True)
        np.testing.assert_almost_equal(float(beam_loads_ts[0, 8]), ref_My,
                                       decimal=3,
                                       err_msg='Pitching moment on wing root not within 3 decimal points of reference.',
                                       verbose=True)

    @classmethod
    def tearDown(self):
        """
            Removes all created files within this test.
        """
        import shutil
        folders = ['hale/output']
        for folder in folders:
            shutil.rmtree(self.route_file_dir + '/' + folder)
        files = ['hale/hale.aero.h5', 'hale/hale.fem.h5', 'hale.hale.sharpy']
        for file in files:
            file_dir = self.route_file_dir + '/' + file
            if os.path.isfile(file_dir):
                os.remove(file_dir)
            
   

if __name__ == '__main__':
    unittest.main()