# import sharpy.utils.settings as settings
# import sharpy.utils.exceptions as exceptions
# import sharpy.utils.cout_utils as cout
import numpy as np
import importlib
import unittest
import os


class TestDynamicUvlm(unittest.TestCase):
    """
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = 'planarwing_dynamic'
        mod = importlib.import_module('tests.uvlm.dynamic.' + case + '.generate_' + case)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_planarwing(self):
        """
        last line of the forces file:
        80, 2.195385e+03, 5.684342e-14, -3.022998e+04, 0.000000e+00, 0.000000e+00, 5.052476e+03,
        153 seconds
        :return:
        """
        import sharpy.sharpy_main
        # from sharpy.utils.cout_utils import cout_wrap
        # suppress screen output
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
                                      '/planarwing_dynamic/planarwing_dynamic.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/planarwing_dynamic/forces/'
        forces_data = np.matrix(np.genfromtxt(output_path + 'planarwing_dynamic_aeroforces.csv', delimiter=','))
        self.assertAlmostEqual(forces_data[-1, 1], 2.195385e3, 1)
        self.assertAlmostEqual(forces_data[-1, 2], 0.0, 2)
        self.assertAlmostEqual(forces_data[-1, 3], -3.022998e4, 1)


