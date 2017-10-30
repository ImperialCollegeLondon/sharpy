# import sharpy.utils.settings as settings
# import sharpy.utils.exceptions as exceptions
# import sharpy.utils.cout_utils as cout
import numpy as np
import importlib
import unittest
import os


class TestSmithUvlm(unittest.TestCase):
    """
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = 'smith_nog_2deg'
        mod = importlib.import_module('tests.coupled.' + case + '.generate_' + case)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_planarwing(self):
        import sharpy.sharpy_main
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
                                      '/smith_nog_2deg/smith_nog_2deg.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        # output_path = os.path.dirname(solver_path) + 'output/aero/'
        # forces_data = np.genfromtxt(output_path + 'smith_nog_2deg_aeroforces.csv')
        # self.assertAlmostEqual(forces_data[-1, 3], 4.88705e3, 2)

