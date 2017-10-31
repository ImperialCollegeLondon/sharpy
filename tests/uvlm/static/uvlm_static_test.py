# import sharpy.utils.settings as settings
# import sharpy.utils.exceptions as exceptions
# import sharpy.utils.cout_utils as cout
import numpy as np
import importlib
import unittest
import os


class TestStaticUvlm(unittest.TestCase):
    """
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = 'planarwing'
        mod = importlib.import_module('tests.uvlm.static.' + case + '.generate_' + case)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_planarwing(self):
        import sharpy.sharpy_main
        # from sharpy.utils.cout_utils import cout_wrap
        # suppress screen output
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/planarwing/planarwing.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/planarwing/forces/'
        forces_data = np.matrix(np.genfromtxt(output_path + 'planarwing_aeroforces.csv', delimiter=','))
        self.assertAlmostEqual(forces_data[-1, 1], 2.239e1, 1)
        self.assertAlmostEqual(forces_data[-1, 2], 0.0, 2)
        self.assertAlmostEqual(forces_data[-1, 3], 4.901e3, 0)

    def test_planarwing_discrete_wake(self):
        import sharpy.sharpy_main
        # suppress screen output
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/planarwing/planarwing_discretewake.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/planarwing_discretewake/forces/'
        forces_data = np.matrix(np.genfromtxt(output_path + 'planarwing_discretewake_aeroforces.csv', delimiter=','))
        self.assertAlmostEqual(forces_data[-1, 1], 3.88263e1, 0)
        self.assertAlmostEqual(forces_data[-1, 2], 0.0, 2)
        self.assertAlmostEqual(forces_data[-1, 3], 4.555e3, 0)


