# import sharpy.utils.settings as settings
# import sharpy.utils.exceptions as exceptions
# import sharpy.utils.cout_utils as cout
import numpy as np
import importlib
import unittest
import os


class TestPlanarWingUvlm(unittest.TestCase):
    """
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = 'planarwing'
        mod = importlib.import_module('tests.uvlm.' + case + '.generate_' + case)

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
        self.assertAlmostEqual(forces_data[-1, 1], 2.245e1, 2)
        self.assertAlmostEqual(forces_data[-1, 2], 0.0, 2)
        self.assertAlmostEqual(forces_data[-1, 3], 4.88705e3, 1)


class TestPlanarWingWakeUvlm(unittest.TestCase):
    """
    """

    def test_planarwing_discrete_wake(self):
        import sharpy.sharpy_main
        # suppress screen output
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/planarwing/planarwing_discretewake.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/planarwing_discretewake/forces/'
        forces_data = np.matrix(np.genfromtxt(output_path + 'planarwing_discretewake_aeroforces.csv', delimiter=','))
        self.assertAlmostEqual(forces_data[-1, 1], 2.91e1, 1)
        self.assertAlmostEqual(forces_data[-1, 2], 0.0, 2)
        self.assertAlmostEqual(forces_data[-1, 3], 4.764e3, 1)

# class TestDynamic2dXbeam(unittest.TestCase):
#     """
#     Tests the xbeam library for the dynamic 2d beam
#     Validation values taken from...
#     """
#
#     @classmethod
#     def setUpClass(cls):
#         # run all the cases generators
#         case = 'dynamic2d'
#         mod = importlib.import_module('tests.xbeam.' + case + '.generate_' + case)
#
#     @classmethod
#     def tearDownClass(cls):
#         pass
#
#     def test_dynamic2d(self):
#         import sharpy.sharpy_main
#         # suppress screen output
#         sharpy.sharpy_main.cout.cout_quiet()
#         solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/dynamic2d/dynamic2d.solver.txt')
#         sharpy.sharpy_main.main(['', solver_path])
#         sharpy.sharpy_main.cout.cout_talk()
#
#         # read output and compare
#         output_path = os.path.dirname(solver_path) + '/beam/'
#         pos_data = np.genfromtxt(output_path + 'beam_dynamic2d_glob_000999.csv')
#         self.assertAlmostEqual(pos_data[-1, 0], -3.7350, 3)
#         self.assertAlmostEqual(pos_data[-1, 1], 13.9267, 3)
