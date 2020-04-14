# import sharpy.utils.settings as settings
# import sharpy.utils.exceptions as exceptions
# import sharpy.utils.cout_utils as cout
import numpy as np
import importlib
import unittest
import os
import sharpy.utils.cout_utils as cout


cout.cout_wrap.print_screen = True

class TestCoupledPrescribed(unittest.TestCase):
    """
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        # case = 'smith_2deg_prescribed'
        # mod = importlib.import_module('tests.coupled.prescribed.' + case + '.generate_' + case)
        # case = 'rotating_wing'
        # mod1 = importlib.import_module('tests.coupled.prescribed.' + case + '.generate_' + case)
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    # def test_smith2deg_prescribed(self):
    #     import sharpy.sharpy_main
    #     solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
    #                                   '/smith_2deg_prescribed/smith_2deg_prescribed.sharpy')
    #     sharpy.sharpy_main.main(['', solver_path])
    #
    #     # read output and compare
    #     output_path = os.path.dirname(solver_path) + 'output/aero/'
    #     forces_data = np.genfromtxt(output_path + 'smith_2deg_prescribed_aeroforces.csv')
    #     self.assertAlmostEqual(forces_data[-1, 3], -3.728e1, 1)

    def test_rotating_wing(self):
        # import sharpy.sharpy_main
        # solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
                                      # '/rotating_wing/rotating_wing.sharpy')
        # sharpy.sharpy_main.main(['', solver_path])
        cout.cout_wrap('No tests for prescribed dynamic configurations (yet)!', 1)
        pass
