# import sharpy.utils.settings as settings
# import sharpy.utils.exceptions as exceptions
# import sharpy.utils.cout_utils as cout
import numpy as np
import importlib
import unittest
import os


class TestSmithCoupled(unittest.TestCase):
    """
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = 'smith_nog_2deg'
        mod = importlib.import_module('tests.coupled.' + case + '.generate_' + case)
        case = 'smith_g_2deg'
        mod2 = importlib.import_module('tests.coupled.' + case + '.generate_' + case)
        case = 'smith_g_4deg'
        mod3 = importlib.import_module('tests.coupled.' + case + '.generate_' + case)
        case = 'smith_nog_4deg'
        mod4 = importlib.import_module('tests.coupled.' + case + '.generate_' + case)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_smith2deg_nog(self):
        """
        Case and results from:
        Smith, M.J., Patil, M.J. and Hodges, D.H.,
        2001.
        CFD-based analysis of nonlinear aeroelastic behavior of high-aspect
        ratio wings.
        AIAA Paper, 1582, p.2001.
        :return:
        """
        import sharpy.sharpy_main
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
                                      '/smith_nog_2deg/smith_nog_2deg.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/smith_nog_2deg/beam/'
        pos_data = np.genfromtxt(output_path + 'beam_smith_nog_2deg_000000.csv')
        self.assertAlmostEqual((pos_data[20, 1] - 15.581)/15.581, 0.00, 2)
        self.assertAlmostEqual((pos_data[20, 2] - 3.399)/3.399, 0.00, 2)

        # results:
        # N = 10 elements
        # M = 15 elements
        # full wake:
        # Nrollup = 100
        # Mstar = 80
        # pos last beam of the wing [  0.02515625  15.62906166   3.20177985]
        # total forces:
        # tstep | fx_st | fy_st | fz_st
        #     0 | 1.464e+00 | -4.480e-03 | 2.389e+02
        # 521 seconds

        # will use this one for validation.
        # same discretisation, with horseshoe:
        # [  0.02796747  15.58103469   3.39999052]
        # forces:
        # tstep |   fx_st    |   fy_st    |   fz_st
        #     0 |  1.047e+00 | -5.944e-03 |  2.542e+02
        # 26 seconds

    def test_smith2deg_g(self):
        import sharpy.sharpy_main
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
                                      '/smith_g_2deg/smith_g_2deg.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        # output_path = os.path.dirname(solver_path) + 'output/aero/'
        # forces_data = np.genfromtxt(output_path + 'smith_nog_2deg_aeroforces.csv')
        # self.assertAlmostEqual(forces_data[-1, 3], 4.88705e3, 2)

    def test_smith4deg_g(self):
        """
        Case from R. Simpson's PhD thesis.
        His wing tip displacements: 15.627927627927626, 3.3021978021978025
        I always get higher deflection when using my gravity implementation
        instead of his. His results with gravity are not validated.***
        :return:
        """
        import sharpy.sharpy_main
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
                                      '/smith_g_4deg/smith_g_4deg.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        # output_path = os.path.dirname(solver_path) + 'output/aero/'
        # forces_data = np.genfromtxt(output_path + 'smith_nog_2deg_aeroforces.csv')
        # self.assertAlmostEqual(forces_data[-1, 3], 4.88705e3, 2)

        # results:
        # N = 10 elements
        # M = 15 elements
        # full wake:
        # Nrollup = 100
        # Mstar = 80
        # pos last beam of the wing
        # total forces:
        # tstep | fx_st | fy_st | fz_st
        #
        # seconds

        # will use this one for validation.
        # same discretisation, with horseshoe:
        # [  0.06318925  15.44724143   3.88387631]
        # forces:
        # tstep |   fx_st    |   fy_st    |   fz_st
        #    0 |  2.150e+00 |  3.136e-06 |  4.000e+02
        # 52 seconds

    def test_smith4deg_nog(self):
        """
        :return:
        """
        import sharpy.sharpy_main
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
                                      '/smith_nog_4deg/smith_nog_4deg.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        # output_path = os.path.dirname(solver_path) + 'output/aero/'
        # forces_data = np.genfromtxt(output_path + 'smith_nog_2deg_aeroforces.csv')
        # self.assertAlmostEqual(forces_data[-1, 3], 4.88705e3, 2)

        # results:
        # N = 10 elements
        # M = 15 elements
        # full wake:
        # Nrollup = 100
        # Mstar = 80
        # pos last beam of the wing
        # total forces:
        # tstep | fx_st | fy_st | fz_st
        #
        # seconds

        # will use this one for validation.
        # same discretisation, with horseshoe:
        #
        # forces:
        # tstep |   fx_st    |   fy_st    |   fz_st
        #
        #  seconds
