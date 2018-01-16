# import sharpy.utils.settings as settings
# import sharpy.utils.exceptions as exceptions
# import sharpy.utils.cout_utils as cout
import numpy as np
import importlib
import unittest
import os


class TestCoupledStatic(unittest.TestCase):
    """
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = 'smith_nog_2deg'
        mod = importlib.import_module('tests.coupled.static.' + case + '.generate_' + case)
        case = 'smith_g_2deg'
        mod2 = importlib.import_module('tests.coupled.static.' + case + '.generate_' + case)
        case = 'smith_g_4deg'
        mod3 = importlib.import_module('tests.coupled.static.' + case + '.generate_' + case)
        case = 'smith_nog_4deg'
        mod4 = importlib.import_module('tests.coupled.static.' + case + '.generate_' + case)

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
        output_path = os.path.dirname(solver_path) + '/output/smith_g_2deg/beam/'
        pos_data = np.genfromtxt(output_path + 'beam_smith_g_2deg_000002.csv')
        self.assertAlmostEqual((pos_data[20, 1] - 15.9777)/15.9777, 0.00, 2)
        self.assertAlmostEqual((pos_data[20, 2] - 0.781620)/0.781620, 0.00, 2)

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
        output_path = os.path.dirname(solver_path) + '/output/smith_g_4deg/beam/'
        pos_data = np.genfromtxt(output_path + 'beam_smith_g_4deg_000002.csv')
        self.assertAlmostEqual((pos_data[20, 1] - 15.5097)/15.097, 0.00, 2)
        self.assertAlmostEqual((pos_data[20, 2] - 3.66200)/3.66200, 0.00, 2)

        # results:
        # N = 10 elements
        # M = 15 elements
        # full wake:
        # Nrollup = 100
        # Mstar = 80
        # pos last node of the wing
        # 0.05668 15.5422 3.53971
        # total forces:
        # tstep | fx_st | fy_st | fz_st
        #   0   | 3.229 | -1.059e-3| 3.766e2
        # 7500 seconds

        # will use this one for validation.
        # same discretisation, with horseshoe:
        # 5.009157275506188889e-02 1.550972930845460596e+01 3.662002908013482383e+00
        # forces:
        # tstep |   fx_st    |   fy_st    |   fz_st
        #    2 |  3.765174e+00, -3.611481e-04, 4.882051e+02
        # 44 seconds

    def test_smith4deg_nog(self):
        """
        Hodges result for Euler+Nonlinear is
        14.668547249647393, 5.451612903225806
        :return:
        """
        import sharpy.sharpy_main
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) +
                                      '/smith_nog_4deg/smith_nog_4deg.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = os.path.dirname(solver_path) + '/output/smith_nog_4deg/beam/'
        pos_data = np.genfromtxt(output_path + 'beam_smith_nog_4deg_000003.csv')
        self.assertAlmostEqual((pos_data[20, 1] - 14.80636555)/14.80636555, 0.00, 2)
        self.assertAlmostEqual((pos_data[20, 2] - 5.65457501)/5.65457501, 0.00, 2)

        # results:
        # N = 10 elements
        # M = 15 elements
        # full wake:
        # Nrollup = 100
        # Mstar = 80
        # pos last beam of the wing
        # 5.59418e-2 1.49094e1 5.41518
        # total forces:
        # tstep | fx_st | fy_st | fz_st
        # 0     | 3.045 | -1.089e-4 | 3.678e2
        # 7380 seconds

        # will use this one for validation.
        # same discretisation, with horseshoe:
        # [  0.05849521  14.80636555   5.65457501]
        # forces:
        # tstep |   fx_st    |   fy_st    |   fz_st
        #     0 |  1.996e+00 | -2.116e-06 |  3.888e+02
        # 142 seconds
