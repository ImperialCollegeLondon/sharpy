import numpy as np
import importlib
import unittest
import os


class TestCoupledStatic(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = "smith_nog_2deg"
        mod = importlib.import_module(
            "tests.coupled.static." + case + ".generate_" + case
        )
        case = "smith_g_2deg"
        mod2 = importlib.import_module(
            "tests.coupled.static." + case + ".generate_" + case
        )
        case = "smith_g_4deg"
        mod3 = importlib.import_module(
            "tests.coupled.static." + case + ".generate_" + case
        )
        case = "smith_nog_4deg"
        mod4 = importlib.import_module(
            "tests.coupled.static." + case + ".generate_" + case
        )

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

        solver_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/smith_nog_2deg/smith_nog_2deg.sharpy"
        )
        sharpy.sharpy_main.main(["", solver_path])

        # read output and compare
        output_path = (
            os.path.dirname(solver_path) + "/output/smith_nog_2deg/WriteVariablesTime/"
        )
        pos_data = np.genfromtxt(output_path + "struct_pos_node20.dat")
        self.assertAlmostEqual((pos_data[2] - 15.599) / 15.599, 0.00, 2)
        self.assertAlmostEqual((pos_data[3] - 3.32600) / 3.32600, 0.00, 2)

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

    def test_smith2deg_g(self):
        import sharpy.sharpy_main

        solver_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/smith_g_2deg/smith_g_2deg.sharpy"
        )
        sharpy.sharpy_main.main(["", solver_path])

        # read output and compare
        output_path = (
            os.path.dirname(solver_path) + "/output/smith_g_2deg/WriteVariablesTime/"
        )
        pos_data = np.genfromtxt(output_path + "struct_pos_node20.dat")

        self.assertAlmostEqual((pos_data[2] - 15.98295) / 15.98295, 0.00, 2)
        self.assertAlmostEqual((pos_data[3] - 0.682268) / 0.682268, 0.00, 2)

    def test_smith4deg_g(self):
        """
        Case from R. Simpson's PhD thesis.
        His wing tip displacements: 15.627927627927626, 3.3021978021978025
        I always get higher deflection when using my gravity implementation
        instead of his. His results with gravity are not validated.***
        :return:
        """
        import sharpy.sharpy_main

        solver_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/smith_g_4deg/smith_g_4deg.sharpy"
        )
        sharpy.sharpy_main.main(["", solver_path])

        # read output and compare
        output_path = (
            os.path.dirname(solver_path) + "/output/smith_g_4deg/WriteVariablesTime/"
        )
        pos_data = np.genfromtxt(output_path + "struct_pos_node20.dat")
        self.assertAlmostEqual((pos_data[2] - 15.55) / 15.55, 0.00, 2)
        self.assertAlmostEqual((pos_data[3] - 3.671) / 3.671, 0.00, 2)

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

    def test_smith4deg_nog(self):
        """
        Hodges result for Euler+Nonlinear is
        14.668547249647393, 5.451612903225806
        :return:
        """
        import sharpy.sharpy_main

        solver_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/smith_nog_4deg/smith_nog_4deg.sharpy"
        )
        sharpy.sharpy_main.main(["", solver_path])

        # read output and compare
        output_path = (
            os.path.dirname(solver_path) + "/output/smith_nog_4deg/WriteVariablesTime/"
        )
        pos_data = np.genfromtxt(output_path + "struct_pos_node20.dat")
        self.assertAlmostEqual((pos_data[2] - 14.87) / 14.87, 0.00, 2)
        self.assertAlmostEqual((pos_data[3] - 5.5078) / 5.5078, 0.00, 2)

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
