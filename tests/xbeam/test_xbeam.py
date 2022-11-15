import numpy as np
import importlib
import unittest
import os
import glob
import shutil


class TestGeradinXbeam(unittest.TestCase):
    """
    Tests the xbeam library for the geradin clamped beam
    Validation values taken from
    Simpson, R.J. and Palacios, R., 2013.
    Numerical aspects of nonlinear flexible aircraft flight dynamics modeling.
    In 54th AIAA/ASME/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference (p. 1634).
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = "geradin"
        mod = importlib.import_module("tests.xbeam." + case + ".generate_" + case)

    def test_geradin(self):
        import sharpy.sharpy_main

        # suppress screen output
        solver_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)) + "/geradin/geradin.sharpy"
        )
        sharpy.sharpy_main.main(["", solver_path])

        # read output and compare
        output_path = (
            os.path.dirname(solver_path) + "/output/geradin/WriteVariablesTime/"
        )
        # pos_def
        pos_data = np.atleast_2d(np.genfromtxt(output_path + "struct_pos_node-1.dat"))
        self.assertAlmostEqual(pos_data[0, 3], -2.159, 2)
        self.assertAlmostEqual(5.0 - pos_data[0, 1], 0.596, 3)
        # psi_def
        psi_data = np.atleast_2d(np.genfromtxt(output_path + "struct_psi_node-1.dat"))
        self.assertAlmostEqual(psi_data[-1, 2], 0.6720, 3)

    def tearDown(self):
        solver_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)) + "/geradin/"
        )
        files_to_delete = list()
        extensions = ("*.txt", "*.h5", "*.sharpy")
        for f in extensions:
            files_to_delete.extend(glob.glob(solver_path + "/" + f))

        for f in files_to_delete:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        try:
            shutil.rmtree(solver_path + "/output")
        except FileNotFoundError:
            pass
