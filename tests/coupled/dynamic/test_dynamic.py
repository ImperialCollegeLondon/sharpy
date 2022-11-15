import numpy as np
import importlib
import unittest
import os
import sharpy.utils.cout_utils as cout


class TestCoupledDynamic(unittest.TestCase):
    """
    Tests for dynamic coupled problems to identify errors in the unsteady solvers.
    Implemented tests:
    - Gust response of the hale aircraft
    """

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        case = "hale"
        mod = importlib.import_module(
            "tests.coupled.dynamic." + case + ".generate_" + case
        )
        pass

    def test_hale_dynamic(self):
        """
        Case and results from:
        tests/coupled/dynamic/hale
        reference results produced with SHARPy version 1.3
        :return:
        """
        import sharpy.sharpy_main

        case_name = "hale"
        route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        cases_folder = os.path.join(route_test_dir, case_name)
        output_folder = cases_folder + "/output/"

        sharpy.sharpy_main.main(["", cases_folder + "/hale.sharpy"])
        n_tstep = 20

        # compare results with reference values
        ref_Fz = 50.4986064826483
        ref_My = -1833.91402522644
        file = os.path.join(
            output_folder, case_name, "beam/beam_loads_%i.csv" % (n_tstep)
        )
        beam_loads_ts = np.loadtxt(file, delimiter=",")
        np.testing.assert_almost_equal(
            float(beam_loads_ts[0, 6]),
            ref_Fz,
            decimal=3,
            err_msg=(
                "Vertical load on wing root not within 3 decimal points of reference."
            ),
            verbose=True,
        )
        np.testing.assert_almost_equal(
            float(beam_loads_ts[0, 8]),
            ref_My,
            decimal=3,
            err_msg=(
                "Pitching moment on wing root not within 3 decimal points of reference."
            ),
            verbose=True,
        )

    @classmethod
    def tearDownClass(cls):
        import shutil

        list_cases = ["hale"]
        list_file_extensions = [".fem.h5", ".aero.h5", ".sharpy"]
        list_folders = ["output", "__pycache__"]
        for case in list_cases:
            file_path = os.path.join(
                os.path.abspath(os.path.dirname(os.path.realpath(__file__))), case
            )
            for folder in list_folders:
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
            for extension in list_file_extensions:
                os.remove(os.path.join(file_path, case + extension))
        pass
