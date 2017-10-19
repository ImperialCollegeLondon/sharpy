# import sharpy.utils.settings as settings
# import sharpy.utils.exceptions as exceptions
# import sharpy.utils.cout_utils as cout
# from copy import deepcopy
# import numpy as np
# import subprocess
import importlib
import unittest
import os


class TestStaticXbeam(unittest.TestCase):
    """
    Tests the xbeam library for several cases
    """
    cases = ['geradin']

    @classmethod
    def setUpClass(cls):
        # run all the cases generators
        for case in cls.cases:
            mod = importlib.import_module('tests.xbeam.' + case + '.generate_' + case)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_geradin(self):
        import sharpy.sharpy_main
        # suppress screen output
        # sharpy.sharpy_main.cout.cout_quiet()
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/geradin/geradin.solver.txt')
        sharpy.sharpy_main.main(['', solver_path])
        sharpy.sharpy_main.cout.cout_talk()
