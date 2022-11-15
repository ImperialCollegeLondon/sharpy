"""Test Schur Removal of Unstable Eigenvalues

"""

import numpy as np
import unittest
import sharpy.rom.krylov as krylov
import os
import sharpy.utils.cout_utils as coututils


class TestSchurDecomposition(unittest.TestCase):
    test_dir = os.path.abspath(os.path.dirname(__file__))
    A = np.loadtxt(test_dir + "/src/schur_A.dat")
    eigsA = np.linalg.eigvals(A)

    def setUp(self):
        coututils.start_writer()

    def test_dt(self):
        """
        Discrete time system test. Ensures that all eigenvalues inside the unit circle are preserved.
        """
        A = TestSchurDecomposition.A
        eigsA = TestSchurDecomposition.eigsA

        rom = krylov.Krylov()
        TL, TR = rom.stable_realisation(A, ct=False)

        n_stable_fom = np.sum(np.abs(eigsA) <= 1)
        Ap = TL.T.dot(A.dot(TR))

        eigsAp = np.linalg.eigvals(Ap)
        n_stable_rom = np.sum(np.abs(eigsAp) <= 1)

        assert (
            n_stable_rom == n_stable_fom
        ), "Number of stable eigenvalues not preserved during decomposition"

    def test_ct(self):
        """
        Continuous time system test. Ensures that all eigenvalues in the left hand plane are preserved.
        """
        A = TestSchurDecomposition.A
        eigsA = TestSchurDecomposition.eigsA

        rom = krylov.Krylov()
        TL, TR = rom.stable_realisation(A, ct=True)

        n_stable_fom = np.sum(eigsA.real <= 0)
        Ap = TL.T.dot(A.dot(TR))

        eigsAp = np.linalg.eigvals(Ap)
        n_stable_rom = np.sum(eigsAp.real <= 0)

        assert (
            n_stable_rom == n_stable_fom
        ), "Number of stable eigenvalues not preserved during decomposition"

    def tearDown(self):
        coututils.finish_writer()
