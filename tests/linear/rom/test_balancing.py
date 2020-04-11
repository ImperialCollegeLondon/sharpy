import copy
import unittest
import sharpy.linear.src.libss as libss
import sharpy.rom.utils.librom as librom
import numpy as np
import sharpy.linear.src.libsparse as libsp


class TestBalancing(unittest.TestCase):
    """
    Test Balancing ROM methods

    """

    def test_balreal_direct_py(self):
        Nx, Nu, Ny = 6, 4, 2
        ss = libss.random_ss(Nx, Nu, Ny, dt=0.1, stable=True)

        ### direct balancing
        hsv, T, Ti = librom.balreal_direct_py(ss.A, ss.B, ss.C,
                                              DLTI=True, full_outputs=False)
        ssb = copy.deepcopy(ss)

        # Note: notation below is correct and consistent with documentation
        # SHARPy historically uses notation different from regular literature notation (i.e. 'swapped')
        ssb.project(Ti, T)

        # Compare freq. resp. - inconclusive!
        # The system is consistently transformed using T and Tinv - system dynamics do not change, independent of
        # choice of T and Tinv. Frequency response will yield the same response:
        kv = np.linspace(0.01, 10)
        Y = ss.freqresp(kv)
        Yb = ssb.freqresp(kv)
        er_max = np.max(np.abs(Yb - Y))
        assert er_max / np.max(np.abs(Y)) < 1e-10, 'Error too large in frequency response'

        # Compare grammians:
        Wc = linalg.solve_discrete_lyapunov(ssb.A, np.dot(ssb.B, ssb.B.T))
        Wo = linalg.solve_discrete_lyapunov(ssb.A.T, np.dot(ssb.C.T, ssb.C))

        er_grammians = np.max(np.abs(Wc - Wo))
        # Print grammians to compare:
        if er_grammians / np.max(np.abs(Wc)) > 1e-10:
            print('Controllability grammian, Wc:\n', Wc)
            print('Observability grammian, Wo:\n', Wo)

        er_hankel = np.max(np.abs(np.diag(hsv) - Wc))
        # Print hsv to compare:
        if er_hankel / np.max(np.abs(Wc)) > 1e-10:
            print('Controllability grammian, Wc:\n', Wc)
            print('Hankel values matrix, HSV:\n', hsv)

        assert er_grammians / np.max(np.abs(Wc)) < 1e-10, 'Relative error in Wc-Wo is too large -> Wc != Wo'
        assert er_hankel / np.max(np.abs(Wc)) < 1e-10, 'Relative error in Wc-HSV is too large -> Wc != HSV'

        # The test below is inconclusive for the direct procedure! T and Tinv are produced from svd(M)
        # This means that going back from svd(M) to T and Tinv will yield the same result for any choice of T and Tinv
        # Unless something else is wrong (e.g. a typo) - so leaving it in.
        # test full_outputs option
        hsv, U, Vh, Qc, Qo = librom.balreal_direct_py(ss.A, ss.B, ss.C,
                                                      DLTI=True, full_outputs=True)

        # build M matrix and SVD
        sinv = hsv ** (-0.5)
        T2 = libsp.dot(Qc, Vh.T * sinv)
        Ti2 = np.dot((U * sinv).T, Qo.T)
        assert np.linalg.norm(T2 - T) < 1e-13, 'Error too large'
        assert np.linalg.norm(Ti2 - Ti) < 1e-13, 'Error too large'

        ssb2 = copy.deepcopy(ss)
        ssb2.project(Ti2, T2)
        Yb2 = ssb2.freqresp(kv)
        er_max = np.max(np.abs(Yb2 - Y))
        assert er_max / np.max(np.abs(Y)) < 1e-10, 'Error too large'
