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
        ssb.project(Ti, T)

        # compare freq. resp.
        kv = np.array([0., .5, 3., 5.67])
        Y = ss.freqresp(kv)
        Yb = ssb.freqresp(kv)
        er_max = np.max(np.abs(Yb - Y))
        assert er_max / np.max(np.abs(Y)) < 1e-10, 'Error too large in frequency response'

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
