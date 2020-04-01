import unittest
import sharpy.linear.src.libss as libss
import scipy.linalg as sclalg
import numpy as np
import os

import sharpy.utils.frequencyutils as frequencyutils


class TestFrequencyUtils(unittest.TestCase):

    test_dir = os.path.abspath(os.path.dirname(__file__))

    def setUp(self):

        # This particular system is a good test as it requires a few iterations of the Hinf
        # solver. In addition it is known that its SVD peak occurs between 0.1 and 1.0 rad/s
        # allowing us to increase the resolution in the graphical method around that vicinity.
        a = np.load(self.test_dir + '/src/a.npy')
        b = np.load(self.test_dir + '/src/b.npy')
        c = np.load(self.test_dir + '/src/c.npy')
        d = np.load(self.test_dir + '/src/d.npy')

        self.sys = libss.ss(a, b, c, d, dt=None)

    def test_hinfinity_norm(self):

        h_inf_iterative = frequencyutils.h_infinity_mimo(self.sys)

        # Compare against graphical method
        # Hinf norm is the maximum SVD across all frequencies
        wv_vec = np.logspace(-1, 0, 100)
        svd_val = np.zeros_like(wv_vec)

        for i in range(len(svd_val)):
            svd_val[i] = np.max(sclalg.svd(self.sys.transfer_function_evaluation(1j*wv_vec[i]),
                                           compute_uv=False))

        h_inf_graph = np.max(svd_val)

        np.testing.assert_almost_equal(h_inf_iterative, h_inf_graph, decimal=3, verbose=True,
                                       err_msg='H-infinity norm using iterative method vs graphical one not '
                                               'equal to 3 decimal places')

        # >>>>>>>>>>> Useful debugging
        # print(h_inf_iterative)
        # print(h_inf_graph)
        #
        # import matplotlib.pyplot as plt
        # plt.semilogx(wv_vec, svd_val)
        # plt.show()
        #
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<
