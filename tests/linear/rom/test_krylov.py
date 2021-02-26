"""
Test Krylov ROM using Hospital Building Model
"""

import os
import unittest
import numpy as np

import sharpy.utils.frequencyutils
import sharpy.utils.cout_utils as cout
import scipy.io as scio
import sharpy.utils.sharpydir as sharpydir
import sharpy.linear.src.libss as libss
import sharpy.rom.krylov as krylov
import sharpy.linear.src.libsparse as libsp


class TestKrylov(unittest.TestCase):

    test_dir = sharpydir.SharpyDir + '/tests/linear/rom'

    def setUp(self):
        cout.cout_wrap.initialise(False, False)
        A = scio.loadmat(TestKrylov.test_dir + '/src/' + 'A.mat')
        B = scio.loadmat(TestKrylov.test_dir + '/src/' + 'B.mat')
        C = scio.loadmat(TestKrylov.test_dir + '/src/' + 'C.mat')
        A = libsp.csc_matrix(A['A'])
        B = B['B']
        C = C['C']
        D = np.zeros((B.shape[1], C.shape[0]))

        A = A.todense()

        self.ss = libss.StateSpace(A, B, C, D)

        self.rom = krylov.Krylov()

        if not os.path.exists(self.test_dir + '/figs/'):
            os.makedirs(self.test_dir + '/figs/')

    def run_test(self, test_settings):
        self.rom.initialise(test_settings)
        ssrom = self.rom.run(self.ss)

        # self.rom.restart()
        frequency = test_settings['frequency'].imag
        if frequency[0] != 0.:
            wv = np.logspace(np.log10(np.min(frequency))-0.5, np.log10(np.max(frequency))+0.5, 100)
        else:
            wv = np.logspace(-1, 2, 100)
        Y_fom = self.ss.freqresp(wv)
        Y_rom = ssrom.freqresp(wv)

        max_error = sharpy.utils.frequencyutils.frequency_error(Y_fom, Y_rom, wv)

        # can be used for debugging
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # plt.semilogx(wv, Y_fom[0, 0, :].real)
        # plt.semilogx(wv, Y_rom[0, 0, :].real)
        #
        # fig.savefig(self.test_dir + '/figs/%sfreqresp.png' %test_settings['algorithm'])

        assert np.log10(max_error) < -2, 'Significant mismatch in ROM frequency Response'

    def test_krylov(self):
        algorithm_list = {
            'one_sided_arnoldi':
                {'r': 48,
                 'frequency': np.array([0])},
            'dual_rational_arnoldi':
                {'r': 48,
                 'frequency': np.array([0])}
            }
        for algorithm in list(algorithm_list.keys()):
            with self.subTest(algorithm=algorithm):
                test_settings = {'algorithm': algorithm,
                                 'r': algorithm_list[algorithm]['r'],
                                 'frequency': algorithm_list[algorithm]['frequency']}
                self.run_test(test_settings)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir + '/figs/')

if __name__ == '__main__':
    unittest.main()
