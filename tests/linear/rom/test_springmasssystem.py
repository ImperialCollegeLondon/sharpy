"""
Generate a mass spring system

NGoizueta 16 Feb 2019

"""

import numpy as np
import sharpy.linear.src.libss as libss
import sharpy.linear.src.lingebm as lingebm
import sharpy.rom.krylov as krylov
import unittest
# import matplotlib.pyplot as plt
import sharpy.utils.cout_utils as cout

cout.cout_wrap.initialise(False, False)

@unittest.skip('Not a robust test case that is giving too many failures. Use test_krylov instead')
class TestKrylovRom(unittest.TestCase):

    tolerance = 1e-3
    display_output = True

    def test_siso_ct(self):
        system_inputs = 'SISO'
        system_time = 'ct'

        ss = self.build_system(system_inputs, system_time)

        algorithm = 'two_sided_arnoldi'
        interpolation_point = np.array([1.0j])
        r = 7

        print('\nTesting CT, SISO rational Arnoldi...')
        rom = self.run_rom(ss, algorithm, r, interpolation_point)

        wv = np.logspace(-1, 3, 1000)
        freq_error = self.compare_freq_resp(rom, wv, interpolation_point)

        print('Frequency Response Error at %.2f rad/s: %.2e' % (interpolation_point.imag, freq_error))

        self.assertTrue(freq_error < self.tolerance)

    def test_siso_dt(self):
        system_inputs = 'SISO'
        system_time = 'dt'

        ss = self.build_system(system_inputs, system_time)

        algorithm = 'two_sided_arnoldi'
        interpolation_point_ct = np.array([0.8j])
        r = 7

        print('\nTesting DT, SISO rational Arnoldi...')
        rom = self.run_rom(ss, algorithm, r, interpolation_point_ct)

        wv = np.logspace(-1, 3, 1000)
        freq_error = self.compare_freq_resp(rom, wv, interpolation_point_ct)

        print('Frequency Response Error at %.2f rad/s: %.2e' % (interpolation_point_ct.imag, freq_error))

        self.assertTrue(freq_error < self.tolerance)

    def test_siso_dt_multipoint(self):
        system_inputs = 'SISO'
        system_time = 'dt'

        ss = self.build_system(system_inputs, system_time)

        algorithm = 'dual_rational_arnoldi'
        interpolation_point_ct = np.array([0.0, 2.0j, 11.0j])
        r = 7

        print('\nTesting DT, SISO Multipoint rational Arnoldi...')
        rom = self.run_rom(ss, algorithm, r, interpolation_point_ct)

        wv = np.logspace(-1, 3, 1000)
        for i in range(len(interpolation_point_ct)):
            freq_error = self.compare_freq_resp(rom, wv, interpolation_point_ct[i])
            print('Frequency Response Error at %.2f rad/s: %.2e' % (interpolation_point_ct[i].imag, freq_error))
            self.assertTrue(freq_error < self.tolerance)


    def build_system(self, system_inputs, system_time):
        N = 5  # Number of masses/springs/dampers

        k_db = np.linspace(1, 10, N)  # Stiffness database
        m_db = np.logspace(2, 0, N)  # Mass database
        C_db = np.ones(N) * 1e-1  # Damping database

        # Build mass matrix
        m = np.zeros((N, N))
        k = np.zeros((N, N))
        C = np.zeros((N, N))
        m[0, 0] = m_db[0]

        k[0, 0:2] = [k_db[0]+k_db[1], -k_db[1]]
        C[0, 0:2] = [C_db[0] + C_db[1], -C_db[1]]
        for i in range(1, N-1):
            k[i, i-1:i+2] = [-k_db[i-1], k_db[i]+k_db[i+1], -k_db[i+1]]
            C[i, i-1:i+2] = [-C_db[i-1], C_db[i]+C_db[i+1], -C_db[i+1]]
            m[i, i] = m_db[i]
        m[-1, -1] = m_db[-1]
        k[-1, -2:] = [-k_db[-1], k_db[-1]]
        C[-1, -2:] = [-C_db[-1], C_db[-1]]

        # Input: Forces, Output: Displacements
        if system_inputs == 'MIMO':
            b = np.zeros((2*N, N))
            b[N:, :] = np.eye(N)
            # Output rn
            c = np.zeros((N, 2*N))
            c[:, :N] = np.eye(N)
            d = np.zeros((N, N))
        else:
            b = np.zeros((2*N,))
            b[-1] = 1.
            c = np.zeros((1, 2*N))
            c[0, N-1] = 1
            d = np.zeros(1)

        # Plant matrix
        Minv = np.linalg.inv(m)
        MinvK = Minv.dot(k)
        A = np.zeros((2*N, 2*N))
        A[:N, N:] = np.eye(N)
        A[N:, :N] = -MinvK
        A[N:, N:] = -Minv.dot(C)

        # Build State Space
        if system_time == 'ct':
            system = libss.StateSpace(A, b, c, d, dt=None)

        else:
            # Discrete time system
            dt = 1e-2
            Adt, Bdt, Cdt, Ddt = lingebm.newmark_ss(Minv, C, k, dt=dt, num_damp=0)

            system = libss.StateSpace(Adt, Bdt, Cdt, Ddt, dt=dt)

            # SISO Gains for DT system
            if system_inputs == 'SISO':
                b_dt = np.zeros((N))
                b_dt[-1] = 1
                system.addGain(b_dt, 'in')

                system.addGain(c, where='out')

        return system

    def run_rom(self, system, algorithm, r, interpolation_point):
        rom = krylov.Krylov()
        rom_settings = {'algorithm': algorithm,
                        'r': r,
                        'frequency': interpolation_point}

        rom.initialise(in_settings=rom_settings)

        rom.run(system)

        return rom

    def compare_freq_resp(self, rom, wv, interpolation_frequency, show_plots=False):

        Y_fom = rom.ss.freqresp(wv)
        Y_rom = rom.ssrom.freqresp(wv)

        interpol_index = np.argwhere(wv >= interpolation_frequency.imag)[0]

        error = np.abs(Y_fom[0, 0, interpol_index] - Y_rom[0, 0, interpol_index])

        if TestKrylovRom.display_output:
            pass
            # fig, ax = plt.subplots(nrows=2)
            # ax[0].semilogx(wv, np.abs(Y_fom[0, 0, :]), 'k-')
            # ax[0].semilogx(wv, np.abs(Y_rom[0, 0, :]), '--', color='0.2')

            # ax[1].semilogx(wv, np.angle(Y_fom[0, 0, :]), 'k-')
            # ax[1].semilogx(wv, np.angle(Y_rom[0, 0, :]), '--', color='0.2')
            # fig.show()
        return error

    def tearDown(self):
        cout.cout_wrap = cout.Writer()

# evals_DT = np.linalg.eigvals(system_DT.A)
#
# evals_dt_conv = np.log(evals_DT) / dt
# #
# plt.scatter(evals_ss.real, evals_ss.imag, marker='s')
# # plt.scatter(evals_dt_conv.real, evals_dt_conv.imag, marker='^')
# # plt.show()
#
# wv = np.logspace(-1, 1, 1000)
# freqresp = system_DT.freqresp(wv)
# freqresp_ct = system_CT.freqresp(wv)
#
# # fig, ax = plt.subplots(nrows=1)
# # bode_mag_dt = (freqresp[0, 0, :].real)
# # bode_mag_ct = (freqresp_ct[0, 0, :].real)
# # ax.semilogx(wv, bode_mag_dt)
# # ax.semilogx(wv, bode_mag_ct, ls='--')
# #
# # fig.show()
#
# print('Routine Complete')
#
# # ROM
# rom = krylov.KrylovReducedOrderModel()
# rom.initialise(data=None,ss=system_DT)
#
# algorithm = 'dual_rational_arnoldi'
# # algorithm = 'arnoldi'
# r = 1
# # frequency = np.array([1.0, 1.005j])
# # frequency = np.array([np.inf])
# frequency = np.array([0.7j, 1.0j])
# z_interpolation = np.exp(frequency*dt)
#
# rom.run(algorithm,r, frequency=z_interpolation)
#
# plot_freq = freq_plots.FrequencyResponseComparison()
# plot_settings = {'frequency_type': 'w',
#                  'plot_type': 'bode'}
#
# plot_freq.initialise(None, system_DT, rom, plot_settings)
# if system_inputs == 'MIMO':
#     plot_freq.plot_frequency_response(wv, freqresp[:3, :3, :], rom.ssrom.freqresp(wv)[:3, :3, :], frequency)
# else:
#     plot_freq.plot_frequency_response(wv, freqresp, rom.ssrom.freqresp(wv), frequency)

# plot_freq.plot_frequency_response(wv, freqresp[4:, 4:, :], rom.ssrom.freqresp(wv), frequency)
# plot_freq.save_figure('DT_07_1_r2.png')

if __name__=='__main__':
    unittest.main()
