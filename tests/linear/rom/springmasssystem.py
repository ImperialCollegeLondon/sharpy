"""
Generate a mass spring system
NGoizueta 16 Feb 2019

"""

import numpy as np
import matplotlib.pyplot as plt
import sharpy.linear.src.libss as libss
import sharpy.linear.src.lingebm as lingebm
import sharpy.rom.reducedordermodel as ROM
import sharpy.rom.frequencyresponseplot as freq_plots

N = 5

k_db = np.linspace(1, 10, N)
m_db = np.logspace(2, 0, N)
C_db = np.ones(N) * 1e-1

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


# Continuous time SISO
# Input Rn
b = np.zeros((2*N, ))
b[-1] = 1

# Output rn
c = np.zeros((1, 2*N))
c[0, N-1] = 1
d = np.zeros((1))

# Plant matrix
Minv = np.linalg.inv(m)
MinvK = Minv.dot(k)
A = np.zeros((2*N, 2*N))
A[:N, N:] = np.eye(N)
A[N:, :N] = -MinvK
A[N:, N:] = -Minv.dot(C)

system_CT = libss.ss(A, b, c, d, dt=None)

evals_ss = np.linalg.eigvals(system_CT.A)

# Discrete time system
dt = 1e-2
Adt, Bdt, Cdt, Ddt = lingebm.newmark_ss(Minv, C, k, dt=dt, num_damp=0)

system_DT = libss.ss(Adt, Bdt, Cdt, Ddt, dt=dt)

# SISO Gains
b_dt = np.zeros((N))
b_dt[-1] = 1
system_DT.addGain(b_dt, 'in')

system_DT.addGain(c, where='out')

evals_DT = np.linalg.eigvals(system_DT.A)

evals_dt_conv = np.log(evals_DT) / dt
#
# plt.scatter(evals_ss.real, evals_ss.imag, marker='s')
# plt.scatter(evals_dt_conv.real, evals_dt_conv.imag, marker='^')
# plt.show()

wv = np.logspace(-1, 1, 1000)
freqresp = system_DT.freqresp(wv)
freqresp_ct = system_CT.freqresp(wv)

# fig, ax = plt.subplots(nrows=1)
# bode_mag_dt = (freqresp[0, 0, :].real)
# bode_mag_ct = (freqresp_ct[0, 0, :].real)
# ax.semilogx(wv, bode_mag_dt)
# ax.semilogx(wv, bode_mag_ct, ls='--')
#
# fig.show()

print('Routine Complete')

# ROM
rom = ROM.ReducedOrderModel()
rom.initialise(data=None,ss=system_DT)

algorithm = 'dual_rational_arnoldi'
# algorithm = 'arnoldi'
r = 1
# frequency = np.array([1.0, 1.005j])
# frequency = np.array([2.2, 1.])
frequency = np.array([0.7j, 1.0j])
z_interpolation = np.exp(frequency*dt)

rom.run(algorithm,r, frequency=z_interpolation)

rom.compare_frequency_response(wv, plot_figures=False)

plot_freq = freq_plots.FrequencyResponseComparison()
plot_settings = {'frequency_type': 'w',
                 'plot_type': 'bode'}

plot_freq.initialise(None, system_DT, rom, plot_settings)
plot_freq.plot_frequency_response(wv, freqresp, rom.ssrom.freqresp(wv), frequency)
# plot_freq.save_figure('DT_07_1_r2.png')
