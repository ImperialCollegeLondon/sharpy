import numpy as np
import scipy.linalg as sclalg
import sharpy.linear.src.libss as libss
import sharpy.utils.h5utils as h5

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes

class ReducedOrderModel(object):
    """
    Reduced Order Model
    """

    def __init__(self):

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['algorithm'] = 'str'
        self.settings_default['algorithm'] = None

        self.settings_types['frequencies'] = 'list(complex)'
        self.settings_default['frequencies'] = None

        self.frequency = None
        self.algorithm = None
        self.ss = None
        self.r = 100
        self.V = None
        self.H = None
        self.ssrom = None
        self.data = None

    def initialise(self, data, ss):

        self.data = data
        self.ss = ss

    def run(self, algorithm, r, frequency=None):
        self.algorithm = algorithm
        self.frequency = frequency
        self.r = r

        if algorithm == 'arnoldi':
            Ar, Br, Cr = self.single_arnoldi(frequency, r)

        elif algorithm == 'dual_rational_arnoldi':
            Ar, Br, Cr = self.dual_rational_arnoldi_siso(frequency, r)

        elif algorithm =='dual_rational_arnoldi_single_frequency':
            Ar, Br, Cr = self.dual_rational_arnoldi_single_frequency(frequency, r)

        else:
            raise NotImplementedError('Algorithm %s not recognised, check for spelling or it may not be implemented'
                                      %algorithm)

        self.ssrom = libss.ss(Ar, Br, Cr, self.ss.D, self.ss.dt)

    def lanczos(self):
        """
        Warnings:
            Not Implemented

        Returns:

        """

        raise NotImplementedError('Lanczos Interpolation not yet implemented')

        A = self.ss.A
        B = self.ss.B
        C = self.ss.C
        r = self.r

        nx = A.shape[0]
        nu = 1
        ny = 1

        beta = np.sqrt(np.abs(C.dot(B)))
        gamma = np.sign(C.dot(B)) * beta
        v = 1 / beta * B
        gamma = 1 / gamma * C.T

        for j in range(0,r-1):
            pass
            # Appears to be an issue in the algorithm with v(j-1) at iteration j [missing declaration]
            # alpha = w.T.dot(A.dot(v))
            # r = A.dot(v) - alpha * v
            # q = A.T.dot(w) - alpha * w - beta *

    def single_arnoldi(self, frequency, r):
            A = self.ss.A
            B = self.ss.B
            C = self.ss.C


            nx = A.shape[0]

            # Output projection matrices
            V = np.zeros((nx, r),
                         dtype=complex)
            H = np.zeros((r, r),
                         dtype=complex)

            # Declare iterative variables
            f = np.zeros((nx, r),
                         dtype=complex)

            if frequency != np.inf and frequency is not None:
                # LU decomposiotion
                lu_A = sclalg.lu_factor((A - frequency * np.eye(nx)))
                v = sclalg.lu_solve(lu_A, B)
                # v = sclalg.inv(frequency * np.eye(nx) - A).dot(B)
                v = v / np.linalg.norm(v)

                w = sclalg.lu_solve(lu_A, v)
                # w = sclalg.inv((frequency * np.eye(nx) - A)).dot(v)
            else:
                v_arb = B
                v = v_arb / np.linalg.norm(v_arb)
                w = A.dot(v)

            alpha = v.T.dot(w)

            # Initial assembly
            f[:, 0] = w - v.dot(alpha)
            V[:, 0] = v
            H[0, 0] = alpha

            for j in range(0, r-1):

                beta = np.linalg.norm(f[:, j])
                v = 1 / beta * f[:, j]

                V[:, j+1] = v
                H_hat = np.block([[H[:j+1, :j+1]],
                                 [beta * evec(j)]])

                if frequency != np.inf and frequency is not None:
                    w = sclalg.lu_solve(lu_A, v)
                    # w_1 = sclalg.inv(frequency * np.eye(nx) - A).dot(v)
                else:
                    w = A.dot(v)

                h = V[:,:j+2].T.dot(w)
                f[:,j+1] = w - V[:, :j+2].dot(h)

                # Finite precision
                s = V[:, :j+2].T.dot(f[:, j+1])
                f[:, j+1] = f[:, j+1] - V[:, :j+2].dot(s)  #Confusing line in Gugercin's thesis where it states f_{j=1}?
                h += s

                h.shape = (j+2, 1)  # Enforce shape for concatenation
                H[:j+2, :j+2] = np.block([H_hat, h])

            self.V = V
            self.H = H

            # Reduced state space model
            Ar = V.T.dot(A.dot(V))
            Br = V.T.dot(B)
            Cr = C.dot(V)

            return Ar, Br, Cr

    def dual_rational_arnoldi_siso(self, frequency, r):
        """
        Dual Rational Arnoli Interpolation for SISO sytems. Based on the methods of Grimme
        Args:
            frequency:
            r:

        Returns:

        """
        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]

        try:
            nfreq = frequency.shape[0]
        except AttributeError:
            nfreq = 1

        assert nfreq > 1, 'Dual Rational Arnoldi required more than one frequency to interpolate'

        # Allocate matrices
        V = np.zeros((nx, nfreq * r), dtype=complex)
        Z = np.zeros((nx, nfreq * r), dtype=complex)

        # Is it better to store column vectors in matrix or in list?
        # for v_hat and _tilde, probably not required

        # Store tuple of LU decompositions of (A - sigma I) in list
        lu_frequency = [None] * nfreq

        m = 0
        for j in range(0, r):
            for k in range(0, nfreq):
                if j == 0:
                    # Lu factorisation of (A - sigma I) for each sigma (frequency). Calculated only once
                    lu_k = sclalg.lu_factor((A - frequency[k] * np.eye(nx)))
                    lu_frequency[k] = lu_k  # Add LU factorisation for the k-th frequency to the database
                # if j == 1:
                    v_tilde = sclalg.lu_solve(lu_k, B)
                    z_tilde = sclalg.lu_solve(lu_k, C.T, trans=1)
                    z_tilde = z_tilde[:,0]
                else:
                    v_tilde = sclalg.lu_solve(lu_frequency[k], V[:, m-nfreq])
                    z_tilde = sclalg.lu_solve(lu_frequency[k], Z[:, m-nfreq], trans=1)

                v_hat = v_tilde - V[:, :m+1].dot(V[:, :m+1].T.dot(v_tilde))
                z_hat = z_tilde - Z[:, :m+1].dot(Z[:, :m+1].T.dot(z_tilde))

                V[:, m] = v_hat / np.linalg.norm(v_hat)
                Z[:, m] = z_hat / np.linalg.norm(z_hat)

                m += 1

        Ar = Z.T.dot(A.dot(V))
        Br = Z.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

    def dual_rational_arnoldi_single_frequency(self, frequency, r):
        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]

        try:
            nfreq = frequency.shape[0]
        except AttributeError:
            nfreq = 1


        # Allocate matrices
        V = np.zeros((nx, nfreq * r), dtype=float)
        Z = np.zeros((nx, nfreq * r), dtype=float)

        # Is it better to store column vectors in matrix or in list?
        # for v_hat and _tilde, probably not required

        # Store tuple of LU decompositions of (A - sigma I) in list
        lu_k = sclalg.lu_factor((A - frequency * np.eye(nx)))
        v_tilde = sclalg.lu_solve(lu_k, B)
        V[:, 0] = v_tilde / np.linalg.norm(v_tilde)

        for j in range(1, r):
            v_tilde = sclalg.lu_solve(lu_k, V[:, j-1])
            # z_tilde = sclalg.lu_solve(lu_k, Z[:, m], trans=1)

            v_hat = v_tilde - V[:, :j].dot(V[:, :j].T.dot(v_tilde))
            # z_hat = z_tilde - Z[:, :m+1].dot(Z[:, :m+1].T.dot(z_tilde))

            V[:, j] = v_hat / np.linalg.norm(v_hat)
            # Z[:, m] = z_hat / np.linalg.norm(z_hat)


        # Ar = Z.T.dot(A.dot(V))
        # Br = Z.T.dot(B)
        # Cr = C.dot(V)

        Ar = V.T.dot(A.dot(V))
        Br = V.T.dot(B)
        Cr = C.dot(V)
        return Ar, Br, Cr

    def compare_frequency_response(self):
        """
        Computes the frequency response of the full and reduced models up to the Nyquist frequency
        
        Returns:

        """
        Uinf0 = self.data.aero.timestep_info[0].u_ext[0][0, 0, 0]
        c_ref = self.data.aero.timestep_info[0].zeta[0][0, -1, 0] - self.data.aero.timestep_info[0].zeta[0][0, 0, 0]
        ds = 2. / self.data.aero.aero_dimensions[0][0]  # Spatial discretisation
        fs = 1. / ds
        fn = fs / 2.
        ks = 2. * np.pi * fs
        kn = 2. * np.pi * fn  # Nyquist frequency
        Nk = 151  # Number of frequencies to evaluate
        kv = np.linspace(1e-3, kn, Nk)  # Reduced frequency range
        wv = 2. * Uinf0 / c_ref * kv  # Angular frequency range

        nstates = self.ss.states
        rstates = self.r

        # Compute the frequency response
        Y_full_system = libss.freqresp(self.ss, wv)
        Y_freq_rom = libss.freqresp(self.ssrom, wv)

        fig, ax = plt.subplots(nrows=2)


        if self.frequency is None:  # for plotting purposes
            k_rom = np.inf
        else:
            k_rom = c_ref * self.frequency[0].real * 0.5 / Uinf0

        ax[0].plot(kv, np.abs(Y_full_system[0, 0, :]),
                   lw=4,
                   alpha=0.5,
                   color='b',
                   label='UVLM - %g states' % nstates)
        ax[1].plot(kv, np.angle((Y_full_system[0, 0, :])), ls='-',
                   lw=4,
                   alpha=0.5,
                   color='b')

        ax[1].set_xlim(0, kv[-1])
        ax[0].grid()
        ax[1].grid()
        ax[0].plot(kv, np.abs(Y_freq_rom[0, 0, :]), ls='-.',
                   lw=1.5,
                   color='k',
                   label='ROM - %g states' % rstates)
        ax[1].plot(kv, np.angle((Y_freq_rom[0, 0, :])), ls='-.',
                   lw=1.5,
                   color='k')

        axins0 = inset_axes(ax[0], 1, 1, loc=1)
        axins0.plot(kv, np.abs(Y_full_system[0, 0, :]),
                    lw=4,
                    alpha=0.5,
                    color='b')
        axins0.plot(kv, np.abs(Y_freq_rom[0, 0, :]), ls='-.',
                    lw=1.5,
                    color='k')
        axins0.set_xlim([0, 1])
        axins0.set_ylim([0, 0.1])

        axins1 = inset_axes(ax[1], 1, 1.25, loc=1)
        axins1.plot(kv, np.angle((Y_full_system[0, 0, :])), ls='-',
                    lw=4,
                    alpha=0.5,
                    color='b')
        axins1.plot(kv, np.angle((Y_freq_rom[0, 0, :])), ls='-.',
                    lw=1.5,
                    color='k')
        axins1.set_xlim([0, 1])
        axins1.set_ylim([-np.pi, np.pi])

        ax[1].set_xlabel('Reduced Frequency, k')
        # ax.set_ylabel('Normalised Response')
        ax[0].set_title('ROM - %s, r = %g, $\sigma_k$ = %.1f' % (self.algorithm, rstates, self.frequency[0]))
        ax[0].legend()

        fig.show()


def evec(j):
    """j-th unit vector (in row format)"""
    e = np.zeros(j+1)
    e[j] = 1
    return e

if __name__ == "__main__":
    pass
