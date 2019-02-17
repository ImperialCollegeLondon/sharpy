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

        elif algorithm == 'two_sided_arnoldi':
            Ar, Br, Cr = self.two_sided_arnoldi(frequency, r)

        elif algorithm == 'dual_rational_arnoldi':
            Ar, Br, Cr = self.dual_rational_arnoldi_siso(frequency, r)

        elif algorithm =='dual_rational_arnoldi_single_frequency':
            Ar, Br, Cr = self.dual_rational_arnoldi_single_frequency(frequency, r)

        elif algorithm == 'real_rational_arnoldi':
            Ar, Br, Cr = self.real_rational_arnoldi(frequency, r)

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
        """
        Single expansion point Arnoldi method based on the methods of Gugercin
        Args:
            frequency:
            r:

        Returns:

        """
        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]

        if frequency != np.inf and frequency is not None:
            lu_A = (frequency * np.eye(nx) - A)
            V = self.construct_krylov(r, lu_A, B, 'LUA', 'b')
        else:
            V = self.construct_krylov(r, A, B, 'direct', 'b')

        # Reduced state space model
        Ar = V.T.dot(A.dot(V))
        Br = V.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

    def two_sided_arnoldi(self, frequency, r):
        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]

        if frequency != np.inf and frequency is not None:
            lu_A = frequency * np.eye(nx) - A
            V = self.construct_krylov(r, lu_A, B, 'LUA', 'b')
            W = self.construct_krylov(r, lu_A, C.T, 'LUA', 'c')
        else:
            V = self.construct_krylov(r, A, B, 'direct', 'b')
            W = self.construct_krylov(r, A, C.T, 'direct', 'c')

        # Ensure oblique projection to ensure W^T V = I
        # lu_WW = sclalg.lu_factor(W.T.dot(V))
        # W1 = sclalg.lu_solve(lu_WW, W.T, trans=1).T # Verify
        W = W.dot(sclalg.inv(W.T.dot(V)).T)

        # Reduced state space model
        Ar = W.T.dot(A.dot(V))
        Br = W.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

    @staticmethod
    def construct_krylov(r, lu_A, B, lu_A_type='LUA', side='b'):

        nx = B.shape[0]

        # Side indicates projection side. if using C then it needs to be transposed
        if side=='c':
            transpose_mode = 1
            B.shape = (B.shape[0], )
        else:
            transpose_mode = 0

        # Output projection matrices
        V = np.zeros((nx, r),
                     dtype=complex)
        H = np.zeros((r, r),
                     dtype=complex)

        # Declare iterative variables
        f = np.zeros((nx, r),
                     dtype=complex)

        if lu_A_type != 'direct':
            # LU decomposiotion
            lu_A = sclalg.lu_factor(lu_A)
            v = sclalg.lu_solve(lu_A, B, trans=transpose_mode)
            v = v / np.linalg.norm(v)

            w = sclalg.lu_solve(lu_A, v)
            # w = sclalg.inv((frequency * np.eye(nx) - A)).dot(v)
        else:
            A = lu_A
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

            if lu_A_type != 'direct':
                w = sclalg.lu_solve(lu_A, v, trans=transpose_mode)
            else:
                w = A.dot(v)

            h = V[:, :j+2].T.dot(w)
            f[:, j+1] = w - V[:, :j+2].dot(h)

            # Finite precision
            s = V[:, :j+2].T.dot(f[:, j+1])
            f[:, j+1] = f[:, j+1] - V[:, :j+2].dot(s)  #Confusing line in Gugercin's thesis where it states f_{j=1}?
            h += s

            h.shape = (j+2, 1)  # Enforce shape for concatenation
            H[:j+2, :j+2] = np.block([H_hat, h])

        return V

    def rational_arnoldi(self, frequency, r):
        """
        Rational Arnoldi algorithm for real valued expansion points based on methods of Lee(2006)
        Args:
            frequency:
            r (np.ndarray): Moments to match at every expansion point

        Returns:

        """
        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]
        nfreq = frequency.shape[0]

        # Columns of matrix v
        v_ncols = np.sum(r)

        # Output projection matrices
        V = np.zeros((nx, v_ncols),
                     dtype=complex)
        H = np.zeros((r, v_ncols),
                     dtype=complex)
        res = np.zeros((nx, v_ncols),
                       dtype=complex)

        raise NotImplementedError('Not finished writing the algorithm')

        # # Initialise residual
        # lu_A = sclalg.lu_factor((frequency[0]*np.eye(nx)- A))
        # res[:, 0] = sclalg.lu_solve(lu_A, B)
        #
        # for i in range(0, nfreq):
        #     for j in range(0, r[i]):
        #         k = i*r[i] + j
        #
        #         # Generate orthogonal vector
        #         h = np.linalg.norm(res)
        #         v = res / h
        #
        #         # Update residual
        #         if j == r[i] - 1 and i < nfreq:
        #             lu_A = sclalg.lu_factor(frequency[i+1] * np.eye(nx) - A)
        #             res = sclalg.lu_solve(lu_A, B)
        #         else:
        #             res = sclalg.lu_solve(lu_A, v)
        #
        #         for t in range(k):  # change so it does not get called when k = 0
        #             h[t, k] = v.T.dot(res)
        #             res -= h[t, k].dot(v)  # need to make all of these indexable

    def real_rational_arnoldi(self, frequency, r):
        """
        When employing complex frequencies, the projection matrix can be normalised to be real
        Following Algorithm 1b in Lee(2006)
        Args:
            frequency:
            r:

        Returns:

        """

        ### Not working, having trouble with the last column of H. need to investigate the background behind the creation of H and see hwat can be done

        A = self.ss.A
        B = self.ss.B
        C = self.ss.C

        nx = A.shape[0]
        nfreq = frequency.shape[0]

        # Columns of matrix v
        v_ncols = 2 * np.sum(r)

        # Output projection matrices
        V = np.zeros((nx, v_ncols),
                     dtype=float)
        H = np.zeros((v_ncols, v_ncols),
                     dtype=float)
        res = np.zeros((nx,v_ncols+2),
                       dtype=float)

        lu_A = sclalg.lu_factor(frequency[0] * np.eye(nx) - A)
        v_res = sclalg.lu_solve(lu_A, B)

        H[0, 0] = np.linalg.norm(v_res)
        V[:, 0] = v_res.real / H[0, 0]

        k = 0
        for i in range(nfreq):
            for j in range(r[i]):
                # k = 2*(i*r[i] + j)
                print("i = %g\t j = %g\t k = %g" % (i, j, k))

                # res[:, k] = np.imag(v_res)
                # if k > 0:
                #     res[:, k-1] = np.real(v_res)
                #
                # # Working on the last finished column i.e. k-1 only when k>0
                # if k > 0:
                #     for t in range(k):
                #         H[t, k-1] = V[:, t].T.dot(res[:, k-1])
                #         res[:, k-1] -= res[:, k-1] - H[t, k-1] * V[:, t]
                #
                #     H[k, k-1] = np.linalg.norm(res[:, k-1])
                #     V[:, k] = res[:, k-1] / H[k, k-1]
                #
                # # Normalise working column k
                # for t in range(k+1):
                #     H[t, k] = V[:, t].T.dot(res[:, k])
                #     res[:, k] -= H[t, k] * V[:, t]
                #
                # # Subdiagonal term
                # H[k+1, k] = np.linalg.norm(res[:, k])
                # V[:, k + 1] = res[:, k] / np.linalg.norm(res[:, k])
                #
                # if j == r[i] - 1 and i < nfreq - 1:
                #     lu_A = sclalg.lu_factor(frequency[i+1] * np.eye(nx) - A)
                #     v_res = sclalg.lu_solve(lu_A, B)
                # else:
                #     v_res = - sclalg.lu_solve(lu_A, V[:, k+1])

                if k == 0:
                    V[:, 0] = v_res.real / np.linalg.norm(v_res.real)
                else:
                    res[:, k] = np.imag(v_res)
                    res[:, k-1] = np.real(v_res)

                    for t in range(k):
                        H[t, k-1] = np.linalg.norm(res[:, k-1])
                        res[:, k-1] -= H[t, k-1]*V[:, t]

                    H[k, k-1] = np.linalg.norm(res[:, k-1])
                    V[:, k] = res[:, k-1] / H[k, k-1]

                if k == 0:
                    H[0, 0] = V[:, 0].T.dot(v_res.imag)
                    res[:, 0] -= H[0, 0] * V[:, 0]

                else:
                    for t in range(k+1):
                        H[t, k] = V[:, t].T.dot(res[:, k])
                        res[:, k] -= H[t, k] * V[:, t]
                H[k+1, k] = np.linalg.norm(res[:, k])
                V[:, k+1] = res[:, k] / H[k+1, k]

                if j == r[i] - 1 and i < nfreq - 1:
                    lu_A = sclalg.lu_factor(frequency[i+1]*np.eye(nx) - A)
                    v_res = sclalg.lu_solve(lu_A, B)
                else:
                    v_res = - sclalg.lu_solve(lu_A, V[:, k+1])

                k += 2

        # Add last column of H
        print(k)
        res[:, k-1] = - sclalg.lu_solve(lu_A, V[:, k-1])
        for t in range(k-1):
            H[t, k-1] = V[:, t].T.dot(res[:, k-1])
            res[:, k-1] -= H[t, k-1]*V[:, t]

        self.V = V
        self.H = H

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

        V = np.zeros((nx, r*nfreq), dtype=complex)
        W = np.zeros((nx, r*nfreq), dtype=complex)

        we = 0
        for i in range(nfreq):
            sigma = frequency[i]
            lu_A = sigma * np.eye(nx) - A
            V[:, we:we+r] = self.construct_krylov(r, lu_A, B, 'LUA', 'b')
            W[:, we:we+r] = self.construct_krylov(r, lu_A, C.T, 'LUA', 'c')

            we += r

        W = W.dot(sclalg.inv(W.T.dot(V)).T)
        self.W = W
        self.V = V

        # Reduced state space model
        Ar = W.T.dot(A.dot(V))
        Br = W.T.dot(B)
        Cr = C.dot(V)

        return Ar, Br, Cr

        # # Allocate matrices
        # V = np.zeros((nx, nfreq * r), dtype=complex)
        # Z = np.zeros((nx, nfreq * r), dtype=complex)
        #
        # # Is it better to store column vectors in matrix or in list?
        # # for v_hat and _tilde, probably not required
        #
        # # Store tuple of LU decompositions of (A - sigma I) in list
        # lu_frequency = [None] * nfreq
        #
        # m = 0
        # for j in range(0, r):
        #     for k in range(0, nfreq):
        #         if j == 0:
        #             # Lu factorisation of (A - sigma I) for each sigma (frequency). Calculated only once
        #             lu_k = sclalg.lu_factor((A - frequency[k] * np.eye(nx)))
        #             lu_frequency[k] = lu_k  # Add LU factorisation for the k-th frequency to the database
        #         # if j == 1:
        #             v_tilde = sclalg.lu_solve(lu_k, B)
        #             z_tilde = sclalg.lu_solve(lu_k, C.T, trans=1)
        #             z_tilde = z_tilde[:,0]
        #         else:
        #             v_tilde = sclalg.lu_solve(lu_frequency[k], V[:, m-nfreq])
        #             z_tilde = sclalg.lu_solve(lu_frequency[k], Z[:, m-nfreq], trans=1)
        #
        #         v_hat = v_tilde - V[:, :m+1].dot(V[:, :m+1].T.dot(v_tilde))
        #         z_hat = z_tilde - Z[:, :m+1].dot(Z[:, :m+1].T.dot(z_tilde))
        #
        #         V[:, m] = v_hat / np.linalg.norm(v_hat)
        #         Z[:, m] = z_hat / np.linalg.norm(z_hat)
        #
        #         m += 1
        #
        # Ar = Z.T.dot(A.dot(V))
        # Br = Z.T.dot(B)
        # Cr = C.dot(V)

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

    def compare_frequency_response(self, return_error=False, plot_figures=False):
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

        frequency = self.frequency
        # TODO to be modified for plotting purposes when using multi rational interpolation
        try:
            nfreqs = frequency.shape[0]
        except AttributeError:
            nfreqs = 1

        if frequency is None:
            k_rom = np.inf
        else:
            k_rom = c_ref * frequency * 0.5 / Uinf0

        display_frequency = '$\sigma$ ='
        if nfreqs > 1:
            display_frequency += ' ['
            for i in range(nfreqs):
                if type(k_rom[i]) == complex:
                    display_frequency += ' %.1f + %.1fj' % (k_rom[i].real, k_rom[i].imag)
                else:
                    display_frequency += ' %.1f' % k_rom[i]
                display_frequency += ','
            display_frequency += ']'
        else:
            if type(k_rom) == complex:
                display_frequency += ', %.1f + %.1fj' % (k_rom.real, k_rom.imag)
            else:
                display_frequency += ', %.1f' % k_rom

        nstates = self.ss.states
        rstates = self.r

        # Compute the frequency response
        Y_full_system = libss.freqresp(self.ss, wv)
        Y_freq_rom = libss.freqresp(self.ssrom, wv)

        rel_error = (Y_freq_rom[0, 0, :] - Y_full_system[0, 0, :]) / Y_full_system[0, 0, :]

        fig, ax = plt.subplots(nrows=2)

        if plot_figures:

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
            axins1.set_ylim([-3.5, 3.5])

            ax[1].set_xlabel('Reduced Frequency, k')
            # ax.set_ylabel('Normalised Response')
            ax[0].set_title('ROM - %s, r = %g, %s' % (self.algorithm, rstates, display_frequency))
            ax[0].legend()

            fig.show()

            # Relative error
            fig, ax = plt.subplots()

            real_rel_error = np.abs(rel_error.real)
            imag_rel_error = np.abs(rel_error.imag)

            ax.semilogy(kv, real_rel_error,
                        color='k',
                        lw=1.5,
                        label='Real')

            ax.semilogy(kv, imag_rel_error,
                        ls='--',
                        color='k',
                        lw=1.5,
                        label='Imag')

            ax.set_title('ROM - %s, r = %g, %s' % (self.algorithm, rstates, display_frequency))
            ax.set_xlabel('Reduced Frequency, k')
            ax.set_ylabel('Relative Error')
            ax.set_ylim([1e-5, 1])
            ax.legend()
            fig.show()
            fig.savefig('./figs/theo_rolled/DRA_0_04_r2.eps')

        if return_error:
            return kv, rel_error


def evec(j):
    """j-th unit vector (in row format)"""
    e = np.zeros(j+1)
    e[j] = 1
    return e

if __name__ == "__main__":
    pass
