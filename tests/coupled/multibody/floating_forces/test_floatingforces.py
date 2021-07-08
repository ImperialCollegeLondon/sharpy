import numpy as np
import unittest
import os
import shutil
from scipy import fft
import sharpy.generators.floatingforces as ff


class TestFloatingForces(unittest.TestCase):
    """
    Some tests of the floating forces library
    Check references therein
    """

    def test_compute_xf_zf(self):
        """
            This function tests based on by hand computations and data from the MooringLineFD.txt file
            from the OC3 task.
                Jonkman, J.
                Definition of the Floating System for Phase IV of OC3
                NREL/TP-500-47535
        """
    
        # Values for OC3
        l = 902.2 # Initial length [m]
        w = 698.094 # Aparent weight 77.7066*9.81 # Apparent mass per unit length times gravity
        EA = 384243000. # Extensional stiffness
        cb = 0.1 # Seabed friction coefficient
    
        # No mooring line on the Seabed
        vf = 1.1*l*w # 692802.4475
        hf = vf
        xf_byhand = 784.5965853 + 1.626695524
        zf_byhand = 406.9813526 + 0.887288467
        xf, zf = ff.compute_xf_zf(hf, vf, l, w, EA, cb)
        self.assertAlmostEqual(xf_byhand, xf, 4)
        self.assertAlmostEqual(zf_byhand, zf, 4)
    
        # Some mooring line on the Seabed
        lb_div_l = 0.1 # 10% of the mooring line on the seabed
        vf = (1-lb_div_l)*l*w
        hf = vf
        xf_byhand = 90.22 + 715.6577252 + 1.330932701 - 7.298744381e-4
        zf_byhand = 336.3331284 + 0.598919715
        xf, zf = ff.compute_xf_zf(hf, vf, l, w, EA, cb)
        self.assertAlmostEqual(xf_byhand, xf, 4)
        self.assertAlmostEqual(zf_byhand, zf, 4)
    
    
    def test_generate_mooringlinefd(self):
        """
            This function generates a file similar to MoorinLinesFD.txt for compiarison
            File obtained from the google drive directory of the OC3 benchmark
        """
    
        # Values for OC3
        l = 902.2 # Initial length [m]
        w = 698.094 # Aparent weight 77.7066*9.81 # Apparent mass per unit length times gravity
        EA = 384243000. # Extensional stiffness
        cb = 0. # Seabed friction coefficient
    
        zf = 320. - 70.
    
        # xf0 = 853.87
        # xf_list = np.arange(653.00, 902.50 + 1., 1.)
        xf_list = np.arange(653.00, 902.50, 10.)
        npoints = xf_list.shape[0]
        output = np.zeros((npoints, 4))
        for i in range(npoints):
            hf, vf = ff.quasisteady_mooring(xf_list[i], zf, l, w, EA, cb, hf0=None, vf0=None)
            # print(xf0, zf0, hf0, vf0)
            lb = np.maximum(l - vf/w, 0)
            # print("Suspended lenght = %f" % (l - lb))
            output[i, :] = np.array([xf_list[i], np.sqrt(vf**2 + hf**2)*1e-3, hf*1e-3, (l - lb)])

        of_results = np.loadtxt("MooringLineFD.dat", skiprows=9)
        for i in range(npoints):
            for icol in range(4):
                of_value = np.interp(xf_list[i], of_results[:, 0], of_results[:, icol])
                error = np.abs((np.round_(output[i, icol], 1) - of_value)/of_value)
                self.assertLess(error, 0.1)
                
        save_txt = False
        if save_txt:    
            np.savetxt("sharpy_mooringlinefd.txt", output, header="# DISTANCE(m) TENSION(kN) HTENSION(kN) SUSPL(m)")

    
    def test_change_system(self):
        # Wind turbine degrees of freedom: Surge, sway, heave, roll, pitch, yaw.
        # SHARPy axis associated:              z,    y,     x,    z,     y,   x
    
        wt_matrix_num = np.zeros((6,6),)
        for idof in range(6):
            for jdof in range(6):
                wt_matrix_num[idof, jdof] = 10.*idof + jdof
    
        sharpy_matrix = ff.change_of_to_sharpy(wt_matrix_num)
        undo_sharpy_matrix = ff.change_of_to_sharpy(sharpy_matrix)
        
        for idof in range(6):
            for jdof in range(6):
                self.assertEqual(wt_matrix_num[idof, jdof], undo_sharpy_matrix[idof, jdof])
   
 
    def test_time_wave_forces(self):
        Tp = 14.656 #10.
        Hs = 5.49 #6.
        nrealisations = 100
        dt = 2*np.pi/4 # 1./20 To get max freq equal to 10Hz
        ntime_steps = 1000
        time = np.arange(ntime_steps)*dt
    
        # Get the zero-noise specturm
        w_js = np.arange(0, 4, 0.01)
        zero_noise_spectrum = ff.jonswap_spectrum(Tp, Hs, w_js)
    
        # Compute different realisations
        xi = np.zeros((2, 6), dtype=complex)
        xi[0, 0] = 1. + 0j
        xi[1, 0] = 1. + 0j
        w_xi = np.array([0., 4.])
        wave_force = np.zeros((ntime_steps, nrealisations), dtype=np.complex)
        for ireal in range(nrealisations):
            wave_force[:, ireal] = ff.time_wave_forces(Tp, Hs, dt, time, xi, w_xi)[:, 0] # Keep only on dimension
    
        # Compute the spectrum of the realisations
        ns = np.zeros((ntime_steps//2, nrealisations), dtype=np.complex)
        for ireal in range(nrealisations):
            ns[:, ireal] = dt/ntime_steps*np.abs(fft(wave_force[:, ireal])[:ntime_steps//2])**2
            ns[1:, ireal] *= 2
        # To rad/s
        ns /= 2.*np.pi
        w_ns = (np.fft.fftfreq(ntime_steps, d=dt)[:ntime_steps//2])*2.*np.pi
        # Compare the zero noise with the realisations average
        avg_noise_spectrum = np.average(ns, axis=1)

        for iomega in range(w_js.shape[0]):
            error = (np.interp(w_js[iomega], w_ns, avg_noise_spectrum) - zero_noise_spectrum[iomega])
            if error > 0.1:
                error /= zero_noise_spectrum[iomega]
                # 0.3 is a large error but otherwise I need to increment nrealisations a lot
                # Use ``save_fig = True`` for visual inspection
                self.assertLess(error, 0.3)
            
        save_fig = False
        if save_fig:
            np.savetxt("zero_noise_jonswap.txt", np.column_stack((w_js, zero_noise_spectrum)))
            np.savetxt("realisations_jonswap.txt", np.column_stack((w_ns, np.abs(ns))))
            np.savetxt("average_jonswap.txt", np.column_stack((w_ns, avg_noise_spectrum)))
             
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            ax.grid()
            ax.set_xlabel("omega [rad/s]")
            ax.set_ylabel("specturm")
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 12)
            for ireal in range(nrealisations):
                ax.plot(w_ns, np.abs(ns[:, ireal]), 'bo')
            ax.plot(w_ns, avg_noise_spectrum, '-', label="avg")
            ax.plot(w_js, zero_noise_spectrum, '--', label="JONSWAP")
            fig.legend()
            fig.tight_layout()
            fig.show()
            fig.savefig("spectrum.png")
            plt.close()


    # def tearDown(self):
    #     solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    #     solver_path += '/'
    #     files_to_delete = [name + '.aero.h5',
    #                        name + '.dyn.h5',
    #                        name + '.fem.h5',
    #                        name + '.mb.h5',
    #                        name + '.sharpy']
    #     for f in files_to_delete:
    #         os.remove(solver_path + f)

    #     shutil.rmtree(solver_path + 'output/')
