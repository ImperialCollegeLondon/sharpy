import numpy as np
import h5py as h5
import ctypes as ct
import os
from scipy import fft, ifft
from control import forced_response, TransferFunction

import sharpy.utils.cout_utils as cout
import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.constants import deg2rad
import sharpy.utils.h5utils as h5utils
import sharpy.utils.algebra as algebra


def compute_xf_zf(hf, vf, l, w, EA, cb):
    """
        Fairlead location (xf, zf) computation
    """

    # Rename some terms for convenience
    root1 = np.sqrt(1. + (vf/hf)**2)
    root2 = np.sqrt(1. + ((vf - w*l)/hf)**2)
    ln1 = np.log(vf/hf + root1)
    ln2 = np.log((vf - w*l)/hf + root2)
    lb = l - vf/w

    # Define if there is part of the mooring line on the bed
    if lb <= 0:
        nobed = True
    else:
        nobed = False

    # Compute the position of the fairlead
    if nobed:
        xf = hf/w*(ln1 - ln2) + hf*l/EA
        zf = hf/w*(root1 - root2) + 1./EA*(vf*l-w*l**2/2)
    else:
        xf = lb + hf/w*ln1 + hf*l/EA
        if not cb == 0.:
            xf += cb*w/2/EA*(-lb**2 + (lb - hf/cb/w)*np.maximum((lb - hf/cb/w), 0))
        zf = hf/w*(root1 - 1) + vf**2/2/EA/w

    return xf, zf


def test_compute_xf_zf():
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
    xf, zf = compute_xf_zf(hf, vf, l, w, EA, cb)
    print("Case without mooring line on the seabed")
    print("xf=%f and xf_byhand=%f" % (xf, xf_byhand))
    print("zf=%f and zf_byhand=%f" % (zf, zf_byhand))

    # Some mooring line on the Seabed
    lb_div_l = 0.1 # 10% of the mooring line on the seabed
    vf = (1-lb_div_l)*l*w
    hf = vf
    xf_byhand = 90.22 + 715.6577252 + 1.330932701 - 7.298744381e-4
    zf_byhand = 336.3331284 + 0.598919715
    xf, zf = compute_xf_zf(hf, vf, l, w, EA, cb)
    print("Case with %f%% mooring line on the seabed" % (lb_div_l*100))
    print("xf=%f and xf_byhand=%f" % (xf, xf_byhand))
    print("zf=%f and zf_byhand=%f" % (zf, zf_byhand))

    # From solution file
    xf, zf = compute_xf_zf(0.1*1e3, 174.599971*1e3, l, w, EA, cb)
    distance = 653.
    print("Case with mooring line on the seabed from file")
    print("xf=%f and distance=%f. zf=%f" % (xf, distance, zf))
    xf, zf = compute_xf_zf(1415.6*1e3, 730.565288*1e3, l, w, EA, cb)
    distance = 864.
    print("Case without mooring line on the seabed from file")
    print("xf=%f and distance=%f. zf=%f" % (xf, distance, zf))


def compute_jacobian(hf, vf, l, w, EA, cb):
    """
        Analytical computation of the Jacobian of equations
        in function compute_xf_zf
    """

    # Rename some terms for convenience
    root1 = np.sqrt(1. + (vf/hf)**2)
    root2 = np.sqrt(1. + ((vf - w*l)/hf)**2)
    ln1 = np.log(vf/hf + root1)
    ln2 = np.log((vf - w*l)/hf + root2)
    lb = l - vf/w

    # Compute their deivatives
    der_root1_hf = 0.5*(1. + (vf/hf)**2)**(-0.5)*(2*vf/hf*(-vf/hf/hf))
    der_root1_vf = 0.5*(1. + (vf/hf)**2)**(-0.5)*(2*vf/hf/hf)

    der_root2_hf = 0.5*(1. + ((vf - w*l)/hf)**2)**(-0.5)*(2.*(vf - w*l)/hf*(-(vf - w*l)/hf/hf))
    der_root2_vf = 0.5*(1. + ((vf - w*l)/hf)**2)**(-0.5)*(2.*(vf - w*l)/hf/hf)

    der_ln1_hf = 1./(vf/hf + root1)*(vf/hf/hf + der_root1_hf)
    der_ln1_vf = 1./(vf/hf + root1)*(1./hf + der_root1_vf)

    der_ln2_hf = 1./((vf - w*l)/hf + root2)*(-(vf - w*l)/hf/hf + der_root2_hf)
    der_ln2_vf = 1./((vf - w*l)/hf + root2)*(1./hf + der_root2_vf)

    der_lb_hf = 0.
    der_lb_vf = -1./w

    # Define if there is part of the mooring line on the bed
    if lb <= 0:
        nobed = True
    else:
        nobed = False

    # Compute the Jacobian
    if nobed:
        der_xf_hf = 1./w*(ln1 - ln2) + hf/w*(der_ln1_hf + der_ln2_hf) + l/EA
        der_xf_vf = hf/w*(der_ln1_vf + der_ln2_vf)

        der_zf_hf = 1./w*(root1 - root2) + hf/w*(der_root1_hf - der_root2_hf)
        der_zf_vf = hf/w*(der_root1_vf - der_root2_vf) + 1./EA*l
    else:
        der_xf_hf = der_lb_hf + 1./w*ln1 + hf/w*der_ln1_hf + l/EA
        if not cb == 0.:
            arg1_max = l - vf/w - hf/cb/w
            if arg1_max > 0.:
                der_xf_hf += cb*w/2/EA*(2*(arg1_max)*(-1/cb/w))

        der_xf_vf = der_lb_vf + hf/w*der_ln1_vf + cb*w/2/EA*(-2.*lb*der_lb_vf)
        if not cb == 0.:
            arg1_max = l - vf/w - hf/cb/w
            if arg1_max > 0.:
                der_xf_vf += cb*w/2/EA*(2.*(lb - hf/cb/w)*der_lb_vf)

        der_zf_hf = 1/w*(root1 - 1) + hf/w*der_root1_hf
        der_zf_vf = hf/w*der_root1_vf + vf/EA/w

    J = np.array([[der_xf_hf, der_xf_vf],[der_zf_hf, der_zf_vf]])

    return J


def quasisteady_mooring(xf, zf, l, w, EA, cb, hf0=None, vf0=None):
    """
        Computation of the forces generated by the mooring system
        It performs a Newton-Raphson iteration based on the known equations
        in compute_xf_zf function and the Jacobian
    """

    # Initialise guess for hf0 and vf0
    if xf == 0:
        lambda0 = 1e6
    elif np.sqrt(xf**2 + zf**2) > l:
        lambda0 = 0.2
    else:
        lambda0 = np.sqrt(3*((l**2 - zf**2)/xf**2 - 1))

    if hf0 is None:
        hf0 = np.abs(w*xf/2/lambda0)

    if vf0 is None:
        vf0 = w/2*(zf/np.tanh(lambda0) + l)

    # Compute the solution through Newton-Raphson iteration
    hf_est = hf0 + 0.
    vf_est = vf0 + 0.
    xf_est, zf_est = compute_xf_zf(hf_est, vf_est, l, w, EA, cb)
    # print("initial: ", xf_est, zf_est)
    tol = 1e-6
    error = 2*tol
    max_iter = 100
    it = 0
    while ((error > tol) and (it < max_iter)):
        J_est = compute_jacobian(hf_est, vf_est, l, w, EA, cb)
        inv_J_est = np.linalg.inv(J_est)
        hf_est += inv_J_est[0, 0]*(xf - xf_est) + inv_J_est[0, 1]*(zf - zf_est)
        vf_est += inv_J_est[1, 0]*(xf - xf_est) + inv_J_est[1, 1]*(zf - zf_est)
        # hf += (xf - xf_est)/J[0, 0] + (zf - zf_est)/J[1, 0]
        # vf += (xf - xf_est)/J[0, 1] + (zf - zf_est)/J[1, 1]

        xf_est, zf_est = compute_xf_zf(hf_est, vf_est, l, w, EA, cb)
        error = np.maximum(np.abs(xf - xf_est), np.abs(zf - zf_est))
        # print(error)
        it += 1
    if ((it == max_iter - 1) and (error > tol)):
        cout.cout_wrap(("Mooring system did not converge. error %f" % error), 4)

    return hf_est, vf_est


def generate_mooringlinefd():
    """
        This function generates a file similar to MoorinLinesFD.txt for comparison
    """

    # Values for OC3
    l = 902.2 # Initial length [m]
    w = 698.094 # Aparent weight 77.7066*9.81 # Apparent mass per unit length times gravity
    EA = 384243000./A # Extensional stiffness
    cb = 0. # Seabed friction coefficient

    zf = 320. - 70.

    # xf0 = 853.87
    xf_list = np.arange(653.00, 902.50 + 1., 1.)
    npoints = xf_list.shape[0]
    output = np.zeros((npoints, 4))
    for i in range(npoints):
        hf, vf = quasisteady_mooring(xf_list[i], zf, l, w, EA, cb, hf0=None, vf0=None)
        # print(xf0, zf0, hf0, vf0)
        lb = np.maximum(l - vf/w, 0)
        # print("Suspended lenght = %f" % (l - lb))
        output[i, :] = np.array([xf_list[i], np.sqrt(vf**2 + hf**2)*1e-3, hf*1e-3, (l - lb)])

    np.savetxt("sharpy_mooringlinefd.txt", output, header="# DISTANCE(m) TENSION(kN) HTENSION(kN) SUSPL(m)")


def wave_radiation_damping(K, qdot, it, dt):
    """
        This function computes the wave radiation damping assuming K constant
    """
    qdot_int = np.zeros((6,))
    for idof in range(6):
        qdot_int[idof] = np.trapz(np.arange(0, it + 1, 1)*dt, qdot[0:it, idof])

    return np.dot(K, qdot_int)


def test_change_system():
    # Wind turbine degrees of freedom: Surge, sway, heave, roll, pitch, yaw.
    # SHARPy axis associated:              z,    y,     x,    z,     y,   x

    wt_dofs_char = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
    wt_matrix = np.zeros((6,6), dtype=np.object_)
    wt_matrix_num = np.zeros((6,6),)
    for idof in range(6):
        for jdof in range(6):
            wt_matrix[idof, jdof] = ("%s-%s" % (wt_dofs_char[idof], wt_dofs_char[jdof]))
            wt_matrix_num[idof, jdof] = 10.*idof + jdof

    sharpy_matrix_old = change_of_to_sharpy_old(wt_matrix_num)
    sharpy_matrix = change_of_to_sharpy(wt_matrix_num)

    print("wt matrix: ", wt_matrix_num)
    print("sharpy matrix old: ", sharpy_matrix_old)
    print("sharpy matrix: ", sharpy_matrix)

    undo_sharpy_matrix_old = change_of_to_sharpy_old(sharpy_matrix_old)
    undo_sharpy_matrix = change_of_to_sharpy(sharpy_matrix)

    print("undo sharpy matrix old: ", undo_sharpy_matrix_old)
    print("undo sharpy matrix: ", undo_sharpy_matrix)


def change_of_to_sharpy_old(matrix_of):

    # Wind turbine degrees of freedom: Surge, sway, heave, roll, pitch, yaw.
    # SHARPy axis associated:              z,    y,     x,    z,     y,   x

    of_to_sharpy = [2, 1, 0, 5, 4, 3]
    matrix_sharpy = matrix_of[of_to_sharpy, :]
    matrix_sharpy = matrix_sharpy[:, of_to_sharpy]

    return matrix_sharpy


def change_of_to_sharpy(matrix_of):

    sub_mat = np.array([[0., 0, 1],
                        [0., -1, 0],
                        [1., 0, 0]])
    C_of_s = np.zeros((6,6))
    C_of_s[0:3, 0:3] = sub_mat
    C_of_s[3:6, 3:6] = sub_mat

    matrix_sharpy = np.dot(C_of_s.T, np.dot(matrix_of, C_of_s))
    return matrix_sharpy


def interp_1st_dim_matrix(A, vec, value):

    # Make sure vec is ordered in strictly ascending order
    if (np.diff(vec) <= 0).any():
        cout.cout_wrap("ERROR: vec should be in strictly increasing order", 4)
    if not A.shape[0] == vec.shape[0]:
        cout.cout_wrap("ERROR: Incoherent vector and matrix size", 4)

    # Compute the positions to interpolate
    if value <= vec[0]:
        return A[0, ...]
    elif ((value >= vec[-1]) or (value > vec[-2] and np.isinf(vec[-1]))):
        return A[-1, ...]
    else:
        i = 0
        while value > vec[i]:
            i += 1
        dist = vec[i] - vec[i - 1]
        rel_dist_to_im1 = (value - vec[i - 1])/dist
        rel_dist_to_i = (vec[i] - value)/dist

    return A[i - 1, ...]*rel_dist_to_i + A[i, ...]*rel_dist_to_im1

    # Test
    # A = np.ones((3, 3, 3))
    # A[0, :, :] *= 0.
    # A[2, :, :] *= 2.
    # interp_1st_dim_matrix(A, vec, 0.6)


def rfval(num, den, z):
    """
        Evaluate a rational function given by the coefficients of the numerator (num) and
        denominator (den) at z
    """
    return np.polyval(num, z)/np.polyval(den, z)


def matrix_from_rf(dict_rf, w):

    H = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            pos = "%d_%d" % (i, j)
            H[i, j] = rfval(dict_rf[pos]['num'], dict_rf[pos]['den'], w)

    return H


def response_freq_dep_matrix(H, omega_H, q, it_, dt):
    """
    Compute the frequency response of a system with a transfer function depending on the frequency
    F(t) = H(omega) * q(t)
    """
    it = it_ + 1
    omega_fft = np.linspace(0, 1/(2*dt), it//2)[:it//2]
    fourier_q = fft(q[:it, :], axis=0)
    fourier_f = np.zeros_like(fourier_q)

    ndof = q.shape[1]
    f = np.zeros((ndof))

    # Compute the constant component
    if type(H) is np.ndarray:
        H_omega = interp_1st_dim_matrix(H, omega_H, omega_fft[0])
    elif type(H) is tuple:
        H_omega = matrix_from_rf(H, omega_fft[0])
    else:
        cout.cout_wrap(("ERROR: Not implemented response_freq_dep_matrix for type(H) %s" % type(H)), 4)
    fourier_f[0, :] = np.dot(H_omega, fourier_q[0, :])

    # Compute the rest of the terms
    for iomega in range(1, omega_fft.shape[0]):
        # Interpolate H at omega
        if type(H) is np.ndarray:
            H_omega = interp_1st_dim_matrix(H, omega_H, omega_fft[iomega])
        elif type(H) is dict:
            H_omega = matrix_from_rf(H, omega_fft[iomega])
        fourier_f[iomega, :] = np.dot(H_omega, fourier_q[iomega, :])
        fourier_f[-iomega, :] = np.dot(H_omega, fourier_q[-iomega, :])

    # Compute the inverse Fourier tranform
    f[:] = np.real(ifft(fourier_f, axis=0)[it_, :])

    (T, yout, xout) = lsim(H, q[:it_ + 1, :], T, X0=X0)

    return f


def compute_equiv_hd_added_mass(f, q):
    """
        Compute the matrix H that satisfies f = Hq
        H represents the added mass effects so it has to be
        symmetric.
        For the OC3 platfrom the following statements hold:
            - z-y symmetry
            - Non-diagonal non-zero terms: (1,5) and (2,4). Zero-indexed
    """

    if (q == 0).all():
        return np.zeros((6,6))

    q_mat = np.array([[q[0], 0,    0,    0,    0,    0],
                      [0,    q[1], 0,    0,    q[5], 0],
                      [0,    q[2], 0,    0,    0,    q[4]],
                      [0,    0,    q[3], 0,    0,    0],
                      [0,    0,    0,    q[4], 0,    q[2]],
                      [0,    0,    0,    q[5], q[1], 0]])

    hv = np.dot(np.linalg.inv(q_mat), f)

    H = np.array([[hv[0], 0,     0,     0,     0,     0],
                  [0,     hv[1], 0,     0,     0,     hv[4]],
                  [0,     0,     hv[1], 0,     hv[5], 0],
                  [0,     0,     0,     hv[2], 0,     0],
                  [0,     0,     hv[5], 0,     hv[3], 0],
                  [0,     hv[4], 0,     0,     0,     hv[3]]])

    return H


@generator_interface.generator
class FloatingForces(generator_interface.BaseGenerator):
    r"""
    Floating forces generator

    Generates the forces associated the floating support of offshore wind turbines.
    Currently supports spar configurations.

    The hydrostatic forces model includes buoyancy: an initial vertical force and the restoring forces
    associated with heave, roll and pitch. See the implementation in [1].

    The mooring model is the quasisteady implementation of Jonkman [2] .However, equation 2-37b is thought to be wrong (it is just a copy from eq 2-35b)
    This was corrected according to the theory review of MAP++ [3].

    The default values have been obtained from the OC3 platform report [1]

    [1] Jonkman, J. Definition of the Floating System for Phase IV of OC3. 2010. NREL/TP-500-47535

    [2] Jonkman, J. M. Dynamics modeling and loads analysis of an offshore floating wind turbine. 2007. NREL/TP-500-41958

    [3] https://map-plus-plus.readthedocs.io/en/latest/theory.html (accessed on Octorber 14th, 2020)
    """
    generator_id = 'FloatingForces'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['n_time_steps'] = 'int'
    settings_default['n_time_steps'] = None
    settings_description['n_time_steps'] = 'Number of time steps'

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step'

    settings_types['water_density'] = 'float'
    settings_default['water_density'] = 1025 # kg/m3
    settings_description['water_density'] = 'Water density'

    settings_types['gravity'] = 'float'
    settings_default['gravity'] = 9.81
    settings_description['gravity'] = 'Gravity'

    settings_types['gravity_dir'] = 'list(float)'
    settings_default['gravity_dir'] = [1., 0., 0.]
    settings_description['gravity_dir'] = 'Gravity direction'

    settings_types['floating_file_name'] = 'str'
    settings_default['floating_file_name'] = './oc3.floating.h5'
    settings_description['floating_file_name'] = 'File containing the information about the floating dynamics'

    settings_types['method_matrices_freq'] = 'str'
    settings_default['method_matrices_freq'] = 'constant'
    settings_description['method_matrices_freq'] = 'Method to compute frequency-dependent matrices'
    settings_options['method_matrices_freq'] = ['constant', 'rational_function']

    settings_types['matrices_freq'] = 'float'
    settings_default['matrices_freq'] = 4.8 # Close to the upper limit defined in the oc3 report
    settings_description['matrices_freq'] = 'Frequency [rad/s] to interpolate frequency-dependent matrices'

    settings_types['steps_constant_matrices'] = 'int'
    settings_default['steps_constant_matrices'] = 8
    settings_description['steps_constant_matrices'] = 'Time steps to compute with constant matrices computed at ``matrices_freq``. Irrelevant in ``method_matrices_freq``=``constant``'

    settings_types['added_mass_in_mass_matrix'] = 'bool'
    settings_default['added_mass_in_mass_matrix'] = True
    settings_description['added_mass_in_mass_matrix'] = 'Include the platform added mass in the mass matrix of the system'

    settings_types['wave_amplitude'] = 'float'
    settings_default['wave_amplitude'] = 0.
    settings_description['wave_amplitude'] = 'Wave amplitude'

    settings_types['wave_freq'] = 'float'
    settings_default['wave_freq'] = 0.
    settings_description['wave_freq'] = 'Wave circular frequency [rad/s]'

    settings_types['wave_incidence'] = 'float'
    settings_default['wave_incidence'] = 0.
    settings_description['wave_incidence'] = 'Wave incidence in rad'

    settings_types['write_output'] = 'bool'
    settings_default['write_output'] = False
    settings_description['write_output'] = 'Write forces to an output file'

    settings_types['folder'] = 'str'
    settings_default['folder'] = 'output'
    settings_description['folder'] = 'Folder for the output files'

    settings_types['log_filename'] = 'str'
    settings_default['log_filename'] = 'log_floating_forces'
    settings_description['log_filename'] = 'Log file name to write outputs'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()

        self.water_density = None
        self.gravity = None
        self.gravity_dir = None

        self.floating_data = None

        self.mooring_node = None
        self.n_mooring_lines = None
        self.anchor_pos = None
        self.fairlead_pos_A = None
        self.hf_prev = list() # Previous value of hf just for initialisation
        self.vf_prev = list()

        self.buoyancy_node = None
        self.buoy_F0 = None
        self.buoy_rest_mat = None

        self.wave_forces_node = None

        self.q = None
        self.qdot = None
        self.qdotdot = None

        self.log_filename = None
        self.added_mass_in_mass_matrix = None


    def initialise(self, in_dict=None, data=None):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default,
                                 self.settings_options,
                                 no_ctype=True)
        self.settings = self.in_dict

        self.water_density = self.settings['water_density']
        self.gravity = self.settings['gravity']
        self.gravity_dir = self.settings['gravity_dir']

        # Platform dofs
        self.q = np.zeros((self.settings['n_time_steps'] + 1, 6))
        self.qdot = np.zeros_like(self.q)
        self.qdotdot = np.zeros_like(self.q)

        # Read the file with the floating information
        fid = h5.File(self.settings['floating_file_name'], 'r')
        self.floating_data = h5utils.load_h5_in_dict(fid)
        fid.close()

        # Mooringlines parameters
        self.mooring_node = self.floating_data['mooring']['node']
        self.n_mooring_lines = self.floating_data['mooring']['n_lines']
        self.anchor_pos = np.zeros((self.n_mooring_lines, 3))
        self.fairlead_pos_A = np.zeros((self.n_mooring_lines, 3))
        self.hf_prev = [None]*self.n_mooring_lines
        self.vf_prev = [None]*self.n_mooring_lines

        theta = 2.*np.pi/self.n_mooring_lines
        R = algebra.rotation3d_x(theta)
        self.anchor_pos[0, 0] = -self.floating_data['mooring']['anchor_depth']
        self.anchor_pos[0, 2] = self.floating_data['mooring']['anchor_radius']
        self.fairlead_pos_A[0, 0] = -self.floating_data['mooring']['fairlead_depth']
        self.fairlead_pos_A[0, 2] = self.floating_data['mooring']['fairlead_radius']
        for imoor in range(1, self.n_mooring_lines):
            self.anchor_pos[imoor, :] = np.dot(R, self.anchor_pos[imoor - 1, :])
            self.fairlead_pos_A[imoor, :] = np.dot(R, self.fairlead_pos_A[imoor - 1, :])

        # Hydrostatics
        self.buoyancy_node = self.floating_data['hydrostatics']['node']
        self.buoy_F0 = np.zeros((6,), dtype=float)
        self.buoy_F0[0:3] = (self.floating_data['hydrostatics']['V0']*
                              self.settings['water_density']*
                              self.settings['gravity']*self.settings['gravity_dir'])
        self.buoy_rest_mat = self.floating_data['hydrostatics']['buoyancy_restoring_matrix']

        # hydrodynamics
        if self.settings['method_matrices_freq'] == 'constant':
            self.hd_added_mass_const = interp_1st_dim_matrix(self.floating_data['hydrodynamics']['added_mass_matrix'],
                                        self.floating_data['hydrodynamics']['ab_freq_rads'],
                                        self.settings['matrices_freq'])

            self.hd_damping_const = interp_1st_dim_matrix(self.floating_data['hydrodynamics']['damping_matrix'],
                                        self.floating_data['hydrodynamics']['ab_freq_rads'],
                                        self.settings['matrices_freq'])

            # self.hd_added_mass *= 0.
            # self.hd_damping *= 0.
        elif self.settings['method_matrices_freq'] == 'rational_function':
            self.hd_added_mass_const = self.floating_data['hydrodynamics']['added_mass_matrix'][-1, :, :]
            self.hd_damping_const = self.floating_data['hydrodynamics']['damping_matrix'][-1, :, :]

        self.added_mass_in_mass_matrix = self.settings['added_mass_in_mass_matrix']
        if self.added_mass_in_mass_matrix:
        # if ((self.settings['method_matrices_freq'] == 'constant') and
        #     self.added_mass_in_mass_matrix):
                # Include added mass in structure
            data.structure.add_lumped_mass_to_element(self.buoyancy_node,
                                                      self.hd_added_mass_const)
            data.structure.generate_fortran()
            # self.hd_added_mass *= 0.

        # if self.settings['method_matrices_freq'] == 'interp_matrices':
        #     self.hd_added_mass = self.floating_data['hydrodynamics']['added_mass_matrix']
        #     self.hd_damping = self.floating_data['hydrodynamics']['damping_matrix']
        #     self.ab_freq_rads = self.floating_data['hydrodynamics']['ab_freq_rads']

        if self.settings['method_matrices_freq'] == 'rational_function':
            ninput = 6
            noutput = 6
            # hd_added_mass_num = [None]*noutput
            # hd_added_mass_den = [None]*noutput
            # hd_damping_num = [None]*noutput
            # hd_damping_den = [None]*noutput
            hd_K_num = [None]*noutput
            hd_K_den = [None]*noutput
            for ioutput in range(noutput):
                # hd_added_mass_num[ioutput] = [None]*ninput
                # hd_added_mass_den[ioutput] = [None]*ninput
                # hd_damping_num[ioutput] = [None]*ninput
                # hd_damping_den[ioutput] = [None]*ninput
                hd_K_num[ioutput] = [None]*ninput
                hd_K_den[ioutput] = [None]*ninput
                for iinput in range(ninput):
                    # pos = "%d_%d" % (i, j)
                    pos = "%d_%d" % (ioutput, iinput)
                    # hd_added_mass_num[ioutput][iinput] = self.floating_data['hydrodynamics']['added_mass_rf'][pos]['num']
                    # hd_added_mass_den[ioutput][iinput] = self.floating_data['hydrodynamics']['added_mass_rf'][pos]['den']
                    # hd_damping_num[ioutput][iinput] = self.floating_data['hydrodynamics']['damping_rf'][pos]['num']
                    # hd_damping_den[ioutput][iinput] = self.floating_data['hydrodynamics']['damping_rf'][pos]['den']
                    hd_K_num[ioutput][iinput] = self.floating_data['hydrodynamics']['K_rf'][pos]['num']
                    hd_K_den[ioutput][iinput] = self.floating_data['hydrodynamics']['K_rf'][pos]['den']

            # self.hd_added_mass = TransferFunction(hd_added_mass_num, hd_added_mass_den, self.settings['dt'])
            # self.hd_damping = TransferFunction(hd_damping_num, hd_damping_den, self.settings['dt'])
            # self.hd_added_mass = TransferFunction(hd_added_mass_num, hd_added_mass_den)
            # self.hd_damping = TransferFunction(hd_damping_num, hd_damping_den)
            self.hd_K = TransferFunction(hd_K_num, hd_K_den)
            self.ab_freq_rads = self.floating_data['hydrodynamics']['ab_freq_rads']

            # self.x0_added_mass = [None]*(self.settings['n_time_steps'] + 1)
            # self.x0_damping = [None]*(self.settings['n_time_steps'] + 1)
            # self.x0_added_mass[0] = 0.
            # self.x0_damping[0] = 0.
            self.x0_K = [None]*(self.settings['n_time_steps'] + 1)
            self.x0_K[0] = 0.


        # Wave forces
        self.wave_forces_node = self.floating_data['wave_forces']['node']
        xi_matrix2 = interp_1st_dim_matrix(self.floating_data['wave_forces']['xi'],
                                           self.floating_data['wave_forces']['xi_freq_rads'],
                                           self.settings['wave_freq'])
        self.xi = interp_1st_dim_matrix(xi_matrix2,
                                        self.floating_data['wave_forces']['xi_beta_deg']*deg2rad,
                                        self.settings['wave_incidence'])

        # Log file
        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        folder = self.settings['folder'] + '/' + data.settings['SHARPy']['case'] + '/floatingforces/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.log_filename = folder + self.settings['log_filename'] + '.h5'


    def write_output(self, ts, k, mooring, mooring_yaw, hydrostatic,
                     hydrodynamic_qdot, hydrodynamic_qdotdot, hd_correct_grav, waves):

        output = dict()
        output['ts'] = ts
        output['k'] = k
        output['q'] = self.q[ts, :]
        output['qdot'] = self.qdot[ts, :]
        output['qdotdot'] = self.qdotdot[ts, :]
        output['mooring_forces'] = mooring
        output['mooring_yaw'] = mooring_yaw
        output['hydrostatic'] = hydrostatic
        output['hydrodynamic_qdot'] = hydrodynamic_qdot
        output['hydrodynamic_qdotdot'] = hydrodynamic_qdotdot
        output['hydrodynamic_correct_grav'] = hd_correct_grav
        output['waves'] = waves

        fid = h5.File(self.log_filename, 'a')
        group_name = "ts%d_k%d" % (ts, k)
        if fid.__contains__(group_name):
            del fid[group_name]
        group = fid.create_group(group_name)
        for key, value in output.items():
            group.create_dataset(key, data=value)
        fid.close()

        debug_output = False
        if debug_output:
            print("q: ", self.q[ts, :])
            print("qdot: ", self.qdot[ts, :])
            print("qdotdot: ", self.qdotdot[ts, :])

            print("mooring: ", mooring)
            print("mooring_yaw: ", mooring_yaw)
            print("hydrostatic: ", hydrostatic)
            print("hydrodynamic_qdot: ", hydrodynamic_qdot)
            print("hydrodynamic_qdotdot: ", hydrodynamic_qdotdot)
            print("hydrodynamic_correct_grav: ", hd_correct_grav)
            print("waves: ", waves)

        return


    def update_dof_vector(self, beam, struct_tstep, it, k):

        if k:
            cga = struct_tstep.cga()
            # self.q[it, 0:3] = (np.dot(cga, struct_tstep.pos[self.buoyancy_node, :]) -
            #           np.dot(beam.ini_info.cga(), beam.ini_info.pos[self.buoyancy_node, :]) +
            #           struct_tstep.for_pos[0:3] -
            #           beam.ini_info.for_pos[0:3])
            self.q[it, 0:3] = (np.dot(cga, struct_tstep.pos[self.buoyancy_node, :]) +
                      struct_tstep.for_pos[0:3])
            # ams: in my casethe angles are negative in the x and z components
            self.q[it, 3:6] = (algebra.quat2euler(struct_tstep.quat)*
                               np.array([-1., 0., -1.]))
                               # np.array([1., 0., 1.]))

            self.qdot[it, 0:3] = np.dot(cga, struct_tstep.for_vel[0:3])
            self.qdot[it, 3:6] = np.dot(cga, struct_tstep.for_vel[3:6])

            self.qdotdot[it, 0:3] = np.dot(cga, struct_tstep.for_acc[0:3])
            self.qdotdot[it, 3:6] = np.dot(cga, struct_tstep.for_acc[3:6])

        else:
            self.q[it, :] = self.q[it-1, :]
            self.qdot[it, :] = self.qdot[it-1, :]
            self.qdotdot[it, :] = self.qdotdot[it-1, :]

        return


    def generate(self, params):
        # Renaming for convenience
        data = params['data']
        struct_tstep = params['struct_tstep']
        aero_tstep = params['aero_tstep']
        force_coeff = params['force_coeff']
        k = params['fsi_substep']

        # Update dof vector
        self.update_dof_vector(data.structure, struct_tstep, data.ts, k)

        # Mooring lines
        mooring_forces = np.zeros((self.n_mooring_lines, 2))
        cga = struct_tstep.cga()
        ielem, inode_in_elem = data.structure.node_master_elem[self.mooring_node]
        cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem])
        cbg = np.dot(cab.T, cga.T)

        moor_force_out = 0.
        moor_mom_out = 0.

        for imoor in range(self.n_mooring_lines):
            fairlead_pos_G = (np.dot(cga, self.fairlead_pos_A[imoor, :]) +
                              struct_tstep.for_pos[0:3])
            fl_to_anchor_G = self.anchor_pos[imoor, :] - fairlead_pos_G
            xf = np.sqrt(fl_to_anchor_G[1]**2 + fl_to_anchor_G[2]**2)
            zf = np.abs(fl_to_anchor_G[0])
            hf, vf = quasisteady_mooring(xf,
                                 zf,
                                 self.floating_data['mooring']['unstretched_length'],
                                 self.floating_data['mooring']['apparent_weight'],
                                 self.floating_data['mooring']['EA'],
                                 self.floating_data['mooring']['seabed_drag_coef'],
                                 hf0=self.hf_prev[imoor],
                                 vf0=self.vf_prev[imoor])
                                 # hf0=None,
                                 # vf0=None)
            mooring_forces[imoor, :] = np.array([hf, vf])
            # Save the results to initialise the computation in the next time step
            self.hf_prev[imoor] = hf + 0.
            self.vf_prev[imoor] = vf + 0.
            # print(imoor, hf, vf)

            # Convert to the adequate reference system
            horizontal_unit_vec = algebra.unit_vector(fl_to_anchor_G)
            horizontal_unit_vec[0] = 0.
            horizontal_unit_vec = algebra.unit_vector(horizontal_unit_vec)
            force_fl = hf*horizontal_unit_vec + vf*np.array([-1., 0., 0.])
            # print(imoor, force_fl)

            # Move the forces to the mooring node
            force_cl = np.zeros((6,))
            force_cl[0:3] = force_fl
            mooring_node_pos_G = (np.dot(cga, struct_tstep.pos[self.mooring_node, :]) +
                              struct_tstep.for_pos[0:3])
            r_fairlead_G = fairlead_pos_G - mooring_node_pos_G
            force_cl[3:6] = np.cross(r_fairlead_G, force_fl)

            struct_tstep.runtime_generated_forces[self.mooring_node, 0:3] += np.dot(cbg, force_cl[0:3])
            struct_tstep.runtime_generated_forces[self.mooring_node, 3:6] += np.dot(cbg, force_cl[3:6])

        # Yaw moment generated by the mooring system
        yaw = np.array([self.q[data.ts, 3], 0., 0.])
        # if np.abs(yaw[0]) < 1.*deg2rad:
        #     yaw[0] = 0.
        mooring_yaw = -self.floating_data['mooring']['yaw_spring_stif']*yaw
        struct_tstep.runtime_generated_forces[self.mooring_node, 3:6] += np.dot(cbg,
                                                                      mooring_yaw)



        # Hydrostatic model
        hs_f_g = self.buoy_F0 - np.dot(self.buoy_rest_mat, self.q[data.ts, :])

        if not force_coeff == 0.:
            hd_f_qdot_g = -np.dot(self.floating_data['hydrodynamics']['additional_damping'], self.qdot[data.ts, :])

            if ((self.settings['method_matrices_freq'] == 'constant') or
                 (data.ts < self.settings['steps_constant_matrices'])):
                hd_f_qdot_g -= np.dot(self.hd_damping_const, self.qdot[data.ts, :])
                hd_f_qdotdot_g = np.zeros((6))
                # hd_f_qdotdot_g = -np.dot(self.hd_added_mass_const, self.qdotdot[data.ts, :])
                # equiv_hd_added_mass = self.hd_added_mass_const

            elif self.settings['method_matrices_freq'] == 'rational_function':
                # Damping
                (T, yout, xout) = forced_response(self.hd_K,
                                                  T=[0, self.settings['dt']],
                                                  U=self.qdot[data.ts-1:data.ts+1, :].T,
                                                  X0=self.x0_K[data.ts-1])
                                                  # transpose=True)
                self.x0_K[data.ts] = xout[:, 1]
                hd_f_qdot_g -= yout[:, 1]
                hd_f_qdotdot_g = np.zeros((6))

                # (T, yout, xout) = forced_response(self.hd_damping,
                #                                   T=[0, self.settings['dt']],
                #                                   U=self.qdot[data.ts-1:data.ts+1, :].T,
                #                                   X0=self.x0_damping[data.ts-1])
                #                                   # transpose=True)
                # self.x0_damping[data.ts] = xout[:, 1]
                # hd_f_qdot_g -= yout[:, 1]
                # hd_f_qdotdot_g = np.zeros((6))
                # (T, yout, xout) = forced_response(self.hd_added_mass,
                #                                   T=[0, self.settings['dt']],
                #                                   U=self.qdotdot[data.ts-1:data.ts+1, :],
                #                                   X0=self.x0_added_mass[data.ts-1],
                #                                   transpose=True)
                # self.x0_added_mass[data.ts] = xout[1, :]
                # hd_f_qdotdot_g = -yout[1, :]
                # hd_f_qdotdot_g = -np.dot(self.hd_added_mass_const, self.qdotdot[data.ts, :])

                # hd_f_qdot_g -= response_freq_dep_matrix(self.hd_damping, self.ab_freq_rads, self.qdot, data.ts, self.settings['dt'])
                # hd_f_qdotdot_g = -response_freq_dep_matrix(self.hd_added_mass, self.ab_freq_rads, self.qdotdot, data.ts, self.settings['dt'])

                # Compute the equivalent added mass matrix
                # equiv_hd_added_mass = compute_equiv_hd_added_mass(-hd_f_qdotdot_g, self.qdotdot[data.ts, :])
                # if self.added_mass_in_mass_matrix:
                    # data.structure.add_lumped_mass_to_element(self.buoyancy_node,
                    #                                           equiv_hd_added_mass,
                    #                                           replace=True)
                    # data.structure.generate_fortran()
                    # hd_f_qdotdot_g += np.dot(self.hd_added_mass_const, self.qdotdot[data.ts, :])

            else:
                cout.cout_wrap(("ERROR: Unknown method_matrices_freq %s" % self.settings['method_matrices_freq']), 4)

            # Correct gravity forces if needed
            if self.added_mass_in_mass_matrix:
                # Compensate added mass
                # struct_tstep.runtime_generated_forces[self.buoyancy_node, 0:3] += np.dot(cbg,
                #                                                             hd_f_qdotdot_g[0:3])
                # struct_tstep.runtime_generated_forces[self.buoyancy_node, 3:6] += np.dot(cbg,
                #                                                              hd_f_qdotdot_g[3:6])
                # Correct unreal gravity forces from added mass
                gravity_b = np.zeros((6,),)
                gravity_b[0:3] = np.dot(cbg, -self.settings['gravity_dir'])*self.settings['gravity']
                # hd_correct_grav = -np.dot(equiv_hd_added_mass, gravity_b)
                hd_correct_grav = -np.dot(self.hd_added_mass_const, gravity_b)
                struct_tstep.runtime_generated_forces[self.buoyancy_node, :] += hd_correct_grav
                # hd_f_qdotdot_g = np.zeros((6))
            else:
                hd_correct_grav = np.zeros((6))

                # if self.added_mass_in_mass_matrix:
                    # # Include added mass in structure
                    # data.structure.add_lumped_mass_to_element(self.buoyancy_node,
                                                # self.hd_added_mass)
                                                # data.structure.generate_fortran()
                                                # # self.hd_added_mass *= 0.
        else:
            hd_f_qdot_g = np.zeros((6))
            hd_f_qdotdot_g = np.zeros((6))
            hd_correct_grav = np.zeros((6))

        ielem, inode_in_elem = data.structure.node_master_elem[self.buoyancy_node]
        cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem])
        cbg = np.dot(cab.T, cga.T)
        struct_tstep.runtime_generated_forces[self.buoyancy_node, 0:3] += np.dot(cbg, hs_f_g[0:3] + force_coeff*(hd_f_qdot_g[0:3] + hd_f_qdotdot_g[0:3]))
        struct_tstep.runtime_generated_forces[self.buoyancy_node, 3:6] += np.dot(cbg, hs_f_g[3:6] + force_coeff*(hd_f_qdot_g[3:6] + hd_f_qdotdot_g[3:6]))

        # Wave loading
        phase = self.settings['wave_freq']*data.ts*self.settings['dt']
        wave_forces_g = np.real(self.settings['wave_amplitude']*self.xi*(np.cos(phase) + 1j*np.sin(phase)))

        ielem, inode_in_elem = data.structure.node_master_elem[self.wave_forces_node]
        cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem])
        cbg = np.dot(cab.T, cga.T)

        struct_tstep.runtime_generated_forces[self.wave_forces_node, 0:3] += np.dot(cbg, force_coeff*wave_forces_g[0:3])
        struct_tstep.runtime_generated_forces[self.wave_forces_node, 3:6] += np.dot(cbg, force_coeff*wave_forces_g[3:6])

        # Write output
        if self.settings['write_output']:
            self.write_output(data.ts, k, mooring_forces, mooring_yaw, hs_f_g,
                     hd_f_qdot_g, hd_f_qdotdot_g, hd_correct_grav, wave_forces_g)
