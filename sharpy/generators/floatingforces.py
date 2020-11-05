import numpy as np
import h5

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.cout_utils as cout
from sharpy.utils.constants import deg2rad
import sharpy.utils.h5utils as h5utils


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
    tol = 1e-12
    error = 2*tol
    while error > tol:
        J_est = compute_jacobian(hf_est, vf_est, l, w, EA, cb)
        inv_J_est = np.linalg.inv(J_est)
        hf_est += inv_J_est[0, 0]*(xf - xf_est) + inv_J_est[0, 1]*(zf - zf_est)
        vf_est += inv_J_est[1, 0]*(xf - xf_est) + inv_J_est[1, 1]*(zf - zf_est)
        # hf += (xf - xf_est)/J[0, 0] + (zf - zf_est)/J[1, 0]
        # vf += (xf - xf_est)/J[0, 1] + (zf - zf_est)/J[1, 1]

        xf_est, zf_est = compute_xf_zf(hf_est, vf_est, l, w, EA, cb)
        error = np.maximum(np.abs(xf - xf_est), np.abs(zf - zf_est))
        # print(error)

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
    for idof in range(6):
        for jdof in range(6):
            wt_matrix[idof, jdof] = ("%s-%s" % (wt_dofs_char[idof], wt_dofs_char[jdof]))

    wt_to_sharpy = [2, 1, 0, 5, 4, 3]
    sharpy_matrix = wt_matrix[wt_to_sharpy, :]
    sharpy_matrix = sharpy_matrix[:, wt_to_sharpy]

    print("wt matrix: ", wt_matrix)
    print("sharpy matrix: ", sharpy_matrix)


def change_of_to_sharpy(matrix_of):

    of_to_sharpy = [2, 1, 0, 5, 4, 3]
    matrix_sharpy = matrix_of[of_to_sharpy, :]
    matrix_sharpy = matrix_sharpy[:, of_to_sharpy]

    return matrix_sharpy


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

    settings_types['n_time_steps'] = 'int'
    settings_default['n_time_steps'] = None
    settings_description['n_time_steps'] = 'Number of time steps'

    settings_types['water_density'] = 'float'
    settings_default['water_density'] = 1.025 # kg/m3
    settings_description['water_density'] = 'Water density'

    settings_types['gravity'] = 'float'
    settings_default['gravity'] = 9.81
    settings_description['gravity'] = 'Gravity'

    settings_types['gravity_dir'] = 'list(float)'
    settings_default['gravity_dir'] = np.array([1., 0., 0.])
    settings_description['gravity_dir'] = 'Gravity direction'

    settings_types['floating_file_name'] = 'string'
    settings_default['floating_file_name'] = './oc3.floating.h5'
    settings_description['floating_file_name'] = 'File containing the information about the floating dynamics'

    settings_types['wave_amplitude'] = 'float'
    settings_default['wave_amplitude'] = 0.
    settings_description['wave_amplitude'] = 'Wave amplitude'

    settings_types['wave_freq'] = 'float'
    settings_default['wave_freq'] = 0.
    settings_description['wave_freq'] = 'Wave circular frequency [rad/s]'

    settings_types['wave_incidence'] = 'float'
    settings_default['wave_incidence'] = 0.
    settings_description['wave_incidence'] = 'Wave incidence in rad'

    settings_types['hydrodynamic_inertia'] = 'list'
    settings_default['hydrodynamic_inertia'] = np.zeros((36,))
    settings_description['hydrodynamic_inertia'] = 'Hydrodynamic inertia matrix. Surge, sway, heave, roll, pitch, yaw.'

    settings_types['additional_damping'] = 'list'
    settings_default['additional_damping'] = np.zeros((36,))
    settings_description['additional_damping'] = 'Additional damping matrix. Surge, sway, heave, roll, pitch, yaw.'

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
        self.hf_prev = list()
        self.vf_prev = list()

        self.buoyancy_node = None
        self.buoy_F0 = None
        self.buoy_rest_mat = None

        self.q = None
        self.qdot = None
        self.qdotdot = None

    def initialise(self, in_dict=None):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)
        self.settings = self.in_dict

        self.water_density = self.settings['water_density']
        self.gravity = self.settings['gravity']
        self.gravity_dir = self.settings['gravity_dir']

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
        self.fairlead_pos_A[0, 2] = self.floating_data['mooring']['fairlead_radius']
        for imoor in range(1, self.n_mooring_lines):
            self.anchor_pos[imoor, :] = np.dot(R, self.anchor_pos[imoor - 1, :])
            self.fairlead_pos_A[imoor, :] = np.dot(R, self.fairlead_pos_A[imoor - 1, :])

        # Buoyancy parameters
        self.buoyancy_node = self.floating_data['hydrostatics']['bouyancy_node']
        self.buoy_F0 = np.zeros((6,), dtype=float)
        self.buoy_F0[0:3] = -(self.floating_data['hydrostatics']['V0']*
                              self.settings['water_density']*
                              self.settings['gravity']*self.settings['gravity_dir'])
        self.buoy_rest_mat = self.floating_data['hydrostatics']['buoyancy_restoring_matrix']

        self.q = np.zeros((self.settings['n_time_steps'], 6))
        self.qdot = np.zeros_like(self.q)
        self.qdotdot = np.zeros_like(self.q)

        self.wave_freq = self.settings['wave_freq']
        self.wave_incidence = self.settings['wave_incidence']
        self.wave_amplitude = self.settings['wave_amplitude']

        xi_matrix2 = interp_1st_dim_matrix(self.floating_data['wave_forces']['xi'],
                                           self.floating_data['wave_forces']['xi_freq_rads'],
                                           self.wave_freq)
        self.xi = interp_1st_dim_matrix(xi_matrix2,
                                        self.floating_data['wave_forces']['xi_beta_deg']*deg2rad,
                                           self.wave_incidence)

        self.A = interp_1st_dim_matrix(self.floating_data['hydrodynamics']['added_mass'],
                                        self.floating_data['hydrodynamics']['ab_freq_rads'],
                                           self.wave_freq)

        self.B = interp_1st_dim_matrix(self.floating_data['hydrodynamics']['damping'],
                                        self.floating_data['hydrodynamics']['ab_freq_rads'],
                                           self.wave_freq)

        # Wind turbine degrees of freedom: Surge, sway, heave, roll, pitch, yaw.
        # SHARPy axis associated:              z,    y,     x,    z,     y,   x

        [2, 1, 0, 5, 4, 3]

        wt_to_sharpy_dofs = np.array()[[]]

    def interp_1st_dim_matrix(A, vec, value):

        # Make sure vec is ordered in strictly ascending order
        if (np.diff(vec) <= 0).any():
            print("ERROR: vec should be in strictly increasing order")
        if not A.shape[0] == vec.shape[0]:
            print("ERROR: Incoherent vector and matrix size")

        # Compute the positions to interpolate
        if value <= vec[0]:
            return A[0, ...]
        elif value >= vec[-1]:
            return A[-1, ...]
        else:
            i = 0
            while value < vec[i]:
                i += 1
            dist = vec[i + 1] - vec[i]
            rel_dist_to_i = (value - vec[i])/dist
            rel_dist_to_ip1 = (vec[i + 1] - value)/dist

            return A[i, ...]*rel_dist_to_ip1 + A[i + 1, ...]*rel_dist_to_i

        # Test
        # A = np.ones((3, 3, 3))
        # A[0, :, :] *= 0.
        # A[2, :, :] *= 2.
        # interp_1st_dim_matrix(A, vec, 0.6)

    def update_dof_vector(struct_tstep):



    def generate(self, params):
        # Renaming for convenience
        data = params['data']
        struct_tstep = params['struct_tstep']
        aero_tstep = params['aero_tstep']

        # Mooring lines
        cga = struct_tstep.cga()
        ielem, inode_in_elem = data.structure.node_master_elem[self.mooring_node]
        cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem])
        cbg = np.dot(cab.T, cga.T)
        for imoor in range(self.n_mooring_lines):
            fairlead_pos_G = np.dot(cga, self.fairlead_pos_A[imoor, :])
            fl_to_anchor_G = self.anchor_pos[imoor, :] - fairlead_pos_G
            xf = np.sqrt(fl_to_anchor_G[1]**2 + fl_to_anchor_G[2]**2)
            zf = np.abs(fl_to_anchor[0])
            hf, vf = quasisteady_mooring(xf,
                                 zf,
                                 self.floating_data['mooring']['unstretched_length'],
                                 self.floating_data['mooring']['apparent_weight'],
                                 self.floating_data['mooring']['EA'],
                                 self.floating_data['mooring']['seabed_drag_coef'],
                                 hf0=self.hf_prev[imoor],
                                 vf0=self.vf_prev[imoor])
            # Save the results to initialise the computation in the next time step
            self.hf_prev[imoor] = hf + 0.
            self.vf_prev[imoor] = vf + 0.

            # Convert to the adequate reference system
            horizontal_unit_vec = algebra.unit_vector(fl_to_anchor_G)
            horizontal_unit_vec[0] = 0.
            force_fl = hf*horizontal_unit_vec + vf*np.array([1., 0., 0.])

            # Move the forces to the centerline
            force_cl = np.zeros((6,))
            force_cl[0:3] = force_fl
            force_cl[3:6] = np.cross(fairlead_pos_G, force_fl)

            struct_tstep.unsteady_applied_forces[self.mooring_node, 0:3] += np.dot(cbg, force_cl[0:3])
            struct_tstep.unsteady_applied_forces[self.mooring_node, 3:6] += np.dot(cbg, force_cl[3:6])

        # Yaw moment generated by the mooring system
        yaw = algebra.quat2euler(struct_tstpe.quat)[0]
        struct_tstep.unsteady_applied_forces[self.mooring_node, 3:6] += np.dot(cbg,
                                                                      self.floating_data['mooring']['yaw_spring_stif']*yaw)

        # Hydrostatic model
        q = np.zeros((6,), dtype=int)
        q[0:3] = (np.dot(cga, struct_tstep.pos[self.bouyancy_node, :]) -
                  np.dot(data.structure.ini_info.cga, data.structure.ini_info.pos[self.bouyancy_node, :]) +
                  struct_tstep.for_pos[0:3] -
                  data.structure.ini_info.for_pos[0:3])
        q[3:6] = algebra.quat2euler(struct_tstpe.quat)

        hd_forces_g = self.buoy_F0 + np.dot(self.buoy_rest_mat, q)
        hd_forces_g -= np.dot(self.A, self.qdotdot)
        hd_forces_g -= np.dot(self.B, self.qdot)
        ielem, inode_in_elem = data.structure.node_master_elem[self.bouyancy_node]
        cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem])
        cbg = np.dot(cab.T, cga.T)

        struct_tstep.unsteady_applied_forces[self.bouyancy_node, 0:3] += np.dot(cbg, hd_forces_g[0:3])
        struct_tstep.unsteady_applied_forces[self.bouyancy_node, 3:6] += np.dot(cbg, hd_forces_g[3:6])


        # Wave loading
        phase = self.wave_freq*data.it*self.dt
        np.real(self.wave_amplitude*self.xi*(np.cos(phase) + 1j*np.sin(phase))))
