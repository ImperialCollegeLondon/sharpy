import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.cout_utils as cout
from sharpy.utils.constants import deg2rad


def compute_xf_zf(hf, vf, l, w, E, A, cg):

    xf = (l - vf/w + hf/w*np.log(vf/hf + np.sqrt(1 + (vf/hf)**2)) +
          hf*l/E/A +
          cg*w/2/E/A*(-(l-vf/w)**2 +
                      (l-vf/w-hf/cg/w)*np.max(l - vf/w - hf/cg/w, 0)))

    zf = hf/w*(np.sqrt(1+(vf/hf)**2) - np.sqrt(1 + ((vf-w*l)/hf)**2)) + 1./E/A*(vf*l-w*l**2/2)

    return xf, zf

def compute_jacobian(hf, vf, l, w, E, A, cg):

    der_xf_hf = (1/w*np.log(vf/hf + np.sqrt(1 + (vf/hf)**2)) +
                 hf/w/(vf/hf + np.sqrt(1 + (vf/hf)**2))*(-vf/hf**2 + 0.5*(1+(vf/hf)**2)**(-0.5)*(2*vf/hf*(-vf/hf**2))) +
                 l/E/A)
    arg1_max = l - vf/w - hf/cg/w
    if arg1_max > 0.:
        der_xf_hf += cg*w/2/E/A*(2*(arg1_max)*(-1/cg/w))

    def_xf_vf = (1/w + hf/w/(vf/hf + np.sqrt(1 + (vf/hf)**2))*(1/hf + 0.5*(1+(vf/hf)**2)**(-0.5)*(2*vf/hf/hf)) +
                 cg*w/2/E/A*(2*(l-vf/w)/w))
    if arg1_max > 0.:
        der_xf_vf += cg*w/2/E/A*(2*(arg1_max)*(-1/w))

    der_zf_hf = (1/w*(np.sqrt(1+(vf/hf)**2) - np.sqrt(1 + ((vf-w*l)/hf)**2)) +
                hf/w*(0.5*(1+(vf/hf)**2)**(-0.5)*(2*vf/hf*(-vf/hf**2)) -
                0.5*(1 + ((vf-w*l)/hf)**2)**(-0.5)*(2*(vf-w*l)/hf*(vf-w*l)/hf**2)))

    der_zf_vf = ((hf/w)*(0.5*(1+(vf/hf)**2)**(-0.5)*(2*vf/hf/hf) -
                 0.5*(1 + ((vf-w*l)/hf)**2)**(-0.5)*(2*(vf-w*l)/hf/hf)) +
                 1/E/A*l)

    J = np.array([[der_xf_hf, der_xf_vf],[der_zf_hf, der_zf_vf]])

    return J

# def test_jacobian:
#
#     # Values for OC3
#     l = 902.2 # Initial length [m]
#     w = 77.7066*9.81 # Apparent mass per unit length times gravity
#     E = 384243000 # Extensional stiffness
#     A = np.pi*(0.09/2)**2 # Cross-section area
#     cg = 0. # Seabed friction coefficient
#
#     xf =
#     hf =
#
#     dist_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
#     for dist in dist_vec:
#
#     for a in A:
#             # perturbed
#             fv = a * fv1 + (1. - a) * fv0
#             dfv = fv - fv0
#
#             ### Compute relevant quantities
#             dCab = algebra.crv2rotation(fv0 + dfv) - Cab
#             T = algebra.crv2tan(fv0)
#             Tdfv = np.dot(T, dfv)
#             Tdfv_skew = algebra.skew(Tdfv)
#             dCab_an = np.dot(Cab, Tdfv_skew)
#
#             er_tan_new = np.max(np.abs(dCab - dCab_an)) / np.max(np.abs(dCab_an))
#             assert er_tan_new < er_tan, 'crv2tan error not converging to 0'
#             er_tan = er_tan_new
#
#         assert er_tan < A[-2], 'crv2tan error too large'


def jonkman_mooring(xf, zf, l, w, E, A, cg, hf0=None, vf0=None):

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
    hf = hf0 + 0.
    vf = vf0 + 0.
    xf_est, zf_est = compute_xf_zf(hf, vf, l, w, E, A, cg)
    tol = 1e-6
    error = 2*tol
    while error > tol:
        J = compute_jacobian(hf, vf, l, w, E, A, cg):
        hf += (xf - xf_est)/J[0, 0] + (zf - zf_est)/J[1, 0]
        vf += (xf - xf_est)/J[0, 1] + (zf - zf_est)/J[1, 1]

        xf_est, zf_est = compute_xf_zf(hf, vf, l, w, E, A, cg)
        error = np.max(np.abs(xf - xf_est), np.abs(zf - zf_est))

    return hf, vf

def tri_area(p1, p2, p3):
    # Heron's formula
    # https://en.wikipedia.org/wiki/Triangle

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c)/2.0
    area = np.sqrt(s*(s - a)*(s - b)*(s - c))

    return area


def tri_centroid(p1, p2, p3):
    # Computed as intersection of medians
    m12 = 0.5*(p1 + p2) # Median side from 1 to 2
    m13 = 0.5*(p1 + p3) # Median side from 1 to 3

    # Using the vector equation of the medians as m3 = p3 + t3*(m12 - p3)
    #                                             m2 = p2 + t2*(m13 - p2)
    # Forcing m2 = m3, and using the equations for x and y I can create the
    # system for t:

    A = np.array([[m12[0] - p3[0], -1.*(m13[0] - p2[0])],
                  [m12[1] - p3[1], -1.*(m13[1] - p2[1])]])
    b = np.array([p2[0] - p3[0], p2[1] - p3[1]])
    t = np.dot(np.linalg.inv(A), b)

    centroid = p3 + t[0]*(m12 - p3)

    return centroid

def tri_mom_inertia_centroid(p1, p2, p3):
    # Compute the moments of inertia around a set of axis with origin the centroid

    # First compute it with respect to a set of axis such that:
    # - two axis belong to the plane of the triangle (a1, a2) and a1 of them is
    # parallel to the side p2-p3
    # - The third axis is perpendicular to the triangle plane
    # b = np.linalg.norm(p3 - p2)
    # h =
    # I11 = (b*h**3)/36


    # Convert to globla xyz system




def quad_area(p1, p2, p3, p4):
    # p1, p2, p3, p4 are consecutive vertices
    area1 = tri_area(p1, p2, p4)
    area2 = tri_area(p2, p3, p4)

    return area1 + area2


def quad_centroid(p1, p2, p3, p4):
    # p1, p2, p3, p4 are consecutive vertices
    centroid1 = tri_centroid(p1, p2, p4)
    area1 = tri_area(p1, p2, p4)
    centroid2 = tri_centroid(p2, p3, p4)
    area2 = tri_area(p2, p3, p4)

    area = area1 + area2
    centroid = (area1*centroid1 + area2*centroid2)/area

    return centroid


@generator_interface.generator
class FloatingForces(generator_interface.BaseGenerator):
    r"""
    Floating forces generator

    Generates the forces associated the floating support of offshore wind turbines.

    S
    """
    generator_id = 'FloatingForces'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['water_density'] = 'float'
    settings_default['water_density'] = 1.025 # kg/m3
    settings_description['water_density'] = 'Water density'

    settings_types['gravity'] = 'float'
    settings_default['gravity'] = 9.81
    settings_description['gravity'] = 'Gravity'

    settings_types['gravity_dir'] = 'list(float)'
    settings_default['gravity_dir'] = np.array([1., 0., 0.])
    settings_description['gravity_dir'] = 'Gravity direction'

    settings_types['mooring_node'] = 'int'
    settings_default['mooring_node'] = 0
    settings_description['mooring_node'] = 'Structure node where mooring forces are applied'

    settings_types['mooring_K'] = 'float'
    settings_default['mooring_K'] = 0.
    settings_description['mooring_K'] = 'Elastic constant of mooring lines'

    settings_types['mooring_C'] = 'float'
    settings_default['mooring_C'] = 0.
    settings_description['mooring_C'] = 'Damping constant of mooring lines'

    settings_types['COB_node'] = 'int'
    settings_default['COB_node'] = 0
    settings_description['COB_node'] = 'Structure node where buoyancy forces are applied (Centre Of Buoyancy)'

    settings_types['V0'] = 'float'
    settings_default['V0'] = 8026463.788
    settings_description['V0'] = 'Initial volume submerged'

    settings_types['buoy_rest_heave_heave'] = 'float'
    settings_default['buoy_rest_heave_heave'] = 332.941
    settings_description['buoy_rest_heave_heave'] = 'Buoyancy restoring coefficient of the heave-heave motion'

    settings_types['buoy_rest_roll_roll'] = 'float'
    settings_default['buoy_rest_roll_roll'] = -4999180000.
    settings_description['buoy_rest_roll_roll'] = 'Buoyancy restoring coefficient of the roll-roll motion'

    settings_types['buoy_rest_pitch_pitch'] = 'float'
    settings_default['buoy_rest_pitch_pitch'] = -4999180000.
    settings_description['buoy_rest_pitch_pitch'] = 'Buoyancy restoring coefficient of the pitch-pitch motion'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()

        self.water_density = None
        self.gravity = None
        self.gravity_dir = None

        self.mooring_node = None
        self.mooring_K = None
        self.mooring_C = None

        self.cob_node = None
        self.buoy_F0 = None
        self.buoy_rest_mat = None


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

        self.mooring_node = self.settings['mooring_node']
        self.mooring_K = self.settings['mooring_K']
        self.mooring_C = self.settings['mooring_C']

        self.cob_node = self.settings['COB_node']
        self.buoy_F0 = np.zeros((6,), dtype=float)
        self.buoy_F0[0:3] = -(self.settings['V0']*
                              self.settings['water_density']*
                              self.settings['gravity']*self.settings['gravity_dir'])
        self.buoy_rest_mat = np.zeros((6,6), dtype=float)
        self.buoy_rest_mat[0,0] = self.settings['buoy_rest_heave_heave']
        self.buoy_rest_mat[5,5] = self.settings['buoy_rest_roll_roll']
        self.buoy_rest_mat[4,4] = self.settings['buoy_rest_pitch_pitch']


    def generate(self, params):
        # Renaming for convenience
        data = params['data']
        struct_tstep = params['struct_tstep']
        aero_tstep = params['aero_tstep']

        # Mooring lines
        # jonkman_mooring(xf, zf, l, w, E, A, cg, hf0=None, vf0=None):
        base_disp = struct_tstep.for_pos[0:3] - data.beam.ini_info.for_pos[0:3]

        base_vel = struct_tstep.for_pos[0:3]

        mooring = np.array([[0., 0., 1.],
                            [0., np.cos(30*deg2rad), -np.sin(30*deg2rad)],)
                            [0., -np.cos(30*deg2rad), -np.sin(30*deg2rad)]])

        for imooring in range(mooring.shape[0]):
            disp = np.dot(mooring[imooring, :], base_disp)
            if disp < 0.:
                struct_tstep.applied_forces[self.mooring_node, 0:3] += solf.mooring_K*np.abs(disp)*mooring[imooring, :]

            vel = np.dot(mooring[imooring, :], base_vel)
            if vel < 0.:
                struct_tstep.applied_forces[self.mooring_node, 0:3] += solf.mooring_C*np.abs(vel)*mooring[imooring, :]

        # Hydrodynamic model
        cga = struct_tstep.cga()
        q = np.zeros((6,), dtype=int)
        q[0:3] = np.dot(cga, struct_tstep.pos[self.cob_node, :]) + struct_tstep.for_pos[0:3]
        q[3:6] = algebra.quat2euler(struct_tstpe.quat) # roll, pitch, yaw

        hd_forces_g = self.buoy_F0 + np.dot(self.buoy_rest_mat, q)
        ielem, inode_in_elem = data.structure.node_master_elem[self.cob_node]
        cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem])
        cbg = np.dot(cab.T, cga.T)

        struct_tstep.applied_forces[self.cob_node, 0:3] = np.dot(cbg, hd_forces_g[0:3])
        struct_tstep.applied_forces[self.cob_node, 3:6] = np.dot(cbg, hd_forces_g[3:6])
