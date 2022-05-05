"""
Horten Wing Class Generator
N Goizueta Nov 18
"""

import numpy as np
import os
import h5py as h5
import sharpy.utils.geo_utils as geo_utils
import sharpy.utils.algebra as algebra
import configobj
import scipy.linalg as sclalg


class HortenWing:
    """
    Horten Wing Class Generator

    A ``HortenWing`` class contains the basic geometry and properties of a simplified Horten wing, as
    described by Richards (2016)

    This class allows the user to quickly obtain SHARPy cases for the purposes of parametric analyses.

    """

    def __init__(self,
                 M,
                 N,
                 Mstarfactor,
                 u_inf,
                 rho=1.225,
                 alpha_deg=0.,
                 beta_deg=0.,
                 roll_deg=0.,
                 cs_deflection_deg=0.,
                 thrust=5.,
                 physical_time=10,
                 case_name_format=2,
                 case_remarks=None,
                 case_route='./cases/',
                 case_name='horten'):

        # Discretisation
        self.M = M
        self.N = N
        self.Mstarfactor = Mstarfactor

        self.n_node_elem = 3
        self.n_elem_wing = N
        self.n_elem_fuselage = 1
        self.n_surfaces = 4

        # Case admin
        if case_name_format == 0:
            self.case_name = case_name + '_u_inf%.4d_a%.4d' % (int(u_inf*100), int(alpha_deg * 100))
        elif case_name_format == 1:
            self.case_name = case_name + '_u_inf%.4d' % int(u_inf*100)
        elif case_name_format == 2:
            self.case_name = case_name
        else:
            self.case_name = case_name + '_u_inf%.4d_%s' % (int(u_inf*100), case_remarks)

        self.case_route = os.path.abspath(case_route + self.case_name + '/')

        self.config = None

        # Flight conditions
        self.u_inf = u_inf
        self.rho = rho
        self.alpha = alpha_deg * np.pi / 180
        self.roll = roll_deg * np.pi / 180
        self.beta = beta_deg * np.pi / 180
        self.cs_deflection = cs_deflection_deg * np.pi / 180
        self.thrust = thrust

        # Compute number of nodes
        n_node = 0
        self.n_node_wing = self.n_elem_wing * (self.n_node_elem - 1)
        self.n_node_fuselage = self.n_elem_fuselage * self.n_node_elem
        n_node += 2 * self.n_node_fuselage - 1 + 2 * self.n_node_wing
        self.n_node = n_node

        # Compute number of elements
        self.n_elem = 2 * (self.n_elem_wing + self.n_elem_fuselage)

        # Wing geometry
        self.span = 20.0  # [m]
        self.sweep_LE = 20 * np.pi / 180  # [rad] Leading Edge Sweep
        # self.c_root = 1.0  # [m] Root chord - Richards
        self.c_root = 1.5819  # [m] Root chord - Mardanpour 2014
        self.taper_ratio = 0.17 # Mardanpour 2014
        # self.taper_ratio = 0.25 # Richards
        self.thrust_nodes = [self.n_node_fuselage - 1,
                             self.n_node_fuselage + self.n_node_wing + 1]

        self.loc_cg = 0.45  # CG position wrt to LE (from sectional analysis)
        # EA is the reference in NATASHA - defined with respect to the midchord. SHARPy is wrt to LE and as a pct of
        # local chord
        self.main_ea_root = 0.5 + 0.15*0.0254 / self.c_root
        self.main_ea_tip = 0.5 + 0.21*0.0254 / (self.c_root*self.taper_ratio)

        # FUSELAGE GEOMETRY
        # self.fuselage_width = 1.
        self.fuselage_width = 0.8248
        self.c_fuselage = 84*0.0254

        # WASH OUT
        self.washout_root = 0*np.pi/180
        self.washout_tip = -2 * np.pi / 180

        # Horseshoe wake
        self.horseshoe = False
        self.wake_type = 2
        self.dt_factor = 1
        self.dt = 1 / self.M / self.u_inf * self.dt_factor

        # Dynamics
        self.physical_time = physical_time
        self.n_tstep = int(physical_time/self.dt)
        self.gust_intensity = 0.05

        # Numerics
        self.tolerance = 1e-12
        self.fsi_tolerance = 1e-10
        self.relaxation_factor = 0.2

        # H5 Variables initialisation as class attributes
        # coordinates
        self.x = np.zeros((n_node,))
        self.y = np.zeros((n_node,))
        self.z = np.zeros((n_node,))
        # beam number
        self.beam_number = np.zeros(self.n_elem, dtype=int)
        # frame of reference delta
        self.frame_of_reference_delta = np.zeros((self.n_elem, self.n_node_elem, 3))
        # connectivity of beams
        self.connectivities = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        # stiffness
        self.n_stiffness = self.n_elem_wing + self.n_elem_fuselage
        self.base_stiffness = np.zeros((self.n_stiffness, 6, 6))
        self.elem_stiffness = np.zeros((self.n_elem,), dtype=int)
        # mass
        self.n_mass = self.n_elem_wing * 2 // 2
        self.base_mass = np.zeros((self.n_mass, 6, 6))
        self.elem_mass = np.zeros(self.n_elem, dtype=int)
        # boundary conditions
        self.boundary_conditions = np.zeros((n_node,), dtype=int)
        # applied forces
        self.app_forces = np.zeros((n_node, 6))
        self.n_lumped_mass = 3

        self.lumped_mass_nodes = np.zeros((self.n_lumped_mass), dtype=int)
        self.lumped_mass = np.zeros(self.n_lumped_mass)
        self.lumped_mass_inertia = np.zeros((self.n_lumped_mass, 3, 3))
        self.lumped_mass_position = np.zeros((self.n_lumped_mass, 3))

        # Aerodynamic properties
        # H5 AERO FILE VARIABLES INITIALISATION
        # airfoil distribution
        self.airfoil_distribution = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        # surface distribution
        self.surface_distribution = np.zeros((self.n_elem,), dtype=int) - 1
        self.surface_m = np.zeros((self.n_surfaces,), dtype=int)
        self.m_distribution = 'uniform'
        # aerodynamic nodes boolean
        self.aero_nodes = np.zeros((self.n_node,), dtype=bool)
        # aero twist
        self.twist = np.zeros((self.n_elem, self.n_node_elem))
        # chord
        self.chord = np.zeros((self.n_elem, self.n_node_elem))
        # elastic axis
        self.elastic_axis = np.zeros((self.n_elem, self.n_node_elem))

        # control surfaces attributes initialisation
        self.n_control_surfaces = 1
        self.control_surface = np.zeros((self.n_elem, self.n_node_elem), dtype=int) - 1
        self.control_surface_type = np.zeros((self.n_control_surfaces,), dtype=int)
        self.control_surface_deflection = np.zeros((self.n_control_surfaces,))
        self.control_surface_chord = np.zeros((self.n_control_surfaces,), dtype=int)
        self.control_surface_hinge_coord = np.zeros((self.n_control_surfaces,), dtype=float)

        self.settings = dict()

    def initialise(self):

        if not os.path.exists(self.case_route):
            os.makedirs(self.case_route)

        # Compute number of nodes
        n_node = 0
        self.n_node_wing = self.n_elem_wing * (self.n_node_elem - 1)
        self.n_node_fuselage = self.n_elem_fuselage * self.n_node_elem
        n_node += 2 * self.n_node_fuselage - 1 + 2 * self.n_node_wing
        self.n_node = n_node

        # Compute number of elements
        self.n_elem = 2 * (self.n_elem_wing + self.n_elem_fuselage)

        self.dt = 1 / self.M / self.u_inf * self.dt_factor

        # H5 Variables initialisation as class attributes
        # coordinates
        self.x = np.zeros((n_node,))
        self.y = np.zeros((n_node,))
        self.z = np.zeros((n_node,))
        # beam number
        self.beam_number = np.zeros(self.n_elem, dtype=int)
        # frame of reference delta
        self.frame_of_reference_delta = np.zeros((self.n_elem, self.n_node_elem, 3))
        # connectivity of beams
        self.connectivities = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        # stiffness
        self.n_stiffness = self.n_elem_wing + self.n_elem_fuselage
        self.base_stiffness = np.zeros((self.n_stiffness, 6, 6))
        self.elem_stiffness = np.zeros((self.n_elem,), dtype=int)
        # mass
        self.base_mass = np.zeros((self.n_mass, 6, 6))
        self.elem_mass = np.zeros(self.n_elem, dtype=int)
        # boundary conditions
        self.boundary_conditions = np.zeros((n_node,), dtype=int)
        # applied forces
        self.app_forces = np.zeros((n_node, 6))
        self.n_lumped_mass = 3

        self.lumped_mass_nodes = np.zeros((self.n_lumped_mass), dtype=int)
        self.lumped_mass = np.zeros(self.n_lumped_mass)
        self.lumped_mass_inertia = np.zeros((self.n_lumped_mass, 3, 3))
        self.lumped_mass_position = np.zeros((self.n_lumped_mass, 3))

        # Aerodynamic properties
        # H5 AERO FILE VARIABLES INITIALISATION
        # airfoil distribution
        self.airfoil_distribution = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        # surface distribution
        self.surface_distribution = np.zeros((self.n_elem,), dtype=int) - 1
        self.surface_m = np.zeros((self.n_surfaces,), dtype=int)
        self.m_distribution = 'uniform'
        # aerodynamic nodes boolean
        self.aero_nodes = np.zeros((self.n_node,), dtype=bool)
        # aero twist
        self.twist = np.zeros((self.n_elem, self.n_node_elem))
        # chord
        self.chord = np.zeros((self.n_elem, self.n_node_elem))
        # elastic axis
        self.elastic_axis = np.zeros((self.n_elem, self.n_node_elem))

        # control surfaces attributes initialisation
        self.n_control_surfaces = 1
        self.control_surface = np.zeros((self.n_elem, self.n_node_elem), dtype=int) - 1
        self.control_surface_type = np.zeros((self.n_control_surfaces,), dtype=int)
        self.control_surface_deflection = np.zeros((self.n_control_surfaces,))
        self.control_surface_chord = np.zeros((self.n_control_surfaces,), dtype=int)
        self.control_surface_hinge_coord = np.zeros((self.n_control_surfaces,), dtype=float)

        self.settings = dict()

    def dynamic_control_surface(self, *delta):
        """
        Generate dynamic control surface input files

        Args:
            delta (list): list of numpy arrays containing deflection time history

        Returns:

        """
        i = 0
        for cs in delta:
            self.control_surface_type[i] = 1
            np.savetxt(self.case_route + '/' + self.case_name + '.input.txt', cs)
            i += 1

    def planform_area(self):
        S_fus = 0.5 * (self.c_fuselage + self.c_root) * self.fuselage_width
        S_wing = 0.5 * (self.c_root + self.c_root*self.taper_ratio) * self.span / 2
        return 2*S_fus + 2*S_wing

    def update_mass_stiffness(self, sigma=1.):
        r"""
        The mass and stiffness matrices are computed. Both vary over the span of the wing, hence a
        dictionary is created that acts as a database of the different properties along the wing.

        The variation of the stiffness is cubic along the span:

            .. math:: \mathbf{K} = \mathbf{K}_0\bar{c}^3

        where :math:`\mathbf{K}_0 is the stiffness of the wing root section and :math:`\bar{c}` is
        the ratio between the local chord and the root chord.

        The variation of the sectional mass is quadratic along the span:

            .. math:: \mu = \mu_0\,\bar{c}^2

        where :math:`\mu` is the mass per unit span and the zero subscript denotes the root value.

        The sectional inertia is varied linearly along the span, based on the information by Mardanpour 2013.

        Three lumped masses are included with the following properties (Richards, 2013)

        =====  =====  =================  =========  ===============================  ============
        No     Node   Relative Position  Mass [kg]  Inertia [kg m^2]                 Description
        =====  =====  =================  =========  ===============================  ============
        ``0``  ``2``  ``[0,0,0]``        ``5.24``   ``[0.29547, 0.29322, 0.29547]``  Right Engine
        ``1``  ``S``  ``[0,0,0]``        ``5.24``   ``[0.29547, 0.29322, 0.29547]``  Left Engine
        ``2``  ``0``  ``[0,0,0]``        ``15.29``  ``[0.5, 1.0, 1.0]*15.29``        Fuselage
        =====  =====  =================  =========  ===============================  ============


        Args:
            sigma (float): stiffening factor

        Returns:

        """

        n_elem_fuselage = self.n_elem_fuselage
        n_elem_wing = self.n_elem_wing
        n_node_wing = self.n_node_wing
        n_node_fuselage = self.n_node_fuselage
        c_root = self.c_root
        taper_ratio = self.taper_ratio

        # Local chord to root chord initialisation
        c_bar_temp = np.linspace(c_root, taper_ratio * c_root, n_elem_wing)

        # Structural properties at the wing root section from Richards 2016
        ea = 1e6
        ga = 1e6
        gj = 4.24e5
        eiy = 3.84e5
        eiz = 2.46e7

        # Sectional mass from Richards 2016
        mu_0 = 9.761
        # Bending inertia properties from Mardanpour 2013
        j_root = 0.303
        j_tip = 0.2e-2 * 4.448 / 9.81

        # Number of stiffnesses used
        n_stiffness = self.n_stiffness

        # Initialise the stiffness database
        base_stiffness = self.base_stiffness

        # Wing root section stiffness properties
        # stiffness_root = sigma * np.diag([ea, ga, ga, gj, eiy, eiz])
        stiffness_root = np.array([[3.23624595e+09, 0.00000000e+00, 0.00000000e+00,
                                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [0.00000000e+00, 1.00000000e+14, 0.00000000e+00,
                                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+14,
                                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                    8.04987046e+07, -1.69971789e+05, 5.69905411e+07],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                    -1.69971789e+05, 5.03651190e+07, -6.70649560e+06],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                    5.69905411e+07, -6.70649560e+06, 2.24864852e+09]]) * sigma

        stiffness_tip = np.array([[5.86034256e+07, 0.00000000e+00, 0.00000000e+00,
                                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 1.00000000e+15, 0.00000000e+00,
                                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+15,
                                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                   2.93227100e+04, -2.11280834e+03, 2.52782426e+04],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                   -2.11280834e+03, 1.44883639e+05, -2.52913470e+04],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                   2.52782426e+04, -2.52913470e+04, 3.02592902e+05]]) * sigma


        stiffness_root = np.diag(np.diag(stiffness_root))
        stiffness_tip = np.diag(np.diag(stiffness_tip))

        # Rigid fuselage
        sigma_fuselage = 1000
        base_stiffness[0, :, :] = sigma_fuselage * stiffness_root

        # Cubic variation of the stiffness along the span - Richards
        # for stiff_entry in range(1, n_stiffness):
        #     base_stiffness[stiff_entry, :, :] = stiffness_root * c_bar_temp[stiff_entry - 1] ** 3

        # Linear variation between root an tip as per Mardanpour 2014
        alpha = np.linspace(0, 1, self.n_elem_wing)
        for i_elem in range(0, self.n_elem_wing):
            base_stiffness[i_elem + 1, :, :] = stiffness_root*(1-alpha[i_elem]) + stiffness_tip*alpha[i_elem]

        # Mass variation along the span
        n_mass = self.n_mass
        sigma_mass = 1

        # Initialise database
        base_mass = self.base_mass

        mu_root = 2.784472 * sigma_mass
        chi_root_right = np.array([0, -5.29, 0.594]) * 0.0254 * 0
        chi_root_left = np.array([0, +5.29, 0.594]) * 0.0254 * 0
        m_root = np.eye(3) * mu_root
        j_root = np.array([[0.30378797, 0., 0.],
                           [0., 0.0122422, 0.],
                           [0., 0., 0.30016065]])

        j_root = np.diag([0.1, 0.1, 0.2])
        mass_root_right = sclalg.block_diag(np.diag([mu_root, mu_root, mu_root]), j_root)
        mass_root_left = sclalg.block_diag(np.diag([mu_root, mu_root, mu_root]), j_root)

        mass_root_right[:3, -3:] = -mu_root * algebra.skew(chi_root_right)
        mass_root_right[-3:, :3] = mu_root * algebra.skew(chi_root_right)

        mass_root_left[:3, -3:] = -mu_root * algebra.skew(chi_root_left)
        mass_root_left[-3:, :3] = mu_root * algebra.skew(chi_root_left)

        mu_tip = 0.284084 * sigma_mass
        chi_tip_right = np.array([0, -1.644, +0.563]) * 0.0254 * 0
        chi_tip_left = np.array([0, +1.644, +0.563]) * 0.0254 * 0
        j_tip = np.array([[9.06829766e-04, 0.00000000e+00, 0.00000000e+00],
                          [0.00000000e+00, 6.34780836e-05, 0.00000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 8.16146789e-04]])
        j_tip = np.diag([0.1, 0.1, 0.2])
        mass_tip_right = sclalg.block_diag(np.diag([mu_tip, mu_tip, mu_tip]), j_tip)
        mass_tip_left = sclalg.block_diag(np.diag([mu_tip, mu_tip, mu_tip]), j_tip)
        mass_tip_right[:3, -3:] += -algebra.skew(chi_tip_right) * mu_tip
        mass_tip_right[-3:, :3] += algebra.skew(chi_tip_right) * mu_tip

        mass_tip_left[:3, -3:] += -algebra.skew(chi_tip_right) * mu_tip
        mass_tip_left[-3:, :3] += algebra.skew(chi_tip_right) * mu_tip

        mass_tip_left = mass_tip_right
        mass_root_left = mass_root_right

        for i_elem in range(self.n_elem_wing):
            base_mass[i_elem, :, :] = mass_root_right*(1-alpha[i_elem]) + mass_tip_right*alpha[i_elem]

        # for i_elem in range(self.n_elem_wing, 2*self.n_elem_wing):
        #     base_mass[i_elem, :, :] = mass_root_left*(1-alpha[i_elem-self.n_elem_wing]) + mass_tip_left*alpha[i_elem-self.n_elem_wing]

        # Quadratic variation in the mass per unit span - Richards
        # mu_temp = mu_0 * np.ones_like(c_bar_temp)
        # j_temp = np.linspace(j_root, j_tip, n_elem_wing)

        # for mass_entry in range(n_mass):
        #     base_mass[mass_entry, :, :] = np.diag([
        #         mu_temp[mass_entry],
        #         mu_temp[mass_entry],
        #         mu_temp[mass_entry],
        #         j_temp[mass_entry],
        #         j_temp[mass_entry],
        #         j_temp[mass_entry]
        #     ])

        # Lumped mass initialisation
        # 0 - Right engine
        # 1 - Left engine
        # 2 - Fuselage

        lumped_mass_nodes = self.lumped_mass_nodes
        lumped_mass = self.lumped_mass
        lumped_mass_inertia = self.lumped_mass_inertia
        lumped_mass_position = self.lumped_mass_position

        # Lumped masses nodal position
        lumped_mass_nodes[0] = 2
        lumped_mass_nodes[1] = n_node_fuselage + n_node_wing + 1
        lumped_mass_nodes[2] = 0

        # Lumped mass value from Richards 2013
        # lumped_mass[0:2] = 51.445 / 9.81
        # lumped_mass[2] = 150 / 9.81

        # Lumped masses from Mardanpour
        # Pilot mass
        lumped_mass[2] = 16.06
        lumped_mass_position[2] = np.array([0., -0.254, -0.254])

        # Engine mass
        lumped_mass[0:2] = 0.535
        lumped_mass_inertia[0, :, :] = np.diag([0.02994352, 0.02994352, 0.02994352])
        lumped_mass_inertia[1, :, :] = np.diag([0.02994352, 0.02994352, 0.02994352])

        # Lumped mass inertia
        # lumped_mass_inertia[0, :, :] = np.diag([0.29547, 0.29322, 0.29547])
        # lumped_mass_inertia[1, :, :] = np.diag([0.29547, 0.29322, 0.29547])
        # lumped_mass_inertia[2, :, :] = np.diag([0.5, 1, 1]) * lumped_mass[2]

        # Define class attributes
        self.lumped_mass = lumped_mass * 0
        self.lumped_mass_nodes = lumped_mass_nodes * 0
        self.lumped_mass_inertia = lumped_mass_inertia * 0
        self.lumped_mass_position = lumped_mass_position * 0

        self.base_stiffness = base_stiffness
        self.base_mass = base_mass

    def update_fem_prop(self):
        """
        Computes the FEM properties prior to analysis such as the connectivity matrix, coordinates, etc

        Returns:

        """
        # Obtain class attributes
        n_node_elem = self.n_node_elem
        n_elem = self.n_elem
        n_elem_wing = self.n_elem_wing
        n_elem_fuselage = self.n_elem_fuselage
        n_node = self.n_node
        n_node_wing = self.n_node_wing
        n_node_fuselage = self.n_node_fuselage
        fuselage_width = self.fuselage_width
        thrust = self.thrust
        thrust_nodes = self.thrust_nodes
        span = self.span
        sweep_LE = self.sweep_LE

        # mass and stiffness matrices
        stiffness = self.base_stiffness
        mass = self.base_mass
        n_stiffness = stiffness.shape[0]
        n_mass = mass.shape[0]

        # H5 FEM FILE VARIABLES INITIALISATION
        # coordinates
        x = np.zeros((n_node,))
        y = np.zeros((n_node,))
        z = np.zeros((n_node,))
        # twist
        structural_twist = np.zeros_like(x)
        # beam number
        beam_number = np.zeros(n_elem, dtype=int)
        # frame of reference delta
        frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
        # connectivity of beams
        conn = np.zeros((n_elem, n_node_elem), dtype=int)
        # stiffness
        stiffness = np.zeros((n_stiffness, 6, 6))
        elem_stiffness = np.zeros((n_elem,), dtype=int)
        # mass
        mass = np.zeros((n_mass, 6, 6))
        elem_mass = np.zeros(n_elem, dtype=int)
        # boundary conditions
        boundary_conditions = np.zeros((n_node,), dtype=int)
        # applied forces
        app_forces = np.zeros((n_node, 6))

        # assemble connectivites
        # worked elements
        we = 0
        # worked nodes
        wn = 0

        # RIGHT RIGID FUSELAGE
        beam_number[we:we + 1] = 0
        # coordinates
        x[wn:wn + n_node_fuselage] = 0
        y[wn:wn + n_node_fuselage] = np.linspace(0, fuselage_width / 2, n_node_fuselage)

        # connectivities
        elem_mass[0] = 0
        conn[we, :] = [0, 2, 1]

        # frame of reference change
        frame_of_reference_delta[0, 0, :] = [-1.0, 0.0, 0.0]
        frame_of_reference_delta[0, 1, :] = [-1.0, 0.0, 0.0]
        frame_of_reference_delta[0, 2, :] = [-1.0, 0.0, 0.0]

        # element stiffness
        elem_stiffness[0] = 0
        elem_mass[0] = 0

        # boundary conditions
        boundary_conditions[0] = 1

        # applied forces - engine 1
        app_forces[thrust_nodes[0]] = [0, thrust, 0,
                                       0, 0, 0]

        # updated worked nodes and elements
        we += n_elem_fuselage
        wn += n_node_fuselage

        # RIGHT WING
        beam_number[we:we + n_elem_wing] = 1
        # y coordinate (positive right)
        y[wn:wn + n_node_wing] = np.linspace(fuselage_width / 2,
                                             span / 2,
                                             n_node_wing + 1)[1:]
        x[wn:wn + n_node_wing] = 0 + (y[wn:wn + n_node_wing] - fuselage_width / 2) * np.tan(sweep_LE)

        # connectivities
        for ielem in range(n_elem_wing):
            conn[we + ielem, :] = (np.ones(n_node_elem) * (we + ielem) * (n_node_elem - 1) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

            elem_mass[we + ielem] = ielem
            elem_stiffness[we + ielem] = ielem + 1

        # element stiffness and mass
        # elem_stiffness[we:we+n_elem_wing] = 0
        # elem_mass[we:we+n_elem_wing] = 0

        # boundary conditions of free end
        boundary_conditions[wn + n_node_wing - 1] = -1

        # update worked elements and nodes
        we += n_elem_wing
        wn += n_node_wing

        # LEFT FUSELAGE
        beam_number[we:we + n_elem_fuselage] = 2
        # coordinates
        y[wn:wn + n_node_fuselage - 1] = np.linspace(0,
                                                     -fuselage_width / 2,
                                                     n_node_fuselage)[1:]
        x[wn:wn + n_node_fuselage - 1] = 0

        # connectivity
        conn[we, :] = [0, wn + 1, wn]

        # frame of reference delta
        for ielem in range(n_elem_fuselage):
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]

        # element stiffness and mass
        elem_stiffness[we:we + n_elem_fuselage] = 0
        elem_mass[we:we + n_elem_fuselage] = self.n_elem_wing

        # applied forces - engine 2
        app_forces[thrust_nodes[1]] = [0, -thrust, 0,
                                       0, 0, 0]

        # update worked elements and nodes
        we += n_elem_fuselage
        wn += n_node_fuselage - 1

        # LEFT WING
        # coordinates
        beam_number[we:we + n_elem_wing] = 3
        y[wn:wn + n_node_wing] = np.linspace(-fuselage_width / 2,
                                             -span / 2,
                                             n_node_wing + 1)[1:]
        x[wn:wn + n_node_wing] = 0 + -1 * (y[wn:wn + n_node_wing] + fuselage_width / 2) * np.tan(sweep_LE)

        # left wing connectivities
        for ielem in range(n_elem_wing):
            conn[we + ielem, :] = np.ones(n_node_elem) * (we + ielem) * (n_node_elem - 1) + [0, 2, 1]

            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [1.0, 0.0, 0.0]

            elem_mass[we + ielem] = ielem + self.n_elem_wing
            elem_stiffness[we + ielem] = ielem + 1

        # element stiffness and mass
        # elem_stiffness[we:we+n_node_wing] = 0

        # boundary conditions at the free end
        boundary_conditions[wn + n_node_wing - 1] = -1

        # update worked elements and nodes
        we += n_elem_wing
        wn += n_node_wing

        # set attributes
        self.x = x
        self.y = y
        self.z = z
        self.connectivities = conn
        self.elem_stiffness = elem_stiffness
        self.elem_mass = elem_mass
        self.frame_of_reference_delta = frame_of_reference_delta
        self.boundary_conditions = boundary_conditions
        self.beam_number = beam_number
        self.app_forces = app_forces

    def generate_fem_file(self):
        """
        Generates the ``.fem.h5`` folder containing the structural information of the problem

        The file is written to ``self.case_route / self.case_name .fem.h5``

        """


        with h5.File(self.case_route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            coordinates = h5file.create_dataset('coordinates',
                                                data=np.column_stack((self.x, self.y, self.z)))
            connectivities = h5file.create_dataset('connectivities', data=self.connectivities)
            num_nodes_elem_handle = h5file.create_dataset(
                'num_node_elem', data=self.n_node_elem)
            num_nodes_handle = h5file.create_dataset(
                'num_node', data=self.n_node)
            num_elem_handle = h5file.create_dataset(
                'num_elem', data=self.n_elem)
            stiffness_db_handle = h5file.create_dataset(
                'stiffness_db', data=self.base_stiffness)
            stiffness_handle = h5file.create_dataset(
                'elem_stiffness', data=self.elem_stiffness)
            mass_db_handle = h5file.create_dataset(
                'mass_db', data=self.base_mass)
            mass_handle = h5file.create_dataset(
                'elem_mass', data=self.elem_mass)
            frame_of_reference_delta_handle = h5file.create_dataset(
                'frame_of_reference_delta', data=self.frame_of_reference_delta)
            structural_twist_handle = h5file.create_dataset(
                'structural_twist', data=np.zeros((self.n_elem, self.n_node_elem)))
            bocos_handle = h5file.create_dataset(
                'boundary_conditions', data=self.boundary_conditions)
            beam_handle = h5file.create_dataset(
                'beam_number', data=self.beam_number)
            app_forces_handle = h5file.create_dataset(
                'app_forces', data=self.app_forces)
            lumped_mass_nodes_handle = h5file.create_dataset(
                'lumped_mass_nodes', data=self.lumped_mass_nodes)
            lumped_mass_handle = h5file.create_dataset(
                'lumped_mass', data=self.lumped_mass)
            lumped_mass_inertia_handle = h5file.create_dataset(
                'lumped_mass_inertia', data=self.lumped_mass_inertia)
            lumped_mass_position_handle = h5file.create_dataset(
                'lumped_mass_position', data=self.lumped_mass_position)

    def create_linear_simulation(self, delta_e=None, delta_dot=None):

        Kpanels = self.M * (self.n_node - 1)
        Kvertices = (self.M + 1) * self.n_node
        Kpanels_wake = Kpanels * self.Mstarfactor
        n_states_aero = 3 * Kpanels + Kpanels_wake
        # n_inputs_aero = 2 * 3 * Kvertices

        n_states_struct = 2*(6 * (self.n_node - 1) + 9)
        n_inputs_struct = n_states_struct // 2

        x0 = np.zeros((n_states_aero + n_states_struct))
        # x0[-7] = 0.05
        # x0[-4:] = (algebra.euler2quat([ -5*np.pi/180, 0, 0]))
        u = np.zeros((self.n_tstep, n_states_struct + n_inputs_struct + 2 * self.n_control_surfaces))
        # u[0:3, -7] = -1000
        if delta_e is not None:
            u[:, n_states_struct:n_states_struct+self.n_control_surfaces] = delta_e
            u[:, n_states_struct + self.n_control_surfaces:n_states_struct+self.n_control_surfaces + self.n_control_surfaces] = delta_dot

        # u[10:15, -8] = 100

        self.generate_linear_sim_files(x0, u)

    def generate_linear_sim_files(self, x0, input_vec):
        if not os.path.exists(self.case_route):
                os.makedirs(self.case_route)

        with h5.File(self.case_route + '/' + self.case_name + '.lininput.h5', 'a') as h5file:
            x0 = h5file.create_dataset(
                'x0', data=x0)
            u = h5file.create_dataset(
                'u', data=input_vec)

    def clean_test_files(self):
        """
        Clears previously generated files
        """

        case_name = self.case_name
        route = self.case_route

        # FEM
        fem_file_name = route + '/' + case_name + '.fem.h5'
        if os.path.isfile(fem_file_name):
            os.remove(fem_file_name)

        # Dynamics File
        dyn_file_name = route + '/' + case_name + '.dyn.h5'
        if os.path.isfile(dyn_file_name):
            os.remove(dyn_file_name)

        # Aerodynamics File
        aero_file_name = route + '/' + case_name + '.aero.h5'
        if os.path.isfile(aero_file_name):
            os.remove(aero_file_name)

        # Solver file
        solver_file_name = route + '/' + case_name + '.sharpy'
        if os.path.isfile(solver_file_name):
            os.remove(solver_file_name)

        # Flight conditions file
        flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
        if os.path.isfile(flightcon_file_name):
            os.remove(flightcon_file_name)

        # Linear inputs file
        lin_file_name  = self.case_route + '/' + self.case_name + '.lininput.h5'
        if os.path.isfile(lin_file_name):
            os.remove(lin_file_name)

        # if os.path.isdir(route):
        #     os.system('rm -r %s' %route)

    def update_aero_properties(self):
        """
        Updates the aerodynamic properties of the horten wing

        """

        # Retrieve attributes
        n_elem = self.n_elem
        n_node_elem = self.n_node_elem
        n_node_wing = self.n_node_wing
        n_node_fuselage = self.n_node_fuselage
        n_elem_fuselage = self.n_elem_fuselage
        n_elem_wing = self.n_elem_wing
        c_root = self.c_root
        taper_ratio = self.taper_ratio
        washout_root = self.washout_root
        washout_tip = self.washout_tip
        n_control_surfaces = self.n_control_surfaces
        cs_deflection = self.cs_deflection
        m = self.M
        main_ea_root = self.main_ea_root
        main_ea_tip = self.main_ea_tip
        airfoil_distribution = self.airfoil_distribution
        chord = self.chord
        surface_distribution = self.surface_distribution
        surface_m = self.surface_m
        aero_nodes = self.aero_nodes
        elastic_axis = self.elastic_axis
        twist = self.twist

        control_surface = self.control_surface
        control_surface_type = self.control_surface_type
        control_surface_deflection = self.control_surface_deflection
        control_surface_chord = self.control_surface_chord
        control_surface_hinge_coord = self.control_surface_hinge_coord

        self.dt = 1 / self.M / self.u_inf

        # control surface type: 0 = static
        # control surface type: 1 = dynamic
        control_surface_type[0] = self.control_surface_type[0]
        control_surface_deflection[0] = cs_deflection
        control_surface_chord[0] = 2  # m
        control_surface_hinge_coord[0] = 0.25

        # RIGHT FUSELAGE (Surface 0, Beam 0)
        we = 0
        wn = 0

        i_surf = 0
        airfoil_distribution[we:we + n_elem_fuselage] = 0
        surface_distribution[we:we + n_elem_fuselage] = i_surf
        surface_m[i_surf] = m

        aero_nodes[wn:wn + n_node_fuselage] = True

        temp_chord = np.linspace(self.c_fuselage, self.c_root, self.n_node_fuselage)
        temp_washout = washout_root

        # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_fuselage):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = temp_chord[node_counter]
                elastic_axis[ielem, i_local_node] = main_ea_root
                twist[ielem, i_local_node] = -temp_washout

        we += n_elem_fuselage
        wn += n_node_fuselage

        # RIGHT WING (Surface 1, Beam 1)
        # surface_id
        i_surf = 1
        airfoil_distribution[we:we + n_elem_wing, :] = 0
        surface_distribution[we:we + n_elem_wing] = i_surf
        surface_m[i_surf] = m

        # specify aerodynamic characteristics of wing nodes
        aero_nodes[wn:wn + n_node_wing - 1] = True

        # linear taper initialisation
        temp_chord = np.linspace(c_root, taper_ratio * c_root, n_node_wing + 1)

        # linear wash out initialisation
        temp_washout = np.linspace(washout_root, washout_tip, n_node_wing + 1)

        # elastic axis variation
        temp_ea = np.linspace(main_ea_root, main_ea_tip, n_node_wing + 1)

        # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_wing):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = temp_chord[node_counter]
                elastic_axis[ielem, i_local_node] = temp_ea[node_counter]
                twist[ielem, i_local_node] = -temp_washout[node_counter]
                if ielem >= round(((we + n_elem_wing) / 2)):
                    control_surface[ielem, i_local_node] = 0

        # update working element and node
        we += n_elem_wing
        wn += n_node_wing - 1

        # LEFT FUSELAGE (Surface 2, Beam 2)
        i_surf = 2
        airfoil_distribution[we:we + n_elem_fuselage] = 0
        surface_distribution[we:we + n_elem_fuselage] = i_surf
        surface_m[i_surf] = m

        aero_nodes[wn:wn + n_node_fuselage] = True

        temp_chord = np.linspace(self.c_fuselage, self.c_root, self.n_node_fuselage)
        temp_washout = washout_root

        # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_fuselage):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = temp_chord[node_counter]
                elastic_axis[ielem, i_local_node] = main_ea_root
                twist[ielem, i_local_node] = -temp_washout

        we += n_elem_fuselage
        wn += n_node_fuselage

        # LEFT WING (Surface 3, Beam 3)
        i_surf = 3
        airfoil_distribution[we:we + n_elem_wing, :] = 0
        surface_distribution[we: we + n_elem_wing] = i_surf
        surface_m[i_surf] = m

        # linear taper initialisation
        temp_chord = np.linspace(c_root, taper_ratio * c_root, n_node_wing + 1)

        # linear wash out initialisation
        temp_washout = np.linspace(washout_root, washout_tip, n_node_wing + 1)

        # specify aerodynamic characterisics of wing nodes
        aero_nodes[wn:wn + n_node_wing] = True

        # linear taper initialisation
        # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_wing):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = temp_chord[node_counter]
                elastic_axis[ielem, i_local_node] = temp_ea[node_counter]
                twist[ielem, i_local_node] = -temp_washout[node_counter]
                if ielem >= round((we + n_elem_wing / 2)):
                    control_surface[ielem, i_local_node] = 0

        # update working element and node
        we += n_elem_wing
        wn += n_node_wing

        # end node is the middle node
        mid_chord = np.array(chord[:, 1], copy=True)
        chord[:, 1] = chord[:, 2]
        chord[:, 2] = mid_chord

        mid_ea = np.array(elastic_axis[:, 1], copy=True)
        elastic_axis[:, 1] = elastic_axis[:, 2]
        elastic_axis[:, 2] = mid_ea

        # Update aerodynamic attributes of class
        self.chord = chord
        self.twist = twist
        self.aero_nodes = aero_nodes
        self.elastic_axis = elastic_axis
        self.control_surface = control_surface

    def generate_aero_file(self, route=None, case_name=None):
        """
        Generates the ``.aero.h5`` file with the aerodynamic properties of the wing

        Args:
            route (str): route to write case file. If None is specified the default will be used
            case_name (str): name of file. If None is specified the default will be used

        """

        if not route:
            route = self.case_route

        if not case_name:
            case_name = self.case_name

        if not os.path.isdir(self.case_route):
            os.makedirs(self.case_route)

        chord = self.chord
        twist = self.twist
        airfoil_distribution = self.airfoil_distribution
        surface_distribution = self.surface_distribution
        surface_m = self.surface_m
        m_distribution = self.m_distribution
        aero_nodes = self.aero_nodes
        elastic_axis = self.elastic_axis
        control_surface = self.control_surface
        control_surface_deflection = self.control_surface_deflection
        control_surface_chord = self.control_surface_chord
        control_surface_hinge_coord = self.control_surface_hinge_coord
        control_surface_type = self.control_surface_type

        control_surface_deflection[0] = self.cs_deflection

        with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
                geo_utils.generate_naca_camber(P=0, M=0)))
            naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
                geo_utils.generate_naca_camber(P=0, M=0)))
            naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
                geo_utils.generate_naca_camber(P=0, M=0)))

            # chord
            chord_input = h5file.create_dataset('chord', data=chord)
            dim_attr = chord_input.attrs['units'] = 'm'

            # twist
            twist_input = h5file.create_dataset('twist', data=twist)
            dim_attr = twist_input.attrs['units'] = 'rad'

            # airfoil distribution
            airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

            surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
            surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
            m_distribution_input = h5file.create_dataset('m_distribution',
                                                         data=m_distribution.encode('ascii', 'ignore'))

            aero_node_input = h5file.create_dataset('aero_node', data=aero_nodes)
            elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

            control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
            control_surface_deflection_input = h5file.create_dataset('control_surface_deflection',
                                                                     data=control_surface_deflection)
            control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
            control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord',
                                                                      data=control_surface_hinge_coord)
            control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)

    def set_default_config_dict(self, route=None, case_name=None):
        """
        Generates default solver configuration file
        Returns:

        """

        if not route:
            route = self.case_route

        if not case_name:
            case_name = self.case_name

        u_inf = self.u_inf
        rho = self.rho
        dt = self.dt
        tolerance = self.tolerance
        alpha = self.alpha
        beta = self.beta
        thrust = self.thrust
        thrust_nodes = self.thrust_nodes
        cs_deflection = self.cs_deflection
        fsi_tolerance = self.fsi_tolerance
        n_tstep = self.n_tstep
        gust_intensity = self.gust_intensity
        relaxation_factor = self.relaxation_factor

        file_name = route + '/' + case_name + '.sharpy'

        settings = dict()
        settings['SHARPy'] = {'case': case_name,
                              'route': route,
                              'flow': ['BeamLoader',
                                       'AerogridLoader',
                                       'StaticCoupled',
                                       'Modal',
                                       'AerogridPlot',
                                       'BeamPlot',
                                       'SaveData'],
                              'write_screen': 'on',
                              'write_log': 'on',
                              'log_folder': route + '/output/' + case_name + '/',
                              'log_file': case_name + '.log'}

        settings['BeamLoader'] = {'unsteady': 'off',
                                  'orientation': algebra.euler2quat(np.array([self.roll,
                                                                              self.alpha,
                                                                              self.beta]))}

        settings['StaticUvlm'] = {'print_info': 'on',
                                  'horseshoe': self.horseshoe,
                                  'num_cores': 4,
                                  'n_rollup': 1,
                                  'rollup_dt': dt,
                                  'rollup_aic_refresh': 1,
                                  'rollup_tolerance': 1e-4,
                                  'velocity_field_generator': 'SteadyVelocityField',
                                  'velocity_field_input': {'u_inf': u_inf,
                                                           'u_inf_direction': [1., 0, 0]},
                                  'rho': rho}

        settings['StaticCoupled'] = {'print_info': 'on',
                                     'structural_solver': 'NonLinearStatic',
                                     'structural_solver_settings': {'print_info': 'off',
                                                                    'max_iterations': 200,
                                                                    'num_load_steps': 1,
                                                                    'delta_curved': 1e-5,
                                                                    'min_delta': tolerance,
                                                                    'gravity_on': 'on',
                                                                    'gravity': 9.81},
                                     'aero_solver': 'StaticUvlm',
                                     'aero_solver_settings': {'print_info': 'on',
                                                              'horseshoe': self.horseshoe,
                                                              'num_cores': 4,
                                                              'n_rollup': int(0),
                                                              'rollup_dt': dt, #self.c_root / self.M / self.u_inf,
                                                              'rollup_aic_refresh': 1,
                                                              'rollup_tolerance': 1e-4,
                                                              'velocity_field_generator': 'SteadyVelocityField',
                                                              'velocity_field_input': {'u_inf': u_inf,
                                                                                       'u_inf_direction': [1., 0, 0]},
                                                              'rho': rho},
                                     'max_iter': 200,
                                     'n_load_steps': 1,
                                     'tolerance': tolerance,
                                     'relaxation_factor': 0.2}

        if self.horseshoe is True:
            settings['AerogridLoader'] = {'unsteady': 'off',
                                          'aligned_grid': 'on',
                                          'mstar': 1,
                                          'freestream_dir': ['1', '0', '0'],
                                          'control_surface_deflection': ['']}
        else:
            settings['AerogridLoader'] = {'unsteady': 'off',
                                          'aligned_grid': 'on',
                                          'mstar': int(self.M * self.Mstarfactor),
                                          'freestream_dir': ['1', '0', '0'],
                                          'control_surface_deflection': ['']}

        settings['NonLinearStatic'] = {'print_info': 'off',
                                       'max_iterations': 150,
                                       'num_load_steps': 1,
                                       'delta_curved': 1e-8,
                                       'min_delta': tolerance,
                                       'gravity_on': True,
                                       'gravity': 9.81}

        settings['StaticTrim'] = {'solver': 'StaticCoupled',
                                  'solver_settings': settings['StaticCoupled'],
                                  'thrust_nodes': thrust_nodes,
                                  'initial_alpha': alpha,
                                  'initial_deflection': cs_deflection,
                                  'initial_thrust': thrust,
                                  'max_iter': 200,
                                  'fz_tolerance': 1e-2,
                                  'fx_tolerance': 1e-2,
                                  'm_tolerance': 1e-2}

        settings['Trim'] = {'solver': 'StaticCoupled',
                            'solver_settings': settings['StaticCoupled'],
                            'initial_alpha': alpha,
                            'initial_beta': beta,
                            'cs_indices': [0],
                            'initial_cs_deflection': [cs_deflection],
                            'thrust_nodes': thrust_nodes,
                            'initial_thrust': [thrust, thrust]}

        settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                                   'initial_velocity_direction': [-1., 0., 0.],
                                                   'max_iterations': 950,
                                                   'delta_curved': 1e-6,
                                                   'min_delta': tolerance,
                                                   'newmark_damp': 5e-3,
                                                   'gravity_on': True,
                                                   'gravity': 9.81,
                                                   'num_steps': n_tstep,
                                                   'dt': dt,
                                                   'initial_velocity': u_inf * 1}

        settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'on',
                                                   'initial_velocity_direction': [-1., 0., 0.],
                                                   'max_iterations': 950,
                                                   'delta_curved': 1e-6,
                                                   'min_delta': self.tolerance,
                                                   'newmark_damp': 5e-3,
                                                   'gravity_on': True,
                                                   'gravity': 9.81,
                                                   'num_steps': self.n_tstep,
                                                   'dt': self.dt}

        settings['StepLinearUVLM'] = {'dt': self.dt,
                                      'integr_order': 1,
                                      'remove_predictor': True,
                                      'use_sparse': True,
                                      'velocity_field_generator': 'GustVelocityField',
                                      'velocity_field_input': {'u_inf': u_inf,
                                                               'u_inf_direction': [1., 0., 0.],
                                                               'gust_shape': '1-cos',
                                                               'gust_length': 1.,
                                                               'gust_intensity': self.gust_intensity * u_inf,
                                                               'offset': 30.,
                                                               'span': self.span}}

        settings['StepUvlm'] = {'print_info': 'on',
                                'horseshoe': self.horseshoe,
                                'num_cores': 4,
                                'n_rollup': 1,
                                'convection_scheme': self.wake_type,
                                'rollup_dt': dt,
                                'rollup_aic_refresh': 1,
                                'rollup_tolerance': 1e-4,
                                'velocity_field_generator': 'GustVelocityField',
                                'velocity_field_input': {'u_inf': u_inf * 0,
                                                         'u_inf_direction': [1., 0, 0],
                                                         'gust_shape': '1-cos',
                                                         'gust_length': 5.,
                                                         'gust_intensity': gust_intensity * u_inf,
                                                         'offset': 15.0,
                                                         'span': self.span,
                                                         'relative_motion': True},
                                # 'velocity_field_generator': 'SteadyVelocityField',
                                # 'velocity_field_input': {'u_inf': u_inf*1,
                                #                             'u_inf_direction': [1., 0., 0.]},
                                'rho': rho,
                                'n_time_steps': n_tstep,
                                'dt': dt,
                                'gamma_dot_filtering': 3}

        settings['DynamicCoupled'] = {'print_info': 'on',
                                      # 'structural_substeps': 1,
                                      # 'dynamic_relaxation': 'on',
                                      # 'clean_up_previous_solution': 'on',
                                      'structural_solver': 'NonLinearDynamicCoupledStep',
                                      'structural_solver_settings': settings['NonLinearDynamicCoupledStep'],
                                      'aero_solver': 'StepUvlm',
                                      'aero_solver_settings': settings['StepUvlm'],
                                      'fsi_substeps': 200,
                                      'fsi_tolerance': fsi_tolerance,
                                      'relaxation_factor': relaxation_factor,
                                      'minimum_steps': 1,
                                      'relaxation_steps': 150,
                                      'final_relaxation_factor': 0.5,
                                      'n_time_steps': n_tstep,
                                      'dt': dt,
                                      'include_unsteady_force_contribution': 'off',
                                      'postprocessors': ['BeamLoads', 'StallCheck', 'BeamPlot', 'AerogridPlot'],
                                      'postprocessors_settings': {'BeamLoads': {'csv_output': 'off'},
                                                                  'StallCheck': {'output_degrees': True,
                                                                                 'stall_angles': {
                                                                                     '0': [-12 * np.pi / 180,
                                                                                           6 * np.pi / 180],
                                                                                     '1': [-12 * np.pi / 180,
                                                                                           6 * np.pi / 180],
                                                                                     '2': [-12 * np.pi / 180,
                                                                                           6 * np.pi / 180]}},
                                                                  'BeamPlot': {'include_rbm': 'on',
                                                                               'include_applied_forces': 'on'},
                                                                  'AerogridPlot': {
                                                                      'u_inf': u_inf,
                                                                      'include_rbm': 'on',
                                                                      'include_applied_forces': 'on',
                                                                      'minus_m_star': 0},
                                                                  #              'WriteVariablesTime': {
                                                                  #              #     'delimeter': ',',
                                                                  #              #     'structure_nodes': [0],
                                                                  #              #     'structure_variables': ['Z']
                                                                  #                 # settings['WriteVariablesTime'] = {'delimiter': ' ',
                                                                  #     'FoR_variables': ['GFoR_pos', 'GFoR_vel', 'GFoR_acc'],
                                                                  # 'FoR_number': [],
                                                                  # 'structure_variables': ['AFoR_steady_forces', 'AFoR_unsteady_forces','AFoR_position'],
                                                                  # 'structure_nodes': [0,-1],
                                                                  # 'aero_panels_variables': ['gamma', 'gamma_dot'],
                                                                  # 'aero_panels_isurf': [0,1,2],
                                                                  # 'aero_panels_im': [1,1,1],
                                                                  # 'aero_panels_in': [-2,-2,-2],
                                                                  # 'aero_nodes_variables': ['GFoR_steady_force', 'GFoR_unsteady_force'],
                                                                  # 'aero_nodes_isurf': [0,1,2],
                                                                  # 'aero_nodes_im': [1,1,1],
                                                                  # 'aero_nodes_in': [-2,-2,-2]
                                                                  #              }}}
                                                                  }}

        settings['Modal'] = {'print_info': True,
                             'use_undamped_modes': True,
                             'NumLambda': 30,
                             'rigid_body_modes': True,
                             'write_modes_vtk': 'on',
                             'print_matrices': 'on',
                             'save_data': 'on',
                             'continuous_eigenvalues': 'off',
                             'dt': dt,
                             'plot_eigenvalues': False}

        settings['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                       'linear_system_settings': {
                                           'beam_settings': {'modal_projection': False,
                                                             'inout_coords': 'nodes',
                                                             'discrete_time': True,
                                                             'newmark_damp': 0.5,
                                                             'discr_method': 'newmark',
                                                             'dt': dt,
                                                             'proj_modes': 'undamped',
                                                             'use_euler': 'off',
                                                             'num_modes': 40,
                                                             'print_info': 'on',
                                                             'gravity': 'on',
                                                             'remove_dofs': []},
                                           'aero_settings': {'dt': dt,
                                                             'integr_order': 2,
                                                             'density': rho,
                                                             'remove_predictor': False,
                                                             'use_sparse': True,
                                                             'rigid_body_motion': True,
                                                             'use_euler': False,
                                                             'remove_inputs': ['u_gust']},
                                           'rigid_body_motion': True}}

        settings['AsymptoticStability'] = {'sys_id': 'LinearAeroelastic',
                                    'print_info': 'on',
                                    'display_root_locus':'off',
                                    'frequency_cutoff': 0,
                                    'export_eigenvalues': 'on',
                                    'num_evals':100}

        settings['LinDynamicSim'] = {'dt': dt,
                                      'n_tsteps': self.n_tstep,
                                      'sys_id': 'LinearAeroelastic',
                                      'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                      'postprocessors_settings': {'AerogridPlot': {
                                          'u_inf': u_inf,
                                          'include_rbm': 'on',
                                          'include_applied_forces': 'on',
                                          'minus_m_star': 0},
                                          'BeamPlot': {'include_rbm': 'on',
                                                       'include_applied_forces': 'on'}}}

        settings['AerogridPlot'] = {'include_rbm': 'off',
                                    'include_applied_forces': 'on',
                                    'minus_m_star': 0,
                                    'u_inf': u_inf
                                    }
        settings['AeroForcesCalculator'] = {'write_text_file': 'off',
                                            'text_file_name': case_name + '_aeroforces.csv',
                                            'screen_output': 'on',
                                            'unsteady': 'off',
                                            'coefficients': True,
                                            'q_ref': 0.5 * rho * u_inf ** 2,
                                            'S_ref': self.planform_area()
                                            }
        settings['BeamPlot'] = {'include_rbm': 'on',
                                'include_applied_forces': 'on',
                                'include_FoR': 'on'}

        settings['BeamLoads'] = {'csv_output': 'off'}

        settings['SaveData'] = {'save_aero': 'on',
                                'save_structure': 'on',
                                'save_linear': 'off'}

        settings['StabilityDerivatives'] = {'u_inf': self.u_inf,
                                            'S_ref': 12.809,
                                            'b_ref': self.span,
                                            'c_ref': 0.719}

        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()
        self.settings = settings

        self.config = config

if __name__=='__main__':

    ws = HortenWing(M=4,
                    N=11,
                    Mstarfactor=5,
                    u_inf=28,
                    rho=1.225,
                    alpha_deg=4)
