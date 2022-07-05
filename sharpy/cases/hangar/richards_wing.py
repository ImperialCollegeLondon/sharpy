"""
Simple Horten Wing as used by Richards. Baseline and simplified models

Richards, P. W., Yao, Y., Herd, R. A., Hodges, D. H., & Mardanpour, P. (2016). Effect of Inertial and Constitutive
Properties on Body-Freedom Flutter for Flying Wings. Journal of Aircraft. https://doi.org/10.2514/1.C033435
"""
import numpy as np
from sharpy.cases.hangar.horten_wing import HortenWing
import sharpy.utils.algebra as algebra


class Baseline(HortenWing):

    def set_properties(self):

        # Wing geometry
        self.span = 20.0  # [m]
        self.sweep_LE = 20 * np.pi / 180  # [rad] Leading Edge Sweep
        self.c_root = 1.0  # [m] Root chord - Richards
        self.taper_ratio = 0.25  # Richards
        self.thrust_nodes = [self.n_node_fuselage - 1,
                             self.n_node_fuselage + self.n_node_wing + 1]

        self.loc_cg = 0.45 # CG position wrt to LE (from sectional analysis)
        # EA is the reference in NATASHA - defined with respect to the midchord. SHARPy is wrt to LE and as a pct of
        # local chord
        self.main_ea_root = 0.33
        self.main_ea_tip = 0.33
        self.n_mass = 2 * self.n_elem_wing

        # FUSELAGE GEOMETRY
        self.fuselage_width = 1.65/2
        self.c_fuselage = self.c_root

        # WASH OUT
        self.washout_root = 0*np.pi/180
        self.washout_tip = -2 * np.pi / 180

        # Horseshoe wake
        self.horseshoe = False
        self.wake_type = 2
        self.dt_factor = 1
        self.dt = 1 / self.M / self.u_inf * self.dt_factor

        # Dynamics
        self.n_tstep = int(self.physical_time/self.dt)
        self.gust_intensity = 0.1

        # Numerics
        self.tolerance = 1e-12
        self.fsi_tolerance = 1e-10
        self.relaxation_factor = 0.2

    def update_mass_stiffness(self, sigma=1., sigma_mass=1., payload=0):
        """
        Sets the mass and stiffness properties of the default wing

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

        root_i_beam = IBeam()
        root_i_beam.build(c_root)
        root_i_beam.rotation_axes = np.array([0, self.main_ea_root-0.25, 0])

        root_airfoil = Airfoil()
        root_airfoil.build(c_root)
        root_i_beam.rotation_axes = np.array([0, self.main_ea_root-0.25, 0])

        mu_0 = root_i_beam.mass + root_airfoil.mass
        j_xx = root_i_beam.ixx + root_airfoil.ixx
        j_yy = root_i_beam.iyy + root_airfoil.iyy
        j_zz = root_i_beam.izz + root_airfoil.izz

        # Number of stiffnesses used
        n_stiffness = self.n_stiffness

        # Initialise the stiffness database
        base_stiffness = self.base_stiffness

        stiffness_root = sigma * np.diag([ea, ga, ga, gj, eiy, eiz])
        stiffness_tip = taper_ratio ** 2 * stiffness_root

        # Assume a linear variation in the stiffness. Richards et al. use VABS on the linearly tapered wing to find the
        # spanwise properties
        alpha = np.linspace(0, 1, self.n_elem_wing)
        for i_elem in range(0, self.n_elem_wing):
            base_stiffness[i_elem + 1, :, :] = stiffness_root*(1-alpha[i_elem]**2) + stiffness_tip*alpha[i_elem]**2
        base_stiffness[0] = base_stiffness[1]

        # Mass variation along the span
        # Right wing centre of mass - wrt to 0.25c
        cm = (root_airfoil.centre_mass * root_airfoil.mass + root_i_beam.centre_mass * root_i_beam.mass) \
             / np.sum(root_airfoil.mass + root_i_beam.mass)
        cg = np.array([0, -(cm[0] + 0.25 * self.c_root - self.main_ea_root), 0]) * 1
        n_mass = self.n_mass
        # sigma_mass = 1.25

        # Initialise database
        base_mass = self.base_mass

        mass_root_right = np.diag([mu_0, mu_0, mu_0, j_xx, j_yy, j_zz]) * sigma_mass
        mass_root_right[:3, -3:] = -algebra.skew(cg) * mu_0
        mass_root_right[-3:, :3] = algebra.skew(cg) * mu_0

        mass_root_left = np.diag([mu_0, mu_0, mu_0, j_xx, j_yy, j_zz]) * sigma_mass
        mass_root_left[:3, -3:] = -algebra.skew(-cg) * mu_0
        mass_root_left[-3:, :3] = algebra.skew(-cg) * mu_0

        mass_tip_right = taper_ratio * mass_root_right
        mass_tip_left = taper_ratio * mass_root_left

        ixx_dummy = []
        iyy_dummy = []
        izz_dummy = []

        for i_elem in range(self.n_elem_wing):
            # Create full cross section
            c_bar = self.c_root * ((1-alpha[i_elem]) + self.taper_ratio * alpha[i_elem])
            x_section = WingCrossSection(c_bar)
            # print(i_elem)
            # print('Section Mass: %.2f ' %x_section.mass)
            # print('Linear Mass: %.2f' % (mu_0 * (1-alpha[i_elem]) + mu_0 * self.taper_ratio * alpha[i_elem]))
            # print('Section Ixx: %.4f' % x_section.ixx)
            # print('Section Iyy: %.4f' % x_section.iyy)
            # print('Section Izz: %.4f' % x_section.izz)
            # print('Linear Ixx: %.2f' % (j_xx * (1-alpha[i_elem]) + j_xx * self.taper_ratio * alpha[i_elem]))
            # base_mass[i_elem, :, :] = mass_root_right*(1-alpha[i_elem]) + mass_tip_right*alpha[i_elem]
            # base_mass[i_elem + self.n_elem_wing + self.n_elem_fuselage - 1] = mass_root_left*(1-alpha[i_elem]) + mass_tip_left*alpha[i_elem]

            base_mass[i_elem, :, :] = np.diag([x_section.mass, x_section.mass, x_section.mass,
                                               x_section.ixx, x_section.iyy, x_section.izz])
            cg = np.array([0, -(x_section.centre_mass[0] + (0.25 - self.main_ea_root) * c_bar / self.c_root), 0]) * 1
            base_mass[i_elem, :3, -3:] = -algebra.skew(cg) * x_section.mass
            base_mass[i_elem, -3:, :3] = algebra.skew(cg) * x_section.mass

            base_mass[i_elem + self.n_elem_wing + self.n_elem_fuselage - 1, :, :] = np.diag([x_section.mass, x_section.mass, x_section.mass,
                                               x_section.ixx, x_section.iyy, x_section.izz])
            cg = np.array([0, -(x_section.centre_mass[0] + (0.25 - self.main_ea_root) * c_bar / self.c_root), 0]) * 1
            base_mass[i_elem + self.n_elem_wing + self.n_elem_fuselage - 1, :3, -3:] = -algebra.skew(-cg) * x_section.mass
            base_mass[i_elem + self.n_elem_wing + self.n_elem_fuselage - 1, -3:, :3] = algebra.skew(-cg) * x_section.mass

            ixx_dummy.append(x_section.ixx)
            iyy_dummy.append(x_section.iyy)
            izz_dummy.append(x_section.izz)

            # for item in x_section.items:
            #     plt.plot(item.y, item.z)
            # plt.scatter(x_section.centre_mass[0], x_section.centre_mass[1])
            # plt.show()
            # print(x_section.centre_mass)
            # print(cg)

        # plt.plot(range(self.n_elem_wing), ixx_dummy)
        # plt.plot(range(self.n_elem_wing), iyy_dummy)
        # plt.plot(range(self.n_elem_wing), izz_dummy)
        # plt.show()
        # Lumped mass initialisation
        lumped_mass_nodes = self.lumped_mass_nodes
        lumped_mass = self.lumped_mass
        lumped_mass_inertia = self.lumped_mass_inertia
        lumped_mass_position = self.lumped_mass_position

        # Lumped masses nodal position
        # 0 - Right engine
        # 1 - Left engine
        # 2 - Fuselage
        lumped_mass_nodes[0] = 2
        lumped_mass_nodes[1] = n_node_fuselage + n_node_wing + 1
        lumped_mass_nodes[2] = 0

        # Lumped mass value from Richards 2013
        lumped_mass[0:2] = 51.445 / 9.81
        lumped_mass[2] = 150 / 9.81 + payload
        # lumped_mass_position[2] = [0, 0, -10.]

        # Lumped mass inertia
        lumped_mass_inertia[0, :, :] = np.diag([0.29547, 0.29322, 0.29547])
        lumped_mass_inertia[1, :, :] = np.diag([0.29547, 0.29322, 0.29547])
        lumped_mass_inertia[2, :, :] = np.diag([0.5, 1, 1]) * lumped_mass[2]

        # Define class attributes
        self.lumped_mass = lumped_mass * 1
        self.lumped_mass_nodes = lumped_mass_nodes * 1
        self.lumped_mass_inertia = lumped_mass_inertia * 1
        self.lumped_mass_position = lumped_mass_position * 1

        self.base_stiffness = base_stiffness
        self.base_mass = base_mass


class CrossSection(object):
    def __init__(self):
        self.rho = 2770

        self.rotation_axes = np.array([0, 0.33-0.25, 0])

        self.y = np.ndarray((2,))
        self.z = np.ndarray((2,))
        self.t = np.ndarray((2,))

    @property
    def mass(self):
        """
        Mass of the I beam per unit length
        """
        return np.sum(self.t * self.elem_length) * self.rho

    @property
    def ixx(self):
        ixx_ = np.sum(self.elem_length * self.t * self.rho * (self.elem_cm_y ** 2 + self.elem_cm_z ** 2))
        return ixx_ + self.mass * (self.centre_mass[0] - self.rotation_axes[1]) ** 2

    @property
    def elem_length(self):
        elem_length = np.sqrt(np.diff(self.y) ** 2 + np.diff(self.z) ** 2)
        return elem_length

    @property
    def elem_cm_y(self):
        elem_cm_y_ = np.ndarray((self.n_elem, ))
        elem_cm_y_[:] = 0.5 * (self.y[:-1] + self.y[1:])
        return elem_cm_y_

    @property
    def elem_cm_z(self):
        elem_cm_z_ = np.ndarray((self.n_elem, ))
        elem_cm_z_[:] = 0.5 * (self.z[:-1] + self.z[1:])
        return elem_cm_z_

    @property
    def centre_mass(self):
        y_cm = np.sum(self.elem_cm_y * self.elem_length) / np.sum(self.elem_length)
        z_cm = np.sum(self.elem_cm_z * self.elem_length) / np.sum(self.elem_length)

        return np.array([y_cm, z_cm])

    @property
    def iyy(self):
        x_dom = np.linspace(-0.5, 0.5, 100)
        x_cg = 0.5 * (x_dom[:-1].copy() + x_dom[1:].copy())
        dx = np.diff(x_dom)[0]
        iyy_ = 0
        for elem in range(len(self.elem_length)):
            z_cg = np.ones_like(x_cg) * self.elem_cm_z[elem]
            iyy_ += np.sum(self.elem_length[elem] * self.t[elem] * dx * self.rho * (x_cg ** 2 + z_cg ** 2))
        return iyy_ #np.sum(self.elem_length * self.t * self.rho * 1 * self.elem_cm_z ** 2)

    @property
    def izz(self):
        x_dom = np.linspace(-0.5, 0.5, 100)
        x_cg = 0.5 * (x_dom[:-1].copy() + x_dom[1:].copy())
        dx = np.diff(x_dom)[0]
        iyy_ = 0
        izz_ = 0
        for elem in range(len(self.elem_length)):
            y_cg = np.ones_like(x_cg) * self.elem_cm_y[elem]
            izz_ += np.sum(self.elem_length[elem] * self.t[elem] * dx * self.rho * (x_cg ** 2 + y_cg ** 2))
        return izz_ #np.sum(self.elem_length * self.t * self.rho * 1 * self.elem_cm_y ** 2)

    @property
    def n_node(self):
        return self.y.shape[0]

    @property
    def n_elem(self):
        return self.n_node - 1

    def build(self, y, z, t):
        self.y = y
        self.z = z
        self.t = t


class IBeam(CrossSection):

    def build(self, c_root):
        t_skin = 0.127e-2
        t_c = 0.12
        w_I = 10e-2 * c_root  # Width of the Ibeam

        self.rho = 2770

        self.y = np.ndarray((self.n_node, ))
        self.z = np.ndarray((self.n_node, ))
        self.t = np.ndarray((self.n_node, ))

        z_max = t_c * c_root
        y = np.array([-w_I/2, w_I/2, 0, 0, -w_I/2, w_I/2])
        z = np.array([z_max/2, z_max/2, z_max/2, -z_max/2, -z_max/2, -z_max/2])
        t = np.array([t_skin, 0, t_skin, 0, t_skin])

        self.y = y
        self.z = z
        self.t = t


class Airfoil(CrossSection):

    def build(self, c_root):
        t_c = 0.12
        t_skin = 0.127e-2 * 1.5

        y_dom = np.linspace(0, c_root, 100)
        y = np.concatenate((y_dom, y_dom[:-1][::-1]))
        z_dom = 5 * t_c * (0.2969 * np.sqrt(y_dom/c_root) -
                                              0.1260 * y_dom/c_root -
                                              0.3516 * (y_dom/c_root) ** 2 +
                                              0.2843 * (y_dom/c_root) ** 3 -
                                              0.1015 * (y_dom/c_root) ** 4) * c_root
        z = np.concatenate((z_dom, -z_dom[:-1][::-1]))

        self.y = y - 0.25 * c_root
        self.z = z
        self.t = t_skin * np.ones(self.n_elem)


class WingCrossSection:

    def __init__(self, chord):
        self.chord = chord

        self.items = list()

        self.items.append(Airfoil())
        self.items.append(IBeam())

        for item in self.items:
            item.build(chord)

    @property
    def mass(self):
        return np.sum([item.mass for item in self.items])

    @property
    def centre_mass(self):
        y = np.sum([item.mass * item.centre_mass[0] for item in self.items]) / self.mass
        z = np.sum([item.mass * item.centre_mass[1] for item in self.items]) / self.mass
        return np.array([y, z])

    @property
    def ixx(self):
        return np.sum([item.ixx for item in self.items])

    @property
    def iyy(self):
        return np.sum([item.iyy for item in self.items])

    @property
    def izz(self):
        return np.sum([item.izz for item in self.items])


if __name__ == '__main__':

    ws = Baseline(M=4,
                  N=11,
                  Mstarfactor=5,
                  u_inf=28,
                  rho=1.225,
                  alpha_deg=4)

    # ws.clean_test_files()
    ws.update_mass_stiffness()
    ws.update_fem_prop()
    # ws.generate_fem_file()
    ws.update_aero_properties()
    # ws.generate_aero_file()
    # ws.set_default_config_dict()
