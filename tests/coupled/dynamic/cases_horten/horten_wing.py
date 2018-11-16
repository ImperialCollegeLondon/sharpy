"""
Horten Wing Class Generator
N Goizueta Nov 18
"""

import numpy as np

class HortenWing():
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
                 rho = 1.225,
                 alpha_deg=0.,
                 beta_deg=0.,
                 cs_deflection=0.,
                 case_route='.',
                 case_name='horten'):

        # Discretisation
        self.M = M
        self.N = N
        self.Mstarfactor = Mstarfactor

        self.n_node_elem = 3
        self.n_elem_wing = 11
        self.n_elem_fuselage = 1
        self.n_surfaces = 4


        # Flight conditions
        self.u_inf = u_inf
        self.rho = rho
        self.alpha = alpha_deg*np.pi/180

        # Compute number of nodes
        n_node = 0
        n_node_wing = n_elem_wing*(n_node_elem-1)
        n_node_fuselage = n_elem_fuselage*n_node_elem
        n_node += 2*n_node_fuselage - 1 + 2*n_node_wing
        self.n_node = n_node

        # Compute number of elements
        self.n_elem = 2*(n_elem_wing + n_elem_fuselage)

        # Wing geometry
        self.span = 20.0   # [m]
        self.sweep_LE = 20*np.pi/180 # [rad] Leading Edge Sweep
        self.c_root = 1.0 # [m] Root chord
        self.taper_ratio = 0.25

        self.loc_cg = 0.45 # CG position wrt to LE (from sectional analysis)
        self.ea_offset_root = 0.13 # from Mardanpour
        ea_offset_tip = -1.644*0.0254
        main_ea_root = loc_cg-ea_offset_root
        main_ea_tip = loc_cg-ea_offset_tip

        # FUSELAGE GEOMETRY
        fuselage_width = 1.

        # WASH OUT
        washout_root = -0.2*np.pi/180
        washout_tip = -2*np.pi/180

    def update_mass_stiffness(self, sigma=1.):
        r"""
        The mass and stiffness matrices are computed. Both vary over the span of the wing, hence a
        dictionary is created that acts as a database of the different properties along the wing.

        The variation of the stiffness is cubic along the span:

            .. math:: \mathbf{K} = \mathbf{K}_0\bar{c}^3

        where :math:`\mathbf{K}_0 is the stiffness of the wing root section and :math:`\bar{c}` is
        the ratio between the local chord and the root chord.

        Args:
            sigma (float): stiffening factor

        Returns:

        """

        n_elem_fuselage = self.n_elem_fuselage
        n_elem_wing = self.n_elem_wing
        c_root = self.c_root
        taper_ratio = self.taper_ratio

        # Local chord to root chord initialisation
        c_bar_temp = np.linspace(c_root, taper_ratio*c_root, n_elem_wing)

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
        j_tip = 0.2e-2*4.448/9.81

        # Number of stiffnesses used
        n_stiffness = n_elem_wing + n_elem_fuselage

        # Initialise the stiffness database
        base_stiffness = np.ndarray((n_stiffness, 6, 6))

        # Wing root section stiffness properties
        stiffness_root = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])

        # Rigid fuselage
        sigma_fuselage = 1000
        base_stiffness[0, :, :] = sigma_fuselage*stiffness_root

        # Cubic variation of the stiffness along the span
        for stiff_entry in range(1, n_stiffness):
            base_stiffness[stiff_entry, :, :] = stiffness_root*c_bar_temp[stiff_entry-1]**3

        
