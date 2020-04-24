"""
Control surface deflector for linear systems
"""
import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.utils.algebra as algebra

@ss_interface.linear_system
class LinControlSurfaceDeflector(object):
    """
    Subsystem that deflects control surfaces for use with linear state space systems

    The current version supports only deflections. Future work will include standalone state-space systems to model
    physical actuators.

    """
    sys_id = 'LinControlSurfaceDeflector'

    def __init__(self):
        # Has the back bone structure for a future actuator model
        # As of now, it simply maps a deflection onto the aerodynamic grid by means of Kzeta_delta
        self.n_control_surfaces = 0
        self.Kzeta_delta = None
        self.Kdzeta_ddelta = None
        self.Kmom = None

        self.linuvlm = None
        self.aero = None
        self.structure = None

        self.tsaero0 = None
        self.tsstruct0 = None

        self.under_development = False

    def initialise(self, data, linuvlm):
        # Tasks:
        # 1 - Generic information
        #   * How many control surfaces (number of actual inputs)
        #   * How many uvlm surfaces
        self.n_control_surfaces = data.aero.n_control_surfaces

        self.linuvlm = linuvlm
        self.aero = data.aero
        self.structure = data.structure
        self.tsaero0 = data.aero.timestep_info[0]
        self.tsstruct0 = data.structure.timestep_info[0]

        #Testing....
        # Kzeta_d = self.generate(data, data.aero, data.structure)
        # self.Kzeta_delta = Kzeta_d

    def assemble(self):
        """
        Warnings:
            Under-development

        Will assemble the state-space for an actuator model
        Returns:

        """
        pass

    def generate(self, linuvlm=None, tsaero0=None, tsstruct0=None, aero=None, structure=None):
        """
        Generates a matrix mapping a linear control surface deflection onto the aerodynamic grid.

        The parsing of arguments is temporary since this state space element will include a full actuator model.

        The parsing of arguments is optional if the class has been previously initialised.

        Args:
            linuvlm:
            tsaero0:
            tsstruct0:
            aero:
            structure:

        Returns:

        """

        if self.aero is not None:
            aero = self.aero
            structure = self.structure
            linuvlm = self.linuvlm
            tsaero0 = self.tsaero0
            tsstruct0 = self.tsstruct0

        # Find the vertices corresponding to a control surface from beam coordinates to aerogrid
        aero_dict = aero.aero_dict
        n_surf = aero.timestep_info[0].n_surf
        n_control_surfaces = self.n_control_surfaces

        if self.under_development:
            try:
                import matplotlib.pyplot as plt  # Part of the testing process
            except ModuleNotFoundError:
                import warnings
                warnings.warn('Unable to import matplotlib, skipping plots')
                self.under_development = False

        Kdisp = np.zeros((3 * linuvlm.Kzeta, n_control_surfaces))
        Kvel = np.zeros((3 * linuvlm.Kzeta, n_control_surfaces))
        Kmom = np.zeros((3 * linuvlm.Kzeta, n_control_surfaces))
        zeta0 = np.concatenate([tsaero0.zeta[i_surf].reshape(-1, order='C') for i_surf in range(n_surf)])

        Cga = algebra.quat2rotation(tsstruct0.quat).T
        Cag = Cga.T

        # Initialise these parameters
        hinge_axis = None  # Will be set once per control surface to the hinge axis
        with_control_surface = False  # Will be set to true if the spanwise node contains a control surface

        for global_node in range(structure.num_node):

            # Retrieve elements and local nodes to which a single node is attached
            for i_elem in range(structure.num_elem):
                if global_node in structure.connectivities[i_elem, :]:
                    i_local_node = np.where(structure.connectivities[i_elem, :] == global_node)[0][0]

                    for_delta = structure.frame_of_reference_delta[i_elem, :, 0]

                    # CRV to transform from G to B frame
                    psi = tsstruct0.psi[i_elem, i_local_node]
                    Cab = algebra.crv2rotation(psi)
                    Cba = Cab.T
                    Cbg = np.dot(Cab.T, Cag)
                    Cgb = Cbg.T

                    # print(global_node)
                    if self.under_development:
                        print('\n\nNode -- ' + str(global_node))
                        print('i_elem = %g' % i_elem)
                    # Map onto aerodynamic coordinates. Some nodes may be part of two aerodynamic surfaces.
                    for structure2aero_node in aero.struct2aero_mapping[global_node]:
                        # Retrieve surface and span-wise coordinate
                        i_surf, i_node_span = structure2aero_node['i_surf'], structure2aero_node['i_n']

                        # Although a node may be part of 2 aerodynamic surfaces, we need to ensure that the current
                        # element for the given node is indeed part of that surface.
                        elems_in_surf = np.where(aero_dict['surface_distribution'] == i_surf)[0]
                        if i_elem not in elems_in_surf:
                            continue

                        if self.under_development:
                            print("i_surf = %g" % i_surf)
                            print("i_node_span = %g" % i_node_span)

                        # Surface panelling
                        M = aero.aero_dimensions[i_surf][0]
                        N = aero.aero_dimensions[i_surf][1]

                        K_zeta_start = 3 * sum(linuvlm.MS.KKzeta[:i_surf])
                        shape_zeta = (3, M + 1, N + 1)

                        if self.under_development:
                            print('Surface dimensions, M = %g, N = %g' % (M, N))

                        i_control_surface = aero_dict['control_surface'][i_elem, i_local_node]
                        if i_control_surface >= 0:
                            if self.under_development:
                                print('Control surface present, i_control_surface = %g' % i_control_surface)
                            if not with_control_surface:
                                i_start_of_cs = i_node_span.copy()
                                with_control_surface = True
                            if self.under_development:
                                print('Control surface span start index = %g' % i_start_of_cs)
                            control_surface_chord = aero_dict['control_surface_chord'][i_control_surface]
                            i_node_hinge = M - control_surface_chord
                            i_vertex_hinge = [K_zeta_start +
                                              np.ravel_multi_index((i_axis, i_node_hinge, i_node_span), shape_zeta)
                                              for i_axis in range(3)]
                            i_vertex_next_hinge = [K_zeta_start +
                                                   np.ravel_multi_index((i_axis, i_node_hinge, i_start_of_cs + 1),
                                                                        shape_zeta) for i_axis in range(3)]
                            zeta_hinge = zeta0[i_vertex_hinge]
                            zeta_next_hinge = zeta0[i_vertex_next_hinge]

                            if hinge_axis is None:
                                # Hinge axis not yet set for current control surface
                                # Hinge axis is in G frame
                                hinge_axis = zeta_next_hinge - zeta_hinge
                                hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)
                            for i_node_chord in range(M + 1):
                                i_vertex = [K_zeta_start +
                                            np.ravel_multi_index((i_axis, i_node_chord, i_node_span), shape_zeta)
                                            for i_axis in range(3)]

                                if i_node_chord > i_node_hinge:
                                    # Zeta in G frame
                                    zeta_node = zeta0[i_vertex]  # Gframe
                                    zeta_nodeA = Cag.dot(zeta_node)
                                    zeta_hingeA = Cag.dot(zeta_hinge)
                                    zeta_hingeB = Cbg.dot(zeta_hinge)
                                    zeta_nodeB = Cbg.dot(zeta_node)
                                    chord_vec = (zeta_node - zeta_hinge)
                                    if self.under_development:
                                        print('G Frame')
                                        print('Hinge axis = ' + str(hinge_axis))
                                        print('\tHinge = ' + str(zeta_hinge))
                                        print('\tNode = ' + str(zeta_node))
                                        print('A Frame')
                                        print('\tHinge = ' + str(zeta_hingeA))
                                        print('\tNode = ' + str(zeta_nodeA))
                                        print('B Frame')
                                        print('\tHinge axis = ' + str(Cbg.dot(hinge_axis)))
                                        print('\tHinge = ' + str(zeta_hingeB))
                                        print('\tNode = ' + str(zeta_nodeB))
                                        print('Chordwise Vector')
                                        print('GVec = ' + str(chord_vec/np.linalg.norm(chord_vec)))
                                        print('BVec = ' + str(Cbg.dot(chord_vec/np.linalg.norm(chord_vec))))
                                        # pass
                                    # Removing the += because cs where being added twice
                                    Kdisp[i_vertex, i_control_surface] = \
                                        Cgb.dot(der_R_arbitrary_axis_times_v(Cbg.dot(hinge_axis),
                                                                             0,
                                                                             -for_delta * Cbg.dot(chord_vec)))
                                    # Kdisp[i_vertex, i_control_surface] = \
                                    #     der_R_arbitrary_axis_times_v(hinge_axis, 0, chord_vec)

                                    # Flap velocity
                                    Kvel[i_vertex, i_control_surface] = -algebra.skew(chord_vec).dot(
                                        hinge_axis)

                                    # Flap hinge moment - future work
                                    # Kmom[i_vertex, i_control_surface] += algebra.skew(chord_vec)

                                    # Testing progress
                                    if self.under_development:
                                        plt.scatter(zeta_hingeB[1], zeta_hingeB[2], color='k')
                                        plt.scatter(zeta_nodeB[1], zeta_nodeB[2], color='b')
                                        # plt.scatter(zeta_hinge[1], zeta_hinge[2], color='k')
                                        # plt.scatter(zeta_node[1], zeta_node[2], color='b')

                                        # Testing out
                                        delta = 5*np.pi/180
                                        # zeta_newB = Cbg.dot(Kdisp[i_vertex, 1].dot(delta)) + zeta_nodeB
                                        zeta_newB = Cbg.dot(Kdisp[i_vertex, -1].dot(delta)) + zeta_nodeB
                                        plt.scatter(zeta_newB[1], zeta_newB[2], color='r')

                                        old_vector = zeta_nodeB - zeta_hingeB
                                        new_vector = zeta_newB - zeta_hingeB

                                        angle = np.arccos(new_vector.dot(old_vector) /
                                                          (np.linalg.norm(new_vector) * np.linalg.norm(old_vector)))
                                        print(angle)

                            if self.under_development:
                                plt.axis('equal')
                                plt.show()
                        else:
                            with_control_surface = False
                            hinge_axis = None  # Reset for next control surface

        self.Kzeta_delta = Kdisp
        self.Kdzeta_ddelta = Kvel
        # self.Kmom = Kmom
        return Kdisp, Kvel


def der_Cx_by_v(delta, v):
    sd = np.sin(delta)
    cd = np.cos(delta)
    v2 = v[1]
    v3 = v[2]
    return np.array([0, -v2 * sd - v3 * cd, v2 * cd - v3 * sd])

def der_Cy_by_v(delta, v):
    s = np.sin(delta)
    c = np.cos(delta)
    v1 = v[0]
    v3 = v[2]
    return np.array([-s*v1 + v*v3, 0, -c*v1 - s*v3])


def der_R_arbitrary_axis_times_v(u, theta, v):
    r"""
    Linearised rotation vector of the vector ``v`` by angle ``theta`` about an arbitrary axis ``u``.

    The rotation of a vector :math:`\mathbf{v}` about the axis :math:`\mathbf{u}` by an
    angle :math:`\boldsymbol{\theta}` can be expressed as

    .. math:: \mathbf{w} = \mathbf{R}(\mathbf{u}, \theta) \mathbf{v},

    where :math:`\mathbf{R}` is a :math:`\mathbb{R}^{3\times 3}` matrix.

    This expression can be linearised for it to be included in the linear solver as

    .. math:: \delta\mathbf{w} = \frac{\partial}{\partial\theta}\left(\mathbf{R}(\mathbf{u}, \theta_0)\right)\delta\theta

    The matrix :math:`\mathbf{R}` is

    .. math::

        \mathbf{R} =
        \begin{bmatrix}\cos \theta +u_{x}^{2}\left(1-\cos \theta \right) &
        u_{x}u_{y}\left(1-\cos \theta \right)-u_{z}\sin \theta &
        u_{x}u_{z}\left(1-\cos \theta \right)+u_{y}\sin \theta \\
        u_{y}u_{x}\left(1-\cos \theta \right)+u_{z}\sin \theta &
        \cos \theta +u_{y}^{2}\left(1-\cos \theta \right)&
        u_{y}u_{z}\left(1-\cos \theta \right)-u_{x}\sin \theta \\
        u_{z}u_{x}\left(1-\cos \theta \right)-u_{y}\sin \theta &
        u_{z}u_{y}\left(1-\cos \theta \right)+u_{x}\sin \theta &
        \cos \theta +u_{z}^{2}\left(1-\cos \theta \right)\end{bmatrix},

    and its linearised expression becomes

    .. math::

        \frac{\partial}{\partial\theta}\left(\mathbf{R}(\mathbf{u}, \theta_0)\right) =
        \begin{bmatrix}
        -\sin \theta +u_{x}^{2}\sin \theta \mathbf{v}_1 +
        u_{x}u_{y}\sin \theta-u_{z} \cos \theta \mathbf{v}_2 +
        u_{x}u_{z}\sin \theta +u_{y}\cos \theta \mathbf{v}_3 \\
        u_{y}u_{x}\sin \theta+u_{z}\cos \theta\mathbf{v}_1
        -\sin \theta +u_{y}^{2}\sin \theta\mathbf{v}_2 +
        u_{y}u_{z}\sin \theta-u_{x}\cos \theta\mathbf{v}_3 \\
        u_{z}u_{x}\sin \theta-u_{y}\cos \theta\mathbf{v}_1 +
        u_{z}u_{y}\sin \theta+u_{x}\cos \theta\mathbf{v}_2
        -\sin \theta +u_{z}^{2}\sin\theta\mathbf{v}_3\end{bmatrix}_{\theta=\theta_0}

    and is of dimension :math:`\mathbb{R}^{3\times 1}`.

    Args:
        u (numpy.ndarray): Arbitrary rotation axis
        theta (float): Rotation angle (radians)
        v (numpy.ndarray): Vector to rotate

    Returns:
        numpy.ndarray: Linearised rotation vector of dimensions :math:`\mathbb{R}^{3\times 1}`.
    """

    u = u / np.linalg.norm(u)
    c = np.cos(theta)
    s = np.sin(theta)

    ux, uy, uz = u
    v1, v2, v3 = v

    dR11 = -s + ux ** 2 * s
    dR12 = ux * uy * s - uz * c
    dR13 = ux * uz * s + uy * c

    dR21 = uy * ux * s + uz * c
    dR22 = -s + uy ** 2 * s
    dR23 = uy * uz * s - ux * c

    dR31 = uz * ux * s - uy * c
    dR32 = uz * uy * s + ux * c
    dR33 = -s + uz ** 2

    dRv = np.zeros((3, ))
    dRv[0] = dR11 * v1 + dR12 * v2 + dR13 * v3
    dRv[1] = dR21 * v1 + dR22 * v2 + dR23 * v3
    dRv[2] = dR31 * v1 + dR32 * v2 + dR33 * v3

    return dRv

