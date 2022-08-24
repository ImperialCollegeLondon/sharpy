"""
Control surface deflector for linear systems
"""
import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.utils.algebra as algebra
import sharpy.linear.src.libsparse as libsp
import scipy.sparse as sp
import sharpy.linear.src.libss as libss


@ss_interface.linear_system
class LinControlSurfaceDeflector(object):
    """
    Subsystem that deflects control surfaces for use with linear state space systems.

    Note:
        The control surface sign convention is different to the convention in the nonlinear solver. See
        https://www.github.com/imperialcollegelondon/sharpy/issues/193 for more details.

    In the linear implementation, the control surface deflection sign convention follows the :math:`x_B` vector,
    thus, in order to have symmetric control surface deflections, additional inputs may be required compared to the
    implementation in the nonlinear solver.

    The current version supports only deflections. Future work will include standalone state-space systems to model
    physical actuators.

    """
    sys_id = 'LinControlSurfaceDeflector'

    def __init__(self):
        # Has the back bone structure for a future actuator model
        # As of now, it simply maps a deflection onto the aerodynamic grid by means of Kzeta_delta
        self.n_control_surfaces = 0
        self.Kzeta_delta = None  # type: np.ndarray
        self.Kdzeta_ddelta = None  # type: np.ndarray

        self.linuvlm = None  # type: sharpy.linear.src.linuvlm.Dynamic
        self.aero = None  # type: sharpy.aero.models.aerogrid.Aerogrid
        self.structure = None  # type: sharpy.structure.models.beam.Beam

        self.tsaero0 = None
        self.tsstruct0 = None

        self.gain_cs = None  # type: np.ndarray # input gain to the UVLM

        self.print_info = False  # used for debugging purposes

    def initialise(self, data, linuvlm):
        self.n_control_surfaces = data.aero.n_control_surfaces

        self.linuvlm = linuvlm
        self.aero = data.aero
        self.structure = data.structure
        self.tsaero0 = data.aero.timestep_info[0]
        self.tsstruct0 = data.structure.timestep_info[0]

    def generate(self):
        """
        Generates a matrix mapping a linear control surface deflection onto the aerodynamic grid.

        This generates two matrices:

            * `Kzeta_delta` maps the deflection angle onto displacements. It has as many columns as independent control
              surfaces.

            * `Kdzeta_ddelta` maps the deflection rate onto grid velocities. Again, it has as many columns as
              independent control surfaces.

        Returns:
            tuple: Tuple containing `Kzeta_delta` and `Kdzeta_ddelta`.

        """
        # For future development
        # In hindsight, building this matrix iterating through structural node was a big mistake that
        # has led to very messy code. Would rework by element and in B frame

        aero = self.aero
        structure = self.structure
        linuvlm = self.linuvlm
        tsaero0 = self.tsaero0
        tsstruct0 = self.tsstruct0

        # Find the vertices corresponding to a control surface from beam coordinates to aerogrid
        data_dict = aero.data_dict
        n_surf = tsaero0.n_surf
        n_control_surfaces = self.n_control_surfaces

        Kdisp = np.zeros((3 * linuvlm.Kzeta, n_control_surfaces))
        Kvel = np.zeros((3 * linuvlm.Kzeta, n_control_surfaces))
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

                    for_delta = structure.frame_of_reference_delta[i_elem, i_local_node, :]

                    # CRV to transform from G to B frame
                    psi = tsstruct0.psi[i_elem, i_local_node]
                    Cab = algebra.crv2rotation(psi)
                    Cba = Cab.T
                    Cbg = np.dot(Cab.T, Cag)
                    Cgb = Cbg.T

                    # Map onto aerodynamic coordinates. Some nodes may be part of two aerodynamic surfaces.
                    for structure2aero_node in aero.struct2aero_mapping[global_node]:
                        # Retrieve surface and span-wise coordinate
                        i_surf, i_node_span = structure2aero_node['i_surf'], structure2aero_node['i_n']

                        # Although a node may be part of 2 aerodynamic surfaces, we need to ensure that the current
                        # element for the given node is indeed part of that surface.
                        elems_in_surf = np.where(data_dict['surface_distribution'] == i_surf)[0]
                        if i_elem not in elems_in_surf:
                            continue

                        # Surface panelling
                        M = aero.dimensions[i_surf][0]
                        N = aero.dimensions[i_surf][1]

                        K_zeta_start = 3 * sum(linuvlm.MS.KKzeta[:i_surf])
                        shape_zeta = (3, M + 1, N + 1)

                        i_control_surface = data_dict['control_surface'][i_elem, i_local_node]
                        if i_control_surface >= 0:
                            if not with_control_surface:
                                i_start_of_cs = i_node_span.copy()
                                with_control_surface = True

                            control_surface_chord = data_dict['control_surface_chord'][i_control_surface]

                            try:
                                control_surface_hinge_coord = \
                                    data_dict['control_surface_hinge_coord'][i_control_surface] * \
                                    data_dict['chord'][i_elem, i_local_node]
                            except KeyError:
                                control_surface_hinge_coord = None

                            i_node_hinge = M - control_surface_chord
                            i_vertex_hinge = [K_zeta_start +
                                              np.ravel_multi_index((i_axis, i_node_hinge, i_node_span), shape_zeta)
                                              for i_axis in range(3)]
                            i_vertex_next_hinge = [K_zeta_start +
                                                   np.ravel_multi_index((i_axis, i_node_hinge, i_start_of_cs + 1),
                                                                        shape_zeta) for i_axis in range(3)]

                            if control_surface_hinge_coord is not None and M == control_surface_chord:  # fully articulated control surface
                                zeta_hinge = Cgb.dot(Cba.dot(tsstruct0.pos[global_node]) + for_delta * np.array([0, control_surface_hinge_coord, 0]))
                                zeta_next_hinge = Cgb.dot(Cbg.dot(zeta_hinge) + np.array([1, 0, 0]))  # parallel to the x_b vector
                            else:
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

                                if i_node_chord >= i_node_hinge:
                                    # Zeta in G frame
                                    zeta_node = zeta0[i_vertex]  # Gframe
                                    chord_vec = (zeta_node - zeta_hinge)

                                    if self.print_info:
                                        print(f'i_node = {global_node}')
                                        print(f'i_node_chord = {i_node_chord}')
                                        print(f'zeta_node = {zeta_node}')
                                        print(f'chord_vec = {chord_vec}')
                                        print(f'zeta_hinge = {zeta_hinge}')
                                        print(f'zeta_next_hinge = {zeta_next_hinge}')
                                        print(f'hinge_axis = {hinge_axis}')

                                    # Flap displacement
                                    Kdisp[i_vertex, i_control_surface] = \
                                        der_R_arbitrary_axis_times_v(hinge_axis, 0, chord_vec)

                                    # Flap velocity
                                    Kvel[i_vertex, i_control_surface] = -algebra.skew(chord_vec).dot(
                                        hinge_axis)

                                    if self.print_info:
                                        print('Matrix entries:')
                                        print('Kdisp:')
                                        print(Kdisp[i_vertex, i_control_surface])
                                        print('Kvel:')
                                        print(Kvel[i_vertex, i_control_surface])

                        else:
                            with_control_surface = False
                            hinge_axis = None  # Reset for next control surface

        self.Kzeta_delta = Kdisp
        self.Kdzeta_ddelta = Kvel

        return Kdisp, Kvel

    def apply(self, ss):
        """
        Applies the control surface deflection to the UVLM state space

        Args:
            ss (libss.StateSpace): UVLM state space

        Returns:
            libss.StateSpace: UVLM state-space with control surfaces and control surface deflection rate as inputs
        """

        Kzeta_delta, Kdzeta_ddelta = self.generate()
        n_zeta, n_ctrl_sfc = Kzeta_delta.shape

        if type(ss.A) is libsp.csc_matrix:
            gain_cs = sp.eye(ss.inputs, ss.inputs + 2 * self.n_control_surfaces,
                             format='lil')
            gain_cs[:n_zeta, ss.inputs: ss.inputs + n_ctrl_sfc] = Kzeta_delta
            gain_cs[n_zeta: 2*n_zeta, ss.inputs + n_ctrl_sfc: ss.inputs + 2 * n_ctrl_sfc] = Kdzeta_ddelta
            gain_cs = libsp.csc_matrix(gain_cs)
        else:
            gain_cs = np.eye(ss.inputs, ss.inputs + 2 * self.n_control_surfaces)
            gain_cs[:n_zeta, ss.inputs: ss.inputs + n_ctrl_sfc] = Kzeta_delta
            gain_cs[n_zeta: 2*n_zeta, ss.inputs + n_ctrl_sfc: ss.inputs + 2 * n_ctrl_sfc] = Kdzeta_ddelta

        control_surface_gain = libss.Gain(gain_cs)
        in_vars = ss.input_variables.copy()
        in_vars.append('control_surface_deflection', size=n_ctrl_sfc)
        in_vars.append('dot_control_surface_deflection', size=n_ctrl_sfc)
        control_surface_gain.input_variables = in_vars
        control_surface_gain.output_variables = ss_interface.LinearVector.transform(ss.input_variables,
                                                                                    to_type=ss_interface.OutputVariable)

        self.gain_cs = control_surface_gain

        return ss
    # def generator():
    # future feature idea: instead of defining the inputs for the time domain simulations as the whole input vector
    # etc, we could add a generate() method to these systems that can be called from the LinDynamicSim to apply
    # the gust and generate the correct input vector.


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

