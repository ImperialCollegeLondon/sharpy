import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.linear.src.libss as libss
import scipy.signal as scsig
import sharpy.utils.settings as settings

dict_of_linear_gusts = {}


def linear_gust(arg):
    global dict_of_linear_gusts
    try:
        arg.gust_id
    except AttributeError:
        raise AttributeError('Class defined as gust has no gust_id attribute')
    dict_of_linear_gusts[arg.gust_id] = arg
    return arg


def gust_from_string(gust_name):
    return dict_of_linear_gusts[gust_name]()


class LinearGust:
    # Base class from which to develop the desired gusts
    settings_types = {}
    settings_default = {}
    settings_description = {}

    def __init__(self):
        self.aero = None  #: aerogrid
        self.linuvlm = None # :linuvlm
        self.tsaero0 = None  # timestep info at the linearisation point

        self.state_to_uext = None
        self.gust_ss = None # :libss.ss containing the gust state-space system

        self.settings = None

    def initialise(self, aero, linuvlm, tsaero0, custom_settings=None):
        self.aero = aero
        self.linuvlm = linuvlm
        self.tsaero0 = tsaero0

        if custom_settings is not None:
            self.settings = custom_settings
            settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def get_x_max(self):
        max_chord_surf = []
        min_chord_surf = []
        for i_surf in range(len(self.tsaero0.zeta)):
            max_chord_surf.append(np.max(self.tsaero0.zeta[i_surf][0, :, :]))
            min_chord_surf.append(np.min(self.tsaero0.zeta[i_surf][0, :, :]))
        return min(min_chord_surf), max(max_chord_surf)

    def assemble(self):
        pass

    def apply(self, ssuvlm):
        r"""
        Couples in series the gust system assembled by the LinearGust with the Linear UVLM.

        It generates an augmented gust system which feeds through the other UVLM inputs
        (grid displacements and velocities). The resulting augmented gust state space takes the form
        of:

        .. math::
            \boldsymbol{B}_{aug} = \begin{pmatrix} \boldsymbol{0}_{K_\zeta} & \boldsymbol{0}_{K_\zeta}
            & \boldsymbol{B}_{gust} \end{pmatrix}

        .. math::
            \boldsymbol{C}_{aug} = \begin{pmatrix} \boldsymbol{0}_{K_\zeta} \\ \boldsymbol{0}_{K_\zeta} \\
            \boldsymbol{C}_{gust} \end{pmatrix}

        .. math::
            \boldsymbol{D}_{aug} = \begin{pmatrix} \boldsymbol{I}_{K_\zeta} \\ \ &  \boldsymbol{I}_{K_\zeta} \\
            \boldsymbol{0}_{K_\zeta} \end{pmatrix}

        where :math:`K_\zeta` is 3 times the number of vertices in the UVLM lattice and the size of the input
        sets displacements and velocities, as well as external velocity inputs.

        Therefore, the inputs to the resulting coupled system become

        .. math:: \boldsymbol{u} = \begin{bmatrix} \boldsymbol{\zeta} & \boldsymbol{\dot{\zeta}} & \boldsymbol{u}_{gust}

        where the size of :math:`\boldsymbol{u}_{gust} will depend on the chosen gust assembly scheme.

        Args:
            ssuvlm (libss.ss): State space object of the linear UVLM.

        Returns:
            libss.ss: Coupled gust system with Linear UVLM.
        """
        ssgust = self.assemble()
        #
        # Feed through UVLM inputs
        b_aug = np.zeros((ssgust.states, ssuvlm.inputs - ssgust.outputs + ssgust.inputs))
        c_aug = np.zeros((ssuvlm.inputs, ssgust.states))
        d_aug = np.zeros((ssuvlm.inputs, b_aug.shape[1]))
        b_aug[:, -ssgust.inputs:] = ssgust.B
        c_aug[-ssgust.outputs:, :] = ssgust.C
        d_aug[:-ssgust.outputs, :-ssgust.inputs] = np.eye(ssuvlm.inputs - ssgust.outputs)

        self.gust_ss = libss.ss(ssgust.A, b_aug, c_aug, d_aug, dt=ssgust.dt)
        input_variables = ssuvlm.input_variables.copy()
        input_variables.remove('u_gust')
        self.gust_ss.input_variables = \
            ss_interface.LinearVector.merge(input_variables, ssgust.input_variables)
        self.gust_ss.state_variables = ssgust.state_variables.copy()
        self.gust_ss.output_variables = ss_interface.LinearVector.transform(ssuvlm.input_variables,
                                                                            to_type=ss_interface.OutputVariable)
        # np.testing.assert_array_equal(b_aug, self.gust_ss.B)
        # np.testing.assert_array_equal(c_aug, self.gust_ss.C)
        # np.testing.assert_array_equal(d_aug, self.gust_ss.D)
        ss = libss.series(self.gust_ss, ssuvlm)

        return ss


@linear_gust
class LeadingEdge(LinearGust):
    """
    Reduces the gust input to a single input for the vertical component of the gust at the leading edge.

    This is vertical velocity is then convected downstream with the free stream velocity. The gust is uniform in span.

    """
    gust_id = 'LeadingEdge'

    def assemble(self):
        """
        Assembles the gust state space system, creating the (A, B, C and D) matrices that convect the single gust input
        at the leading edge downstream and uniformly across the span
        """
        return self.leading_edge()

        # Kzeta = self.linuvlm.Kzeta
        # M = self.linuvlm.MS.MM[0]
        #
        # # Create state-space to convect gust downstream
        # A_gust = np.zeros((M+1, M+1))
        # A_gust[1:, :-1] = np.eye(M, M)
        #
        # B_gust = np.zeros((M+1, 6 * Kzeta + 1))
        # B_gust[0, 6 * Kzeta] = 1
        #
        # C_gust = np.zeros((9 * Kzeta, M+1))
        #
        # D_gust = np.zeros((9 * Kzeta, 6 * Kzeta + 1))
        #
        # Kout = np.zeros((3 * Kzeta, M+1))
        #
        # for i_surf in range(self.aero.n_surf):
        #
        #     M_surf, N_surf = self.aero.aero_dimensions[i_surf]
        #     Kzeta_start = 3 * sum(self.linuvlm.MS.KKzeta[:i_surf])  # number of coordinates up to current surface
        #     shape_zeta = (3, M_surf + 1, N_surf + 1)
        #
        #     for i_node_span in range(N_surf + 1):
        #         for i_node_chord in range(M_surf + 1):
        #             i_vertex = [Kzeta_start + np.ravel_multi_index((i_axis, i_node_chord, i_node_span),
        #                                                            shape_zeta) for i_axis in range(3)]
        #             Kout[i_vertex, i_node_chord] = np.array([0, 0, 1])
        #
        # C_gust[-3 * Kzeta:] = Kout
        # D_gust[:6 * Kzeta, :6 * Kzeta] = np.eye(6 * Kzeta)
        #
        # self.gust_ss = libss.ss(A_gust, B_gust, C_gust, D_gust, dt=self.linuvlm.SS.dt)
        # gust_input_variables = self.linuvlm.SS.input_variables.copy()
        # gust_input_variables.modify('u_gust', size=1)
        # self.gust_ss.input_variables = gust_input_variables
        #
        # self.gust_ss.state_variables = ss_interface.LinearVector(
        #     [ss_interface.StateVariable('gust', size=self.gust_ss.states, index=0)])
        #
        # self.gust_ss.output_variables = ss_interface.LinearVector.transform(self.linuvlm.SS.input_variables,
        #                                                                     to_type=ss_interface.OutputVariable)
    def leading_edge(self):

        n_inputs = 1
        # Number of inputs: defined at the leading edge
        # probably need a tuple of input and spanwise location.
        # Question arises for swept wings: is the input defined at the same x reference or at the leading edge?
        # Bi-linear interpolation probably best (nonlinear?) one axis first then the other?
        Kzeta = self.linuvlm.Kzeta

        # Convection system: needed for as many inputs (since it carries their time histories)
        x_min, x_max = self.get_x_max()  # G frame
        # TODO: project onto free stream vector
        # TODO: apply to time scaled systems
        N = int(np.ceil((x_max - x_min) / self.linuvlm.SS.dt))
        x_domain = np.linspace(x_min, x_max, N)
        # State Equation
        # for each input...
        a_i = np.zeros((N, N))
        a_i[1:, :-1] = np.eye(N-1)
        b_i = np.zeros((N, 1))
        b_gust = np.zeros((N, 6 * Kzeta + 1))
        b_gust[0, -1] = 1
        b_i[0, 0] = 1

        # Output Equation
        # check order of input into the UVLM. is it ux1 uy1 uz1 ux2 uy2 uz2...?
        c_gust = np.zeros((9 * Kzeta, N))
        d_gust = np.zeros((9 * Kzeta, 6 * Kzeta + 1))

        c_i = np.zeros((3 * Kzeta, N))
        for i_surf in range(self.aero.n_surf):

            M_surf, N_surf = self.aero.aero_dimensions[i_surf]
            Kzeta_start = 3 * sum(self.linuvlm.MS.KKzeta[:i_surf])  # number of coordinates up to current surface
            shape_zeta = (3, M_surf + 1, N_surf + 1)

            for i_node_span in range(N_surf + 1):
                for i_node_chord in range(M_surf + 1):
                    i_vertex = [Kzeta_start + np.ravel_multi_index((i_axis, i_node_chord, i_node_span),
                                                                   shape_zeta) for i_axis in range(3)]
                    x_vertex = self.tsaero0.zeta[i_surf][0, i_node_chord, i_node_span]
                    interpolation_weights, column_indices = chordwise_interpolation(x_vertex, x_domain)
                    c_i[i_vertex, column_indices[0]] = np.array([0, 0, interpolation_weights[0]])
                    c_i[i_vertex, column_indices[1]] = np.array([0, 0, interpolation_weights[1]])
                    # if i_node_span == 0:
                    #     print('x vertex', x_vertex)
                    #     print(column_indices)
                    #     print(interpolation_weights)

        # c_gust[-3 * Kzeta:] = c_i
        # d_gust[:6 * Kzeta, :6 * Kzeta] = np.eye(6 * Kzeta)
        #
        # self.gust_ss = libss.ss(a_i, b_gust, c_gust, d_gust, dt=self.linuvlm.SS.dt)
        # gust_input_variables = self.linuvlm.SS.input_variables.copy()
        # gust_input_variables.modify('u_gust', size=1)
        # self.gust_ss.input_variables = gust_input_variables
        #
        # self.gust_ss.state_variables = ss_interface.LinearVector(
        #     [ss_interface.StateVariable('gust', size=self.gust_ss.states, index=0)])
        #
        # self.gust_ss.output_variables = ss_interface.LinearVector.transform(self.linuvlm.SS.input_variables,
        #                                                                     to_type=ss_interface.OutputVariable)
        self.state_to_uext = c_i

        gustss = libss.ss(a_i, b_i, c_i, np.zeros((c_i.shape[0], b_i.shape[1])),
                          dt=self.linuvlm.SS.dt)
        gustss.input_variables = ss_interface.LinearVector(
            [ss_interface.InputVariable('u_gust', size=1, index=0)])
        gustss.state_variables = ss_interface.LinearVector(
            [ss_interface.StateVariable('gust', size=gustss.states, index=0)])
        gustss.output_variables = ss_interface.LinearVector(
            [ss_interface.OutputVariable('u_gust', size=gustss.outputs, index=0)])
        return gustss


@linear_gust
class MultiLeadingEdge(LinearGust):
    """
    Gust input channels defined at user-defined spanwise locations. Linearly interpolated in between
    the spanwise input positions.

    """

    gust_id = 'MultiLeadingEdge'

    settings_types = {}
    settings_default = {}
    settings_description = {}

    settings_types['span_location'] = 'list(float)'
    settings_default['span_location'] = [-10., 10.]
    settings_description['span_location'] = 'Spanwise location of the input streams of the gust'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        super().__init__()

        self.span_loc = None
        self.n_gust = None

    def initialise(self, aero, linuvlm, tsaero0, custom_settings=None):
        super().initialise(aero, linuvlm, tsaero0, custom_settings)

        self.span_loc = self.settings['span_location']
        self.n_gust = len(self.span_loc)

        assert self.n_gust > 1, 'Use LeadingEdgeGust for single inputs'

    def assemble(self):

        n_gust = self.n_gust
        Kzeta = self.linuvlm.Kzeta
        span_loc = self.span_loc

        # Convection system: needed for as many inputs (since it carries their time histories)
        x_min, x_max = self.get_x_max()  # G frame
        # TODO: project onto free stream vector
        # TODO: apply to time scaled systems
        N = int(np.ceil((x_max - x_min) / self.linuvlm.SS.dt))
        x_domain = np.linspace(x_min, x_max, N)

        # State Equation
        # for each input...
        gust_a = np.zeros((N * n_gust, N * n_gust))
        gust_b = np.zeros((N * n_gust, n_gust))
        for ith_gust in range(n_gust):
            gust_a[ith_gust * N + 1: ith_gust * N + N, ith_gust * N: ith_gust * N + N - 1] = np.eye(N-1)
            gust_b[ith_gust * N, ith_gust] = 1

        gust_c = np.zeros((3 * Kzeta, n_gust * N))
        gust_d = np.zeros((gust_c.shape[0], gust_b.shape[1]))
        for i_surf in range(self.aero.n_surf):

            M_surf, N_surf = self.aero.aero_dimensions[i_surf]
            Kzeta_start = 3 * sum(self.linuvlm.MS.KKzeta[:i_surf])  # number of coordinates up to current surface
            shape_zeta = (3, M_surf + 1, N_surf + 1)

            for i_node_span in range(N_surf + 1):
                for i_node_chord in range(M_surf + 1):
                    i_vertex = [Kzeta_start + np.ravel_multi_index((i_axis, i_node_chord, i_node_span),
                                                                   shape_zeta) for i_axis in range(3)]
                    x_vertex = self.tsaero0.zeta[i_surf][0, i_node_chord, i_node_span]
                    y_vertex = self.tsaero0.zeta[i_surf][1, i_node_chord, i_node_span]
                    # print('x', x_vertex)
                    # print('y', y_vertex)
                    interpolation_weights, column_indices = spanwise_interpolation(y_vertex, span_loc, x_vertex, x_domain)
                    # print('weights', interpolation_weights)
                    # print('col ind', column_indices)
                    for i in range(len(column_indices)):
                        gust_c[i_vertex, column_indices[i]] = np.array([0, 0, interpolation_weights[i]])
                        # print(gust_c[i_vertex, column_indices[i]])
        gustss = libss.ss(gust_a, gust_b, gust_c, gust_d, dt=self.linuvlm.SS.dt)
        # b_gust = np.zeros((N * n_gust, 6 * Kzeta + n_gust))
        # c_gust = np.zeros((9 * Kzeta, N * n_gust))
        # d_gust = np.zeros((9 * Kzeta, 6 * Kzeta + n_gust))
        # d_gust[:6 * Kzeta, :6 * Kzeta] = np.eye(6 * Kzeta)
        #
        # b_gust[:, -gust_b.shape[1]:] = gust_b
        # c_gust[-3 * Kzeta:, :gust_a.shape[0]] = gust_c
        # self.gust_ss = libss.ss(gust_a, b_gust, c_gust, d_gust, dt=self.linuvlm.SS.dt)
        # gust_input_variables = self.linuvlm.SS.input_variables.copy()
        # gust_input_variables.modify('u_gust', size=n_gust)
        # self.gust_ss.input_variables = gust_input_variables
        #
        # self.gust_ss.state_variables = ss_interface.LinearVector(
        #     [ss_interface.StateVariable('gust', size=self.gust_ss.states, index=0)])
        #
        # self.gust_ss.output_variables = ss_interface.LinearVector.transform(self.linuvlm.SS.input_variables,
        #                                                                     to_type=ss_interface.OutputVariable)
        self.state_to_uext = gust_c
        # gustss = libss.ss(gust_a, , c_i, np.zeros((c_i.shape[0], b_i.shape[1])),
        #                   dt=self.linuvlm.SS.dt)
        gustss.input_variables = ss_interface.LinearVector(
            [ss_interface.InputVariable('u_gust', size=gustss.inputs, index=0)])
        gustss.state_variables = ss_interface.LinearVector(
            [ss_interface.StateVariable('gust', size=gustss.states, index=0)])
        gustss.output_variables = ss_interface.LinearVector(
            [ss_interface.OutputVariable('u_gust', size=gustss.outputs, index=0)])
        return gustss


def linear_interpolation_weights(x_vertex, x_domain):

    column_ind_left = np.argwhere(x_domain >= x_vertex)[0][0] - 1
    if column_ind_left == - 1:
        column_ind_left = 0
    column_indices = (column_ind_left, column_ind_left + 1)
    interpolation_weights = np.array([x_domain[column_ind_left + 1] - x_vertex, x_vertex - x_domain[column_ind_left]])
    interpolation_weights /= (x_domain[column_ind_left + 1] - x_domain[column_ind_left])
    return interpolation_weights, column_indices


def chordwise_interpolation(x_vertex, x_domain):

    column_ind_left = np.argwhere(x_domain >= x_vertex)[0][0] - 1
    column_indices = (column_ind_left, column_ind_left + 1)
    interpolation_weights = np.array([x_domain[column_ind_left + 1] - x_vertex, x_vertex - x_domain[column_ind_left]])
    interpolation_weights /= (x_domain[column_ind_left + 1] - x_domain[column_ind_left])
    return interpolation_weights, column_indices


def spanwise_interpolation(y_vertex, span_loc, x_vertex, x_domain):
    # span_ind_left = np.argwhere(span_loc >= y_vertex)[0][0] - 1  # will tell which column sets to pick
    # span_indices = (span_ind_left, span_ind_left + 1)

    N = len(x_domain)
    span_weights, span_indices = linear_interpolation_weights(y_vertex, span_loc)
    # print('Span weights', span_weights)
    # print('Span indices', span_indices)
    interpolation_weights = np.zeros(4)
    interpolation_columns = []
    chord_weights, chord_ind = linear_interpolation_weights(x_vertex, x_domain)
    # print('Chord weights', chord_weights)
    # print('Chord indices', chord_ind)
    for i_span, span_location in enumerate(span_indices):
        interpolation_weights[2 * i_span - 1] = span_weights[i_span] * chord_weights[0]
        interpolation_columns.append(span_location * N + chord_ind[0])
        interpolation_weights[2 * i_span] = span_weights[i_span] * chord_weights[1]
        interpolation_columns.append(span_location * N + chord_ind[1])

    # print('Weights', interpolation_weights)
    # print('Sum', np.sum(interpolation_weights))
    # print('Columns', interpolation_columns)
    return interpolation_weights, interpolation_columns


def campbell(sigma_w, length_scale, velocity, dt=None):
    """
    Campbell approximation to the Von Karman turbulence filter.

    Args:
        sigma_w (float): Turbulence intensity in feet/s
        length_scale (float): Turbulence length scale in feet
        velocity (float): Flight velocity in feet/s
        dt (float (optional)): Discrete-time time step for discrete time systems

    Returns:
        libss.ss: SHARPy State space representation of the Campbell approximation to the Von Karman
          filter

    References:
        Palacios, R. and Cesnik, C.. Dynamics of Flexible Aircraft: Coupled flight dynamics, aeroelasticity and control.
        pg 28.
    """
    a = 1.339
    time_scale = a * length_scale / velocity
    num_tf = np.array([91/12, 52, 60]) * sigma_w * np.sqrt(time_scale / a / np.pi)
    den_tf = np.array([935/216, 561/12, 102, 60])
    if dt is None:
        camp_tf = scsig.ltisys.TransferFunction(num_tf, den_tf)
    else:
        camp_tf = scsig.ltisys.TransferFunction(num_tf, den_tf, dt=dt)

    return libss.ss.from_scipy(camp_tf.to_ss())


if __name__ == '__main__':
    pass
    # sys = campbell(90, 1750, 30, dt=0.1)

    import unittest

    class TestInterpolation(unittest.TestCase):

        def test_interpolation(self):

            x_domain = np.linspace(0, 1, 4)
            span_loc = np.linspace(0, 1, 2)

            # mesh coordinates
            x_grid = np.linspace(0.25, 0.75, 3)
            y_grid = np.linspace(0, 1, 3)
            x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

            print(x_mesh)
            print(y_mesh)
            print(y_mesh.shape)
            for i in range(len(x_grid)):
                for j in range(len(y_grid)):
                    print(i)
                    print(j)
                    print('Vertex x({:g}) = {:.2f}, y({:g}) = {:.2f}'.format(j, x_mesh[j, i],
                                                                             i, y_mesh[j, i]))
                    weights, columns = spanwise_interpolation(y_mesh[j, i], span_loc,
                                                                              x_mesh[j, i], x_domain)

                    print('Weights', weights)
                    print('Columns', columns)
                    print('\n')

    unittest.main()