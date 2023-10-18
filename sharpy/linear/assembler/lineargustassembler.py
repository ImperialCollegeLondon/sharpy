import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.linear.src.libss as libss
import scipy.signal as scsig
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
from abc import ABCMeta, abstractmethod
from scipy import signal
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
    """
    Returns an instance of the ``gust_name`` class

    Args:
        gust_name (str): Name of gust

    Returns:
        LinearGust: instance of linear gust
    """
    return dict_of_linear_gusts[gust_name]()


class LinearGust(metaclass=ABCMeta):
    # Base class from which to develop the desired gusts
    settings_types = {}
    settings_default = {}
    settings_description = {}

    print_info = True  # for debugging

    def __init__(self):
        self.aero = None  #: aerogrid
        self.tsaero0 = None  # timestep info at the linearisation point

        self.dt = None  # type: float
        self.remove_predictor = None  # type: bool
        self.Kzeta = None  # type: int  # total number of lattice vertices
        self.KKzeta = None  # type: list  # list of number of lattice vertices in each surface

        self.state_to_uext = None  #type: np.array # Gust system C matrix (for unpacking into timestep_info)
        self.uin_to_uext = None  #type: np.array # Gust system D matrix (for unpacking into timestep_info)
        self.gust_ss = None  #type: libss.StateSpace # containing the gust state-space system

        self.settings = None

        self.u_ext_direction = None  # np.ndarray Unit external velocity vector in G frame
        self.u_inf = None  # float Free stream velocity magnitude

    def initialise(self, aero, linuvlm, tsaero0, u_ext, custom_settings=None):
        """
        Initialise gust class

        Args:
            aero (sharpy.aero.models.Aerogrid):
            linuvlm (sharpy.linear.src.linuvlm.Dynamic) :
            tsaero0 (sharpy.utils.datstructures.AeroTimestepInfo) : time step at reference state
            u_ext (np.ndarray): free-stream velocity
            custom_settings (dict):
        """
        self.aero = aero
        self.tsaero0 = tsaero0
        self.dt = linuvlm.dt
        self.remove_predictor = linuvlm.remove_predictor
        self.KKzeta = linuvlm.MS.KKzeta
        self.Kzeta = sum(self.KKzeta)

        if custom_settings is not None:
            self.settings = custom_settings
            settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # find free stream velocity
        self.u_inf = np.linalg.norm(u_ext)
        self.u_ext_direction = u_ext / self.u_inf

    def get_x_max(self):
        """
        Returns the maximum and minimum and minimum coordinates of the UVLM lattice projected onto the free-stream
        velocity vector

        Returns:
            tuple: minimum and maximum coordinates of lattice projected onto velocity vector
        """
        max_chord_surf = []
        min_chord_surf = []
        for zeta in self.tsaero0.zeta:
            zeta_proj = np.zeros_like(zeta)
            _, m, n = zeta.shape
            for i_m in range(m):
                for i_n in range(n):
                    zeta_proj[:, i_m, i_n] = zeta[:, i_m, i_n].dot(self.u_ext_direction) * self.u_ext_direction
            max_chord_surf.append(np.max(zeta_proj))
            min_chord_surf.append(np.min(zeta_proj))
        return min(min_chord_surf), max(max_chord_surf)

    @abstractmethod
    def assemble(self):
        """
        Assemble gust system according to specific gust requirements in inherited classes

        Returns:
            libss.StateSpace
        """
        return libss.StateSpace(0, 0, 0, 0)  # placeholder for state space

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
            ssuvlm (libss.StateSpace): State space object of the linear UVLM.

        Returns:
            libss.StateSpace: Coupled gust system with Linear UVLM.
        """
        ssgust = self.assemble()  # libss.StateSpace()
        #
        # Feed through UVLM inputs
        b_aug = np.zeros((ssgust.states, ssuvlm.inputs - ssgust.outputs + ssgust.inputs))
        c_aug = np.zeros((ssuvlm.inputs, ssgust.states))
        d_aug = np.zeros((ssuvlm.inputs, b_aug.shape[1]))
        b_aug[:, -ssgust.inputs:] = ssgust.B
        c_aug[-ssgust.outputs:, :] = ssgust.C
        d_aug[:-ssgust.outputs, :-ssgust.inputs] = np.eye(ssuvlm.inputs - ssgust.outputs)

        self.gust_ss = libss.StateSpace(ssgust.A, b_aug, c_aug, d_aug, dt=ssgust.dt)
        input_variables = ssuvlm.input_variables.copy()
        input_variables.remove('u_gust')
        self.gust_ss.input_variables = \
            ss_interface.LinearVector.merge(input_variables, ssgust.input_variables)
        self.gust_ss.state_variables = ssgust.state_variables.copy()
        self.gust_ss.output_variables = ss_interface.LinearVector.transform(ssuvlm.input_variables,
                                                                            to_type=ss_interface.OutputVariable)
        ss = libss.series(self.gust_ss, ssuvlm)

        return ss

    def assemble_gust_statespace(self, a_i, b_i, c_i, d_i):
        """
        Assembles the individual gust system in discrete time and removes the predictor term if that is specified
        in the linear UVLM settings.

        Args:
            a_i (np.array): Gust A matrix
            b_i (np.array): Gust B matrix
            c_i (np.array): Gust C matrix
            d_i (np.array): Gust D matrix

        Returns:
            libss.StateSpace: StateSpace object of the individual gust system
        """
        if self.remove_predictor:
            a_mod, b_mod, c_mod, d_mod = libss.SSconv(a_i, None, b_i, c_i, d_i)
            gustss = libss.StateSpace(a_mod, b_mod, c_mod, d_mod, dt=self.dt)
            self.state_to_uext = c_mod
            self.uin_to_uext = d_mod
        else:
            gustss = libss.StateSpace(a_i, b_i, c_i, d_i,
                                      dt=self.dt)
            self.state_to_uext = c_i
            self.uin_to_uext = d_i
        gustss.input_variables = ss_interface.LinearVector(
            [ss_interface.InputVariable('u_gust', size=1, index=0)])
        gustss.state_variables = ss_interface.LinearVector(
            [ss_interface.StateVariable('gust', size=gustss.states, index=0)])
        gustss.output_variables = ss_interface.LinearVector(
            [ss_interface.OutputVariable('u_gust', size=gustss.outputs, index=0)])

        return gustss

    def discretise_domain(self):
        """
        Generates a "gust-station" domain, aligned with the free-stream velocity equispaced in
        :math:`\Delta t U_\infty` increments.

        Returns:
            tuple: domain and number of elements in the domain
        """
        x_min, x_max = self.get_x_max()  # G frame, min and max of panels x coordinates

        N = int(np.ceil((x_max - x_min) / self.dt / self.u_inf))
        x_domain = np.linspace(x_min, x_max, N)

        return x_domain, N


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
        Kzeta = self.Kzeta  # total number of lattice vertices

        # Convection system: needed for as many inputs (since it carries their time histories)
        x_domain, N = self.discretise_domain()

        # State Equation
        # for each input...
        a_i = np.zeros((N, N))
        a_i[1:, :-1] = np.eye(N-1)
        b_i = np.zeros((N, 1))
        b_i[0, 0] = 1

        c_i = np.zeros((3 * Kzeta, N))
        for i_surf in range(self.aero.n_surf):

            M_surf, N_surf = self.aero.dimensions[i_surf]
            Kzeta_start = 3 * sum(self.KKzeta[:i_surf])  # number of coordinates up to current surface
            shape_zeta = (3, M_surf + 1, N_surf + 1)

            if self.print_info:
                cout.cout_wrap(f'Aero surface {i_surf}')
                cout.cout_wrap(f'Gust monitoring station domain:\n{x_domain}', 1)

            for i_node_span in range(N_surf + 1):
                for i_node_chord in range(M_surf + 1):
                    i_vertex = [Kzeta_start + np.ravel_multi_index((i_axis, i_node_chord, i_node_span),
                                                                   shape_zeta) for i_axis in range(3)]
                    zeta = self.tsaero0.zeta[i_surf][:, i_node_chord, i_node_span]
                    x_vertex = zeta.dot(self.u_ext_direction)
                    interpolation_weights, column_indices = linear_interpolation_weights(x_vertex, x_domain)
                    c_i[i_vertex, column_indices[0]] = np.array([0, 0, interpolation_weights[0]])
                    c_i[i_vertex, column_indices[1]] = np.array([0, 0, interpolation_weights[1]])

                    if self.print_info:
                        cout.cout_wrap(f'Vertex info: i_n = {i_node_span}\ti_m = {i_node_chord}')
                        cout.cout_wrap(f'\tCoordinate: {zeta}', 1)
                        cout.cout_wrap(f'\tProjected coordinate: {x_vertex}', 1)
                        cout.cout_wrap(f'\tInterpolation weights: {interpolation_weights}', 2)
                        cout.cout_wrap(f'\tC matrix column indices: {column_indices}', 2)
                        cout.cout_wrap(f'\ti_vertex: {i_vertex}', 2)



        d_i = np.zeros((c_i.shape[0], b_i.shape[1]))

        gustss = self.assemble_gust_statespace(a_i, b_i, c_i, d_i)

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

    def initialise(self, aero, linuvlm, tsaero0, u_ext, custom_settings=None):
        super().initialise(aero, linuvlm, tsaero0, u_ext, custom_settings)

        self.span_loc = self.settings['span_location']
        self.n_gust = len(self.span_loc)

        assert self.n_gust > 1, 'Use LeadingEdge gust for single inputs'

    def assemble(self):

        n_gust = self.n_gust
        Kzeta = self.Kzeta
        span_loc = self.span_loc

        # Convection system: needed for as many inputs (since it carries their time histories)
        x_domain, N = self.discretise_domain()

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

            M_surf, N_surf = self.aero.dimensions[i_surf]
            Kzeta_start = 3 * sum(self.KKzeta[:i_surf])  # number of coordinates up to current surface
            shape_zeta = (3, M_surf + 1, N_surf + 1)

            for i_node_span in range(N_surf + 1):
                for i_node_chord in range(M_surf + 1):
                    i_vertex = [Kzeta_start + np.ravel_multi_index((i_axis, i_node_chord, i_node_span),
                                                                   shape_zeta) for i_axis in range(3)]
                    x_vertex = self.tsaero0.zeta[i_surf][0, i_node_chord, i_node_span]
                    y_vertex = self.tsaero0.zeta[i_surf][1, i_node_chord, i_node_span]
                    interpolation_weights, column_indices = spanwise_interpolation(y_vertex, span_loc, x_vertex, x_domain)
                    for i in range(len(column_indices)):
                        gust_c[i_vertex, column_indices[i]] = np.array([0, 0, interpolation_weights[i]])

        gustss = self.assemble_gust_statespace(gust_a, gust_b, gust_c, gust_d)
        return gustss


def get_freestream_velocity(data):
    try:
        u_inf = data.settings['StaticUvlm']['aero_solver_settings']['u_inf']
        u_inf_direction = data.settings['StaticCoupled']['aero_solver_settings']['u_inf_direction']
    except KeyError:
        try:
            u_inf = data.settings['StaticCoupled']['aero_solver_settings']['velocity_field_input']['u_inf']
            u_inf_direction = data.settings['StaticCoupled']['aero_solver_settings']['velocity_field_input']['u_inf_direction']
        except KeyError:
            cout.cout_wrap('Unable to find free stream velocity settings in StaticUvlm or StaticCoupled,'
                           'please ensure these settings are provided in the config .sharpy file. If'
                           'you are running a restart simulation make sure they are included too, regardless'
                           'of these solvers being present in the SHARPy flow', 4)
            raise KeyError

    try:
        v0 = u_inf * u_inf_direction
    except TypeError:
        # For restart solutions, where the settings may have not been processed and thus may
        # exist but in string format
        try:
            u_inf_direction = np.array(u_inf_direction, dtype=float)
        except ValueError:
            if u_inf_direction.find(',') < 0:
                u_inf_direction = np.fromstring(u_inf_direction.strip('[]'), sep=' ', dtype=float)
            else:
                u_inf_direction = np.fromstring(u_inf_direction.strip('[]'), sep=',', dtype=float)
        finally:
            v0 = np.array(u_inf_direction, dtype=float) * float(u_inf)

    return v0


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
    if column_ind_left == - 1:
        column_ind_left = 0
    column_indices = (column_ind_left, column_ind_left + 1)
    interpolation_weights = np.array([x_domain[column_ind_left + 1] - x_vertex, x_vertex - x_domain[column_ind_left]])
    interpolation_weights /= (x_domain[column_ind_left + 1] - x_domain[column_ind_left])
    return interpolation_weights, column_indices


def spanwise_interpolation(y_vertex, span_loc, x_vertex, x_domain):
    r"""
    Returns the interpolation weights and indices of said weights for a span wise interpolation. The indices refer to
    column indices of a row vector containing the gust disturbance at each chordwise control point.

    This is used in non-span uniform gusts with multiple inputs, where each set of chordwise
    control points in the state vector corresponds to one gust (i.e. its time history).

    The interpolation is performed as follows.

    For an arbitrary point within a grid delimited by ``(0, 0)``, ``(0, 1)``, ``(1, 0)`` and ``(1, 1)``, the velocity
    at a given point is:

    .. math::

        u_z = \alpha_1 u_1 + \alpha_0 u_0

    where :math:`\alpha_1 = \frac{y - y_0}{y_{1} - y_{0}}` and :math:`\alpha_0 = \frac{y_1-y}{y_1-y_0}`.

    The velocities :math:`u_1` and :math:`u_0` are at the current chordwise point, thus translating into

    .. math::

        u_1 = \beta_{11} u_{11} + \beta_{10} u_{10} \\
        u_0 = \beta_{01} u_{01} + \beta_{00} u_{00}

    where :math:`\beta_{11} = \frac{x - x_0}{x_1 - x_0}` and the rest follow a similar pattern.

    The resulting interpolation scheme gives

    .. math::

        u_z = \begin{pmatrix}
        \alpha_1 \beta_{11} u_{11} &  \alpha_1 \beta{10} &  \alpha_0 \beta_{01} & \alpha_0 \beta_{00}
        \end{pmatrix}
        \begin{bmatrix} u_{11} \\ u_{10} \\ u_{01} \\ u_{00} \end{bmatrix}.

    The indices returned as part of the output correspond to the column indices above to match with the
    corresponding interpolation control points.

    Args:
        y_vertex (np.float): y (span) coordinate of point
        span_loc (np.array): Domain of y-coordinates
        x_vertex (np.float): x (chord) coordinate of point
        x_domain (np.array): Domain of x coordinates

    Returns:
        tuple: 2-tuple containing i) the 4-array of interpolation weights and ii) the 4-array of column
          indicies to place such weights.
    """
    N = len(x_domain)
    span_weights, span_indices = linear_interpolation_weights(y_vertex, span_loc)
    interpolation_weights = np.zeros(4)
    interpolation_columns = []

    chord_weights, chord_ind = linear_interpolation_weights(x_vertex, x_domain)

    for i_span, span_location in enumerate(span_indices):
        interpolation_weights[2 * i_span - 1] = span_weights[i_span] * chord_weights[0]
        interpolation_columns.append(span_location * N + chord_ind[0])
        interpolation_weights[2 * i_span] = span_weights[i_span] * chord_weights[1]
        interpolation_columns.append(span_location * N + chord_ind[1])

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
        libss.StateSpace: SHARPy State space representation of the Campbell approximation to the Von Karman
          filter

    References:
        Palacios, R. and Cesnik, C.. Dynamics of Flexible Aircraft: Coupled flight dynamics, aeroelasticity and control.
        pg 28.
    """
    a = 1.339
    time_scale = a * length_scale / velocity
    num_tf = np.array([91/12, 52, 60]) * sigma_w * np.sqrt(time_scale / a / np.pi)
    den_tf = np.array([935/216, 561/12, 102, 60])
    if dt is not None:
        num_tf, den_tf, dt = scsig.cont2discrete((num_tf, den_tf), dt, method='bilinear')
        camp_tf = signal.TransferFunction(num_tf, den_tf, dt=dt)
    else:
        camp_tf = signal.TransferFunction(num_tf, den_tf)
    ss_turb = libss.StateSpace.from_scipy(camp_tf.to_ss())

    ss_turb.initialise_variables(({'name': 'noise_in', 'size': 1}), var_type='in')
    ss_turb.initialise_variables(({'name': 'u_gust', 'size': 1}), var_type='out')
    ss_turb.initialise_variables(({'name': 'campbell_state', 'size': ss_turb.states}), var_type='state')

    return ss_turb


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