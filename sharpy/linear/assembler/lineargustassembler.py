import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.utils.algebra as algebra
import scipy.sparse as scsp
import sharpy.linear.src.libss as libss
import scipy.signal as scsig

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
    def __init__(self):
        self.aero = None  #: aerogrid
        self.linuvlm = None # :linuvlm
        self.tsaero0 = None  # timestep info at the linearisation point

        self.state_to_uext = None

    def initialise(self, aero, linuvlm, tsaero0):
        self.aero = aero
        self.linuvlm = linuvlm
        self.tsaero0 = tsaero0

    def apply(self, ss):
        pass


@linear_gust
class LeadingEdgeGust(LinearGust):
    """
    Reduces the gust input to a single input for the vertical component of the gust at the leading edge.

    This is vertical velocity is then convected downstream with the free stream velocity. The gust is uniform in span.

    Warning:
        This gust assembly method has only been tested on single, straight wings (i.e. unswept, no tailplane)
    """
    gust_id = 'LeadingEdge'

    def __init__(self):
        super().__init__()

        self.gust_ss = None # :libss.ss containing the gust state-space system

    def __assemble(self):
        """
        Assembles the gust state space system, creating the (A, B, C and D) matrices that convect the single gust input
        at the leading edge downstream and uniformly across the span
        """
        return self.leading_edge()

        Kzeta = self.linuvlm.Kzeta
        M = self.linuvlm.MS.MM[0]

        # Create state-space to convect gust downstream
        A_gust = np.zeros((M+1, M+1))
        A_gust[1:, :-1] = np.eye(M, M)

        B_gust = np.zeros((M+1, 6 * Kzeta + 1))
        B_gust[0, 6 * Kzeta] = 1

        C_gust = np.zeros((9 * Kzeta, M+1))

        D_gust = np.zeros((9 * Kzeta, 6 * Kzeta + 1))

        Kout = np.zeros((3 * Kzeta, M+1))

        for i_surf in range(self.aero.n_surf):

            M_surf, N_surf = self.aero.aero_dimensions[i_surf]
            Kzeta_start = 3 * sum(self.linuvlm.MS.KKzeta[:i_surf])  # number of coordinates up to current surface
            shape_zeta = (3, M_surf + 1, N_surf + 1)

            for i_node_span in range(N_surf + 1):
                for i_node_chord in range(M_surf + 1):
                    i_vertex = [Kzeta_start + np.ravel_multi_index((i_axis, i_node_chord, i_node_span),
                                                                   shape_zeta) for i_axis in range(3)]
                    Kout[i_vertex, i_node_chord] = np.array([0, 0, 1])

        C_gust[-3 * Kzeta:] = Kout
        D_gust[:6 * Kzeta, :6 * Kzeta] = np.eye(6 * Kzeta)

        self.gust_ss = libss.ss(A_gust, B_gust, C_gust, D_gust, dt=self.linuvlm.SS.dt)
        gust_input_variables = self.linuvlm.SS.input_variables.copy()
        gust_input_variables.modify('u_gust', size=1)
        self.gust_ss.input_variables = gust_input_variables

        self.gust_ss.state_variables = ss_interface.LinearVector(
            [ss_interface.StateVariable('gust', size=self.gust_ss.states, index=0)])

        self.gust_ss.output_variables = ss_interface.LinearVector.transform(self.linuvlm.SS.input_variables,
                                                                            to_type=ss_interface.OutputVariable)
    def leading_edge(self):

        n_inputs = 1
        # Number of inputs: defined at the leading edge
        # probably need a tuple of input and spanwise location.
        # Question arises for swept wings: is the input defined at the same x reference or at the leading edge?
        # Bi-linear interpolation probably best (nonlinear?) one axis first then the other?
        Kzeta = self.linuvlm.Kzeta

        # Convection system: needed for as many inputs (since it carries their time histories)
        x_min, x_max = self.__get_x_max()  # G frame
        # TODO: project onto free stream vector
        # TODO: apply to time scaled systems
        N = int(np.ceil((x_max - x_min) / self.linuvlm.SS.dt))
        x_domain = np.linspace(x_min, x_max, N)
        # State Equation
        # for each input...
        a_i = np.zeros((N, N))
        a_i[1:, :-1] = np.eye(N-1)
        b_gust = np.zeros((N, 6 * Kzeta + 1))
        b_gust[0, -1] = 1

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
                    interpolation_weights, column_indices = self.chordwise_interpolation(x_vertex, x_domain)
                    c_i[i_vertex, column_indices[0]] = np.array([0, 0, interpolation_weights[0]])
                    c_i[i_vertex, column_indices[1]] = np.array([0, 0, interpolation_weights[1]])
                    # if i_node_span == 0:
                    #     print('x vertex', x_vertex)
                    #     print(column_indices)
                    #     print(interpolation_weights)

        c_gust[-3 * Kzeta:] = c_i
        d_gust[:6 * Kzeta, :6 * Kzeta] = np.eye(6 * Kzeta)

        self.gust_ss = libss.ss(a_i, b_gust, c_gust, d_gust, dt=self.linuvlm.SS.dt)
        gust_input_variables = self.linuvlm.SS.input_variables.copy()
        gust_input_variables.modify('u_gust', size=1)
        self.gust_ss.input_variables = gust_input_variables

        self.gust_ss.state_variables = ss_interface.LinearVector(
            [ss_interface.StateVariable('gust', size=self.gust_ss.states, index=0)])

        self.gust_ss.output_variables = ss_interface.LinearVector.transform(self.linuvlm.SS.input_variables,
                                                                            to_type=ss_interface.OutputVariable)
        self.state_to_uext = c_i

    @staticmethod
    def chordwise_interpolation(x_vertex, x_domain):

        column_ind_left = np.argwhere(x_domain >= x_vertex)[0][0] - 1
        column_indices = (column_ind_left, column_ind_left + 1)
        interpolation_weights = np.array([x_domain[column_ind_left + 1] - x_vertex, x_vertex - x_domain[column_ind_left]])
        interpolation_weights /= (x_domain[column_ind_left + 1] - x_domain[column_ind_left])
        return interpolation_weights, column_indices

    def __get_x_max(self):
        max_chord_surf = []
        min_chord_surf = []
        for i_surf in range(len(self.tsaero0.zeta)):
            max_chord_surf.append(np.max(self.tsaero0.zeta[i_surf][0, :, :]))
            min_chord_surf.append(np.min(self.tsaero0.zeta[i_surf][0, :, :]))
        return min(min_chord_surf), max(max_chord_surf)

    def apply(self, ss):

        self.__assemble()

        ss = libss.series(self.gust_ss, ss)

        return ss

    # def generator():
    # future feature idea: instead of defining the inputs for the time domain simulations as the whole input vector
    # etc, we could add a generate() method to these systems that can be called from the LinDynamicSim to apply
    # the gust and generate the correct input vector.


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

    sys = campbell(90, 1750, 30, dt=0.1)
