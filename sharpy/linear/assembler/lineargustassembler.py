import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.utils.algebra as algebra
import scipy.sparse as scsp
import sharpy.linear.src.libss as libss

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

    def initialise(self, aero, linuvlm, tsaero0):
        self.aero = aero
        self.linuvlm = linuvlm
        self.tsaero0 = tsaero0

    def apply(self, params):
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

    def apply(self, ss, input_variables=None, state_variables=None):

        self.__assemble()

        ss = libss.series(self.gust_ss, ss)

        input_variables.modify('u_gust', size=1)
        state_variables.add('gust', size=self.gust_ss.states, index=-1)

        input_variables.update()
        state_variables.update()

        return ss

    # def generator():
    # future feature idea: instead of defining the inputs for the time domain simulations as the whole input vector
    # etc, we could add a generate() method to these systems that can be called from the LinDynamicSim to apply
    # the gust and generate the correct input vector.


