import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.utils.algebra as algebra
import scipy.sparse as scsp
import sharpy.linear.src.libss as libss

@ss_interface.linear_system
class LinearGustGenerator(object):
    """Reduces the entire gust field input to a user-defined set of more comprehensive inputs
    """
    sys_id = 'LinearGustGenerator'

    def __init__(self):
        self.aero = None
        self.ss_gust = None
        self.Kout = None

    def initialise(self, aero):
        self.aero = aero

    def generate(self, linuvlm, aero):

        if aero is None:
            aero = self.aero

        Kzeta = linuvlm.Kzeta
        M = linuvlm.MS.MM[0]

        # Create state-space to convect gust downstream
        A_gust = np.zeros((M+1, M+1))
        A_gust[1:, :-1] = np.eye(M, M)

        B_gust = np.zeros((M+1, 6 * Kzeta + 1))
        B_gust[0, 6 * Kzeta] = 1

        C_gust = np.zeros((9 * Kzeta, M+1))

        D_gust = np.zeros((9*Kzeta, 6*Kzeta + 1))

        Kout = np.zeros((3 * Kzeta, M+1))

        for i_surf in range(aero.n_surf):

            M_surf, N_surf = aero.aero_dimensions[i_surf]
            Kzeta_start = 3 * sum(linuvlm.MS.KKzeta[:i_surf])  # number of coordinates up to current surface
            shape_zeta = (3, M_surf + 1, N_surf + 1)

            for i_node_span in range(N_surf + 1):
                for i_node_chord in range(M_surf + 1):
                    i_vertex = [Kzeta_start + np.ravel_multi_index((i_axis, i_node_chord, i_node_span),
                                                                   shape_zeta) for i_axis in range(3)]
                    Kout[i_vertex, i_node_chord] = np.array([0, 0, 1])

        C_gust[-3 * Kzeta:] = Kout
        D_gust[:6 * Kzeta, :6 * Kzeta] = np.eye(6 * Kzeta)

        self.Kout = Kout
        return A_gust, B_gust, C_gust, D_gust
