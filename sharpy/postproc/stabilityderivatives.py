import sharpy.utils.solver_interface as solver_interface
import numpy as np
import scipy.sparse as scsp
import sharpy.linear.src.libsparse as libsp
import sharpy.utils.cout_utils as cout

@solver_interface.solver
class StabilityDerivatives(solver_interface.BaseSolver):
    """
    Outputs the stability derivatives of a free-flying aircraft

    Warnings:
        Under Development

    To Do:
        * Coefficient of stability derivatives
        * Option to output in NED frame

    """
    solver_id = 'StabilityDerivatives'

    def __init__(self):
        self.data = None

        self.inputs = 0

    def initialise(self, data):
        self.data = data

        # Get rigid body + control surface inputs
        try:
            n_ctrl_sfc = self.data.linear.linear_system.uvlm.control_surface.n_control_surfaces
        except AttributeError:
            n_ctrl_sfc = 0

        self.inputs = 10 + n_ctrl_sfc

    def run(self):

        Y_freq = self.uvlm_steady_state_transfer_function()
        self.derivatives(Y_freq)

        return self.data

    def uvlm_steady_state_transfer_function(self):
        """
        Stability derivatives calculated using the transfer function of the UVLM projected onto the structural
        degrees of freedom at zero frequency (steady state).

        Returns:
            np.array: matrix containing the steady state values of the transfer function between the force output
              (columns) and the velocity / control surface inputs (rows).
        """
        ssuvlm = self.data.linear.linear_system.uvlm.ss

        in_matrix = np.zeros((ssuvlm.inputs, self.inputs))
        in_matrix[-self.inputs:, :] = np.eye(self.inputs)

        nout = 6
        out_matrix = np.zeros((nout, ssuvlm.outputs))
        out_matrix[:, -10:-4] = np.eye(nout)

        ssuvlm.addGain(in_matrix, where='in')
        ssuvlm.addGain(out_matrix, where='out')

        A, B, C, D = ssuvlm.get_mats()
        if type(A) == libsp.csc_matrix:
            Y_freq = C.dot(scsp.linalg.inv(scsp.eye(ssuvlm.states, format='csc') - A).dot(B)) + D
        else:
            Y_freq = C.dot(np.linalg.inv(np.eye(ssuvlm.states) - A).dot(B)) + D


        return Y_freq

    def derivatives(self, Y_freq):

        Cng = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # Project SEU on NED - TODO implementation

        der_matrix = np.zeros((6, self.inputs - 4))
        der_col = 0
        for i in list(range(6))+list(range(10, self.inputs)):
            der_matrix[:3, der_col] = Y_freq[:3, i]
            der_matrix[3:6, der_col] = Y_freq[3:6, i]
            der_col += 1

        labels_force = {0: 'X',
                        1: 'Y',
                        2: 'Z',
                        3: 'L',
                        4: 'M',
                        5: 'N'}

        labels_velocity = {0: 'u',
                           1: 'v',
                           2: 'w',
                           3: 'p',
                           4: 'q',
                           5: 'r',
                           6: 'flap1',
                           7: 'flap2',
                           8: 'flap3'}

        table = cout.TablePrinter(n_fields=7, field_length=12, field_types=['s', 'f', 'f', 'f', 'f', 'f', 'f'])
        table.print_header(['der'] + list(labels_force.values()))
        for i in range(der_matrix.shape[1]):
            table.print_line([labels_velocity[i]] + list(der_matrix[:, i]))

        return der_matrix
