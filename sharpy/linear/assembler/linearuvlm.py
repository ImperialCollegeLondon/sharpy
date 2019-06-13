"""
Linear UVLM State Space System
"""

import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.linear.src.linuvlm as linuvlm
import sharpy.linear.src.libsparse as libsp
import sharpy.utils.settings as settings

@ss_interface.linear_system
class LinearUVLM(ss_interface.BaseElement):
    sys_id = 'LinearUVLM'

    def __init__(self):

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['remove_inputs'] = 'list'
        self.settings_default['remove_inputs'] = []

        self.data = None
        self.sys = None
        self.ss = None

        self.settings = dict()
        self.state_variables = None
        self.input_variables = None
        self.output_variables = None

    def initialise(self, data, custom_settings=None):

        self.data = data

        if custom_settings:
            self.settings = custom_settings
        else:
            try:
                self.settings = self.data.settings['LinearSpace'][self.sys_id]  # Load settings, the settings should be stored in data.linear.settings
            except KeyError:
                pass

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.data.linear.tsaero0.rho = float(self.settings['density'])
        uvlm = linuvlm.Dynamic(self.data.linear.tsaero0, dt=None, dynamic_settings=self.settings)

        self.sys = uvlm

        input_variables_database = {'zeta': [0, 3*self.sys.Kzeta],
                                    'zeta_dot': [3*self.sys.Kzeta, 6*self.sys.Kzeta],
                                    'u_gust': [6*self.sys.Kzeta, 9*self.sys.Kzeta]}
        state_variables_database = {'gamma': [0, self.sys.K],
                                    'gamma_w': [self.sys.K, self.sys.K_star],
                                    'dtgamma_dot': [self.sys.K + self.sys.K_star, 2*self.sys.K + self.sys.K_star],
                                    'gamma_m1': [2*self.sys.K + self.sys.K_star, 3*self.sys.K + self.sys.K_star]}

        self.input_variables = ss_interface.LinearVector(input_variables_database, self.sys_id)
        self.state_variables = ss_interface.LinearVector(state_variables_database, self.sys_id)

    def assemble(self):

        self.sys.assemble_ss()
        self.ss = self.sys.SS

        if self.settings['remove_inputs']:
            self.remove_inputs(self.settings['remove_inputs'])


    def remove_inputs(self, remove_list):

        self.input_variables.remove(remove_list)

        i = 0
        for variable in self.input_variables.vector_vars:
            if i == 0:
                trim_array = self.input_variables.vector_vars[variable].cols_loc
            else:
                trim_array = np.hstack((trim_array, self.input_variables.vector_vars[variable].cols_loc))
            i += 1

        self.sys.SS.B = libsp.csc_matrix(self.sys.SS.B[:, trim_array])
        self.sys.SS.D = libsp.csc_matrix(self.sys.SS.D[:, trim_array])

    def unpack_ss_vectors(self, y_n, x_n, u_n, aero_tstep):
        r"""
        Transform column vectors used in the state space formulation into SHARPy format

        The column vectors are transformed into lists with one entry per aerodynamic surface. Each entry contains a
        matrix with the quantities at each grid vertex.

        .. math::
            \mathbf{y}_n \longrightarrow \mathbf{f}_{aero}

        .. math:: \mathbf{x}_n \longrightarrow \mathbf{\Gamma}_n,\,
            \mathbf{\Gamma_w}_n,\,
            \mathbf{\dot{\Gamma}}_n

        If the ``track_body`` option is on, the output forces are projected from
        the linearization frame, to the G frame. Note that the linearisation
        frame is:
            a. equal to the FoR G at time 0 (linearisation point)
            b. rotates as the body frame specified in the ``track_body_number``

        Args:
            y_n (np.ndarray): Column output vector of linear UVLM system
            x_n (np.ndarray): Column state vector of linear UVLM system
            u_n (np.ndarray): Column input vector of linear UVLM system
            aero_tstep (AeroTimeStepInfo): aerodynamic timestep information class instance

        Returns:
            tuple: Tuple containing:

                forces (list):
                    Aerodynamic forces in a list with ``n_surf`` entries.
                    Each entry is a ``(6, M+1, N+1)`` matrix, where the first 3
                    indices correspond to the components in ``x``, ``y`` and ``z``. The latter 3 are zero.

                gamma (list):
                    Bound circulation list with ``n_surf`` entries. Circulation is stored in an ``(M+1, N+1)``
                    matrix, corresponding to the panel vertices.

                gamma_dot (list):
                    Bound circulation derivative list with ``n_surf`` entries.
                    Circulation derivative is stored in an ``(M+1, N+1)`` matrix, corresponding to the panel
                    vertices.

                gamma_star (list):
                    Wake (free) circulation list with ``n_surf`` entries. Wake circulation is stored in an
                    ``(M_star+1, N+1)`` matrix, corresponding to the panel vertices of the wake.

        """

        # project forces from uvlm FoR to FoR G
        if self.settings['track_body'].value:
            Cg_uvlm = np.dot( self.Cga, self.Cga0.T )

        if y_n:
            f_aero = y_n
        else:
            f_aero = np.zeros(3*self.sys.Kzeta)

        gamma_vec, gamma_star_vec, gamma_dot_vec = self.sys.unpack_state(x_n)
        # gamma_vec = self.data.aero.linear['gamma_0'] + dgamma_vec
        # gamma_star_vec = self.data.aero.linear['gamma_star_0'] + dgamma_star_vec
        # gamma_dot_vec = self.data.aero.linear['gamma_dot_0'] + dgamma_dot_vec

        # Reshape output into forces[i_surface] where forces[i_surface] is a (6,M+1,N+1) matrix and circulation terms
        # where gamma is a [i_surf](M+1, N+1) matrix
        forces = []
        gamma = []
        gamma_star = []
        gamma_dot = []

        worked_points = 0
        worked_panels = 0
        worked_wake_panels = 0

        for i_surf in range(aero_tstep.n_surf):
            # Tuple with dimensions of the aerogrid zeta, which is the same shape for forces
            dimensions = aero_tstep.zeta[i_surf].shape
            dimensions_gamma = self.data.aero.aero_dimensions[i_surf]
            dimensions_wake = self.data.aero.aero_dimensions_star[i_surf]

            # Number of entries in zeta
            points_in_surface = aero_tstep.zeta[i_surf].size
            panels_in_surface = aero_tstep.gamma[i_surf].size
            panels_in_wake = aero_tstep.gamma_star[i_surf].size

            # Append reshaped forces to each entry in list (one for each surface)
            forces.append(f_aero[worked_points:worked_points+points_in_surface].reshape(dimensions, order='C'))

            ### project forces.
            # - forces are in UVLM linearisation frame. Hence, these  are projected
            # into FoR (using rotation matrix Cag0 time 0) A and back to FoR G
            if self.settings['track_body'].value:
                for mm in range(dimensions[1]):
                    for nn in range(dimensions[2]):
                        forces[i_surf][:,mm,nn] = np.dot(Cg_uvlm, forces[i_surf][:,mm,nn])

            # Add the null bottom 3 rows to to the forces entry
            forces[i_surf] = np.concatenate((forces[i_surf], np.zeros(dimensions)))

            # Reshape bound circulation terms
            gamma.append(gamma_vec[worked_panels:worked_panels+panels_in_surface].reshape(
                dimensions_gamma, order='C'))
            gamma_dot.append(gamma_dot_vec[worked_panels:worked_panels+panels_in_surface].reshape(
                dimensions_gamma, order='C'))

            # Reshape wake circulation terms
            gamma_star.append(gamma_star_vec[worked_wake_panels:worked_wake_panels+panels_in_wake].reshape(
                dimensions_wake, order='C'))

            worked_points += points_in_surface
            worked_panels += panels_in_surface
            worked_wake_panels += panels_in_wake

        return forces, gamma, gamma_dot, gamma_star
