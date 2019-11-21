"""
Linear UVLM State Space System
"""

import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.linear.src.linuvlm as linuvlm
import sharpy.linear.src.libsparse as libsp
import sharpy.utils.settings as settings
import scipy.sparse as sp
import sharpy.utils.rom_interface as rom_interface
import sharpy.linear.src.libss as libss

@ss_interface.linear_system
class LinearUVLM(ss_interface.BaseElement):
    r"""
    Linear UVLM System Assembler

    Produces state-space model of the form

        .. math::

            \mathbf{x}_{n+1} &= \mathbf{A}\,\mathbf{x}_n + \mathbf{B} \mathbf{u}_{n+1} \\
            \mathbf{y}_n &= \mathbf{C}\,\mathbf{x}_n + \mathbf{D} \mathbf{u}_n

    where the state, inputs and outputs are:

        .. math:: \mathbf{x}_n = \{ \delta \mathbf{\Gamma}_n,\, \delta \mathbf{\Gamma_{w_n}},\,
            \Delta t\,\delta\mathbf{\Gamma}'_n,\, \delta\mathbf{\Gamma}_{n-1} \}

        .. math:: \mathbf{u}_n = \{ \delta\mathbf{\zeta}_n,\, \delta\mathbf{\zeta}'_n,\,
            \delta\mathbf{u}_{ext,n} \}

        .. math:: \mathbf{y} = \{\delta\mathbf{f}\}

    with :math:`\mathbf{\Gamma}\in\mathbb{R}^{MN}` being the vector of vortex circulations,
    :math:`\mathbf{\zeta}\in\mathbb{R}^{3(M+1)(N+1)}` the vector of vortex lattice coordinates and
    :math:`\mathbf{f}\in\mathbb{R}^{3(M+1)(N+1)}` the vector of aerodynamic forces and moments. Note
    that :math:`(\bullet)'` denotes a derivative with respect to time.

    Note that the input is atypically defined at time ``n+1``. If the setting
    ``remove_predictor = True`` the predictor term ``u_{n+1}`` is eliminated through
    the change of state[1]:

        .. math::
            \mathbf{h}_n &= \mathbf{x}_n - \mathbf{B}\,\mathbf{u}_n \\

    such that:

        .. math::
            \mathbf{h}_{n+1} &= \mathbf{A}\,\mathbf{h}_n + \mathbf{A\,B}\,\mathbf{u}_n \\
            \mathbf{y}_n &= \mathbf{C\,h}_n + (\mathbf{C\,B}+\mathbf{D})\,\mathbf{u}_n

    which only modifies the equivalent :math:`\mathbf{B}` and :math:`\mathbf{D}` matrices.

    The ``integr_order`` setting refers to the finite differencing scheme used to calculate the bound circulation
    derivative with respect to time :math:`\dot{\mathbf{\Gamma}}`. A first order scheme is used when
    ``integr_order == 1``

    .. math:: \dot{\mathbf{\Gamma}}^{n+1} = \frac{\mathbf{\Gamma}^{n+1}-\mathbf{\Gamma}^n}{\Delta t}

    If ``integr_order == 2`` a higher order scheme is used (but it isn't exactly second order accurate [1]).

    .. math:: \dot{\mathbf{\Gamma}}^{n+1} = \frac{3\mathbf{\Gamma}^{n+1}-4\mathbf{\Gamma}^n + \mathbf{\Gamma}^{n-1}}
        {2\Delta t}

    References:
        [1] Franklin, GF and Powell, JD. Digital Control of Dynamic Systems, Addison-Wesley Publishing Company, 1980

        [2] Maraniello, S., & Palacios, R.. State-Space Realizations and Internal Balancing in Potential-Flow
            Aerodynamics with
            Arbitrary Kinematics. AIAA Journal, 57(6), 1–14. 2019. https://doi.org/10.2514/1.J058153

    """
    sys_id = 'LinearUVLM'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.1
    settings_description['dt'] = 'Time step'

    settings_types['integr_order'] = 'int'
    settings_default['integr_order'] = 2
    settings_description['integr_order'] = 'Integration order of the circulation derivative. Either ``1`` or ``2``.'

    settings_types['ScalingDict'] = 'dict'
    settings_default['ScalingDict'] = dict()
    settings_description['ScalingDict'] = 'Dictionary of scaling factors to achieve normalised UVLM realisation.'

    settings_types['remove_predictor'] = 'bool'
    settings_default['remove_predictor'] = True
    settings_description['remove_predictor'] = 'Remove the predictor term from the UVLM equations'

    settings_types['use_sparse'] = 'bool'
    settings_default['use_sparse'] = True
    settings_description['use_sparse'] = 'Assemble UVLM plant matrix in sparse format'

    settings_types['density'] = 'float'
    settings_default['density'] = 1.225
    settings_description['density'] = 'Air density'

    settings_types['remove_inputs'] = 'list(str)'
    settings_default['remove_inputs'] = []
    settings_description['remove_inputs'] = 'List of inputs to remove. ``u_gust`` to remove external velocity input.'

    settings_types['gust_assembler'] = 'str'
    settings_default['gust_assembler'] = ''
    settings_description['gust_assembler'] = 'Selected gust assembler. ``leading_edge`` for now'

    settings_types['rom_method'] = 'list(str)'
    settings_default['rom_method'] = []
    settings_description['rom_method'] = 'List of model reduction methods to reduce UVLM'

    settings_types['rom_method_settings'] = 'dict'
    settings_default['rom_method_settings'] = dict()
    settings_description['rom_method_settings'] = 'Dictionary with settings for the desired ROM methods, ' \
                                                  'where the name is the key to the dictionary'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    scaling_settings_types = dict()
    scaling_settings_default = dict()
    scaling_settings_description = dict()

    scaling_settings_types['length'] = 'float'
    scaling_settings_default['length'] = 1.0
    scaling_settings_description['length'] = 'Reference length to be used for UVLM scaling'

    scaling_settings_types['speed'] = 'float'
    scaling_settings_default['speed'] = 1.0
    scaling_settings_description['speed'] = 'Reference speed to be used for UVLM scaling'

    scaling_settings_types['density'] = 'float'
    scaling_settings_default['density'] = 1.0
    scaling_settings_description['density'] = 'Reference density to be used for UVLM scaling'

    __doc__ += settings_table.generate(scaling_settings_types,
                                       scaling_settings_default,
                                       scaling_settings_description)

    def __init__(self):

        self.sys = None
        self.ss = None
        self.tsaero0 = None
        self.rom = None

        self.settings = dict()
        self.state_variables = None
        self.input_variables = None
        self.output_variables = None
        self.C_to_vertex_forces = None

        self.control_surface = None
        self.gust_assembler = None
        self.gain_cs = None
        self.scaled = None

        self.linearisation_vectors = dict()  # reference conditions at the linearisation

    def initialise(self, data, custom_settings=None):

        if custom_settings:
            self.settings = custom_settings
        else:
            try:
                self.settings = data.settings['LinearAssembler']['linear_system_settings']  # Load settings, the settings should be stored in data.linear.settings
            except KeyError:
                pass

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True)
        settings.to_custom_types(self.settings['ScalingDict'], self.scaling_settings_types,
                                 self.scaling_settings_default, no_ctype=True)

        data.linear.tsaero0.rho = float(self.settings['density'])

        self.scaled = not all(scale == 1.0 for scale in self.settings['ScalingDict'].values())

        for_vel = data.linear.tsstruct0.for_vel
        cga = data.linear.tsstruct0.cga()
        uvlm = linuvlm.Dynamic(data.linear.tsaero0,
                               dt=None,
                               dynamic_settings=self.settings,
                               for_vel=np.hstack((cga.dot(for_vel[:3]), cga.dot(for_vel[3:]))))

        self.tsaero0 = data.linear.tsaero0
        self.sys = uvlm

        input_variables_database = {'zeta': [0, 3*self.sys.Kzeta],
                                    'zeta_dot': [3*self.sys.Kzeta, 6*self.sys.Kzeta],
                                    'u_gust': [6*self.sys.Kzeta, 9*self.sys.Kzeta]}
        state_variables_database = {'gamma': [0, self.sys.K],
                                    'gamma_w': [self.sys.K, self.sys.K_star],
                                    'dtgamma_dot': [self.sys.K + self.sys.K_star, 2*self.sys.K + self.sys.K_star],
                                    'gamma_m1': [2*self.sys.K + self.sys.K_star, 3*self.sys.K + self.sys.K_star]}

        self.linearisation_vectors['zeta'] = np.concatenate([self.tsaero0.zeta[i_surf].reshape(-1, order='C')
                                                             for i_surf in range(self.tsaero0.n_surf)])
        self.linearisation_vectors['zeta_dot'] = np.concatenate([self.tsaero0.zeta_dot[i_surf].reshape(-1, order='C')
                                                                 for i_surf in range(self.tsaero0.n_surf)])
        self.linearisation_vectors['u_ext'] = np.concatenate([self.tsaero0.u_ext[i_surf].reshape(-1, order='C')
                                                              for i_surf in range(self.tsaero0.n_surf)])
        self.linearisation_vectors['forces_aero'] = np.concatenate([self.tsaero0.forces[i_surf][:3].reshape(-1, order='C')
                                                                    for i_surf in range(self.tsaero0.n_surf)])

        self.input_variables = ss_interface.LinearVector(input_variables_database, self.sys_id)
        self.state_variables = ss_interface.LinearVector(state_variables_database, self.sys_id)

        if data.aero.n_control_surfaces >= 1:
            import sharpy.linear.assembler.lincontrolsurfacedeflector as lincontrolsurfacedeflector
            self.control_surface = lincontrolsurfacedeflector.LinControlSurfaceDeflector()
            self.control_surface.initialise(data, uvlm)

        if self.settings['rom_method'] != '':
            # Initialise ROM
            self.rom = dict()
            for rom_name in self.settings['rom_method']:
                self.rom[rom_name] = rom_interface.initialise_rom(rom_name)
                self.rom[rom_name].initialise(self.settings['rom_method_settings'][rom_name])

        if 'u_gust' not in self.settings['remove_inputs'] and self.settings['gust_assembler'] == 'leading_edge':
            import sharpy.linear.assembler.lineargustassembler as lineargust
            self.gust_assembler = lineargust.LinearGustGenerator()
            self.gust_assembler.initialise(data.aero)

    def assemble(self, track_body=False):
        r"""
        Assembles the linearised UVLM system, removes the desired inputs and adds linearised control surfaces
        (if present).

        With all possible inputs present, these are ordered as

        .. math:: \mathbf{u} = [\boldsymbol{\zeta},\,\dot{\boldsymbol{\zeta}},\,\mathbf{w},\,\delta]

        Control surface inputs are ordered last as:

        .. math:: [\delta_1, \delta_2, \dots, \dot{\delta}_1, \dot{\delta_2}]
        """

        self.sys.assemble_ss()

        if self.scaled:
            self.sys.nondimss()
        self.ss = self.sys.SS
        self.C_to_vertex_forces = self.ss.C.copy()

        nzeta = 3 * self.sys.Kzeta

        if self.settings['remove_inputs']:
            self.remove_inputs(self.settings['remove_inputs'])

        if self.gust_assembler is not None:
            A, B, C, D = self.gust_assembler.generate(self.sys, aero=None)
            ss_gust = libss.ss(A, B, C, D, dt=self.ss.dt)
            self.gust_assembler.ss_gust = ss_gust
            self.ss = libss.series(ss_gust, self.ss)

        if self.control_surface is not None:
            Kzeta_delta, Kdzeta_ddelta = self.control_surface.generate()
            n_zeta, n_ctrl_sfc = Kzeta_delta.shape

            # Modify the state space system with a gain at the input side
            # such that the control surface deflections are last
            if self.sys.use_sparse:
                gain_cs = sp.eye(self.ss.inputs, self.ss.inputs + 2 * self.control_surface.n_control_surfaces,
                                 format='lil')
                gain_cs[:n_zeta, self.ss.inputs: self.ss.inputs + n_ctrl_sfc] = Kzeta_delta
                gain_cs[n_zeta: 2*n_zeta, self.ss.inputs + n_ctrl_sfc: self.ss.inputs + 2 * n_ctrl_sfc] = Kdzeta_ddelta
                gain_cs = libsp.csc_matrix(gain_cs)
            else:
                gain_cs = np.eye(self.ss.inputs, self.ss.inputs + 2 * self.control_surface.n_control_surfaces)
                gain_cs[:n_zeta, self.ss.inputs: self.ss.inputs + n_ctrl_sfc] = Kzeta_delta
                gain_cs[n_zeta: 2*n_zeta, self.ss.inputs + n_ctrl_sfc: self.ss.inputs + 2 * n_ctrl_sfc] = Kdzeta_ddelta
            self.ss.addGain(gain_cs, where='in')
            self.gain_cs = gain_cs

    def remove_inputs(self, remove_list=list):
        """
        Remove certain inputs from the input vector

        To do:
            * Support for block UVLM

        Args:
            remove_list (list): Inputs to remove
        """

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

    def unpack_ss_vector(self, data, x_n, aero_tstep, track_body=False):
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
        if track_body:
            Cga = data.structure.timestep_info[-1].cga()
            # print(data.structure.timestep_info[-1].quat)
            Cga0 = data.structure.timestep_info[0].cga()
            Cg_uvlm = np.dot(Cga, Cga0.T)

        else:
            Cg_uvlm = np.eye(3)
        y_n = self.C_to_vertex_forces.dot(x_n)
        # y_n = np.zeros((3 * self.sys.Kzeta))

        gamma_vec, gamma_star_vec, gamma_dot_vec = self.sys.unpack_state(x_n)

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
            dimensions_gamma = data.aero.aero_dimensions[i_surf]
            dimensions_wake = data.aero.aero_dimensions_star[i_surf]

            # Number of entries in zeta
            points_in_surface = aero_tstep.zeta[i_surf].size
            panels_in_surface = aero_tstep.gamma[i_surf].size
            panels_in_wake = aero_tstep.gamma_star[i_surf].size

            # Append reshaped forces to each entry in list (one for each surface)
            f_aero = y_n
            forces.append(f_aero[worked_points:worked_points+points_in_surface].reshape(dimensions, order='C'))

            ### project forces.
            # - forces are in UVLM linearisation frame. Hence, these  are projected
            # into FoR (using rotation matrix Cag0 time 0) A and back to FoR G
            if track_body:
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

    def unpack_input_vector(self, u_n):
        """
        Unpacks the input vector into the corresponding grid coordinates, velocities and external velocities.

        Args:
            u_n (np.ndarray): UVLM input vector. May contain control surface deflections and external velocities.

        Returns:
            tuple: Tuple containing ``zeta``, ``zeta_dot`` and ``u_ext``, accounting for the effect of control surfaces.
        """

        # if self.gust_assembler is not None:
        #     u_n = self.gust_assembler.ss_gust

        if self.control_surface is not None:
            u_n = self.gain_cs.dot(u_n)

        input_vars = self.input_variables.vector_vars
        tsaero0 = self.tsaero0

        input_vectors = dict()
        for var in input_vars:
            input_vectors[input_vars[var].name] = u_n[input_vars[var].cols_loc]

        zeta = []
        zeta_dot = []
        u_ext = []
        worked_vertices = 0

        for i_surf in range(tsaero0.n_surf):
            vertices_in_surface = tsaero0.zeta[i_surf].size
            dimensions_zeta = tsaero0.zeta[i_surf].shape
            zeta.append(input_vectors['zeta'][worked_vertices:worked_vertices+vertices_in_surface].reshape(
                dimensions_zeta, order='C'))
            zeta_dot.append(input_vectors['zeta_dot'][worked_vertices:worked_vertices+vertices_in_surface].reshape(
                dimensions_zeta, order='C'))
            try:
                u_gust = input_vectors['u_gust']
            except KeyError:
                u_gust = np.zeros(3*vertices_in_surface*tsaero0.n_surf)
            u_ext.append(u_gust[worked_vertices:worked_vertices+vertices_in_surface].reshape(
                dimensions_zeta, order='C'))

            zeta[i_surf] += tsaero0.zeta[i_surf]
            zeta_dot[i_surf] += tsaero0.zeta_dot[i_surf]
            u_ext[i_surf] += tsaero0.u_ext[i_surf]
            worked_vertices += vertices_in_surface

        return zeta, zeta_dot, u_ext
