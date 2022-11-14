"""
Time domain solver to integrate the linear UVLM aerodynamic system developed by S. Maraniello
N Goizueta
Nov 18
"""
from sharpy.utils.solver_interface import BaseSolver, solver
import numpy as np
import sharpy.utils.settings as settings_utils
import sharpy.utils.generator_interface as gen_interface
import sharpy.utils.algebra as algebra
import sharpy.linear.src.linuvlm as linuvlm
from sharpy.utils.constants import vortex_radius_def


@solver
class StepLinearUVLM(BaseSolver):
    r"""
    Time domain aerodynamic solver that uses a linear UVLM formulation to be used with the
    :class:`solvers.DynamicCoupled` solver.

    To use this solver, the ``solver_id = StepLinearUVLM`` must be given as the name for the ``aero_solver``
    is the case of an aeroelastic solver, where the setting below would be parsed through ``aero_solver_settings``.

    Notes:

        The ``integr_order`` variable refers to the finite differencing scheme used to calculate the bound circulation
        derivative with respect to time :math:`\dot{\mathbf{\Gamma}}`. A first order scheme is used when
        ``integr_order == 1``

        .. math:: \dot{\mathbf{\Gamma}}^{n+1} = \frac{\mathbf{\Gamma}^{n+1}-\mathbf{\Gamma}^n}{\Delta t}

        If ``integr_order == 2`` a higher order scheme is used (but it isn't exactly second order accurate [1]).

        .. math:: \dot{\mathbf{\Gamma}}^{n+1} = \frac{3\mathbf{\Gamma}^{n+1}-4\mathbf{\Gamma}^n + \mathbf{\Gamma}^{n-1}}
            {2\Delta t}

        If ``track_body`` is ``True``, the UVLM is projected onto a frame ``U`` that is:

            * Coincident with ``G`` at the linearisation timestep.

            * Thence, rotates by the same quantity as the FoR ``A``.


        It is similar to a stability axes and is recommended any time rigid body dynamics are included.

    See Also:

        :class:`sharpy.sharpy.linear.assembler.linearuvlm.LinearUVLM`

    References:

        [1] Maraniello, S., & Palacios, R.. State-Space Realizations and Internal Balancing in Potential-Flow
        Aerodynamics with Arbitrary Kinematics. AIAA Journal, 57(6), 1–14. 2019. https://doi.org/10.2514/1.J058153

    """
    solver_id = 'StepLinearUVLM'
    solver_classification = 'aero'

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

    settings_types['track_body'] = 'bool'
    settings_default['track_body'] = True
    settings_description['track_body'] = 'UVLM inputs and outputs projected to coincide with lattice at linearisation'

    settings_types['track_body_number'] = 'int'
    settings_default['track_body_number'] = -1
    settings_description['track_body_number'] = 'Frame of reference number to follow. If ``-1`` track ``A`` frame.'

    settings_types['velocity_field_generator'] = 'str'
    settings_default['velocity_field_generator'] = 'SteadyVelocityField'
    settings_description['velocity_field_generator'] = 'Name of the velocity field generator to be used in the ' \
                                                       'simulation'

    settings_types['velocity_field_input'] = 'dict'
    settings_default['velocity_field_input'] = {}
    settings_description['velocity_field_input'] = 'Dictionary of settings for the velocity field generator'

    settings_types['vortex_radius'] = 'float'
    settings_default['vortex_radius'] = vortex_radius_def
    settings_description['vortex_radius'] = 'Distance between points below which induction is not computed'

    settings_types['vortex_radius_wake_ind'] = 'float'
    settings_default['vortex_radius_wake_ind'] = vortex_radius_def
    settings_description['vortex_radius_wake_ind'] = 'Distance between points below which induction is not computed in the wake convection'

    settings_types['cfl1'] = 'bool'
    settings_default['cfl1'] = True
    settings_description['cfl1'] = 'If it is ``True``, it assumes that the discretisation complies with CFL=1'
    
    settings_table = settings_utils.SettingsTable()
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
                                       scaling_settings_description, header_line='The settings that ``ScalingDict`` '
                                                                                 'accepts are the following:')

    def __init__(self):
        self.data = None
        self.settings = None
        self.lin_uvlm_system = None
        self.velocity_generator = None

    def initialise(self, data, custom_settings=None, restart=False):
        r"""
        Initialises the Linear UVLM aerodynamic solver and the chosen velocity generator.

        Settings are parsed into the standard SHARPy settings format for solvers. It then checks whether there is
        any previous information about the linearised system (in order for a solution to be restarted without
        overwriting the linearisation).

        If a linearised system does not exist, a linear UVLM system is created linearising about the current time step.

        The reference values for the input and output are transformed into column vectors :math:`\mathbf{u}`
        and :math:`\mathbf{y}`, respectively.

        The information pertaining to the linear system is stored in a dictionary ``self.data.aero.linear`` within
        the main ``data`` variable.

        Args:
            data (PreSharpy): class containing the problem information
            custom_settings (dict): custom settings dictionary

        """

        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True)
        settings_utils.to_custom_types(self.settings['ScalingDict'], self.scaling_settings_types,
                                 self.scaling_settings_default, no_ctype=True)

        # Initialise velocity generator
        velocity_generator_type = gen_interface.generator_from_string(self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(self.settings['velocity_field_input'], restart=restart)

        # Check whether linear UVLM has been initialised
        try:
            self.data.aero.linear
        except AttributeError:
            self.data.aero.linear = dict()
            aero_tstep = self.data.aero.timestep_info[-1]

            ### Record body orientation/velocities at time 0
            # This option allows to rotate the linearised UVLM with the A frame
            # or a specific body (multi-body solution)
            if self.settings['track_body']:

                self.num_body_track = self.settings['track_body_number']

                # track A frame
                if self.num_body_track == -1:
                    self.quat0 = self.data.structure.timestep_info[-1].quat.copy()
                    self.for_vel0 = self.data.structure.timestep_info[-1].for_vel.copy()
                else: # track a specific body
                    self.quat0 = \
                        self.data.structure.timestep_info[-1].mb_quat[self.num_body_track,:].copy()
                    self.for_vel0 = \
                        self.data.structure.timestep_info[-1].mb_FoR_vel[self.num_body_track ,:].copy()

                # convert to G frame
                self.Cga0 = algebra.quat2rotation(self.quat0)
                self.Cga = self.Cga0.copy()
                self.for_vel0[:3] = self.Cga0.dot(self.for_vel0[:3])
                self.for_vel0[3:] = self.Cga0.dot(self.for_vel0[3:])

            else: # check/record initial rotation speed
                self.num_body_track = None
                self.quat0 = None
                self.Cag0 = None
                self.Cga = None
                self.for_vel0 = np.zeros((6,))

            # TODO: verify of a better way to implement rho
            aero_tstep.rho = self.settings['density']

            # Generate instance of linuvlm.Dynamic()
            lin_uvlm_system = linuvlm.DynamicBlock(aero_tstep,
                                                   dynamic_settings=self.settings,
                                              # dt=self.settings['dt'],
                                              # integr_order=self.settings['integr_order'],
                                              # ScalingDict=self.settings['ScalingDict'],
                                              # RemovePredictor=self.settings['remove_predictor'],
                                              # UseSparse=self.settings['use_sparse'],
                                              for_vel=self.for_vel0)

            # add rotational speed
            for ii in range(lin_uvlm_system.MS.n_surf):
                lin_uvlm_system.MS.Surfs[ii].omega = self.for_vel0[3:]

            # Save reference values
            # System Inputs
            u_0 = self.pack_input_vector()

            # Linearised state
            dt = self.settings['dt']
            x_0 = self.pack_state_vector(aero_tstep, None, dt, self.settings['integr_order'])

            # Reference forces
            f_0 = np.concatenate([aero_tstep.forces[ss][0:3].reshape(-1, order='C')
                                  for ss in range(aero_tstep.n_surf)])

            # Assemble the state space system
            wake_prop_settings = {'dt': self.settings['dt'],
                                  'ts': self.data.ts,
                                  't': self.data.ts*self.settings['dt'],
                                  'for_pos':self.data.structure.timestep_info[-1].for_pos,
                                  'cfl1': self.settings['cfl1'],
                                  'vel_gen': self.velocity_generator}
            lin_uvlm_system.assemble_ss(wake_prop_settings=wake_prop_settings)
            self.data.aero.linear['System'] = lin_uvlm_system
            self.data.aero.linear['SS'] = lin_uvlm_system.SS
            self.data.aero.linear['x_0'] = x_0
            self.data.aero.linear['u_0'] = u_0
            self.data.aero.linear['y_0'] = f_0
            # self.data.aero.linear['gamma_0'] = gamma
            # self.data.aero.linear['gamma_star_0'] = gamma_star
            # self.data.aero.linear['gamma_dot_0'] = gamma_dot

            # TODO: Implement in AeroTimeStepInfo a way to store the state vectors
            # aero_tstep.linear.x = x_0
            # aero_tstep.linear.u = u_0
            # aero_tstep.linear.y = f_0

    def run(self, **kwargs):
        r"""
        Solve the linear aerodynamic UVLM model at the current time step ``n``. The step increment is solved as:

        .. math::
            \mathbf{x}^n &= \mathbf{A\,x}^{n-1} + \mathbf{B\,u}^n \\
            \mathbf{y}^n &= \mathbf{C\,x}^n + \mathbf{D\,u}^n

        A change of state is possible in order to solve the system without the predictor term. In which case the system
        is solved by:

        .. math::
            \mathbf{h}^n &= \mathbf{A\,h}^{n-1} + \mathbf{B\,u}^{n-1} \\
            \mathbf{y}^n &= \mathbf{C\,h}^n + \mathbf{D\,u}^n


        Variations are taken with respect to initial reference state. The state and input vectors for the linear
        UVLM system are of the form:

                If ``integr_order==1``:
                    .. math:: \mathbf{x}_n = [\delta\mathbf{\Gamma}^T_n,\,
                        \delta\mathbf{\Gamma_w}_n^T,\,
                        \Delta t \,\delta\mathbf{\dot{\Gamma}}_n^T]^T

                Else, if ``integr_order==2``:
                    .. math:: \mathbf{x}_n = [\delta\mathbf{\Gamma}_n^T,\,
                        \delta\mathbf{\Gamma_w}_n^T,\,
                        \Delta t \,\delta\mathbf{\dot{\Gamma}}_n^T,\,
                        \delta\mathbf{\Gamma}_{n-1}^T]^T

                And the input vector:
                    .. math:: \mathbf{u}_n = [\delta\mathbf{\zeta}_n^T,\,
                        \delta\dot{\mathbf{\zeta}}_n^T,\,\delta\mathbf{u_{ext}}^T_n]^T

        where the subscript ``n`` refers to the time step.

        The linear UVLM system is then solved as detailed in :func:`sharpy.linear.src.linuvlm.Dynamic.solve_step`.
        The output is a column vector containing the aerodynamic forces at the panel vertices.

        To Do: option for impulsive start?

        Args:
            aero_tstep (AeroTimeStepInfo): object containing the aerodynamic data at the current time step
            structure_tstep (StructTimeStepInfo): object containing the structural data at the current time step
            convect_wake (bool): for backward compatibility only. The linear UVLM assumes a frozen wake geometry
            dt (float): time increment
            t (float): current time
            unsteady_contribution (bool): (backward compatibily). Unsteady aerodynamic effects are always included

        Returns:
            PreSharpy: updated ``self.data`` class with the new forces and circulation terms of the system

        """

        aero_tstep = settings_utils.set_value_or_default(kwargs, 'aero_step', self.data.aero.timestep_info[-1])
        structure_tstep = settings_utils.set_value_or_default(kwargs, 'structural_step', self.data.structure.timestep_info[-1])
        convect_wake = settings_utils.set_value_or_default(kwargs, 'convect_wake', False)
        dt= settings_utils.set_value_or_default(kwargs, 'dt', self.settings['dt'])                                                                                                    
        t = settings_utils.set_value_or_default(kwargs, 't', self.data.ts*dt)
        unsteady_contribution = settings_utils.set_value_or_default(kwargs, 'unsteady_contribution', False)

        integr_order = self.settings['integr_order']

        ### Define Input

        # Generate external velocity field u_ext
        self.velocity_generator.generate({'zeta': aero_tstep.zeta,
                                          'override': True,
                                          't': t,
                                          'ts': self.data.ts,
                                          'dt': dt,
                                          'for_pos': structure_tstep.for_pos},
                                         aero_tstep.u_ext)

        ### Proj from FoR G to linearisation frame
        # - proj happens in self.pack_input_vector and unpack_ss_vectors
        if self.settings['track_body']:
            # track A frame
            if self.num_body_track  == -1:
                self.Cga = algebra.quat2rotation( structure_tstep.quat )
            else: # track a specific body
                self.Cga = algebra.quat2rotation(
                                structure_tstep.mb_quat[self.num_body_track,:] )

        # Column vector that will be the input to the linearised UVLM system
        # Input is at time step n, since it is updated in the aeroelastic solver prior to aerodynamic solver
        u_n = self.pack_input_vector()

        du_n = u_n - self.data.aero.linear['u_0']

        if self.settings['remove_predictor']:
            u_m1 = self.pack_input_vector()
            du_m1 = u_m1 - self.data.aero.linear['u_0']
        else:
            du_m1 = None

        # Retrieve State vector at time n-1
        if len(self.data.aero.timestep_info) < 2:
            x_m1 = self.pack_state_vector(aero_tstep, None, dt, integr_order)
        else:
            x_m1 = self.pack_state_vector(aero_tstep, self.data.aero.timestep_info[-2], dt, integr_order)

        # dx is at timestep n-1
        dx_m1 = x_m1 - self.data.aero.linear['x_0']

        ### Solve system - output is the variation in force
        dx_n, dy_n = self.data.aero.linear['System'].solve_step(dx_m1, du_m1, du_n, transform_state=True)

        x_n = self.data.aero.linear['x_0'] + dx_n
        y_n = self.data.aero.linear['y_0'] + dy_n

        # if self.settings['physical_model']:
        forces, gamma, gamma_dot, gamma_star = self.unpack_ss_vectors(y_n, x_n, u_n, aero_tstep)
        aero_tstep.forces = forces
        aero_tstep.gamma = gamma
        aero_tstep.gamma_dot = gamma_dot
        aero_tstep.gamma_star = gamma_star

        return self.data

    def add_step(self):
        self.data.aero.add_timestep()

    def update_grid(self, beam):
        self.data.aero.generate_zeta(beam, self.data.aero.aero_settings, -1, beam_ts=-1)

    def update_custom_grid(self, structure_tstep, aero_tstep):
        self.data.aero.generate_zeta_timestep_info(structure_tstep, aero_tstep, self.data.structure, self.data.aero.aero_settings)

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

        ### project forces from uvlm FoR to FoR G
        if self.settings['track_body']:
            Cg_uvlm = np.dot( self.Cga, self.Cga0.T )

        f_aero = y_n

        gamma_vec, gamma_star_vec, gamma_dot_vec = self.data.aero.linear['System'].unpack_state(x_n)

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
            if self.settings['track_body']:
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


    def pack_input_vector(self):
        r"""
        Transform a SHARPy AeroTimestep instance into a column vector containing the input to the linear UVLM system.

        .. math:: [\zeta,\, \dot{\zeta}, u_{ext}] \longrightarrow \mathbf{u}

        If the ``track_body`` option is on, the function projects all the input
        into a frame that:

            a. is equal to the FoR G at time 0 (linearisation point)
            b. rotates as the body frame specified in the ``track_body_number``

        Returns:
            np.ndarray: Input vector
        """

        aero_tstep = self.data.aero.timestep_info[-1]

        ### re-compute projection in G frame as if A was not rotating
        # - u_n is in FoR G. Hence, this is project in FoR A and back to FoR G
        # using rotation matrix aat time 0 (as if FoR A was not rotating).
        if self.settings['track_body']:

            Cuvlm_g = np.dot( self.Cga0, self.Cga.T )
            zeta_uvlm, zeta_dot_uvlm, u_ext_uvlm = [], [], []

            for i_surf in range(aero_tstep.n_surf):

                Mp1, Np1 = aero_tstep.dimensions[i_surf] + 1

                zeta_uvlm.append( np.empty((3,Mp1,Np1)) )
                zeta_dot_uvlm.append( np.empty((3,Mp1,Np1)) )
                u_ext_uvlm.append( np.empty((3,Mp1,Np1)) )

                for mm in range(Mp1):
                    for nn in range(Np1):
                        zeta_uvlm[i_surf][:,mm,nn] = \
                            np.dot(Cuvlm_g, aero_tstep.zeta[i_surf][:,mm,nn])
                        zeta_dot_uvlm[i_surf][:,mm,nn] = \
                            np.dot(Cuvlm_g, aero_tstep.zeta_dot[i_surf][:,mm,nn])
                        u_ext_uvlm[i_surf][:,mm,nn] = \
                            np.dot(Cuvlm_g, aero_tstep.u_ext[i_surf][:,mm,nn])

            zeta = np.concatenate([zeta_uvlm[i_surf].reshape(-1, order='C')
                                   for i_surf in range(aero_tstep.n_surf)])
            zeta_dot = np.concatenate([zeta_dot_uvlm[i_surf].reshape(-1, order='C')
                                       for i_surf in range(aero_tstep.n_surf)])
            u_ext = np.concatenate([u_ext_uvlm[i_surf].reshape(-1, order='C')
                                    for i_surf in range(aero_tstep.n_surf)])

        else:

            zeta = np.concatenate([aero_tstep.zeta[i_surf].reshape(-1, order='C')
                                   for i_surf in range(aero_tstep.n_surf)])
            zeta_dot = np.concatenate([aero_tstep.zeta_dot[i_surf].reshape(-1, order='C')
                                       for i_surf in range(aero_tstep.n_surf)])
            u_ext = np.concatenate([aero_tstep.u_ext[i_surf].reshape(-1, order='C')
                                   for i_surf in range(aero_tstep.n_surf)])

        u = np.concatenate((zeta, zeta_dot, u_ext))

        return u

    @staticmethod
    def pack_state_vector(aero_tstep, aero_tstep_m1, dt, integr_order):
        r"""
        Transform SHARPy Aerotimestep format into column vector containing the state information.

        The state vector is of a different form depending on the order of integration chosen. If a second order
        scheme is chosen, the state includes the bound circulation at the previous timestep,
        hence the timestep information for the previous timestep shall be parsed.

        The transformation is of the form:

        - If ``integr_order==1``:

                .. math:: \mathbf{x}_n = [\mathbf{\Gamma}^T_n,\,
                    \mathbf{\Gamma_w}_n^T,\,
                    \Delta t \,\mathbf{\dot{\Gamma}}_n^T]^T

        - Else, if ``integr_order==2``:

                .. math:: \mathbf{x}_n = [\mathbf{\Gamma}_n^T,\,
                    \mathbf{\Gamma_w}_n^T,\,
                    \Delta t \,\mathbf{\dot{\Gamma}}_n^T,\,
                    \mathbf{\Gamma}_{n-1}^T]^T

        For the second order integration scheme, if the previous timestep information is not parsed, a first order
        stencil is employed to estimate the bound circulation at the previous timestep:

            .. math:: \mathbf{\Gamma}^{n-1} = \mathbf{\Gamma}^n - \Delta t \mathbf{\dot{\Gamma}}^n

        Args:
            aero_tstep (AeroTimeStepInfo): Aerodynamic timestep information at the current timestep ``n``.
            aero_tstep_m1 (AeroTimeStepInfo) Aerodynamic timestep information at the previous timestep ``n-1``.

        Returns:
            np.ndarray: State vector

        """

        # Extract current state...
        gamma = np.concatenate([aero_tstep.gamma[ss].reshape(-1, order='C')
                                for ss in range(aero_tstep.n_surf)])
        gamma_star = np.concatenate([aero_tstep.gamma_star[ss].reshape(-1, order='C')
                                    for ss in range(aero_tstep.n_surf)])
        gamma_dot = np.concatenate([aero_tstep.gamma_dot[ss].reshape(-1, order='C')
                                    for ss in range(aero_tstep.n_surf)])

        if integr_order == 1:
            gamma_m1 = []

        else:
            if aero_tstep_m1:
                gamma_m1 = np.concatenate([aero_tstep_m1.gamma[ss].reshape(-1, order='C')
                                    for ss in range(aero_tstep.n_surf)])
            else:
                gamma_m1 = gamma - dt * gamma_dot

        x = np.concatenate((gamma, gamma_star, dt * gamma_dot, gamma_m1))

        return x
