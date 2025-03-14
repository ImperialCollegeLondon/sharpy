import numpy as np
import typing

from sharpy.utils.solver_interface import solver, solver_from_string
import sharpy.utils.settings as settings_utils
import sharpy.utils.solver_interface as solver_interface
import sharpy.structure.utils.xbeamlib as xbeamlib
import sharpy.utils.multibodyjax as mb
import sharpy.structure.utils.lagrangeconstraintsjax as lagrangeconstraints

_BaseStructural = solver_from_string('_BaseStructural')


@solver
class NonLinearDynamicMultibodyJAX(_BaseStructural):
    """
    Nonlinear dynamic multibody

    Nonlinear dynamic step solver for multibody structures.

    """
    solver_id = 'NonLinearDynamicMultibodyJAX'
    solver_classification = 'structural'

    settings_types = _BaseStructural.settings_types.copy()
    settings_default = _BaseStructural.settings_default.copy()
    settings_description = _BaseStructural.settings_description.copy()
    settings_options = dict()

    settings_types['time_integrator'] = 'str'
    settings_default['time_integrator'] = 'NewmarkBetaJAX'
    settings_description['time_integrator'] = 'Method to perform time integration'
    settings_options['time_integrator'] = ['NewmarkBetaJAX', 'GeneralisedAlphaJAX']

    settings_types['time_integrator_settings'] = 'dict'
    settings_default['time_integrator_settings'] = dict()
    settings_description['time_integrator_settings'] = 'Settings for the time integrator'

    settings_types['jacobian_method'] = 'str'
    settings_default['jacobian_method'] = 'forward'
    settings_description['jacobian_method'] = 'Autodifferentiation method used to calculate system jacobians'
    settings_options['jacobian_method'] = ['forward', 'reverse']

    settings_types['write_lm'] = 'bool'
    settings_default['write_lm'] = False
    settings_description['write_lm'] = 'Write lagrange multipliers to file'

    settings_types['relax_factor_lm'] = 'float'
    settings_default['relax_factor_lm'] = 0.
    settings_description['relax_factor_lm'] = ('Relaxation factor for Lagrange Multipliers. '
                                               '0 no relaxation. 1 full relaxation')

    settings_types['allow_skip_step'] = 'bool'
    settings_default['allow_skip_step'] = False
    settings_description['allow_skip_step'] = 'Allow skip step when NaN is found while solving the system'

    settings_types['zero_ini_dot_ddot'] = 'bool'
    settings_default['zero_ini_dot_ddot'] = False
    settings_description['zero_ini_dot_ddot'] = 'Set to zero the position and crv derivatives at the first time step'

    settings_types['fix_prescribed_quat_ini'] = 'bool'
    settings_default['fix_prescribed_quat_ini'] = False
    settings_description['fix_prescribed_quat_ini'] = 'Set to initial the quaternion for prescibed bodies'

    # initial speed direction is given in inertial FOR!!! also in a lot of cases coincident with global A frame
    settings_types['initial_velocity_direction'] = 'list(float)'
    settings_default['initial_velocity_direction'] = [-1., 0., 0.]
    settings_description[
        'initial_velocity_direction'] = 'Initial velocity of the reference node given in the inertial FOR'

    settings_types['initial_velocity'] = 'float'
    settings_default['initial_velocity'] = 0
    settings_description['initial_velocity'] = 'Initial velocity magnitude of the reference node'

    # restart sim after dynamictrim
    settings_types['dyn_trim'] = 'bool'
    settings_default['dyn_trim'] = False
    settings_description['dyn_trim'] = 'flag for dyntrim prior to dyncoup'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

        self.sys_size: typing.Optional[int] = None  # total number of unknowns in the system
        self.num_lm_tot: typing.Optional[int] = None  # total number of LMs in the system
        self.num_eq_tot: typing.Optional[int] = None

        # Total number of equations associated to the Lagrange multipliers
        self.lc_list: typing.Optional[list[lagrangeconstraints.Constraint]] = None
        self.num_lm_eq: typing.Optional[int] = None
        self.lambda_h: typing.Optional[np.ndarray] = None
        self.lambda_n: typing.Optional[np.ndarray] = None

        # function called to generate contributions of all LMs to equations
        self.lc_all_run: typing.Optional[typing.Callable] = None

        self.gamma: typing.Optional[int] = None
        self.beta: typing.Optional[int] = None
        self.prev_dq: typing.Optional[np.ndarray] = None

        self.time_integrator = None

        self.out_files = None  # dict: containing output_variable:file_path if desired to write output

    def initialise(self, data, custom_settings=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                       self.settings_options, no_ctype=True)

        # load info from dyn dictionary
        self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['num_steps'])

        if self.settings['initial_velocity']:
            new_direction = (self.data.structure.timestep_info[-1].cag()
                             @ self.settings['initial_velocity_direction'] * self.settings['initial_velocity'])
            self.data.structure.timestep_info[-1].for_vel[:3] = new_direction
            num_body = self.data.structure.timestep_info[0].mb_FoR_vel.shape[0]
            for ibody in range(num_body):
                self.data.structure.timestep_info[-1].mb_FoR_vel[ibody, :] \
                    = self.data.structure.timestep_info[-1].for_vel

        # find total number of LC equations
        dict_of_lc = lagrangeconstraints.DICT_OF_LC
        lc_cls_list: list[typing.Type[lagrangeconstraints.Constraint]] = []
        lc_settings: list[dict] = []
        self.num_lm_tot = 0
        for i in range(self.data.structure.ini_mb_dict['num_constraints']):
            lc_settings.append(self.data.structure.ini_mb_dict[f'constraint_{i:02d}'])
            lc_cls_list.append(dict_of_lc[lc_settings[-1]['behaviour']])
            self.num_lm_tot += lc_cls_list[-1].get_n_lm()

        # find total number of equations
        mb_dict = self.data.structure.ini_mb_dict
        self.sys_size = self.data.structure.num_dof.value
        for ibody in range(self.data.structure.num_bodies):
            if mb_dict[f'body_{ibody:02d}']['FoR_movement'] == 'free':
                self.sys_size += 10
        self.num_eq_tot = self.sys_size + self.num_lm_tot

        i_start_lc = 0
        self.lc_list = []
        for i_lc, lc_cls in enumerate(lc_cls_list):
            self.lc_list.append(lc_cls(self, i_start_lc, lc_settings[i_lc]))
            i_start_lc += self.lc_list[-1].num_lm

        # create a single function for all constraints
        self.lc_all_run = lagrangeconstraints.combine_constraints(self.lc_list)

        # lambda values
        self.lambda_h = np.zeros(self.num_lm_tot)
        self.lambda_n = np.zeros(self.num_lm_tot)

        self.prev_dq = np.zeros(self.sys_size + self.num_lm_tot)

        try:
            self.num_lm_eq = self.lc_list[0].num_lm_tot
        except IndexError:
            self.num_lm_eq = 0

        self.settings['time_integrator_settings']['sys_size'] = self.sys_size
        self.settings['time_integrator_settings']['num_LM_eq'] = self.num_lm_eq

        # Initialise time integrator
        if not restart:
            self.time_integrator = solver_interface.initialise_solver(self.settings['time_integrator'])

        self.time_integrator.initialise(self.data, self.settings['time_integrator_settings'], restart=restart)


    def assembly_mb_eq_system(self, mb_beam, mb_tstep, ts, dt, lambda_h, lambda_n, mb_dict, mb_prescribed_dict):
        """
        This function generates the matrix and vector associated to the linear system to solve a structural iteration
        It usses a Newmark-beta scheme for time integration. Being M, C and K the mass, damping
        and stiffness matrices of the system:

        .. math::
            MB_Asys = mb_k + mb_c \frac{\gamma}{\beta dt} + \frac{1}{\beta dt^2} mb_m

        Args:
            mb_beam (list(:class:`~sharpy.structure.models.beam.Beam`)): each entry represents a body
            mb_tstep (list(:class:`~sharpy.utils.datastructures.StructTimeStepInfo`)): each entry represents a body
            ts (int): Time step number
            dt(int): time step
            lambda_ (np.ndarray): Lagrange Multipliers array
            lambda_dot (np.ndarray): Time derivarive of ``Lambda``
            mb_dict (dict): Dictionary including the multibody information

        Returns:
            MB_Asys (np.ndarray): Matrix of the systems of equations
            mb_q (np.ndarray): Vector of the systems of equations
        """

        mb_m = np.zeros((self.num_eq_tot, self.num_eq_tot))
        mb_c = np.zeros((self.num_eq_tot, self.num_eq_tot))
        mb_k = np.zeros((self.num_eq_tot, self.num_eq_tot))
        mb_rhs = np.zeros(self.num_eq_tot)

        first_dof = 0
        for ibody in range(len(mb_beam)):
            # Generate the matrices for each body
            if mb_beam[ibody].FoR_movement in ('prescribed', 'prescribed_trim'):
                last_dof = first_dof + mb_beam[ibody].num_dof.value
                m, c, k, rhs = xbeamlib.cbeam3_asbly_dynamic(mb_beam[ibody], mb_tstep[ibody], self.settings)
            elif mb_beam[ibody].FoR_movement == 'free':
                last_dof = first_dof + mb_beam[ibody].num_dof.value + 10
                m, c, k, rhs = xbeamlib.xbeam3_asbly_dynamic(mb_beam[ibody], mb_tstep[ibody], self.settings)
            else:
                raise KeyError(f"Body FoR movement {mb_beam[ibody].FoR_movement} is invalid")

            mb_m[first_dof:last_dof, first_dof:last_dof] = m
            mb_c[first_dof:last_dof, first_dof:last_dof] = c
            mb_k[first_dof:last_dof, first_dof:last_dof] = k
            mb_rhs[first_dof:last_dof] = rhs

            first_dof = last_dof

        q = np.hstack([beam.q for beam in mb_tstep])
        q_dot = np.hstack([beam.dqdt for beam in mb_tstep])

        u = []
        u_dot = []
        for i_lc in range(len(self.lc_list)):
            try:
                ctrl_id = self.lc_list[i_lc].settings['controller_id'].decode('UTF-8')
                u.append(mb_prescribed_dict[ctrl_id]['psi'])
                u_dot.append(mb_prescribed_dict[ctrl_id]['psi_dot'])
            except KeyError:
                u.append(None)
                u_dot.append(None)

        lc_c, lc_k, lc_rhs = self.call_lm_generate(q, q_dot, u, u_dot, lambda_h, lambda_n)

        mb_c += lc_c
        mb_k += lc_k
        mb_rhs += lc_rhs

        return mb_m, mb_c, mb_k, mb_rhs

    # added to make profiling easier
    def call_lm_generate(self, *args):
        return self.lc_all_run(*args)

    def integrate_position(self, mb_beam, mb_tstep, dt):
        """
        This function integrates the position of each local A FoR after the
        structural iteration has been solved.

        It uses a Newmark-beta approximation.

        Args:
            mb_beam (list(:class:`~sharpy.structure.models.beam.Beam`)): each entry represents a body
            mb_tstep (list(:class:`~sharpy.utils.datastructures.StructTimeStepInfo`)): each entry represents a body
            dt(int): time step
        """
        vel = np.zeros(6)
        acc = np.zeros(6)
        for ibody in range(len(mb_tstep)):
            acc[:3] = ((0.5 - self.beta) * mb_beam[ibody].timestep_info.cga()
                        @ mb_beam[ibody].timestep_info.for_acc[:3] + self.beta * mb_tstep[ibody].cga()
                        @ mb_tstep[ibody].for_acc[:3])
            vel[:3] = mb_beam[ibody].timestep_info.cga() @ mb_beam[ibody].timestep_info.for_vel[:3]
            mb_tstep[ibody].for_pos[:3] += dt * (vel[:3] + dt * acc[:3])

    def extract_resultants(self, tstep):
        if tstep is None:
            tstep = self.data.structure.timestep_info[self.data.ts]
        steady, unsteady, grav = tstep.extract_resultants(self.data.structure,
                                                          force_type=['steady', 'unsteady', 'grav'])
        totals = steady + unsteady + grav
        return totals[:3], totals[3:6]

    def run(self, **kwargs):
        structural_step = settings_utils.set_value_or_default(kwargs, 'structural_step',
                                                              self.data.structure.timestep_info[-1])
        dt = settings_utils.set_value_or_default(kwargs, 'dt', self.settings['dt'])

        if structural_step.mb_dict is not None:
            mb_dict = structural_step.mb_dict
        else:
            mb_dict = self.data.structure.ini_mb_dict

        mb_prescribed_dict = structural_step.mb_prescribed_dict
        mb_beam, mb_tstep = mb.split_multibody(self.data.structure, structural_step, mb_dict, self.data.ts)

        if self.data.ts == 1 and self.settings['zero_ini_dot_ddot']:
            for ibody in range(len(mb_tstep)):
                mb_beam[ibody].ini_info.pos_dot.fill(0.)
                mb_beam[ibody].ini_info.pos_ddot.fill(0.)
                mb_beam[ibody].ini_info.psi_dot.fill(0.)
                mb_beam[ibody].ini_info.psi_dot_local.fill(0.)
                mb_beam[ibody].ini_info.psi_ddot.fill(0.)
                mb_tstep[ibody].pos_dot.fill(0.)
                mb_tstep[ibody].pos_ddot.fill(0.)
                mb_tstep[ibody].psi_dot.fill(0.)
                mb_tstep[ibody].psi_dot_local.fill(0.)
                mb_tstep[ibody].psi_ddot.fill(0.)

        # Predictor step
        q, dqdt, dqddt = mb.disp_and_accel2state(mb_beam, mb_tstep, self.lambda_h, self.lambda_n,
                                                 self.sys_size, self.num_lm_eq)
        self.time_integrator.predictor(q, dqdt, dqddt)

        res_sys = 0.
        res_lm = 0.

        for iteration in range(self.settings['max_iterations']):
            if iteration == self.settings['max_iterations'] - 1:   # Check if the maximum of iterations has been reached
                print(f'Solver did not converge in {iteration} iterations.\n res_sys = {res_sys} \n res_lm = {res_lm}')
                break

            # Update positions and velocities
            lambda_h, lambda_n = mb.state2disp_and_accel(q, dqdt, dqddt, mb_beam, mb_tstep, self.num_lm_eq)

            mb_m, mb_c, mb_k, mb_rhs = self.assembly_mb_eq_system(mb_beam, mb_tstep, self.data.ts, dt, lambda_h,
                                                                  lambda_n, mb_dict, mb_prescribed_dict)

            a_sys = self.time_integrator.build_matrix(mb_m, mb_c, mb_k)

            dq = np.linalg.solve(a_sys, -mb_rhs)

            # Relaxation
            relax_dq = np.zeros_like(dq)
            relax_dq[:self.sys_size] = dq[:self.sys_size].copy()
            relax_dq[self.sys_size:] = ((1. - self.settings['relax_factor_lm']) * dq[self.sys_size:] +
                                        self.settings['relax_factor_lm'] * self.prev_dq[self.sys_size:])
            self.prev_dq = dq.copy()

            # Corrector step
            self.time_integrator.corrector(q, dqdt, dqddt, relax_dq)

            res_sys = np.max(np.abs(dq[:self.sys_size]))
            try:
                res_lm = np.max(np.abs(dq[self.sys_size:]))
            except ValueError:
                res_lm = 0.
            if iteration > 0 and (res_sys < self.settings['min_delta']) and (res_lm < self.settings['min_delta']):
                break

        lambda_h, lambda_n = mb.state2disp_and_accel(q, dqdt, dqddt, mb_beam, mb_tstep, self.num_lm_eq)

        for lc in self.lc_list:
            lc.postprocess(mb_beam, mb_tstep)

        # End of Newmark-beta iterations
        if self.settings['gravity_on']:
            for ibody in range(len(mb_beam)):
                xbeamlib.cbeam3_correct_gravity_forces(mb_beam[ibody], mb_tstep[ibody], self.settings)
        mb.merge_multibody(mb_tstep, mb_beam, self.data.structure, structural_step, mb_dict, dt)

        self.lambda_h = lambda_h
        self.lambda_n = lambda_n
        self.data.Lambda = lambda_h
        self.data.Lambda_dot = lambda_n

        return self.data

    def add_step(self):
        self.data.structure.next_step()

    def next_step(self):
        raise NotImplementedError
