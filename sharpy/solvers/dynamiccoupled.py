import ctypes as ct
import time
import copy
import threading
import logging
import concurrent.futures
import queue

import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.controller_interface as controller_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra
import sharpy.utils.exceptions as exc
import sharpy.io.network_interface as network_interface
import sharpy.utils.generator_interface as gen_interface


@solver
class DynamicCoupled(BaseSolver):
    """
    The :class:`~sharpy.solvers.dynamiccoupled.DynamicCoupled` solver couples the aerodynamic and structural solvers
    of choice to march forward in time the aeroelastic system's solution.

    Using the :class:`~sharpy.solvers.dynamiccoupled.DynamicCoupled` solver requires that an instance of the
    ``StaticCoupled`` solver is called in the SHARPy solution ``flow`` when defining the problem case.

    Input data (from external controllers) can be received and data sent using the SHARPy network
    interface, specified through the setting ``network_settings`` of this solver. For more detail on how to send
    and receive data see the :class:`~sharpy.io.network_interface.NetworkLoader` documentation.

    Changes to the structural properties or external forces that depend on the instantaneous situation of the system
    can be applied through ``runtime_generators``. These runtime generators are parsed through dictionaries, with the
    key being the name of the generator and the value the settings for such generator. The currently available
    ``runtime_generators`` are :class:`~sharpy.generators.externalforces.ExternalForces` and
    :class:`~sharpy.generators.modifystructure.ModifyStructure`.

    """
    solver_id = 'DynamicCoupled'
    solver_classification = 'Coupled'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write status to screen'

    settings_types['structural_solver'] = 'str'
    settings_default['structural_solver'] = None
    settings_description['structural_solver'] = 'Structural solver to use in the coupled simulation'

    settings_types['structural_solver_settings'] = 'dict'
    settings_default['structural_solver_settings'] = None
    settings_description['structural_solver_settings'] = 'Dictionary of settings for the structural solver'

    settings_types['aero_solver'] = 'str'
    settings_default['aero_solver'] = None
    settings_description['aero_solver'] = 'Aerodynamic solver to use in the coupled simulation'

    settings_types['aero_solver_settings'] = 'dict'
    settings_default['aero_solver_settings'] = None
    settings_description['aero_solver_settings'] = 'Dictionary of settings for the aerodynamic solver'

    settings_types['n_time_steps'] = 'int'
    settings_default['n_time_steps'] = None
    settings_description['n_time_steps'] = 'Number of time steps for the simulation'

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step'

    settings_types['fsi_substeps'] = 'int'
    settings_default['fsi_substeps'] = 70
    settings_description['fsi_substeps'] = 'Max iterations in the FSI loop'

    settings_types['fsi_tolerance'] = 'float'
    settings_default['fsi_tolerance'] = 1e-5
    settings_description['fsi_tolerance'] = 'Convergence threshold for the FSI loop'

    settings_types['structural_substeps'] = 'int'
    settings_default['structural_substeps'] = 0 # 0 is normal coupled sim.
    settings_description['structural_substeps'] = 'Number of extra structural time steps per aero time step. ``0`` ' \
                                                  'is a fully coupled simulation.'

    settings_types['relaxation_factor'] = 'float'
    settings_default['relaxation_factor'] = 0.2
    settings_description['relaxation_factor'] = 'Relaxation parameter in the FSI iteration. ``0`` is no relaxation ' \
                                                'and -> ``1`` is very relaxed'

    settings_types['final_relaxation_factor'] = 'float'
    settings_default['final_relaxation_factor'] = 0.0
    settings_description['final_relaxation_factor'] = 'Relaxation factor reached in ``relaxation_steps`` with ' \
                                                      '``dynamic_relaxation`` on'

    settings_types['minimum_steps'] = 'int'
    settings_default['minimum_steps'] = 3
    settings_description['minimum_steps'] = 'Number of minimum FSI iterations before convergence'

    settings_types['relaxation_steps'] = 'int'
    settings_default['relaxation_steps'] = 100
    settings_description['relaxation_steps'] = 'Length of the relaxation factor ramp between ``relaxation_factor`` ' \
                                               'and ``final_relaxation_factor`` with ``dynamic_relaxation`` on'

    settings_types['dynamic_relaxation'] = 'bool'
    settings_default['dynamic_relaxation'] = False
    settings_description['dynamic_relaxation'] = 'Controls if relaxation factor is modified during the FSI iteration ' \
                                                 'process'

    settings_types['postprocessors'] = 'list(str)'
    settings_default['postprocessors'] = list()
    settings_description['postprocessors'] = 'List of the postprocessors to run at the end of every time step'

    settings_types['postprocessors_settings'] = 'dict'
    settings_default['postprocessors_settings'] = dict()
    settings_description['postprocessors_settings'] = 'Dictionary with the applicable settings for every ' \
                                                      '' \
                                                      '``postprocessor``. Every ``postprocessor`` needs its entry, ' \
                                                      'even if empty'

    settings_types['controller_id'] = 'dict'
    settings_default['controller_id'] = dict()
    settings_description['controller_id'] = 'Dictionary of id of every controller (key) and its type (value)'

    settings_types['controller_settings'] = 'dict'
    settings_default['controller_settings'] = dict()
    settings_description['controller_settings'] = 'Dictionary with settings (value) of every controller id (key)'

    settings_types['cleanup_previous_solution'] = 'bool'
    settings_default['cleanup_previous_solution'] = False
    settings_description['cleanup_previous_solution'] = 'Controls if previous ``timestep_info`` arrays are ' \
                                                        'reset before running the solver'

    settings_types['include_unsteady_force_contribution'] = 'bool'
    settings_default['include_unsteady_force_contribution'] = False
    settings_description['include_unsteady_force_contribution'] = 'If on, added mass contribution is added to the ' \
                                                                  'forces. This depends on the time derivative of ' \
                                                                  'the bound circulation. Check ``filter_gamma_dot`` ' \
                                                                  'in the aero solver'

    settings_types['steps_without_unsteady_force'] = 'int'
    settings_default['steps_without_unsteady_force'] = 0
    settings_description['steps_without_unsteady_force'] = 'Number of initial timesteps that don\'t include unsteady ' \
                                                           'forces contributions. This avoids oscillations due to ' \
                                                           'no perfectly trimmed initial conditions'

    settings_types['pseudosteps_ramp_unsteady_force'] = 'int'
    settings_default['pseudosteps_ramp_unsteady_force'] = 0
    settings_description['pseudosteps_ramp_unsteady_force'] = 'Length of the ramp with which unsteady force ' \
                                                              'contribution is introduced every time step during ' \
                                                              'the FSI iteration process'

    settings_types['correct_forces_method'] = 'str'
    settings_default['correct_forces_method'] = ''
    settings_description['correct_forces_method'] = 'Function used to correct aerodynamic forces. ' \
                                                    'See :py:mod:`sharpy.generators.polaraeroforces`'
    settings_options['correct_forces_method'] = ['EfficiencyCorrection', 'PolarCorrection']

    settings_types['correct_forces_settings'] = 'dict'
    settings_default['correct_forces_settings'] = {}
    settings_description['correct_forces_settings'] = 'Settings for corrected forces evaluation'

    settings_types['network_settings'] = 'dict'
    settings_default['network_settings'] = dict()
    settings_description['network_settings'] = 'Network settings. See ' \
                                               ':class:`~sharpy.io.network_interface.NetworkLoader` for supported ' \
                                               'entries'

    settings_types['runtime_generators'] = 'dict'
    settings_default['runtime_generators'] = dict()
    settings_description['runtime_generators'] = 'The dictionary keys are the runtime generators to be used. ' \
                                                 'The dictionary values are dictionaries with the settings ' \
                                                 'needed by each generator.'

    settings_types['nonlifting_body_interactions'] = 'bool'
    settings_default['nonlifting_body_interactions'] = False
    settings_description['nonlifting_body_interactions'] = 'Effect of Nonlifting Bodies on Lifting bodies are considered'
    
    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None
        self.print_info = False

        self.res = 0.0
        self.res_dqdt = 0.0
        self.res_dqddt = 0.0

        self.previous_force = None

        self.dt = 0.
        self.substep_dt = 0.
        self.initial_n_substeps = None

        self.predictor = False
        self.residual_table = None
        self.postprocessors = dict()
        self.with_postprocessors = False
        self.controllers = None

        self.time_aero = 0.
        self.time_struc = 0.

        self.correct_forces = False
        self.correct_forces_generator = None

        self.logger = logging.getLogger(__name__)  # used with the network interface

        # variables to send and receive
        self.network_loader = None
        self.set_of_variables = None

        self.runtime_generators = dict()
        self.with_runtime_generators = False

    def get_g(self):
        """
        Getter for ``g``, the gravity value
        """
        return self.structural_solver.settings['gravity']

    def set_g(self, new_g):
        """
        Setter for ``g``, the gravity value
        """
        self.structural_solver.settings['gravity'] = ct.c_double(new_g)

    def get_rho(self):
        """
        Getter for ``rho``, the density value
        """
        return self.aero_solver.settings['rho']

    def set_rho(self, new_rho):
        """
        Setter for ``rho``, the density value
        """
        self.aero_solver.settings['rho'] = ct.c_double(new_rho)

    def initialise(self, data, custom_settings=None, restart=False):
        """
        Controls the initialisation process of the solver, including processing
        the settings and initialising the aero and structural solvers, postprocessors
        and controllers.
        """
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           options=self.settings_options)

        self.original_settings = copy.deepcopy(self.settings)

        self.dt = self.settings['dt']
        self.substep_dt = (
            self.dt/(self.settings['structural_substeps'] + 1))
        self.initial_n_substeps = self.settings['structural_substeps']

        self.print_info = self.settings['print_info']
        if self.settings['cleanup_previous_solution']:
            # if there's data in timestep_info[>0], copy the last one to
            # timestep_info[0] and remove the rest
            self.cleanup_timestep_info()

        if not restart:
            self.structural_solver = solver_interface.initialise_solver(
                self.settings['structural_solver'])
            self.aero_solver = solver_interface.initialise_solver(
                self.settings['aero_solver'])
        self.structural_solver.initialise(
            self.data, self.settings['structural_solver_settings'],
            restart=restart)
        self.aero_solver.initialise(self.structural_solver.data,
                                    self.settings['aero_solver_settings'],
                                    restart=restart)
        self.data = self.aero_solver.data

        # initialise postprocessors
        if self.settings['postprocessors']:
            self.with_postprocessors = True
            # Remove previous postprocessors not required on restart
            old_list = list(self.postprocessors.keys())
            for old_list_name in old_list:
                if old_list_name not in self.settings['postprocessors']:
                    del self.postprocessors[old_list_name] 
        for postproc in self.settings['postprocessors']:
            if not postproc in self.postprocessors.keys():
                self.postprocessors[postproc] = solver_interface.initialise_solver(
                    postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc], caller=self,
                restart=restart)

        # initialise controllers
        self.with_controllers = False
        if self.settings['controller_id']:
            self.with_controllers = True
            # Remove previous controllers not required on restart
            if self.controllers is not None:
                old_list = list(self.controllers.keys())
                for old_list_name in old_list:
                    if old_list_name not in self.settings['controller_id']:
                        del self.controllers[old_list_name] 
        for controller_id, controller_type in self.settings['controller_id'].items():
            if self.controllers is not None:
                if not controller_id in self.controllers.keys():
                    self.controllers[controller_id] = (
                        controller_interface.initialise_controller(controller_type))
            else:
                self.controllers = dict()
                self.controllers[controller_id] = (
                       controller_interface.initialise_controller(controller_type))
            self.controllers[controller_id].initialise(self.data,
                    self.settings['controller_settings'][controller_id],
                    controller_id, restart=restart)
        
        # print information header
        if self.print_info:
            self.residual_table = cout.TablePrinter(8, 12, ['g', 'f', 'g', 'f', 'f', 'f', 'e', 'e'])
            self.residual_table.field_length[0] = 5
            self.residual_table.field_length[1] = 6
            self.residual_table.field_length[2] = 4
            self.residual_table.print_header(['ts', 't', 'iter', 'struc ratio', 'iter time', 'residual vel',
                                              'FoR_vel(x)', 'FoR_vel(z)'])

        # Define the function to correct aerodynamic forces
        if self.settings['correct_forces_method'] != '':
            self.correct_forces = True
            self.correct_forces_generator = gen_interface.generator_from_string(self.settings['correct_forces_method'])()
            self.correct_forces_generator.initialise(in_dict=self.settings['correct_forces_settings'],
                                                     aero=self.data.aero,
                                                     structure=self.data.structure,
                                                     rho=self.settings['aero_solver_settings']['rho'],
                                                     vortex_radius=self.settings['aero_solver_settings']['vortex_radius'],
                                                     output_folder = self.data.output_folder)

        # check for empty dictionary
        if self.settings['network_settings']:
            self.network_loader = network_interface.NetworkLoader()
            self.network_loader.initialise(in_settings=self.settings['network_settings'])

        # initialise runtime generators
        if self.settings['runtime_generators']:
            self.with_runtime_generators = True
            # Remove previous runtime generators not required on restart
            old_list = list(self.runtime_generators.keys())
            for old_list_name in old_list:
                if old_list_name not in self.settings['runtime_generators']:
                    del self.runtime_generators[old_list_name] 
        for rg_id, param in self.settings['runtime_generators'].items():
            if not rg_id in self.runtime_generators.keys():
                gen = gen_interface.generator_from_string(rg_id)
                self.runtime_generators[rg_id] = gen()
            self.runtime_generators[rg_id].initialise(param, data=self.data, restart=restart)

    def cleanup_timestep_info(self):
        if max(len(self.data.aero.timestep_info), len(self.data.structure.timestep_info)) > 1:
            self.remove_old_timestep_info(self.data.structure.timestep_info)
            self.remove_old_timestep_info(self.data.aero.timestep_info)            
            if self.settings['nonlifting_body_interactions']:
                self.remove_old_timestep_info(self.data.nonlifting_body.timestep_info)

        self.data.ts = 0

    def remove_old_timestep_info(self, tstep_info):
        # copy last info to first
        tstep_info[0] = tstep_info[-1].copy()
        # delete all the rest
        while len(tstep_info) - 1:
            del tstep_info[-1]

    def process_controller_output(self, controlled_state):
        """
        This function modified the solver properties and parameters as
        requested from the controller.

        This keeps the main loop much cleaner, while allowing for flexibility

        Please, if you add options in here, always code the possibility of
        that specific option not being there without the code complaining to
        the user.

        If it possible, use the same Key for the new setting as for the
        setting in the solver. For example, if you want to modify the
        `structural_substeps` variable in settings, use that Key in the
        `info` dictionary.

        As a convention: a value of None returns the value to the initial
        one specified in settings, while the key not being in the dict
        is ignored, so if any change was made before, it will stay there.
        """
        try:
            info = controlled_state['info']
        except KeyError:
            return controlled_state['structural'], controlled_state['aero']

        # general copy-if-exists, restore if == None
        for info_k, info_v in info.items():
            if info_k in self.settings:
                if info_v is not None:
                    self.settings[info_k] = info_v
                else:
                    self.settings[info_k] = self.original_settings[info_k]

        # specifics of every option
        for info_k, info_v in info.items():
            if info_k in self.settings:

                if info_k == 'structural_substeps':
                    if info_v is not None:
                        self.substep_dt = (
                            self.settings['dt']/(
                                self.settings['structural_substeps'] + 1))

                elif info_k == 'structural_solver':
                    if info_v is not None:
                        self.structural_solver = solver_interface.initialise_solver(
                            info['structural_solver'])
                        self.structural_solver.initialise(
                            self.data, self.settings['structural_solver_settings'])

            elif info_k == 'rotor_vel':
                for lc in self.structural_solver.lc_list:
                    if lc._lc_id == 'hinge_node_FoR_pitch':
                        lc.set_rotor_vel(info_v)

            elif info_k == 'pitch_vel':
                for lc in self.structural_solver.lc_list:
                    if lc._lc_id == 'hinge_node_FoR_pitch':
                        lc.set_pitch_vel(info_v)

        return controlled_state['structural'], controlled_state['aero']

    def run(self, **kwargs):
        """
        Run the time stepping procedure with controllers and postprocessors
        included.
        """
        solvers = settings_utils.set_value_or_default(kwargs, 'solvers', None)
        if self.network_loader is not None:
            self.set_of_variables = self.network_loader.get_inout_variables()

            incoming_queue = queue.Queue(maxsize=1)
            outgoing_queue = queue.Queue(maxsize=1)

            finish_event = threading.Event()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                netloop = executor.submit(self.network_loop, incoming_queue, outgoing_queue, finish_event)
                timeloop = executor.submit(self.time_loop, incoming_queue, outgoing_queue, finish_event, solvers)

                # TODO: improve exception handling to get exceptions when they happen from each thread
                for t1 in [netloop, timeloop]:
                    try:
                        t1.result()
                    except Exception as e:
                        print(e)
                        raise Exception

        else:
            self.time_loop(solvers=solvers)

        if self.print_info:
            cout.cout_wrap('...Finished', 1)

        for postproc in self.postprocessors:
            try:
                self.postprocessors[postproc].shutdown()
            except AttributeError:
                pass

        return self.data

    def network_loop(self, in_queue, out_queue, finish_event):
        # runs in a separate thread from time_loop()
        out_network, in_network = self.network_loader.get_networks()
        out_network.set_queue(out_queue)

        in_network.set_message_length(self.set_of_variables.input_msg_len)
        in_network.set_queue(in_queue)

        previous_queue_empty = True
        while not finish_event.is_set():

            # selector version
            events = network_interface.sel.select(timeout=1)
            if out_network.queue.empty() and not previous_queue_empty:
                out_network.set_selector_events_mask('r')
                previous_queue_empty = True
            elif not out_network.queue.empty() and previous_queue_empty:
                out_network.set_selector_events_mask('w')
                previous_queue_empty = False

            try:
                for key, mask in events:
                    key.data.process_events(mask)
            except KeyboardInterrupt:
                break

        # close sockets
        in_network.close()
        out_network.close()

    def time_loop(self, in_queue=None, out_queue=None, finish_event=None, solvers=None):
        self.logger.debug('Inside time loop')
        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        for self.data.ts in range(
                len(self.data.structure.timestep_info),
                self.settings['n_time_steps'] + 1):
            initial_time = time.perf_counter()

            # network only
            # get input from the other thread
            if in_queue:
                self.logger.info('Time Loop - Waiting for input')
                values = in_queue.get()  # should be list of tuples
                self.logger.debug('Time loop - received {}'.format(values))
                self.set_of_variables.update_timestep(self.data, values)

            structural_kstep = self.data.structure.timestep_info[-1].copy()
            aero_kstep = self.data.aero.timestep_info[-1].copy()
            if self.settings['nonlifting_body_interactions']:
                nl_body_kstep = self.data.nonlifting_body.timestep_info[-1].copy()
            else:
                nl_body_kstep = None
            self.logger.debug('Time step {}'.format(self.data.ts))

            # Add the controller here
            if self.with_controllers:
                state = {'structural': structural_kstep,
                         'aero': aero_kstep}
                for k, v in self.controllers.items():
                    state = v.control(self.data, state)
                    # this takes care of the changes in options for the solver
                    structural_kstep, aero_kstep = self.process_controller_output(
                        state)

            # Add external forces
            if self.with_runtime_generators:
                structural_kstep.runtime_steady_forces.fill(0.)
                structural_kstep.runtime_unsteady_forces.fill(0.)
                params = dict()
                params['data'] = self.data
                params['struct_tstep'] = structural_kstep
                params['aero_tstep'] = aero_kstep
                params['fsi_substep'] = -1
                for id, runtime_generator in self.runtime_generators.items():
                    runtime_generator.generate(params)

            self.time_aero = 0.0
            self.time_struc = 0.0

            # Copy the controlled states so that the interpolation does not
            # destroy the previous information
            controlled_structural_kstep = structural_kstep.copy()
            controlled_aero_kstep = aero_kstep.copy()

            for k in range(self.settings['fsi_substeps'] + 1):
                if (k == self.settings['fsi_substeps'] and
                        self.settings['fsi_substeps']):
                    print_res = 0 if self.res == 0. else np.log10(self.res)
                    print_res_dqdt = 0 if self.res_dqdt == 0. else np.log10(self.res_dqdt)
                    cout.cout_wrap(("The FSI solver did not converge!!! residuals: %f %f" % (print_res, print_res_dqdt)))
                    self.aero_solver.update_custom_grid(
                        structural_kstep,
                        aero_kstep,
                        nl_body_kstep)
                    break

                # generate new grid (already rotated)
                aero_kstep = controlled_aero_kstep.copy()
                
                self.aero_solver.update_custom_grid(
                        structural_kstep,
                        aero_kstep,
                        nl_body_kstep)

                # compute unsteady contribution
                force_coeff = 0.0
                unsteady_contribution = False
                if self.settings['include_unsteady_force_contribution']:
                    if self.data.ts > self.settings['steps_without_unsteady_force']:
                        unsteady_contribution = True
                        if k < self.settings['pseudosteps_ramp_unsteady_force']:
                            force_coeff = k/self.settings['pseudosteps_ramp_unsteady_force']
                        else:
                            force_coeff = 1.

                previous_runtime_steady_forces = structural_kstep.runtime_steady_forces.astype(dtype=ct.c_double, order='F', copy=True)
                previous_runtime_unsteady_forces = structural_kstep.runtime_unsteady_forces.astype(dtype=ct.c_double, order='F', copy=True)
                # Add external forces
                if self.with_runtime_generators:
                    structural_kstep.runtime_steady_forces.fill(0.)
                    structural_kstep.runtime_unsteady_forces.fill(0.)
                    params = dict()
                    params['data'] = self.data
                    params['struct_tstep'] = structural_kstep
                    params['aero_tstep'] = aero_kstep
                    params['fsi_substep'] = k
                    for id, runtime_generator in self.runtime_generators.items():
                        runtime_generator.generate(params)

                # run the solver
                ini_time_aero = time.perf_counter()
                self.data = self.aero_solver.run(aero_step=aero_kstep,
                                                 structural_step=structural_kstep,
                                                 convect_wake=True,
                                                 unsteady_contribution=unsteady_contribution,
                                                 nl_body_tstep = nl_body_kstep)
                self.time_aero += time.perf_counter() - ini_time_aero

                previous_kstep = structural_kstep.copy()
                structural_kstep = controlled_structural_kstep.copy()
                structural_kstep.runtime_steady_forces = previous_kstep.runtime_steady_forces.astype(dtype=ct.c_double, order='F', copy=True)
                structural_kstep.runtime_unsteady_forces = previous_kstep.runtime_unsteady_forces.astype(dtype=ct.c_double, order='F', copy=True)
                previous_kstep.runtime_steady_forces = previous_runtime_steady_forces.astype(dtype=ct.c_double, order='F', copy=True)
                previous_kstep.runtime_unsteady_forces = previous_runtime_unsteady_forces.astype(dtype=ct.c_double, order='F', copy=True)

                # move the aerodynamic surface according the the structural one
                self.aero_solver.update_custom_grid(
                        structural_kstep,
                        aero_kstep,
                        nl_body_kstep)
                
                self.map_forces(aero_kstep,
                            structural_kstep,
                            nl_body_kstep = nl_body_kstep,
                            unsteady_forces_coeff = force_coeff)

                # relaxation
                relax_factor = self.relaxation_factor(k)
                relax(self.data.structure,
                      structural_kstep,
                      previous_kstep,
                      relax_factor)

                # check if nan anywhere.
                # if yes, raise exception
                if np.isnan(structural_kstep.steady_applied_forces).any():
                    raise exc.NotConvergedSolver('NaN found in steady_applied_forces!')
                if np.isnan(structural_kstep.unsteady_applied_forces).any():
                    raise exc.NotConvergedSolver('NaN found in unsteady_applied_forces!')

                copy_structural_kstep = structural_kstep.copy()
                ini_time_struc = time.perf_counter()
                for i_substep in range(
                        self.settings['structural_substeps'] + 1):
                    # run structural solver
                    coeff = ((i_substep + 1)/
                             (self.settings['structural_substeps'] + 1))

                    structural_kstep = self.interpolate_timesteps(
                        step0=self.data.structure.timestep_info[-1],
                        step1=copy_structural_kstep,
                        out_step=structural_kstep,
                        coeff=coeff)

                    self.data = self.structural_solver.run(
                        structural_step=structural_kstep,
                        dt=self.substep_dt)

                self.time_struc += time.perf_counter() - ini_time_struc

                # check convergence
                if self.convergence(k,
                                    structural_kstep,
                                    previous_kstep,
                                    self.structural_solver,
                                    self.aero_solver,
                                    self.with_runtime_generators):
                    # move the aerodynamic surface according to the structural one
                    self.aero_solver.update_custom_grid(structural_kstep,
                                                        aero_kstep,
                                                        nl_body_tstep = nl_body_kstep)
                    break

            # move the aerodynamic surface according the the structural one
            self.aero_solver.update_custom_grid(structural_kstep,
                                                aero_kstep,
                                                nl_body_tstep = nl_body_kstep)

            self.aero_solver.add_step()
            self.data.aero.timestep_info[-1] = aero_kstep.copy()
            if self.settings['nonlifting_body_interactions']:
                self.data.nonlifting_body.timestep_info[-1] = nl_body_kstep.copy()
            self.structural_solver.add_step()
            self.data.structure.timestep_info[-1] = structural_kstep.copy()

            final_time = time.perf_counter()

            if self.print_info:
                print_res = 0 if self.res_dqdt == 0. else np.log10(self.res_dqdt)
                self.residual_table.print_line([self.data.ts,
                                                self.data.ts*self.dt,
                                                k,
                                                self.time_struc/(self.time_aero + self.time_struc),
                                                final_time - initial_time,
                                                print_res,
                                                structural_kstep.for_vel[0],
                                                structural_kstep.for_vel[2],
                                                np.sum(structural_kstep.steady_applied_forces[:, 0]),
                                                np.sum(structural_kstep.steady_applied_forces[:, 2])])
            (self.data.structure.timestep_info[self.data.ts].total_forces[0:3],
             self.data.structure.timestep_info[self.data.ts].total_forces[3:6]) = (
                        self.structural_solver.extract_resultants(self.data.structure.timestep_info[self.data.ts]))
            # run postprocessors
            if self.with_postprocessors:
                for postproc in self.postprocessors:
                    self.data = self.postprocessors[postproc].run(online=True, solvers=solvers)

            # network only
            # put result back in queue
            if out_queue:
                self.logger.debug('Time loop - about to get out variables from data')
                self.set_of_variables.get_value(self.data)
                if out_queue.full():
                    # clear the queue such that it always contains the latest time step
                    out_queue.get()  # clear item from queue
                    self.logger.debug('Data output Queue is full - clearing output')
                out_queue.put(self.set_of_variables)

        if finish_event:
            finish_event.set()
            self.logger.info('Time loop - Complete')

    def convergence(self, k, tstep, previous_tstep,
                    struct_solver, aero_solver, with_runtime_generators):
        r"""
        Check convergence in the FSI loop.

        Convergence is determined as:

        .. math:: \epsilon_q^k = \frac{|| q^k - q^{k - 1} ||}{q^0}
        .. math:: \epsilon_\dot{q}^k = \frac{|| \dot{q}^k - \dot{q}^{k - 1} ||}{\dot{q}^0}

        FSI converged if :math:`\epsilon_q^k < \mathrm{FSI\ tolerance}` and :math:`\epsilon_\dot{q}^k < \mathrm{FSI\ tolerance}`

        """
        # check for non-convergence
        if not all(np.isfinite(tstep.q)):
            import pdb
            pdb.set_trace()
            raise Exception(
                '***Not converged! There is a NaN value in the forces!')

        if not k:
            # save the value of the vectors for normalising later
            self.base_q = np.linalg.norm(tstep.q.copy())
            self.base_dqdt = np.linalg.norm(tstep.dqdt.copy())
            if self.base_dqdt == 0:
                self.base_dqdt = 1.
            if with_runtime_generators:
                self.base_res_forces = np.linalg.norm(tstep.runtime_steady_forces +
                                                      tstep.runtime_unsteady_forces)
                if self.base_res_forces == 0:
                    self.base_res_forces = 1.
            return False

        # Check the special case of no aero and no runtime generators
        if (aero_solver.solver_id.lower() == "noaero"\
             or struct_solver.solver_id.lower()  == "nostructural")\
            and not with_runtime_generators:
            return True

        # relative residuals
        self.res = (np.linalg.norm(tstep.q-
                                   previous_tstep.q)/
                    self.base_q)
        self.res_dqdt = (np.linalg.norm(tstep.dqdt-
                                        previous_tstep.dqdt)/
                         self.base_dqdt)

        if with_runtime_generators:
            res_forces = (np.linalg.norm(tstep.runtime_steady_forces -
                                        previous_tstep.runtime_steady_forces +
                                        tstep.runtime_unsteady_forces -
                                        previous_tstep.runtime_unsteady_forces)/
                                        self.base_res_forces)
        else:
            res_forces = 0.

        # we don't want this to converge before introducing the gamma_dot forces!
        if self.settings['include_unsteady_force_contribution']:
            if k < self.settings['pseudosteps_ramp_unsteady_force'] \
                    and self.data.ts > self.settings['steps_without_unsteady_force']:
                return False

        # convergence
        rigid_solver = False
        if "rigid" in struct_solver.solver_id.lower():
            rigid_solver = True
        elif "NonLinearDynamicMultibody" == struct_solver.solver_id.lower() and struct_solver.settings['rigid_bodies']:
            rigid_solver = True
        if k > self.settings['minimum_steps'] - 1:
            if self.res < self.settings['fsi_tolerance'] or rigid_solver:
                if self.res_dqdt < self.settings['fsi_tolerance']:
                    if res_forces < self.settings['fsi_tolerance']:
                        return True


    def map_forces(self, aero_kstep, structural_kstep, nl_body_kstep = None, unsteady_forces_coeff=1.0):
        # set all forces to 0
        structural_kstep.steady_applied_forces.fill(0.0)
        structural_kstep.unsteady_applied_forces.fill(0.0)

        # aero forces to structural forces
        struct_forces = mapping.aero2struct_force_mapping(
            aero_kstep.forces,
            self.data.aero.struct2aero_mapping,
            aero_kstep.zeta,
            structural_kstep.pos,
            structural_kstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.connectivities,
            structural_kstep.cag(),
            self.data.aero.data_dict)
        dynamic_struct_forces = unsteady_forces_coeff*mapping.aero2struct_force_mapping(
            aero_kstep.dynamic_forces,
            self.data.aero.struct2aero_mapping,
            aero_kstep.zeta,
            structural_kstep.pos,
            structural_kstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.connectivities,
            structural_kstep.cag(),
            self.data.aero.data_dict)

        if self.correct_forces:
            struct_forces = \
                self.correct_forces_generator.generate(aero_kstep=aero_kstep,
                                                       structural_kstep=structural_kstep,
                                                       struct_forces=struct_forces,
                                                       ts=self.data.ts)

        aero_kstep.aero_steady_forces_beam_dof = struct_forces
        structural_kstep.postproc_node['aero_steady_forces'] = struct_forces
        structural_kstep.postproc_node['aero_unsteady_forces'] = dynamic_struct_forces

        # if self.settings['nonlifting_body_interactions']:
        #     struct_forces +=  mapping.aero2struct_force_mapping(
        #         nl_body_kstep.forces,
        #         self.data.nonlifting_body.struct2aero_mapping,
        #         nl_body_kstep.zeta,
        #         structural_kstep.pos,
        #         structural_kstep.psi,
        #         self.data.structure.node_master_elem,
        #         self.data.structure.connectivities,
        #         structural_kstep.cag(),
        #         self.data.nonlifting_body.data_dict)
        # prescribed forces + aero forces
        # prescribed forces + aero forces + runtime generated
        structural_kstep.steady_applied_forces += struct_forces
        structural_kstep.steady_applied_forces += self.data.structure.ini_info.steady_applied_forces
        structural_kstep.steady_applied_forces += structural_kstep.runtime_steady_forces

        structural_kstep.unsteady_applied_forces += dynamic_struct_forces
        if len(self.data.structure.dynamic_input) > 0:
            structural_kstep.unsteady_applied_forces += self.data.structure.dynamic_input[max(self.data.ts - 1, 0)]['dynamic_forces']
        structural_kstep.unsteady_applied_forces += structural_kstep.runtime_unsteady_forces

        # Apply unsteady force coefficient
        structural_kstep.unsteady_applied_forces *= unsteady_forces_coeff

    def relaxation_factor(self, k):
        initial = self.settings['relaxation_factor']
        if not self.settings['dynamic_relaxation']:
            return initial

        final = self.settings['final_relaxation_factor']
        if k >= self.settings['relaxation_steps']:
            return final

        value = initial + (final - initial)/self.settings['relaxation_steps']*k
        return value

    @staticmethod
    def interpolate_timesteps(step0, step1, out_step, coeff):
        """
        Performs a linear interpolation between step0 and step1 based on coeff
        in [0, 1]. 0 means info in out_step == step0 and 1 out_step == step1.

        Quantities interpolated:
        * `steady_applied_forces`
        * `unsteady_applied_forces`
        * `velocity` input in Lagrange constraints

        """
        if not 0.0 <= coeff <= 1.0:
            return out_step

        # forces
        out_step.steady_applied_forces[:] = (
            (1.0 - coeff)*step0.steady_applied_forces +
            (coeff)*(step1.steady_applied_forces))

        out_step.unsteady_applied_forces[:] = (
            (1.0 - coeff)*step0.unsteady_applied_forces +
            (coeff)*(step1.unsteady_applied_forces))

        # multibody if necessary
        if out_step.mb_dict is not None:
            for key in step1.mb_dict.keys():
                if 'constraint_' in key:
                    try:
                        out_step.mb_dict[key]['velocity'][:] = (
                            (1.0 - coeff)*step0.mb_dict[key]['velocity'] +
                            (coeff)*step1.mb_dict[key]['velocity'])
                    except KeyError:
                        pass

        return out_step

    def teardown(self):
        
        self.structural_solver.teardown()
        self.aero_solver.teardown()
        if self.with_postprocessors:
            for pp in self.postprocessors.values():
                pp.teardown()
        if self.with_controllers:
            for cont in self.controllers.values():
                cont.teardown()
        if self.with_runtime_generators:
            for rg in self.runtime_generators.values():
                rg.teardown()


def relax(beam, timestep, previous_timestep, coeff):
    timestep.steady_applied_forces = ((1.0 - coeff)*timestep.steady_applied_forces +
            coeff*previous_timestep.steady_applied_forces)
    timestep.unsteady_applied_forces = ((1.0 - coeff)*timestep.unsteady_applied_forces +
            coeff*previous_timestep.unsteady_applied_forces)
    timestep.runtime_steady_forces = ((1.0 - coeff)*timestep.runtime_steady_forces +
            coeff*previous_timestep.runtime_steady_forces)
    timestep.runtime_unsteady_forces = ((1.0 - coeff)*timestep.runtime_unsteady_forces +
            coeff*previous_timestep.runtime_unsteady_forces)


def normalise_quaternion(tstep):
    tstep.dqdt[-4:] = algebra.unit_vector(tstep.dqdt[-4:])
    tstep.quat = tstep.dqdt[-4:].astype(dtype=ct.c_double, order='F', copy=True)
