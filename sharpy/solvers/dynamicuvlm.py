"""
Time Domain Aerodynamic Solver

N Goizueta Jan 19
"""
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout

@solver
class DynamicUVLM(BaseSolver):
    """
    Dynamic Aerodynamic Time Domain Simulation

    Provides an aerodynamic only simulation in time by time stepping the solution. The type of aerodynamic solver is
    parsed as a setting.

    To Do:
        Clean timestep information for memory efficiency

    Warnings:
        Under development. Issues encountered when using the linear UVLM as the aerodynamic solver with integration
        order = 1.

    """
    solver_id = 'DynamicUVLM'
    solver_classification = 'Aero'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

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

    settings_types['include_unsteady_force_contribution'] = 'bool'
    settings_default['include_unsteady_force_contribution'] = False
    settings_description['include_unsteady_force_contribution'] = 'If on, added mass contribution is added to the forces. This depends on the time derivative of the bound circulation. Check ``filter_gamma_dot`` in the aero solver'
    settings_types['postprocessors'] = 'list(str)'
    settings_default['postprocessors'] = list()
    settings_description['postprocessors'] = 'List of the postprocessors to run at the end of every time step'

    settings_types['postprocessors_settings'] = 'dict'
    settings_default['postprocessors_settings'] = dict()
    settings_description['postprocessors_settings'] = 'Dictionary with the applicable settings for every ``psotprocessor``. Every ``postprocessor`` needs its entry, even if empty'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)


    def __init__(self):

        self.data = None
        self.settings = None
        self.aero_solver = None
        self.print_info = False
        self.dt = None
        self.residual_table = None

        self.postprocessors = dict()
        self.with_postprocessors = False

    def initialise(self, data, custom_settings=None, restart=False):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt']
        self.print_info = self.settings['print_info']

        self.aero_solver = solver_interface.initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.data, self.settings['aero_solver_settings'], restart=False)
        self.data = self.aero_solver.data

        # initialise postprocessors
        self.postprocessors = dict()
        if len(self.settings['postprocessors']) > 0:
            self.with_postprocessors = True
        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = solver_interface.initialise_solver(postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc], caller=self, restart=False)

        if self.print_info:
            self.residual_table = cout.TablePrinter(2, 14, ['g', 'f'])
            self.residual_table.print_header(['ts', 't'])

    def run(self, **kwargs):

        # struct info - only for orientation, no structural solution is performed
        struct_ini_step = self.data.structure.timestep_info[-1]

        for self.data.ts in range(len(self.data.aero.timestep_info),
                                  len(self.data.aero.timestep_info) + self.settings['n_time_steps']):

            aero_tstep = self.data.aero.timestep_info[-1]
            self.aero_solver.update_custom_grid(struct_ini_step, aero_tstep)

            force_coeff = 0.0
            if self.settings['include_unsteady_force_contribution']:
                force_coeff = 1.0
            if self.data.ts < 5:
                force_coeff = 0.0

            # run the solver
            if force_coeff == 0.:
                unsteady_contribution = False
            else:
                unsteady_contribution = True

            self.data = self.aero_solver.run(aero_tstep=aero_tstep,
                                             structure_tstep=struct_ini_step,
                                             convect_wake=True,
                                             unsteady_contribution=unsteady_contribution)

            self.aero_solver.add_step()
            self.data.aero.timestep_info[-1] = aero_tstep.copy()
            self.data.structure.timestep_info.append(struct_ini_step.copy())

            if self.print_info:
                self.residual_table.print_line([self.data.ts,
                                                self.data.ts * self.dt])

            if self.with_postprocessors:
                for postproc in self.postprocessors:
                    self.data = self.postprocessors[postproc].run(online=True)

        if self.print_info:
            cout.cout_wrap('...Finished', 1)

        return self.data
