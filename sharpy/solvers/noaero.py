import sharpy.utils.settings as settings_utils
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class NoAero(BaseSolver):
    """
    Skip UVLM evaluation

    Solver to be used with either :class:`~sharpy.solvers.staticcoupled.StaticCoupled` or
    :class:`~sharpy.solvers.dynamiccoupled.DynamicCoupled` when aerodynamics are not of interest.

    An example would be running a structural-only simulation where you would like to keep an aerodynamic grid for
    visualisation purposes.
    """
    solver_id = 'NoAero'
    solver_classification = 'Aero'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.1
    settings_description['dt'] = 'Time step'

    settings_types['update_grid'] = 'bool'
    settings_default['update_grid'] = True
    settings_description['update_grid'] = 'Update aerodynamic grid as the structure deforms.'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data, custom_settings=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default,
                           no_ctype=True)

        if len(self.data.aero.timestep_info) == 0:  # initialise with zero timestep for static sims
            self.update_step()

    def run(self, **kwargs):

        aero_tstep = settings_utils.set_value_or_default(kwargs, 'aero_step', self.data.aero.timestep_info[-1])
        structure_tstep = settings_utils.set_value_or_default(kwargs, 'structural_step', self.data.structure.timestep_info[-1])
        convect_wake = settings_utils.set_value_or_default(kwargs, 'convect_wake', True)
        dt= settings_utils.set_value_or_default(kwargs, 'dt', self.settings['dt'])                                                                                                                             
        t = settings_utils.set_value_or_default(kwargs, 't', self.data.ts*dt)
        unsteady_contribution = settings_utils.set_value_or_default(kwargs, 'unsteady_contribution', False)

        # generate the wake because the solid shape might change
        self.data.aero.wake_shape_generator.generate({'zeta': aero_tstep.zeta,
                                            'zeta_star': aero_tstep.zeta_star,
                                            'gamma': aero_tstep.gamma,
                                            'gamma_star': aero_tstep.gamma_star,
                                            'dist_to_orig': aero_tstep.dist_to_orig})

        return self.data

    def add_step(self):
        self.data.aero.add_timestep()

    def update_grid(self, beam):
        # called by DynamicCoupled
        if self.settings['update_grid']:
            self.data.aero.generate_zeta(beam,
                                         self.data.aero.aero_settings,
                                         -1,
                                         beam_ts=-1)

    def update_custom_grid(self, structure_tstep, aero_tstep, nl_body_tstep = None):
        # called by DynamicCoupled
        if self.settings['update_grid']:
            self.data.aero.generate_zeta_timestep_info(structure_tstep,
                                                       aero_tstep,
                                                       self.data.structure,
                                                       self.data.aero.aero_settings,
                                                       dt=None)

    def update_step(self):
        # called by StaticCoupled
        if self.settings['update_grid']:
            self.data.aero.generate_zeta(self.data.structure,
                                         self.data.aero.aero_settings,
                                         self.data.ts)
        else:
            self.add_step()

    def next_step(self):
        # called by StaticCoupled
        self.add_step()
