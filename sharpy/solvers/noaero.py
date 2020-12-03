import sharpy.utils.settings as settings
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

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)

        if len(self.data.aero.timestep_info) == 0:  # initialise with zero timestep for static sims
            self.update_step()

    def run(self,
            aero_tstep=None,
            structure_tstep=None,
            convect_wake=True,
            dt=None,
            t=None,
            unsteady_contribution=False):

        # generate the wake because the solid shape might change
        if aero_tstep is None:
            aero_tstep = self.data.aero.timestep_info[self.data.ts]
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

    def update_custom_grid(self, structure_tstep, aero_tstep):
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
