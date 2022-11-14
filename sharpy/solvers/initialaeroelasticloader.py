from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.h5utils as h5utils
import sharpy.utils.exceptions as exceptions


@solver
class InitialAeroelasticLoader(BaseSolver):
    r"""
        This solver prescribes pos, pos_dot, psi, psi_dot and for_vel
        at each time step from a .h5 file
    """
    solver_id = 'InitialAeroelasticLoader'
    solver_classification = 'loader'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['input_file'] = 'str'
    settings_default['input_file'] = None
    settings_description['input_file'] = 'Input file containing the simulation data'

    settings_types['include_forces'] = 'bool'
    settings_default['include_forces'] = True
    settings_description['include_forces'] = 'Map the forces'

    settings_types['generate_aero'] = 'bool'
    settings_default['generate_aero'] = False
    settings_description['generate_aero'] = 'Generate the aerodynamics grids from scratch'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.file_info = None

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
        # Load simulation data
        self.file_info = h5utils.readh5(self.settings['input_file'])

    def run(self, **kwargs):

        aero_step = settings_utils.set_value_or_default(kwargs, 'aero_step', self.data.aero.timestep_info[-1])
        structural_step = settings_utils.set_value_or_default(kwargs, 'structural_step', self.data.structure.timestep_info[-1])
        
        # Copy structural information
        attributes = ['pos', 'pos_dot', 'pos_ddot',
                      'psi', 'psi_dot', 'psi_ddot',
                      'for_pos', 'for_vel', 'for_acc', 'quat',
                      'mb_FoR_pos', 'mb_FoR_vel', 'mb_FoR_acc', 'mb_quat']

        if self.settings['include_forces']:
            attributes.extend(['runtime_steady_forces',
                      'runtime_unsteady_forces',
                      'steady_applied_forces',
                      'unsteady_applied_forces'])

        for att in attributes:
            new_attr = getattr(structural_step, att)
            db_attr = getattr(self.file_info.structure, att)
            if new_attr.shape == db_attr.shape:
                new_attr[...] = db_attr
            else:
                error_msg = "Non matching shapes in attribute %s" % att
                exceptions.NotValidInputFile(error_msg)

        # Copy aero information
        if self.settings['generate_aero']:
            # Generate aerodynamic surface
            self.data.aero.generate_zeta_timestep_info(structural_step,
                                                       aero_step,
                                                       self.data.structure,
                                                       self.data.aero.aero_settings)
            # generate the wake because the solid shape might change
            self.data.aero.wake_shape_generator.generate({'zeta': aero_step.zeta,
                                            'zeta_star': aero_step.zeta_star,
                                            'gamma': aero_step.gamma,
                                            'gamma_star': aero_step.gamma_star,
                                            'dist_to_orig': aero_step.dist_to_orig})

        else:
            attributes = ['zeta', 'zeta_star', 'normals',
                          'gamma', 'gamma_star',
                          'u_ext', 'u_ext_star',]
                          # 'dist_to_orig', 'gamma_dot', 'zeta_dot',

            if self.settings['include_forces']:
                attributes.extend(['dynamic_forces', 'forces',])

            for att in attributes:
                for isurf in range(aero_step.n_surf):
                    new_attr = getattr(aero_step, att)[isurf]
                    db_attr = getattr(self.file_info.aero, att)[isurf]
                    if new_attr.shape == db_attr.shape:
                        new_attr[...] = db_attr
                    else:
                        error_msg = "Non matching shapes in attribute %s" % att
                        exceptions.NotValidInputFile(error_msg)

        return self.data
