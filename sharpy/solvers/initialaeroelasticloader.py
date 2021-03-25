import numpy as np
import h5py as h5

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.utils.h5utils as h5utils

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

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.file_info = None

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
        # Load simulation data
        self.file_info = h5utils.readh5(self.settings['input_file'])

    def run(self, structural_step=None, aero_step=None):

        if structural_step is None:
            structural_step = self.data.structure.timestep_info[-1]
        if aero_step is None:
            aero_step = self.data.aero.timestep_info[-1]

        # Copy structural information
        attributes = ['pos', 'pos_dot', 'pos_ddot',
                      'psi', 'psi_dot', 'psi_ddot',
                      'for_pos', 'for_vel', 'for_acc',
                      'mb_FoR_pos', 'mb_FoR_vel', 'mb_FoR_acc', 'mb_quat',
                      'runtime_generated_forces',
                      'steady_applied_forces',
                      'unsteady_applied_forces']
        for att in attributes:
            getattr(structural_step, att)[...] = getattr(self.file_info.structure, att)
        # structural_step.pos_dot = self.file_info.structure.pos_dot
        # structural_step.pos_ddot = self.file_info.structure.pos_ddot
        # structural_step.psi = self.file_info.structure.psi
        # structural_step.psi_dot = self.file_info.structure.psi_dot
        # structural_step.psi_ddot = self.file_info.structure.psi_ddot
#
        # structural_step.for_pos = self.file_info.structure.for_pos
        # structural_step.for_vel = self.file_info.structure.for_vel
        # structural_step.for_acc = self.file_info.structure.for_acc
        # structural_step.quat = self.file_info.structure.quat

        # Copy multibody information
        # mb_FoR_pos vel acc quat

        # Copy aero information
        attributes = ['zeta', 'zeta_star', 'normals',
                      'gamma', 'gamma_star',
                      'u_ext', 'u_ext_star',
                      'dynamic_forces', 'forces',]
                      # 'dist_to_orig', 'gamma_dot', 'zeta_dot',
        for att in attributes:
            for isurf in range(aero_step.n_surf):
                getattr(aero_step, att)[isurf][...] = getattr(self.file_info.aero, att)[isurf]

        return self.data
