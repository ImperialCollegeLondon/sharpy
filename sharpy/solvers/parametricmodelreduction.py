from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.rom.interpolation.pmorlibrary as pmorlibrary


@solver
class ParametricModelReduction(BaseSolver):
    """
    Warnings:
        Under development

    Standalone solver to

    * Load parametric ROM libraries

    * Create parametric ROMs

    * Interpolate ROMs

    Notes:
        This solver can be used as a standalone solver. I.e it could be the only
        solver in the SHARPy ``flow`` variable.

    See Also:
        How to create pROM libraries in :class:`~sharpy.rom.interpolation.pmorlibrary.ROMLibrary`
    """
    solver_id = 'ParametricModelOrderReduction'
    solver_classification = 'model reduction'

    settings_types = dict()
    settings_description = dict()
    settings_default = dict()
    settings_options = dict()

    settings_types['library_filepath'] = 'str'
    settings_default['library_filepath'] = ''
    settings_description['library_filepath'] = 'Filepath to .pkl file containing pROM library.'

    settings_types['projection_method'] = 'str'
    settings_default['projection_method'] = None
    settings_description['projection_method'] = 'Projection method employed in the transformation of the ' \
                                                'reduced bases to a set of generalised coordinates.'
    settings_options['projection_method'] = ['leastsq',
                                             'strongMAC',
                                             'strongMAC_BT',
                                             'maraniello_BT',
                                             'weakMAC_right_orth',
                                             'weakMAC']

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None

        self.rom_library = None

    def initialise(self, data):

        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 self.settings_options)

        self.rom_library = pmorlibrary.ROMLibrary()

        if self.settings['library_filepath'] is '':
            self.rom_library.interface()
        else:
            self.rom_library.load_library(path=self.settings['library_filepath'])

    def run(self):
        pass
