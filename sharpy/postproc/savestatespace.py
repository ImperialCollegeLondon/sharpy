import os

import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import sharpy.utils.frequencyutils as frequencyutils


@solver_interface.solver
class SaveStateSpace(solver_interface.BaseSolver):
    """
    Save Linear State Space system to h5
    """
    solver_id = 'SaveStateSpace'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder.'

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = False
    settings_description['print_info'] = 'Write output to screen.'

    settings_types['target_system'] = 'list(str)'
    settings_default['target_system'] = ['aeroelastic']
    settings_description['target_system'] = 'System or systems for which to find frequency response.'
    settings_options['target_system'] = ['aeroelastic', 'aerodynamic', 'structural']

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.settings = None
        self.folder = None

        self.print_info = False

        self.data = None

    def initialise(self, data, custom_settings=None):

        self.data = data

        if not custom_settings:
            self.settings = self.data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 self.settings_options,
                                 no_ctype=True)

        self.print_info = self.settings['print_info']

        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = os.path.abspath(self.settings['folder']) + '/' + self.data.settings['SHARPy']['case'] + '/statespace/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run(self, ss=None, ss_name=None):

        filename = self.folder + '/'

        if ss is not None:
            if ss_name is not None:
                filename += ss_name + '.'

            filename += 'statespace.h5'
            if self.print_info:
                cout.cout_wrap('Saving state-space object to')
                cout.cout_wrap('\t{:s}'.format(filename), 1)
            ss.save(filename)
        else:
            for system_name in self.settings['target_system']:
                ss = frequencyutils.find_target_system(self.data, system_name)
                file = filename + system_name + '.statespace.h5'
                if self.print_info:
                    cout.cout_wrap('Saving state-space object to')
                    cout.cout_wrap('\t{:s}'.format(file), 1)
                ss.save(file)
