from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings


@solver
class BeamLoader(BaseSolver):
    solver_id = 'BeamLoader'

    def __init__(self):
        # settings list
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['unsteady'] = 'bool'
        self.settings_default['unsteady'] = True

        self.settings_types['test_list'] = 'list(str)'
        self.settings_default['test_list'] = ['a', 'b']

        self.data = None
        self.settings = None
        self.fem_file_name = ''

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]

        # init settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # open fem file
        self.fem_file_name = self.data.case_route + '/' + self.data.case_name + '.fem.h5'


    def run(self):
        pass
