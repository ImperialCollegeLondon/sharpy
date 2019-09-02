import sharpy.linear.utils.ss_interface as ssinterface
import sys
import importlib
import sharpy.utils.settings as settings

@ssinterface.linear_system
class LinearCustom(ssinterface.BaseElement):
    sys_id = 'LinearCustom'

    def __init__(self):

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_default['solver_path'] = ''
        self.settings_types['solver_path'] = 'str'

        self.settings_default['solver_name'] = 'generate_ss'
        self.settings_types['solver_name'] = 'str'

        self.data = None
        self.lsys = dict()
        self.uvlm = None
        self.beam = None
        self.ss = None

        self.settings = dict()
        self.state_variables = None
        self.input_variables = None
        self.output_variables = None

    def initialise(self, data, custom_settings=None):

        self.data = data

        if custom_settings:
            self.settings = custom_settings
        else:
            try:
                self.settings = self.data.settings['LinearAssembler']['linear_system_settings']  # Load settings, the settings should be stored in data.linear.settings
            except KeyError:
                pass

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def assemble(self):


        if self.settings['solver_path']:
            custom_solver_path = self.settings['solver_path'] + '/'
        else:
            custom_solver_path = self.data.settings['SHARPy']['route'] + '/'

        solver_name = self.settings['solver_name']
        # solver_name_and_path = custom_solver_path + '/' + solver_name
        # solver_name_and_path = solver_name_and_path.replace('/', '.')
        sys.path.append(custom_solver_path)
        custom_solver = importlib.import_module(solver_name, solver_name)

        custom_solver_output = custom_solver.run(self.data)

        self.data = custom_solver_output['data']
        ss = custom_solver_output['ss']
        self.state_variables = custom_solver_output.get('state_variables', None)
        self.input_variables = custom_solver_output.get('input_variables', None)
        self.output_variables = custom_solver_output.get('output_variables', None)
        self.lsys = custom_solver_output.get('lsys', dict())
        self.uvlm = custom_solver_output.get('uvlm', None)
        self.beam = custom_solver_output.get('beam', None)

        return ss
