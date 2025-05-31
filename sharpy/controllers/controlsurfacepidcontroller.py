import numpy as np
import os

import sharpy.utils.controller_interface as controller_interface
import sharpy.utils.settings as settings
import sharpy.utils.control_utils as control_utils
import sharpy.utils.cout_utils as cout


@controller_interface.controller
class ControlSurfacePidController(controller_interface.BaseController):
    r"""


    """
    controller_id = 'ControlSurfacePidController'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['time_history_input_file'] = 'str'
    settings_default['time_history_input_file'] = None
    settings_description['time_history_input_file'] = 'Route and file name of the time history of desired state'

    settings_types['P'] = 'float'
    settings_default['P'] = None
    settings_description['P'] = 'Proportional gain of the controller'

    settings_types['I'] = 'float'
    settings_default['I'] = 0.0
    settings_description['I'] = 'Integral gain of the controller'

    settings_types['D'] = 'float'
    settings_default['D'] = 0.0
    settings_description['D'] = 'Differential gain of the controller'

    settings_types['input_type'] = 'str'
    settings_default['input_type'] = None
    settings_description['input_type'] = (
            'Quantity used to define the' +
            ' reference state. Supported: `pitch`')

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step of the simulation'

    settings_types['controlled_surfaces'] = 'list(int)'
    settings_default['controlled_surfaces'] = None
    settings_description['controlled_surfaces'] = (
            'Control surface indices to be actuated by this controller')

    settings_types['controlled_surfaces_coeff'] = 'list(float)'
    settings_default['controlled_surfaces_coeff'] = [1.]
    settings_description['controlled_surfaces_coeff'] = (
            'Control surface deflection coefficients. ' +
            'For example, for antisymmetric deflections => [1, -1].')

    settings_types['write_controller_log'] = 'bool'
    settings_default['write_controller_log'] = True
    settings_description['write_controller_log'] = (
            'Write a time history of input, required input, ' +
            'and control')

    supported_input_types = ['pitch', 'roll', 'pos_']

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types,
                                       settings_default,
                                       settings_description)

    def __init__(self):
        self.in_dict = None
        self.data = None
        self.settings = None

        self.prescribed_input_time_history = None

        # Time histories are ordered such that the [i]th element of each
        # is the state of the controller at the time of returning.
        # That means that for the timestep i,
        # state_input_history[i] == input_time_history_file[i] + error[i]
        self.p_error_history = list()
        self.i_error_history = list()
        self.d_error_history = list()
        self.real_state_input_history = list()
        self.control_history = list()

        self.controller_implementation = None

        self.n_control_surface = 0

        self.log = None

    def initialise(self, data, in_dict, controller_id=None, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default)

        self.settings = self.in_dict
        self.controller_id = controller_id

        # validate that the input_type is in the supported ones
        valid = False
        for t in self.supported_input_types:
            if t in self.settings['input_type']:
                valid = True
                break
        if not valid:
            cout.cout_wrap('The input_type {} is not supported by {}'.format(
                self.settings['input_type'], self.controller_id), 3)
            cout.cout_wrap('The supported ones are:', 3)
            for i in self.supported_input_types:
                cout.cout_wrap('    {}'.format(i), 3)
            raise NotImplementedError()

        if self.settings['write_controller_log']:
            folder = data.output_folder + '/controllers/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.log = open(folder + self.controller_id + ".log.csv", "w+")
            self.log.write(('#'+ 1*'{:>2},' + 6*'{:>12},' + '{:>12}\n').
                    format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'control'))
            self.log.flush()

        # save input time history
        try:
            self.prescribed_input_time_history = (
                np.loadtxt(self.settings['time_history_input_file'], delimiter=','))
        except OSError:
            raise OSError('File {} not found in Controller'.format(self.settings['time_history_input_file']))

        # Init PID controller
        self.controller_implementation = control_utils.PID(self.settings['P'],
                                                           self.settings['I'],
                                                           self.settings['D'],
                                                           self.settings['dt'])

        # check that controlled_surfaces_coeff has the correct number of parameters
        # if len() == 1 and == 1.0, then expand to number of surfaces.
        # if len(coeff) /= n_surfaces, throw error
        self.n_control_surface = len(self.settings['controlled_surfaces'])
        if (len(self.settings['controlled_surfaces_coeff']) ==
            self.n_control_surface):
            # All good, pass checks
            pass
        elif (len(self.settings['controlled_surfaces_coeff']) == 1 and
            self.settings['controlled_surfaces_coeff'][0] == 1.0):
            # default value, fill with 1.0
            self.settings['controlled_surfaces_coeff'] = np.ones(
                    (self.n_control_surface,), dtype=float)
        else:
            raise ValueError('controlled_surfaces_coeff does not have as many'
                    + ' elements as controller_surfaces')

    def control(self, data, controlled_state):
        r"""
        Main routine of the controller.
        Input is `data` (the self.data in the solver), and
        `currrent_state` which is a dictionary with ['structural', 'aero']
        time steps for the current iteration.

        :param data: problem data containing all the information.
        :param controlled_state: `dict` with two vars: `structural` and `aero`
            containing the `timestep_info` that will be returned with the
            control variables.

        :returns: A `dict` with `structural` and `aero` time steps and control
            input included.
        """

        # get current state input
        self.real_state_input_history.append(self.extract_time_history(controlled_state))

        i_current = len(self.real_state_input_history)
        # apply it where needed.
        control_command, detail = self.controller_wrapper(
                required_input=self.prescribed_input_time_history,
                current_input=self.real_state_input_history,
                control_param={'P': self.settings['P'],
                               'I': self.settings['I'],
                               'D': self.settings['D']},
                i_current=i_current)

        controlled_state['aero'].control_surface_deflection = (
            np.array(self.settings['controlled_surfaces_coeff'])*control_command)

        self.log.write(('{:>6d},'
                        + 6*'{:>12.6f},'
                        + '{:>12.6f}\n').format(i_current,
                                                i_current*self.settings['dt'],
                                                self.prescribed_input_time_history[i_current - 1],
                                                self.real_state_input_history[i_current - 1],
                                                detail[0],
                                                detail[1],
                                                detail[2],
                                                control_command))
        return controlled_state

    def extract_time_history(self, controlled_state):
        output = 0.0
        if self.settings['input_type'] == 'pitch':
            step = controlled_state['structural']
            euler = step.euler_angles()

            output = euler[1]
        elif self.settings['input_type'] == 'roll':
            step = controlled_state['structural']
            euler = step.euler_angles()

            output = euler[0]
        elif 'pos_z(' in self.settings['input_type']:
            node_str = self.settings['input_type']
            node_str = node_str.replace('pos(', '')
            node_str = node_str.replace(')', '')
            node = int(node_str)
            step = controlled_state['structural']
            pos = step.pos[node, :]

            output = pos[2]
        else:
            raise NotImplementedError(
                "input_type {} is not yet implemented in extract_time_history()"
                .format(self.settings['input_type']))
        return output

    def controller_wrapper(self,
                           required_input,
                           current_input,
                           control_param,
                           i_current):
        self.controller_implementation.set_point(required_input[i_current - 1])
        control_param, detailed_control_param = self.controller_implementation(current_input[-1])
        return (control_param, detailed_control_param)

    def __exit__(self, *args):
        self.log.close()
