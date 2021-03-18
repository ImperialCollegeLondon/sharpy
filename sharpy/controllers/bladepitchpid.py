import numpy as np
from control import ss, forced_response

import sharpy.utils.controller_interface as controller_interface
import sharpy.utils.settings as settings
import sharpy.utils.control_utils as control_utils
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra


@controller_interface.controller
class BladePitchPid(controller_interface.BaseController):
    r"""


    """
    controller_id = 'BladePitchPid'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    # PID parameters
    settings_types['P'] = 'float'
    settings_default['P'] = None
    settings_description['P'] = 'Proportional gain of the controller'

    settings_types['I'] = 'float'
    settings_default['I'] = 0.0
    settings_description['I'] = 'Integral gain of the controller'

    settings_types['D'] = 'float'
    settings_default['D'] = 0.0
    settings_description['D'] = 'Differential gain of the controller'

    # Filter
    settings_types['lp_cut_freq'] = 'float'
    settings_default['lp_cut_freq'] = 0.
    settings_description['lp_cut_freq'] = 'Cutting frequency of the low pass filter of the process value. Choose 0 for no filter'

    settings_types['anti_windup_lim'] = 'list(float)'
    settings_default['anti_windup_lim'] = None
    settings_description['anti_windup_lim'] = 'Limits of actuation to apply anti windup'

    # Set point parameters
    settings_types['sp_type'] = 'str'
    settings_default['sp_type'] = None
    settings_description['sp_type'] = (
            'Quantity used to define the' +
            ' set point')
    settings_options['sp_type'] = ['rbm', 'pitch']

    settings_types['sp_source'] = 'str'
    settings_default['sp_source'] = None
    settings_description['sp_source'] = (
            'Source used to define the' +
            ' set point')
    settings_options['sp_source'] = ['file']

    settings_types['sp_time_history_file'] = 'str'
    settings_default['sp_time_history_file'] = None
    settings_description['sp_time_history_file'] = ('Route and file name of the time ' +
                                                    'history of the desired set point.' +
                                                    'Used for ``sp_source = file``')

    # Other parameters
    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step of the simulation'

    settings_types['ntime_steps'] = 'int'
    settings_default['ntime_steps'] = None
    settings_description['ntime_steps'] = 'Number of time steps'

    settings_types['blade_num_body'] = 'list(int)'
    settings_default['blade_num_body'] = [0,]
    settings_description['blade_num_body'] = 'Body number of the blade(s) to pitch'

    settings_types['max_pitch_rate'] = 'float'
    settings_default['max_pitch_rate'] = 0.1396
    settings_description['max_pitch_rate'] = 'Maximum pitch rate [rad/s]'

    # Generator and drive train model
    settings_types['target_power'] = 'float'
    settings_default['target_power'] = 5e6
    settings_description['target_power'] = 'Target power in operation'

    settings_types['GBR'] = 'float'
    settings_default['GBR'] = 97.
    settings_description['GBR'] = 'Gear box ratio'

    settings_types['inertia_dt'] = 'float'
    settings_default['inertia_dt'] =
    settings_description['inertia_dt'] = 'Drive train inertia'

    settings_types['newmark_damp'] = 'float'
    settings_default['newmark_damp'] = 1e-4
    settings_description['newmark_damp'] = 'Damping of the time integration newmark-beta scheme'

    # Output parameters
    settings_types['write_controller_log'] = 'bool'
    settings_default['write_controller_log'] = True
    settings_description['write_controller_log'] = (
            'Write a time history of input, required input, ' +
            'and control')

    settings_types['controller_log_route'] = 'str'
    settings_default['controller_log_route'] = './output/'
    settings_description['controller_log_route'] = (
            'Directory where the log will be stored')

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types,
                                       settings_default,
                                       settings_description)

    def __init__(self):
        self.in_dict = None
        self.data = None
        self.settings = None

        self.prescribed_sp_time_history = None
        self.prescribed_sp = list()
        self.system_pv = list()
        # self.nblades = None

        # Time histories are ordered such that the [i]th element of each
        # is the state of the controller at the time of returning.
        # That means that for the timestep i,
        # state_input_history[i] == input_time_history_file[i] + error[i]
        # self.p_error_history = list()
        # self.i_error_history = list()
        # self.d_error_history = list()
        # self.real_sp_history = list()
        # self.control_history = list()

        self.controller_implementation = None

        # self.n_control_surface = 0

        self.log = None

    def initialise(self, in_dict, controller_id=None):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)

        self.settings = self.in_dict
        self.newmark_damp = 0.5 + self.settings['newmark_damp']
        # self.controller_id = controller_id

        # self.nblades = len(self.settings['blade_num_body'])

        if self.settings['write_controller_log']:
            self.log = open(self.settings['controller_log_route'] + '/' + self.controller_id + '.dat', 'w+')
            self.log.write(('#'+ 1*'{:>2},' + 6*'{:>12},' + '{:>12}\n').
                    format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'control'))
            self.log.flush()

        # save input time history
        if self.settings['sp_source'] == 'file':
            self.prescribed_sp_time_history = np.loadtxt(self.settings['sp_time_history_file'])
        # Init PID controller
        self.controller_implementation = control_utils.PID(self.settings['P'],
                                                           self.settings['I'],
                                                           self.settings['D'],
                                                           self.settings['dt'])

        self.controller_implementation.set_anti_windup_lim(self.settings['anti_windup_lim'])

        if self.settings['lp_cut_freq'] == 0.:
            self.filter_pv = False
        else:
            self.filter_pv = True
            alpha = np.exp(-self.settings['lp_cut_freq']*self.settings['dt'])
            self.filter = ss(alpha, 1.-alpha, alpha, 1.-alpha, self.settings['dt'])

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
        struct_tstep = controlled_state['structural']
        aero_tstep = controlled_state['aero']
        time = self.settings['dt']*data.ts

        prescribed_sp = self.compute_prescribed_sp(time)
        sys_pv = self.compute_system_pv(struct_tstep, data.structure)

        # Apply filter
        if self.filter_pv and (len(self.system_pv) > 1):
            nit = len(self.system_pv)
            time = np.linspace(0, (nit - 1)*self.settings['dt'], nit)
            # print(time.shape, len(self.system_pv))
            T, filtered_pv, xout = forced_response(self.filter,
                                        T=time,
                                        U=self.system_pv)
        else:
            filtered_pv = self.system_pv

        # get current state input
        # self.real_state_input_history.append(self.extract_time_history(controlled_state))

        # i_current = len(self.real_state_input_history)
        # # apply it where needed.
        control_command, detail = self.controller_wrapper(
                required_input=self.prescribed_sp,
                current_input=filtered_pv,
                control_param={'P': self.settings['P'],
                               'I': self.settings['I'],
                               'D': self.settings['D']},
                i_current=data.ts)

        if control_command < -self.settings['max_pitch_rate']:
            control_command = -self.settings['max_pitch_rate']
        elif control_command > self.settings['max_pitch_rate']:
            control_command = self.settings['max_pitch_rate']

        # Apply control order
        # rot_mat = algebra.rotation3d_x(control_command)
        print("control_command: ", control_command)
        # print("init euler: ", algebra.quat2euler(struct_tstep.quat))
        change_quat = False
        if change_quat:
            quat = algebra.rotate_quaternion(struct_tstep.quat, control_command*np.array([1., 0., 0.]))
            # print("final euler: ", algebra.quat2euler(quat))
            # euler = np.array([prescribed_sp[0], 0., 0.])
            struct_tstep.quat = quat
        change_vel = True
        if change_vel:
            # struct_tstep.for_vel[3] = control_command/self.settings['dt']
            # struct_tstep.for_acc[3] = (data.structure.timestep_info[data.ts - 1].for_vel[3] - struct_tstep.for_vel[3])/self.settings['dt']
            # struct_tstep.for_vel[3] = np.sign(control_command)*self.settings['max_pitch_rate']
            # struct_tstep.for_acc[3] = (data.structure.timestep_info[data.ts - 1].for_vel[3] - struct_tstep.for_vel[3])/self.settings['dt']
            struct_tstep.for_vel[3] = control_command
            struct_tstep.for_acc[3] = (data.structure.timestep_info[data.ts - 1].for_vel[3] - struct_tstep.for_vel[3])/self.settings['dt']

            data.structure.dynamic_input[data.ts - 1]['for_vel'] = struct_tstep.for_vel.copy()
            data.structure.dynamic_input[data.ts - 1]['for_acc'] = struct_tstep.for_acc.copy()

        if False:
            data.aero.generate_zeta_timestep_info(struct_tstep,
                                              aero_tstep,
                                              data.structure,
                                              data.aero.aero_settings,
                                              dt=self.settings['dt'])
        # controlled_state['aero'].control_surface_deflection = (
        #     np.array(self.settings['controlled_surfaces_coeff'])*control_command)

        self.log.write(('{:>6d},'
                        + 6*'{:>12.6f},'
                        + '{:>12.6f}\n').format(data.ts,
                                                data.ts*self.settings['dt'],
                                                self.prescribed_sp[-1],
                                                self.system_pv[-1],
                                                detail[0],
                                                detail[1],
                                                detail[2],
                                                control_command))
        return controlled_state


    def compute_prescribed_sp(self, time):
        """
            Compute the set point relevant for the controller
        """
        if self.settings['sp_source'] == 'file':
            sp = np.interp(time,
                           self.prescribed_sp_time_history[:, 0],
                           self.prescribed_sp_time_history[:, 1])
            # vel = np.interp(time,
            #                    self.prescribed_sp_time_history[:, 0],
            #                    self.prescribed_sp_time_history[:, 2])
            # acc = np.interp(time,
            #                    self.prescribed_sp_time_history[:, 0],
            #                    self.prescribed_sp_time_history[:, 3])
            # self.prescribed_sp.append(np.array([pitch, vel, acc]))
            self.prescribed_sp.append(sp)
        return self.prescribed_sp[-1]


    def compute_system_pv(self, struct_tstep, beam):
        """
            Compute the process value relevant for the controller
        """
        if self.settings['sp_type'] == 'pitch':
            pitch = algebra.quat2euler(struct_tstep.quat)[0]
            self.system_pv.append(pitch)
        elif self.settings['sp_type'] == 'rbm':
            steady, unsteady, grav = struct_tstep.extract_resultants(beam, force_type=['steady', 'unsteady', 'gravity'],
                                                                                       ibody=0)
            # rbm = np.linalg.norm(steady[3:6] + unsteady[3:6] + grav[3:6])
            rbm = steady[4] + unsteady[4] + grav[4]
            self.system_pv.append(rbm)
            print("rbm: ", rbm)

        return self.system_pv[-1]
    # def extract_time_history(self, controlled_state):
    #     output = 0.0
    #     if self.settings['input_type'] == 'pitch':
    #         step = controlled_state['structural']
    #         euler = step.euler_angles()

    #         output = euler[1]
    #     elif self.settings['input_type'] == 'roll':
    #         step = controlled_state['structural']
    #         euler = step.euler_angles()

    #         output = euler[0]
    #     elif 'pos_z(' in self.settings['input_type']:
    #         node_str = self.settings['input_type']
    #         node_str = node_str.replace('pos(', '')
    #         node_str = node_str.replace(')', '')
    #         node = int(node_str)
    #         step = controlled_state['structural']
    #         pos = step.pos[node, :]

    #         output = pos[2]
    #     else:
    #         raise NotImplementedError(
    #             "input_type {} is not yet implemented in extract_time_history()"
    #             .format(self.settings['input_type']))
    #     return output

    def controller_wrapper(self,
                           required_input,
                           current_input,
                           control_param,
                           i_current):
        self.controller_implementation.set_point(required_input[i_current - 1])
        control_param, detailed_control_param = self.controller_implementation(current_input[-1])
        return (control_param, detailed_control_param)

    def __exit__(self, *args):
        # self.log.close()
        pass

    def generator_model(aero_torque, ini_rot_vel, ini_rot_acc):

        gen_torque = self.settings['target_power']*ini_rot_vel

        delta_rot_acc = (aero_torque - self.settings['GBR']*gen_torque)/self.settings['inertia_dt']
        rot_acc = ini_tor_acc + delta_rot_acc

        # Integrate according to newmark-beta scheme
        rot_vel = (ini_rot_vel +
                   (1. - self.newmark_beta)*self.settings['dt'] +
                   self.newmark_beta*self.settings['dt']*rot_acc)

        return rot_vel, rot_acc
