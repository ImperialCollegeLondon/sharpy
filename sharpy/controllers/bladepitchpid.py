import numpy as np
import os
from control import ss, forced_response, TransferFunction

import sharpy.utils.controller_interface as controller_interface
import sharpy.utils.settings as settings
import sharpy.utils.control_utils as control_utils
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
from sharpy.utils.constants import deg2rad


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
    settings_description['lp_cut_freq'] = 'Cutting frequency of the low pass filter of the process value in Hz. Choose 0 for no filter'

    settings_types['anti_windup_lim'] = 'list(float)'
    settings_default['anti_windup_lim'] = [-1., -1.]
    settings_description['anti_windup_lim'] = ('Limits of actuation to apply anti windup.' +
                                              'Use the same number to deactivate.')

    # Set point parameters
    settings_types['sp_type'] = 'str'
    settings_default['sp_type'] = None
    settings_description['sp_type'] = (
            'Quantity used to define the' +
            ' set point')
    settings_options['sp_type'] = ['rbm', 'pitch', 'gen_vel']

    settings_types['sp_source'] = 'str'
    settings_default['sp_source'] = None
    settings_description['sp_source'] = (
            'Source used to define the' +
            ' set point')
    settings_options['sp_source'] = ['file', 'const']

    settings_types['sp_time_history_file'] = 'str'
    settings_default['sp_time_history_file'] = ''
    settings_description['sp_time_history_file'] = ('Route and file name of the time ' +
                                                    'history of the desired set point.' +
                                                    'Used for ``sp_source = file``')

    settings_types['sp_const'] = 'float'
    settings_default['sp_const'] = 0.
    settings_description['sp_const'] = 'Constant set point. Only used for ``sp_source`` = `const ``'

    # Other parameters
    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step of the simulation'

    settings_types['ntime_steps'] = 'int'
    settings_default['ntime_steps'] = None
    settings_description['ntime_steps'] = 'Number of time steps'

    #settings_types['tower_body'] = 'int'
    #settings_default['tower_body'] = 0
    #settings_description['tower_body'] = 'Body number of the tower'

    #settings_types['tower_top_node'] = 'int'
    #settings_default['tower_top_node'] = 0
    #settings_description['tower_top_node'] = 'Global node number of the tower top'

    settings_types['blade_num_body'] = 'list(int)'
    settings_default['blade_num_body'] = [0,]
    settings_description['blade_num_body'] = 'Body number of the blade(s) to pitch'

    settings_types['max_pitch_rate'] = 'float'
    settings_default['max_pitch_rate'] = 0.1396
    settings_description['max_pitch_rate'] = 'Maximum pitch rate [rad/s]'

    settings_types['pitch_sp'] = 'float'
    settings_default['pitch_sp'] = 0.
    settings_description['pitch_sp'] = 'Pitch set point [rad]'

    settings_types['initial_pitch'] = 'float'
    settings_default['initial_pitch'] = 0.
    settings_description['initial_pitch'] = 'Initial pitch [rad]'

    settings_types['initial_rotor_vel'] = 'float'
    settings_default['initial_rotor_vel'] = 0.
    settings_description['initial_rotor_vel'] = 'Initial rotor velocity [rad/s]'

    settings_types['min_pitch'] = 'float'
    settings_default['min_pitch'] = 0.
    settings_description['min_pitch'] = 'Minimum pitch [rad]'

    settings_types['max_pitch'] = 'float'
    settings_default['max_pitch'] = 1.5707963267948966
    settings_description['max_pitch'] = 'Maximum pitch [rad]'

    settings_types['nocontrol_steps'] = 'int'
    settings_default['nocontrol_steps'] = -1
    settings_description['nocontrol_steps'] = 'Time steps without control action'

    # Generator and drive train model
    settings_types['gen_model_const_var'] = 'str'
    settings_default['gen_model_const_var'] = ''
    settings_description['gen_model_const_var'] = 'Generator metric to be kept constant at a value `target_gen_value`'
    settings_options['gen_model_const_var'] = ['power', 'torque']

    settings_types['gen_model_const_value'] = 'float'
    settings_default['gen_model_const_value'] = 3945990.325
    settings_description['gen_model_const_value'] = 'Constant value of the generator metric to be kept constant '

    settings_types['GBR'] = 'float'
    settings_default['GBR'] = 97.
    settings_description['GBR'] = 'Gear box ratio'

    settings_types['inertia_dt'] = 'float'
    settings_default['inertia_dt'] = 43776046.25
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

        self.controller_implementation = None

        self.log_fname = None

    def initialise(self, data, in_dict, controller_id=None, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)

        self.settings = self.in_dict
        self.newmark_beta = 0.5 + self.settings['newmark_damp']

        if self.settings['write_controller_log']:
            folder = data.output_folder + '/controllers/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.log_fname = (folder + self.controller_id + ".dat")
            fid = open(self.log_fname, 'a')
            fid.write(('#'+ 1*'{:>2} ' + 10*'{:>12} ' + '{:>12}\n').
                    format('tstep', 'time', 'ref_state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'control', 'gen_torque', 'rotor_vel', 'pitch_vel', 'pitch'))
            fid.close()

        # save input time history
        if self.settings['sp_source'] == 'file':
            self.prescribed_sp_time_history = np.loadtxt(self.settings['sp_time_history_file'])
        # Init PID controller
        self.controller_implementation = control_utils.PID(self.settings['P'],
                                                           self.settings['I'],
                                                           self.settings['D'],
                                                           self.settings['dt'])

        if not self.settings['anti_windup_lim'][0] == self.settings['anti_windup_lim'][1]:
                self.controller_implementation.set_anti_windup_lim(self.settings['anti_windup_lim'])

        if self.settings['lp_cut_freq'] == 0.:
            self.filter_pv = False
        else:
            self.filter_pv = True
            w0 = self.settings['lp_cut_freq']*2*np.pi
            self.filter = TransferFunction(np.array([w0]), np.array([1., w0]))
            self.min_it_filter = int(1./(self.settings['lp_cut_freq']*self.settings['dt']))

        self.pitch = self.settings['initial_pitch']
        self.rotor_vel = self.settings['initial_rotor_vel']
        self.rotor_acc = 0.

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
        # TODO: move this to the initialisation with restart
        if len(self.system_pv) == 0:
            for it in range(data.ts - 1):
                self.system_pv.append(0.)
                self.prescribed_sp.append(0.)

        struct_tstep = controlled_state['structural']
        aero_tstep = controlled_state['aero']
        if not "info" in controlled_state:
            controlled_state['info'] = dict()

        time = self.settings['dt']*data.ts

        # Compute the rotor velocity
        aero_torque = self.compute_aero_torque(data.structure, struct_tstep)
        # if self.settings['variable_speed']:
        if True:
            #ielem, inode_in_elem = data.structure.node_master_elem[self.settings['tower_top_node']
            #node_cga = ag.quat2rotation(struct_tstep.mb_quat[self.settings['tower_body'], :])
            #cab = ag.crv2rotation(struct_tstep.psi[ielem, inode_in_elem, :])
            #FoR_cga = ag.quat2rotation(struct_tstep.mb_quat[self.settings['blade_num_body'][0], :])

            #ini_rotor_vel = ag.multiply_matrices(cab.T, node_cga.T, FoR_cga,
            #                                     struct_tstep.mb_FoR_vel[self.settings['blade_num_body'][0], 3:6])[2]
            #ini_rotor_acc = ag.multiply_matrices(cab.T, node_cga.T, FoR_cga,
            #                                     struct_tstep.mb_FoR_acc[self.settings['blade_num_body'][0], 3:6])[2]
            self.rotor_vel, self.rotor_acc = self.drive_train_model(aero_torque,
                                                                    self.rotor_vel,
                                                                    self.rotor_acc)
        else:
            self.rotor_vel = self.settings['sp_const']
            self.rotor_acc = 0.

        # System set point
        prescribed_sp = self.compute_prescribed_sp(time)
        # System process value
        sys_pv = self.compute_system_pv(struct_tstep,
                                        data.structure,
                                        gen_vel=self.rotor_vel*self.settings['GBR'])

        if data.ts < self.settings['nocontrol_steps']:
            sys_pv = prescribed_sp
            self.system_pv[-1] = sys_pv
            return controlled_state
        else:
            controlled_state['info']['rotor_vel'] = self.rotor_vel

        # Apply filter
        # Filter only after five periods of the cutoff frequency
        if self.filter_pv and (len(self.system_pv) > self.min_it_filter):
            nit = len(self.system_pv)
            time = np.linspace(0, (nit - 1)*self.settings['dt'], nit)
            # print(time.shape, len(self.system_pv))
            T, filtered_pv, xout = forced_response(self.filter,
                                        T=time,
                                        U=self.system_pv)
        else:
            filtered_pv = self.system_pv

        # get current state input
        delta_pitch_ref, detail = self.controller_wrapper(
                required_input=self.prescribed_sp,
                current_input=filtered_pv,
                control_param={'P': self.settings['P'],
                               'I': self.settings['I'],
                               'D': self.settings['D']},
                i_current=data.ts)
        # NREL controller does error = state - reference. Here is done the other way
        delta_pitch_ref *= -1.
        # Limit pitch and pitch rate
        target_pitch = delta_pitch_ref + self.settings['pitch_sp']
        pitch_rate = (target_pitch - self.pitch)/self.settings['dt']
        if pitch_rate < -self.settings['max_pitch_rate']:
            pitch_rate = -self.settings['max_pitch_rate']
        elif pitch_rate > self.settings['max_pitch_rate']:
            pitch_rate = self.settings['max_pitch_rate']

        next_pitch = self.pitch + pitch_rate*self.settings['dt']
        if next_pitch < self.settings['min_pitch']:
            pitch_rate = 0.
            next_pitch = self.settings['min_pitch']
        if next_pitch > self.settings['max_pitch']:
            pitch_rate = 0.
            next_pitch = self.settings['max_pitch']

        self.pitch = next_pitch

        controlled_state['info']['pitch_vel'] = -pitch_rate

        # Apply control order
        change_quat = False
        if change_quat:
            for ibody in self.settings['blade_num_body']:
                quat = algebra.rotate_quaternion(struct_tstep.mb_quat[ibody, :],
                                                 delta_pitch*np.array([1., 0., 0.]))
                struct_tstep.mb_quat[ibody, :] = quat.copy()
                if ibody == 0:
                    struct_tstep.quat = quat.copy()
        change_vel = False
        if change_vel:
            for ibody in self.settings['blade_num_body']:
                struct_tstep.mb_FoR_vel[ibody, 3] = pitch_rate
                struct_tstep.mb_FoR_acc[ibody, 3] = (data.structure.timestep_info[data.ts - 1].mb_FoR_vel[ibody, 3] -
                                                     struct_tstep.mb_FoR_vel[ibody, 3])/self.settings['dt']
                if ibody == 0:
                    struct_tstep.for_vel[3] = struct_tstep.mb_FoR_vel[ibody, 3]
                    struct_tstep.for_acc[3] = struct_tstep.mb_FoR_acc[ibody, 3]

            data.structure.dynamic_input[data.ts - 1]['for_vel'] = struct_tstep.for_vel.copy()
            data.structure.dynamic_input[data.ts - 1]['for_acc'] = struct_tstep.for_acc.copy()

        if False:
            data.aero.generate_zeta_timestep_info(struct_tstep,
                                              aero_tstep,
                                              data.structure,
                                              data.aero.aero_settings,
                                              dt=self.settings['dt'])

        fid = open(self.log_fname, 'a')
        fid.write(('{:>6d} '
                        + 10*'{:>12.6f} '
                        + '{:>12.6f}\n').format(data.ts,
                                                data.ts*self.settings['dt'],
                                                self.prescribed_sp[-1],
                                                self.system_pv[-1],
                                                detail[0],
                                                detail[1],
                                                detail[2],
                                                delta_pitch_ref,
                                                aero_torque/self.settings['GBR'],
                                                self.rotor_vel,
                                                pitch_rate,
                                                self.pitch))
        fid.close()
        return controlled_state


    def compute_prescribed_sp(self, time):
        """
            Compute the set point relevant for the controller
        """
        if self.settings['sp_source'] == 'file':
            sp = np.interp(time,
                           self.prescribed_sp_time_history[:, 0],
                           self.prescribed_sp_time_history[:, 1])
            self.prescribed_sp.append(sp)
        elif self.settings['sp_source'] == 'const':
            self.prescribed_sp.append(self.settings['sp_const'])

        return self.prescribed_sp[-1]


    def compute_system_pv(self, struct_tstep, beam, **kwargs):
        """
            Compute the process value relevant for the controller
        """
        if self.settings['sp_type'] == 'pitch':
            pitch = algebra.quat2euler(struct_tstep.mb_quat[self.settings['blade_num_body'][0]])[0]
            self.system_pv.append(pitch)
        elif self.settings['sp_type'] == 'rbm':
            steady, unsteady, grav = struct_tstep.extract_resultants(beam, force_type=['steady', 'unsteady', 'gravity'],
                                                                                       ibody=0)
            rbm = steady[4] + unsteady[4] + grav[4]
            self.system_pv.append(rbm)
        elif self.settings['sp_type'] == 'gen_vel':
            self.system_pv.append(kwargs['gen_vel'])

        return self.system_pv[-1]

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

    def drive_train_model(self, aero_torque, ini_rot_vel, ini_rot_acc):

        # Assuming contant generator torque demand
        if self.settings['gen_model_const_var'] == 'power':
            gen_torque = self.settings['gen_model_const_value']/self.settings['GBR']/ini_rot_vel
        elif self.settings['gen_model_const_var'] == 'torque':
            gen_torque = self.settings['gen_model_const_value']

        rot_acc = (aero_torque - self.settings['GBR']*gen_torque)/self.settings['inertia_dt']
        rot_vel = ini_rot_vel + rot_acc*self.settings['dt']

        return rot_vel, rot_acc

    def compute_aero_torque(self, beam, struct_tstep):
        # Compute total forces
        total_forces = np.zeros((6))
        for ibody in self.settings['blade_num_body']:
            steady, unsteady, grav = struct_tstep.extract_resultants(beam,
                                                      force_type=['steady', 'unsteady', 'grav'],
                                                      ibody=ibody)
            # total_forces += steady + unsteady + grav
            total_forces += steady + unsteady

        # Compute equivalent forces at hub position
        hub_elem = np.where(beam.body_number == self.settings['blade_num_body'][0])[0][0]
        hub_node = beam.connectivities[hub_elem, 0]
        hub_pos = struct_tstep.pos[hub_node, :]

        hub_forces = np.zeros((6))
        hub_forces[0:3] = total_forces[0:3].copy()
        hub_forces[3:6] = total_forces[3:6] - np.cross(hub_pos, total_forces[0:3])

        return hub_forces[5]

    @staticmethod
    def compute_blade_pitch(beam, struct_tstep, tower_ibody=0, blade_ibody=1):
        # Tower top
        tt_elem = np.where(beam.body_number == tower_ibody)[0][-1]
        tt_node = beam.connectivities[tt_elem, 1]
        ielem, inode_in_elem = beam.node_master_elem[tt_node]
        ca0b = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem, :])
        cga0 = algebra.quat2rotation(struct_tstep.mb_quat[tower_ibody, :])
        zg_tower_top = algebra.multiply_matrices(cga0, ca0b, np.array([0., 0., 1.]))

        # blade root
        cga = algebra.quat2rotation(struct_tstep.mb_quat[blade_ibody, :])
        zg_hub = algebra.multiply_matrices(cga, np.array([0., 0., 1.]))

        pitch = algebra.angle_between_vectors(zg_tower_top, zg_hub)
        return pitch
