import ctypes as ct
import numpy as np
import os
import scipy.interpolate as interpolate

import sharpy.utils.controller_interface as controller_interface
import sharpy.utils.settings as settings
import sharpy.utils.control_utils as control_utils
import sharpy.utils.cout_utils as cout
import sharpy.structure.utils.lagrangeconstraints as lc


@controller_interface.controller
class TakeOffTrajectoryController(controller_interface.BaseController):
    r"""


    """
    controller_id = 'TakeOffTrajectoryController'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['trajectory_input_file'] = 'str'
    settings_default['trajectory_input_file'] = None
    settings_description['trajectory_input_file'] = 'Route and file name of the trajectory file given as a csv with columns: time, x, y, z'

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step of the simulation'

    settings_types['trajectory_method'] = 'str'
    settings_default['trajectory_method'] = 'lagrange'
    settings_description['trajectory_method'] = (
            'Trajectory controller method. For now, "lagrange" is the supported option')

    settings_types['controlled_constraint'] = 'str'
    settings_default['controlled_constraint'] = None
    settings_description['controlled_constraint'] = ('Name of the controlled constraint in the multibody context' +
        ' Usually, it is something like `constraint_00`.')

    settings_types['controller_log_route'] = 'str'
    settings_default['controller_log_route'] = './output/'
    settings_description['controller_log_route'] = (
        'Directory where the log will be stored')

    settings_types['write_controller_log'] = 'bool'
    settings_default['write_controller_log'] = True
    settings_description['write_controller_log'] = (
        'Controls if the log from the controller is written or not.')

    settings_types['free_trajectory_structural_solver'] = 'str'
    settings_default['free_trajectory_structural_solver'] = ''
    settings_description['free_trajectory_structural_solver'] = (
        'If different than and empty string, the structural solver' +
        ' will be changed after the end of the trajectory has been reached')

    settings_types['free_trajectory_structural_substeps'] = 'int'
    settings_default['free_trajectory_structural_substeps'] = 0
    settings_description['free_trajectory_structural_substeps'] = (
        'Controls the structural solver' +
        ' structural substeps once the end of the trajectory has been reached')

    settings_types['initial_ramp_length_structural_substeps'] = 'int'
    settings_default['initial_ramp_length_structural_substeps'] = 10
    settings_description['initial_ramp_length_structural_substeps'] = (
        'Controls the number of timesteps that are used to increase the' +
        ' structural substeps from 0')

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types,
                                       settings_default,
                                       settings_description)

    def __init__(self):
        self.in_dict = None
        self.data = None
        self.settings = None

        self.input_history = None
        self.trajectory_interp = None
        self.trajectory_vel_interp = None
        self.t_limits = np.zeros((2,))

        self.controlled_body = None
        self.controlled_node = None

        self.log = None

    def initialise(self, data, in_dict, controller_id=None, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default)

        self.settings = self.in_dict
        self.controller_id = controller_id

        if self.settings['write_controller_log']:
            # TODO substitute for table writer in cout_utils.
            folder = data.output_folder + '/controllers/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.log = open(folder + self.controller_id + ".log.csv", "w+")
            self.log.write(('#'+ 1*'{:>2},' + 6*'{:>12},' + '{:>12}\n').
                    format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'control'))
            self.log.flush()

        # save input time history
        try:
            self.input_history = (
                np.loadtxt(
                    self.settings['trajectory_input_file'], delimiter=','))
        except OSError:
            raise OSError('File {} not found in {}'.format(
                self.settings['time_history_input_file'], self.controller_id))

        self.process_trajectory()

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
        # note: with or without the -1?
        time = (data.ts - 1)*self.settings['dt']
        i_current = data.ts

        try:
            constraint = controlled_state['structural'].\
                    mb_dict[self.settings['controlled_constraint']]
        except KeyError:
            return controlled_state
        except TypeError:
            import pdb
            pdb.set_trace()

        if self.controlled_body is None or self.controlled_node is None:
            self.controlled_body = constraint['body_number']
            self.controlled_node = constraint['node_number']

        # reset info to include only fresh info
        controlled_state['info'] = dict()

        # apply it where needed.
        traj_command, end_of_traj = self.controller_wrapper(time)
        if end_of_traj:
            lc.remove_constraint(controlled_state['structural'].mb_dict,
                                 self.settings['controlled_constraint'])

            if not self.settings['free_trajectory_structural_solver'] == '':
                controlled_state['info']['structural_solver'] = (
                    self.settings['free_trajectory_structural_solver'])

            controlled_state['info']['structural_substeps'] = (
                self.settings['free_trajectory_structural_substeps'])

            return controlled_state

        constraint['velocity'][:] = traj_command

        if self.settings['write_controller_log']:
            self.log.write(('{:>6d},'
                            + 3*'{:>12.6f},'
                            + '{:>12.6f}\n').format(i_current,
                                                    time,
                                                    traj_command[0],
                                                    traj_command[1],
                                                    traj_command[2]))

        if self.settings['initial_ramp_length_structural_substeps'] >= 0:
            if (i_current <
                    self.settings['initial_ramp_length_structural_substeps']):
                controlled_state['info']['structural_substeps'] = \
                        ct.c_int(i_current - 1)
            elif (i_current ==
                  self.settings['initial_ramp_length_structural_substeps']):
                controlled_state['info']['structural_substeps'] = None

        return controlled_state

    def process_trajectory(self, dxdt=True):
        """
        See https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.interpolate.UnivariateSpline.html
        """
        self.trajectory_interp = []
        # Make sure s = 0.5 is ok.
        self.t_limits[:] = (np.min(self.input_history[:, 0]),
                            np.max(self.input_history[:, 0]))
        for i_dim in range(3):
            self.trajectory_interp.append(
                interpolate.UnivariateSpline(self.input_history[:, 0],
                                             self.input_history[:, i_dim + 1],
                                             k=1,
                                             s=0.,
                                             ext='raise'))
        if dxdt:
            self.trajectory_vel_interp = []
            for i_dim in range(3):
                self.trajectory_vel_interp.append(
                    self.trajectory_interp[i_dim].derivative())

    def controller_wrapper(self, t):
        output_traj = np.zeros((3,))
        end_of_traj = False
        if self.settings['trajectory_method'] == 'lagrange':
            # check that t is in input limits
            if self.t_limits[0] <= t <= self.t_limits[1]:
                # return velocities
                for i_dim in range(3):
                    output_traj[i_dim] = self.trajectory_vel_interp[i_dim](t)
            else:
                for i_dim in range(3):
                    output_traj[i_dim] = np.nan
                    end_of_traj = True
        else:
            raise NotImplementedError('The trajectory_method ' +
                                      self.settings['trajectory_method'] +
                                      ' is not yet implemented.')
        return output_traj, end_of_traj

    def __exit__(self, *args):
        self.log.close()
