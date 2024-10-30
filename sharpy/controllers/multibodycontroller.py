import numpy as np
import os

import sharpy.utils.controller_interface as controller_interface
import sharpy.utils.settings as settings


@controller_interface.controller
class MultibodyController(controller_interface.BaseController):
    r""" """

    controller_id = "MultibodyController"

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types["ang_history_input_file"] = "str"
    settings_default["ang_history_input_file"] = None
    settings_description["ang_history_input_file"] = "Route and file name of the time history of desired CRV rotation"

    settings_types["ang_vel_history_input_file"] = "str"
    settings_default["ang_vel_history_input_file"] = ""
    settings_description["ang_vel_history_input_file"] = ("Route and file name of the time history of desired CRV "
                                                          "velocity")

    settings_types["psi_dot_init"] = "list(float)"
    settings_default["psi_dot_init"] = [0., 0., 0.]
    settings_description["psi_dot_init"] = "Initial rotation velocity of hinge"

    settings_types["dt"] = "float"
    settings_default["dt"] = None
    settings_description["dt"] = "Time step of the simulation"

    settings_types["write_controller_log"] = "bool"
    settings_default["write_controller_log"] = True
    settings_description["write_controller_log"] = (
        "Write a time history of input, required input, " + "and control"
    )

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(
        settings_types, settings_default, settings_description, settings_options
    )

    def __init__(self):
        self.in_dict = None
        self.data = None
        self.settings = None

        self.prescribed_ang_time_history = None
        self.prescribed_ang_vel_time_history = None

        # Time histories are ordered such that the [i]th element of each
        # is the state of the controller at the time of returning.
        # That means that for the timestep i,
        # state_input_history[i] == input_time_history_file[i] + error[i]

        self.real_state_input_history = list()
        self.control_history = list()

        self.controller_implementation = None
        self.log = None

    def initialise(self, data, in_dict, controller_id=None, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(
            self.in_dict, self.settings_types, self.settings_default
        )

        self.settings = self.in_dict
        self.controller_id = controller_id

        # whilst PID control is not here implemented, I have left the remains for if it gets implemented in future
        if self.settings["write_controller_log"]:
            folder = data.output_folder + "/controllers/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.log = open(folder + self.controller_id + ".log.csv", "w+")
            self.log.write(
                ("#" + 1 * "{:>2}," + 6 * "{:>12}," + "{:>12}\n").format(
                    "tstep",
                    "time",
                    "Ref. state",
                    "state",
                    "Pcontrol",
                    "Icontrol",
                    "Dcontrol",
                    "control",
                )
            )
            self.log.flush()

        # save input time history
        try:
            self.prescribed_ang_time_history = np.loadtxt(
                self.settings["ang_history_input_file"], delimiter=","
            )
        except:
            try:
                self.prescribed_ang_time_history = np.load(
                    self.settings["ang_history_input_file"]
                )
            except:
                raise OSError(
                    "File {} not found in Controller".format(
                        self.settings["ang_history_input_file"]
                    )
                )

        if self.settings["ang_vel_history_input_file"]:
            try:
                self.prescribed_ang_vel_time_history = np.loadtxt(
                    self.settings["ang_vel_history_input_file"], delimiter=","
                )
            except:
                try:
                    self.prescribed_ang_vel_time_history = np.load(
                        self.settings["ang_vel_history_input_file"]
                    )
                except:
                    raise OSError(
                        "File {} not found in Controller".format(
                            self.settings["ang_vel_history_input_file"]
                        )
                    )

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

        control_command = self.prescribed_ang_time_history[data.ts - 1, :]

        if self.prescribed_ang_vel_time_history is None:
            if data.ts == 1:
                psi_dot = self.settings["psi_dot_init"]
            else:
                psi_dot = (
                    self.prescribed_ang_time_history[data.ts - 1, :]
                    - self.prescribed_ang_time_history[data.ts - 2, :]
                ) / self.settings["dt"]
        else:
            psi_dot = self.prescribed_ang_vel_time_history[data.ts - 1, :]

        if controlled_state["structural"].mb_prescribed_dict is None:
            controlled_state["structural"].mb_prescribed_dict = dict()
        controlled_state["structural"].mb_prescribed_dict[self.controller_id] = {
            "psi": control_command,
            "psi_dot": psi_dot,
        }
        controlled_state["structural"].mb_prescribed_dict[self.controller_id].update(
            {"delta_psi": control_command - self.prescribed_ang_time_history[0, :]}
        )

        return controlled_state, control_command

    def controller_wrapper(
        self, required_input, current_input, control_param, i_current
    ):
        self.controller_implementation.set_point(required_input[i_current - 1])
        control_param, detailed_control_param = self.controller_implementation(
            current_input[-1]
        )
        return control_param, detailed_control_param

    def __exit__(self, *args):
        self.log.close()
