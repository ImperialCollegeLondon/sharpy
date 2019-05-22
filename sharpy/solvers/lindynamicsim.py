import numpy as np
import h5py as h5
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.linear.src.libss as libss
import sharpy.utils.algebra as algebra
import scipy.linalg as sclalg
import sharpy.utils.h5utils as h5utils
from sharpy.utils.datastructures import LinearTimeStepInfo


@solver
class LinearDynamicSimulation(BaseSolver):
    solver_id = 'LinearDynamicSimulation'

    def __init__(self):

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_default['struct_states'] = 0
        self.settings_types['struct_states'] = 'int'

        self.settings_types['modal'] = 'bool'
        self.settings_default['modal'] = False

        self.settings_default['time_final'] = 10.0
        self.settings_types['time_final'] = 'float'

        self.settings_default['dt'] = 0.001
        self.settings_types['dt'] = 'float'

        self.data = None
        self.settings = dict()

        self.input_data_dict = dict()
        self.input_file_name = ""


    def initialise(self, data, custom_settings=None):

        self.data = data
        if custom_settings:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # Read initial state and input data and store in dictionary
        self.read_files()

    def run(self):

        T = self.settings['time_final']
        dt = self.settings['dt']
        n_steps = int(T/dt)
        t_dom = np.linspace(0, T, n_steps)

        x0 = self.input_data_dict['x0']
        try:
            u = self.input_data_dict['U']
        except KeyError:
            u = np.zeros((n_steps, len(x0)))

        ss = self.data.linear.ss

        # Use the scipy linear solver
        sys = libss.ss_to_scipy(ss)
        out = sys.output(u, t_dom, x0)

        t_out = out[0]
        x_out = out[2]
        y_out = out[1]

        # Pack state variables into linear timestep info
        for n in range(len(t_out)):
            tstep = LinearTimeStepInfo()
            tstep.x = x_out[n, :]
            tstep.y = y_out[n, :]
            tstep.t = t_out[n]
            tstep.u = u[n, :]
            self.data.linear.timestep_info.append(tstep)
            # TODO: option to save to h5

            # Pack variables into respective aero or structural time step infos (with the + f0 from lin)
            # Need to obtain information from the variables in a similar fashion as done with the database
            # for the beam case

    def read_files(self):

        self.input_file_name = self.data.case_root + '/' + self.data.case_name + '.lininput.h5'

        # Check that the file exists
        h5utils.check_file_exists(self.input_file_name)

        # Read and store
        with h5.File(self.input_file_name, 'r') as input_file_handle:
            self.input_data_dict = h5utils.load_h5_in_dict(input_file_handle)

    def structural_state_to_timestep(self, x):
        """
        Convert SHARPy beam state to original beam structure for plotting purposes.

        Args:
            x:

        Returns:

        """

        tstep = self.data.structure.timestep_info[0].copy()

        num_node = tstep.num_node

        q = x[:len(x)//2]
        dq = x[len(x)//2:]

        coords_a = np.array([q[6*i_node + [0, 1, 2]] for i_node in range(num_node-1)])
        crv = np.array([q[6*i_node + [3, 4, 5]] for i_node in range(num_node-1)])
        for_vel = dq[6*num_node:6*num_node+6]

        orient = dq[6*num_node:6*num_node+6:]
        if len(orient) == 3:
            quat = algebra.euler2quat(orient)
        else:
            quat = orient


