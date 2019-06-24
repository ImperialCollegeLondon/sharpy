import numpy as np
import h5py as h5
from sharpy.utils.solver_interface import solver, BaseSolver, initialise_solver
import sharpy.utils.settings as settings
import sharpy.linear.src.libss as libss
import sharpy.utils.algebra as algebra
import scipy.linalg as sclalg
import sharpy.utils.h5utils as h5utils
from sharpy.utils.datastructures import LinearTimeStepInfo
import pandas as pd


@solver
class LinearDynamicSimulation(BaseSolver):
    solver_id = 'LinDynamicSim'

    def __init__(self):

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_default['struct_states'] = 0
        self.settings_types['struct_states'] = 'int'

        self.settings_types['modal'] = 'bool'
        self.settings_default['modal'] = False

        self.settings_default['n_tsteps'] = 10
        self.settings_types['n_tsteps'] = 'int'

        self.settings_default['dt'] = 0.001
        self.settings_types['dt'] = 'float'
        
        self.settings_default['sys_id'] = ''
        self.settings_types['sys_id'] = 'str'

        self.settings_types['postprocessors'] = 'list(str)'
        self.settings_default['postprocessors'] = list()

        self.settings_types['postprocessors_settings'] = 'dict'
        self.settings_default['postprocessors_settings'] = dict()

        self.settings_types['output'] = 'str'
        self.settings_default['output'] = './cases/output/'

        self.data = None
        self.settings = dict()
        self.postprocessors = dict()
        self.with_postprocessors = False

        self.input_data_dict = dict()
        self.input_file_name = ""


    def initialise(self, data, custom_settings=None):

        self.data = data
        if custom_settings:
            self.settings = custom_settings
        else:
            self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # Read initial state and input data and store in dictionary
        self.read_files()

        # initialise postprocessors
        self.postprocessors = dict()
        if len(self.settings['postprocessors']) > 0:
            self.with_postprocessors = True
        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = initialise_solver(postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc])

    def run(self):

        dt = self.settings['dt'].value
        n_steps = self.settings['n_tsteps'].value
        T = n_steps*dt
        t_dom = np.linspace(0, T, n_steps)
        sys_id = self.settings['sys_id']

        x0 = self.input_data_dict['x0']
        u = self.input_data_dict['u']

        ss = self.data.linear.ss

        # Use the scipy linear solver
        sys = libss.ss_to_scipy(ss)
        out = sys.output(u, t=t_dom, x0=x0)

        t_out = out[0]
        x_out = out[2]
        y_out = out[1]

        process = True
        if process:

            # Pack state variables into linear timestep info
            for n in range(len(t_out)-1):
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

                if 'LinearUVLM' in self.data.linear.lsys:
                    x_aero = tstep.x[:self.data.linear.lsys['LinearUVLM'].ss.states]
                    # u_aero = TO DO
                    phi = self.data.linear.lsys[sys_id].lsys['LinearBeam'].sys.U
                    Kas = self.data.linear.lsys[sys_id].couplings['Kas']
                    u_q = tstep.u[:self.data.linear.lsys['LinearUVLM'].ss.inputs] + tstep.y[-self.data.linear.lsys[sys_id].lsys['LinearUVLM'].ss.inputs:]
                    u_aero = Kas.dot(sclalg.block_diag(phi, phi).dot(u_q))

                    # Unpack input
                    zeta, zeta_dot, u_ext = self.data.linear.lsys[sys_id].lsys['LinearUVLM'].unpack_input_vector(u_aero)


                    # Also add the beam forces. I have a feeling there is a minus there as well....
                    # Aero
                    forces, gamma, gamma_dot, gamma_star = self.data.linear.lsys[sys_id].lsys['LinearUVLM'].unpack_ss_vector(self.data,
                                                                                                                 x_n=x_aero,
                                                                                                                 aero_tstep=self.data.linear.tsaero0,
                                                                                                                             track_body=True)
                    current_aero_tstep = self.data.aero.timestep_info[-1].copy()
                    current_aero_tstep.forces = [forces[i_surf] + self.data.linear.tsaero0.forces[i_surf] for i_surf in range(len(gamma))]
                    current_aero_tstep.gamma = [gamma[i_surf] + self.data.linear.tsaero0.gamma[i_surf] for i_surf in range(len(gamma))]
                    current_aero_tstep.gamma_dot = [gamma_dot[i_surf] + self.data.linear.tsaero0.gamma_dot[i_surf] for i_surf in range(len(gamma))]
                    current_aero_tstep.gamma_star = [gamma_star[i_surf] + self.data.linear.tsaero0.gamma_star[i_surf] for i_surf in range(len(gamma))]
                    current_aero_tstep.zeta = zeta
                    current_aero_tstep.zeta_dot = zeta_dot
                    current_aero_tstep.u_ext = u_ext

                    self.data.aero.timestep_info.append(current_aero_tstep)

                # Structural states
                if 'BeamPlot' in self.postprocessors:
                    x_struct = tstep.x[-self.data.linear.lsys[sys_id].lsys['LinearBeam'].ss.states:]
                    aero_forces = self.data.linear.lsys['LinearUVLM'].C_to_vertex_forces.dot(x_aero)
                    beam_forces = self.data.linear.lsys[sys_id].couplings['Ksa'].dot(aero_forces)
                    u_struct = tstep.u[-self.data.linear.lsys[sys_id].lsys['LinearBeam'].ss.inputs:]
                    # y_struct = tstep.y[:self.data.linear.lsys[sys_id].lsys['LinearBeam'].ss.outputs]

                    # Reconstruct the state if modal
                    phi = self.data.linear.lsys['LinearBeam'].sys.U
                    x_s = sclalg.block_diag(phi, phi).dot(x_struct)
                    y_s = beam_forces #+ phi.dot(u_struct)
                    # y_s = self.data.linear.lsys['LinearBeam'].sys.U.T.dot(y_struct)

                    current_struct_step = self.data.linear.lsys[sys_id].lsys['LinearBeam'].unpack_ss_vector(x_s, y_s, self.data.linear.tsstruct0)
                    self.data.structure.timestep_info.append(current_struct_step)

                # run postprocessors
                if self.with_postprocessors:
                    for postproc in self.postprocessors:
                        self.data = self.postprocessors[postproc].run(online=True)

        export = True
        if export:
            y_pd = pd.DataFrame(data=y_out)
            x_pd = pd.DataFrame(data=x_out)
            t_pd = pd.DataFrame(data=t_out)
            u_pd = pd.DataFrame(data=u)
            y_pd.to_csv(self.data.settings['SHARPy']['route'] + '/output/y_out.csv')
            x_pd.to_csv(self.data.settings['SHARPy']['route'] + '/output/x_out.csv')
            t_pd.to_csv(self.data.settings['SHARPy']['route'] + '/output/t_out.csv')
            u_pd.to_csv(self.data.settings['SHARPy']['route'] + '/output/u_out.csv')
        return self.data

    def read_files(self):

        self.input_file_name = self.data.settings['SHARPy']['route'] + '/' + self.data.settings['SHARPy']['case'] + '.lininput.h5'

        # Check that the file exists
        try:
            h5utils.check_file_exists(self.input_file_name)
            # Read and store
            with h5.File(self.input_file_name, 'r') as input_file_handle:
                self.input_data_dict = h5utils.load_h5_in_dict(input_file_handle)
        except FileNotFoundError:
            pass


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


