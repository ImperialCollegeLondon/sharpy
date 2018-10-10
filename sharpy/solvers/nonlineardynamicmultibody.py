import ctypes as ct
import numpy as np


from sharpy.utils.settings import str2bool
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
#from sharpy.solvers.nonlineardynamicprescribedstep import NonLinearDynamicPrescribedStep
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout

from IPython import embed
import scipy.linalg
import sharpy.structure.utils.xbeamlib as xbeamlib
import sharpy.utils.algebra as algebra
import h5py as h5
import sharpy.utils.h5utils as h5utils
import sharpy.utils.multibody as mb
import ipdb
import sharpy.utils.utils_ams as uams
import sharpy.utils.lagrangemultipliers as lagrangemultipliers
import matplotlib.pyplot as plt

@solver
class NonLinearDynamicMultibody(BaseSolver):
    solver_id = 'NonLinearDynamicMultibody'

    def __init__(self):

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['max_iterations'] = 'int'
        self.settings_default['max_iterations'] = 100

        self.settings_types['num_load_steps'] = 'int'
        self.settings_default['num_load_steps'] = 5

        self.settings_types['delta_curved'] = 'float'
        self.settings_default['delta_curved'] = 1e-5

        self.settings_types['min_delta'] = 'float'
        self.settings_default['min_delta'] = 1e-5

        self.settings_types['newmark_damp'] = 'float'
        self.settings_default['newmark_damp'] = 1e-4

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.01

        self.settings_types['num_steps'] = 'int'
        self.settings_default['num_steps'] = 500

        self.settings_types['gravity_on'] = 'bool'
        self.settings_default['gravity_on'] = False

        self.settings_types['gravity'] = 'float'
        self.settings_default['gravity'] = 9.81

        self.data = None
        self.settings = None

        # Total number of unknowns in the Multybody sistem
        self.sys_size = None

        # Total number of equations associated to the Lagrange multipliers
        self.num_LM_eq = None

    def initialise(self, data, custom_settings=None):

        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # load info from dyn dictionary
        self.data.structure.add_unsteady_information(self.data.structure.dyn_dict, self.settings['num_steps'].value)

        # initialise postprocessors
        self.postprocessors = dict()
        if len(self.settings['postprocessors']) > 0:
            self.with_postprocessors = True
        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = solver_interface.initialise_solver(postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc])

        # Define Newmark constants
        self.gamma = 0.5 + self.settings['newmark_damp'].value
        self.beta = 0.25*(self.gamma + 0.5)*(self.gamma + 0.5)

        # Define the number of equations
        self.num_LM_eq = lagrangemultipliers.define_num_LM_eq(self.data.structure.mb_dict)

        # Define the number of dofs
        self.define_sys_size()

    def add_step(self):
        self.data.structure.next_step()

    def next_step(self):
        pass

    def define_sys_size(self):

        MBdict = self.data.structure.mb_dict
        self.sys_size = self.data.structure.num_dof.value

        for ibody in range(self.data.structure.num_bodies):
            if (MBdict['body_%02d' % ibody]['FoR_movement'] == 'free'):
                self.sys_size += 10

    def assembly_MB_eq_system(self, MB_beam, MB_tstep, dt, Lambda, Lambda_dot):

        #print("Lambda: ", Lambda)
        #print("LambdaDot: ", Lambda_dot)
        MBdict = self.data.structure.mb_dict
        MB_M = np.zeros((self.sys_size+self.num_LM_eq, self.sys_size+self.num_LM_eq), dtype=ct.c_double, order='F')
        MB_C = np.zeros((self.sys_size+self.num_LM_eq, self.sys_size+self.num_LM_eq), dtype=ct.c_double, order='F')
        MB_K = np.zeros((self.sys_size+self.num_LM_eq, self.sys_size+self.num_LM_eq), dtype=ct.c_double, order='F')
        MB_Asys = np.zeros((self.sys_size+self.num_LM_eq, self.sys_size+self.num_LM_eq), dtype=ct.c_double, order='F')
        MB_Q = np.zeros((self.sys_size+self.num_LM_eq,), dtype=ct.c_double, order='F')
        #ipdb.set_trace()
        first_dof = 0
        last_dof = 0
        # Loop through the different bodies
        for ibody in range(len(MB_beam)):

            # Initialize matrices
            M = None
            C = None
            K = None
            Q = None

            # Generate the matrices for each body
            if MB_beam[ibody].FoR_movement == 'prescribed':
                last_dof = first_dof + MB_beam[ibody].num_dof.value
                M, C, K, Q = xbeamlib.cbeam3_asbly_dynamic(MB_beam[ibody], MB_tstep[ibody], self.settings)

            elif MB_beam[ibody].FoR_movement == 'free':
                last_dof = first_dof + MB_beam[ibody].num_dof.value + 10
                M, C, K, Q = xbeamlib.xbeam3_asbly_dynamic(MB_beam[ibody], MB_tstep[ibody], self.settings)


            ############### Assembly into the global matrices
            # Flexible and RBM contribution to Asys
            MB_M[first_dof:last_dof, first_dof:last_dof] = M.astype(dtype=ct.c_double, copy=True, order='F')
            MB_C[first_dof:last_dof, first_dof:last_dof] = C.astype(dtype=ct.c_double, copy=True, order='F')
            MB_K[first_dof:last_dof, first_dof:last_dof] = K.astype(dtype=ct.c_double, copy=True, order='F')

            #Q
            MB_Q[first_dof:last_dof] = Q

            first_dof = last_dof


        # Generate matrices associated to Lagrange multipliers
        LM_C, LM_K, LM_Q = lagrangemultipliers.generate_lagrange_matrix(MBdict, MB_beam, MB_tstep, self.num_LM_eq, self.sys_size, dt, Lambda, Lambda_dot)

        #LM_C, LM_K, LM_Q = self.generate_lagrange_matrix(MB_beam, MB_tstep, dt, Lambda, Lambda_dot)

        # Include the matrices associated to Lagrange Multipliers
        MB_C += LM_C
        MB_K += LM_K
        MB_Q += LM_Q

        MB_Asys = MB_K + MB_C*self.gamma/(self.beta*dt) + MB_M/(self.beta*dt*dt)
        # sys_size = self.sys_size
        # MB_Asys[:sys_size,:sys_size] = MB_K[:sys_size,:sys_size] + MB_C[:sys_size,:sys_size]*self.gamma/(self.beta*dt) + MB_M[:sys_size,:sys_size]/(self.beta*dt*dt)
        # # MB_Asys[sys_size:,:] = MB_K[sys_size:,:] + MB_C[sys_size:,:] + MB_M[sys_size:,:]
        # # MB_Asys[:,sys_size:] = MB_K[:,sys_size:] + MB_C[:,sys_size:] + MB_M[:,sys_size:]
        # MB_Asys[sys_size:,:] = MB_C[sys_size:,:]
        # MB_Asys[:,sys_size:] = MB_C[:,sys_size:]

        return MB_Asys, MB_Q

    def integrate_position(self, MB_beam, MB_tstep, dt):

        vel = np.zeros((6,),)
        acc = np.zeros((6,),)
        for ibody in range(len(MB_tstep)):
            # I think this is the right way to do it, but to make it match the rest I change it temporally
            if False:
                MB_tstep[ibody].mb_quat[ibody,:] =  algebra.quaternion_product(MB_tstep[ibody].quat, MB_tstep[ibody].mb_quat[ibody,:])
                acc[0:3] = (0.5-self.beta)*np.dot(MB_beam[ibody].timestep_info.cga(),MB_beam[ibody].timestep_info.for_acc[0:3])+self.beta*np.dot(MB_tstep[ibody].cga(),MB_tstep[ibody].for_acc[0:3])
                vel[0:3] = np.dot(MB_beam[ibody].timestep_info.cga(),MB_beam[ibody].timestep_info.for_vel[0:3])
                MB_tstep[ibody].for_pos[0:3] = dt*(vel[0:3] + dt*acc[0:3])
            else:
                # print("quat: ", MB_tstep[ibody].quat)
                # print("cga: ", MB_tstep[ibody].cga())
                # print("for_vel: ", MB_tstep[ibody].for_vel)
                MB_tstep[ibody].for_pos[0:3] += dt*np.dot(MB_tstep[ibody].cga(),MB_tstep[ibody].for_vel[0:3])

    def run(self):

        # Initialize varaibles
        MBdict = self.data.structure.mb_dict
        dt = self.settings['dt'].value

        # TODO: only working for constant forces
        self.data.structure.timestep_info[-1].unsteady_applied_forces = self.data.structure.dynamic_input[1]['dynamic_forces'].astype(dtype=ct.c_double, order='F', copy=True)
        MB_beam, MB_tstep = mb.split_multibody(self.data.structure, self.data.structure.timestep_info[-1], MBdict)

        # Lagrange multipliers parameters
        num_LM_eq = self.num_LM_eq
        Lambda = np.zeros((num_LM_eq,), dtype=ct.c_double, order='F')
        Lambda_dot = np.zeros((num_LM_eq,), dtype=ct.c_double, order='F')

        for ts in range(self.settings['num_steps'].value):
            # ipdb.set_trace()
            # Initialize
            q = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')
            dqdt = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')
            dqddt = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')

            # Predictor step
            mb.disp2state(MB_beam, MB_tstep, q, dqdt, dqddt)

            # dqddt = np.zeros_like(dqdt)
            q = q + dt*dqdt + (0.5 - self.beta)*dt*dt*dqddt
            dqdt = dqdt + (1.0 - self.gamma)*dt*dqddt
            dqddt = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')
            Lambda = q[-num_LM_eq:].astype(dtype=ct.c_double, copy=True, order='F')
            Lambda_dot = dqdt[-num_LM_eq:].astype(dtype=ct.c_double, copy=True, order='F')
            # TODO: what to do with lambda

            # Newmark-beta iterations
            old_Dq = 1.0
            LM_old_Dq = 1.0

            for iter in range(self.settings['max_iterations'].value):

                # Check if the maximum of iterations has been reached
                if (iter == self.settings['max_iterations'].value):
                    print('Solver did not converge in ', iter, ' iterations.')
                    break

                # Update positions and velocities
                mb.state2disp(q, dqdt, dqddt, MB_beam, MB_tstep)
                # Define matrices
                # if num_LM_eq == 0:
                #     Lambda = np.zeros((num_LM_eq,),)
                #     LambdaDot = np.zeros((num_LM_eq,),)
                # else:
                #     Lambda = np.zeros((num_LM_eq,),)
                #     LambdaDot = q[-num_LM_eq:]

                MB_Asys, MB_Q = self.assembly_MB_eq_system(MB_beam, MB_tstep, dt, Lambda, Lambda_dot)

                # Compute the correction
                Dq = np.zeros((self.sys_size+num_LM_eq,), dtype=ct.c_double, order='F')
                if np.isnan(MB_Asys).any():
                    print("ERROR: Nan in Asys")
                    embed()
                # print("cond: ", np.linalg.cond(MB_Asys))
                Dq = scipy.linalg.solve(MB_Asys, -MB_Q)
                if np.isnan(Dq).any():
                    print("ERROR: Nan in DX")



                # Evaluate convergence
                if (iter > 0):
                    res = np.max(np.abs(Dq[0:self.sys_size]))/old_Dq
                    if not num_LM_eq == 0:
                        LM_res = np.max(np.abs(Dq[self.sys_size:self.sys_size+num_LM_eq]))/LM_old_Dq
                    else:
                        LM_res = 0.0
                    if (res < self.settings['min_delta'].value) and (LM_res < self.settings['min_delta'].value):
                        # print("res: ", res)
                        # print("LMres: ", LM_res)
                        break

                # Compute variables from previous values and increments
                # TODO:decide If I want other way of updating lambda
                q = q + Dq
                dqdt = dqdt + self.gamma/(self.beta*dt)*Dq
                dqddt = dqddt + 1.0/(self.beta*dt*dt)*Dq
                Lambda = q[-num_LM_eq:].astype(dtype=ct.c_double, copy=True, order='F')
                Lambda_dot = dqdt[-num_LM_eq:].astype(dtype=ct.c_double, copy=True, order='F')

                # MB_tstep[1].for_pos = q[self.sys_size-10:self.sys_size-4].astype(dtype=ct.c_double, copy=True, order='F')

                if iter == 0:
                    old_Dq = np.max(np.abs(Dq[0:self.sys_size]))
                    if old_Dq < 1.0:
                        old_Dq = 1.0
                    if not num_LM_eq == 0:
                        LM_old_Dq = np.max(np.abs(Dq[self.sys_size:self.sys_size+num_LM_eq]))
                    else:
                        LM_old_Dq = np.abs(Dq[self.sys_size:self.sys_size+num_LM_eq])
                    if LM_old_Dq < 1.0:
                        LM_old_Dq = 1.0

            mb.state2disp(q, dqdt, dqddt, MB_beam, MB_tstep)

            # BC Check
            # print("last_point AFoR vel: ", MB_tstep[0].pos_dot[-1,:])
            # print("FoR AFoR vel       : ", np.dot(algebra.quat2rotation(MB_tstep[1].quat),MB_tstep[1].for_vel[0:3]))
            # print("FoR acc theta_z:     ", np.dot(algebra.quat2rotation(MB_tstep[1].quat),MB_tstep[1].for_acc[0:3]))
            # print("FoR AFoR rot vel xA: ", MB_tstep[1].for_vel[3])
            # print("psi camping Aslave: ", MB_tstep[1].psi[0,0,:])
            # print("psi camping G: ",np.dot(algebra.quat2rotation(MB_tstep[1].quat), MB_tstep[1].psi[0,0,:]))


        # End of Newmark-beta iterations

            self.integrate_position(MB_beam, MB_tstep, dt)

            # I do this to be able to write variables, but I want them to be in GFoR in the future
            self.data.ts += 1
            for ibody in range(len(MB_tstep)):
                self.data.structure.timestep_info[-1].mb_FoR_pos[ibody,:] = MB_tstep[ibody].for_pos.astype(dtype=ct.c_double, copy=True, order='F')
                self.data.structure.timestep_info[-1].mb_FoR_vel[ibody,:] = MB_tstep[ibody].for_vel.astype(dtype=ct.c_double, copy=True, order='F')
                self.data.structure.timestep_info[-1].mb_FoR_acc[ibody,:] = MB_tstep[ibody].for_acc.astype(dtype=ct.c_double, copy=True, order='F')
                self.data.structure.timestep_info[-1].mb_quat[ibody,:] = MB_tstep[ibody].quat.astype(dtype=ct.c_double, copy=True, order='F')
            #print("BC error: ", np.dot(np.transpose(algebra.crv2tan(MB_tstep[0].psi[-1,1,:])),MB_tstep[0].psi_dot[-1,1,:]) - np.dot(algebra.quat2rotation(MB_tstep[1].quat),MB_tstep[1].for_vel[3:6]))
            # print("rotation error: ", MB_tstep[0].psi[-1,1,:] - algebra.quat2crv(MB_tstep[1].quat))
            # print("tip pos: ", MB_tstep[1].pos[-1,1])
            # run postprocessors
            if self.with_postprocessors:
                for postproc in self.postprocessors:
                    self.data = self.postprocessors[postproc].run(online=True)

            print("ts: ", ts, " finished")
            if ts == 999:
                embed()


        self.data.ts -= 1
        return self.data
