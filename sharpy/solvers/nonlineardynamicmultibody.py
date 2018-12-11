"""
Nonlinear dynamic multibody

Nonlinear dynamic step solver for multibody structures

Args:

Returns:

Examples:

Notes:

"""
import ctypes as ct
import numpy as np


from sharpy.utils.settings import str2bool
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
#from sharpy.solvers.nonlineardynamicprescribedstep import NonLinearDynamicPrescribedStep
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout

# from IPython import embed
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
        # self.postprocessors = dict()
        # if len(self.settings['postprocessors']) > 0:
        #     self.with_postprocessors = True
        # for postproc in self.settings['postprocessors']:
        #     self.postprocessors[postproc] = solver_interface.initialise_solver(postproc)
        #     self.postprocessors[postproc].initialise(
        #         self.data, self.settings['postprocessors_settings'][postproc])

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

    def assembly_MB_eq_system(self, MB_beam, MB_tstep, ts, dt, Lambda, Lambda_dot):

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

                # TEST quaternion equations
                # CQR_solver = C[-4:,-10:-4]
                # CQQ_solver = C[-4:,-4:]
                # QQ_solver = Q[-4:]

                # CQQ_han = np.zeros((4,4),)
                # CQQ_han[0,1:4] = -1.0*MB_tstep[0].for_vel[3:6]
                # CQQ_han[1:4,0] = 1.0*MB_tstep[0].for_vel[3:6]
                # CQQ_han[1:4,1:4] = -1.0*algebra.skew(MB_tstep[0].for_vel[3:6])
                # CQQ_han = -0.5*CQQ_han
                #
                # CQR_han = np.zeros((4,6),)
                # CQR_han[0,3] = -1.0*MB_tstep[0].quat[1]
                # CQR_han[0,4] = -1.0*MB_tstep[0].quat[2]
                # CQR_han[0,5] = -1.0*MB_tstep[0].quat[3]
                #
                # CQR_han[1,3] = 1.0*MB_tstep[0].quat[0]
                # CQR_han[1,4] = -1.0*MB_tstep[0].quat[3]
                # CQR_han[1,5] = 1.0*MB_tstep[0].quat[2]
                #
                # CQR_han[2,3] = 1.0*MB_tstep[0].quat[3]
                # CQR_han[2,4] = 1.0*MB_tstep[0].quat[0]
                # CQR_han[2,5] = -1.0*MB_tstep[0].quat[1]
                #
                # CQR_han[3,3] = -1.0*MB_tstep[0].quat[2]
                # CQR_han[3,4] = 1.0*MB_tstep[0].quat[1]
                # CQR_han[3,5] = 1.0*MB_tstep[0].quat[0]

                # CQQ_han = np.zeros((4,4),)
                # CQQ_han[0,1:4] = -1.0*MB_tstep[0].dqdt[-7:-4]
                # CQQ_han[1:4,0] = 1.0*MB_tstep[0].dqdt[-7:-4]
                # CQQ_han[1:4,1:4] = -1.0*algebra.skew(MB_tstep[0].dqdt[-7:-4])
                # CQQ_han = 0.5*CQQ_han
                #
                # CQR_han = np.zeros((4,6),)
                #
                # aux_quat = MB_tstep[0].dqdt[-4:]
                # CQR_han[0,3] = -1.0*aux_quat[1]
                # CQR_han[0,4] = -1.0*aux_quat[2]
                # CQR_han[0,5] = -1.0*aux_quat[3]
                #
                # CQR_han[1,3] = 1.0*aux_quat[0]
                # CQR_han[1,4] = -1.0*aux_quat[3]
                # CQR_han[1,5] = 1.0*aux_quat[2]
                #
                # CQR_han[2,3] = 1.0*aux_quat[3]
                # CQR_han[2,4] = 1.0*aux_quat[0]
                # CQR_han[2,5] = -1.0*aux_quat[1]
                #
                # CQR_han[3,3] = -1.0*aux_quat[2]
                # CQR_han[3,4] = 1.0*aux_quat[1]
                # CQR_han[3,5] = 1.0*aux_quat[0]
                #
                # CQR_han = 0.5*CQR_han
                #
                # QQ_han = -1.0*(np.dot(CQQ_han,MB_tstep[0].dqdt[-4:]) - MB_tstep[0].dqddt[-4:])

                # np.dot(CQQ_han,MB_tstep[0].dqdt[-4:])


                # print("CQQ solver: ", CQQ_solver)
                # print("CQQ han: ", CQQ_han)
                # print("CQR solver: ", CQR_solver)
                # print("CQR han: ", CQR_han)
                # print("QQ solver: ", QQ_solver)
                # print("QQ han: ", QQ_han)
                # embed()

                # C[-4:,-10:-4] = -1.0*CQR_han.astype(dtype=ct.c_double, copy=True, order='F')
                # C[-4:,-4:] = -1.0*CQQ_han.astype(dtype=ct.c_double, copy=True, order='F')
                # Q[-4:] = QQ_han.astype(dtype=ct.c_double, copy=True, order='F')

            ############### Assembly into the global matrices
            # Flexible and RBM contribution to Asys
            MB_M[first_dof:last_dof, first_dof:last_dof] = M.astype(dtype=ct.c_double, copy=True, order='F')
            MB_C[first_dof:last_dof, first_dof:last_dof] = C.astype(dtype=ct.c_double, copy=True, order='F')
            MB_K[first_dof:last_dof, first_dof:last_dof] = K.astype(dtype=ct.c_double, copy=True, order='F')

            #Q
            MB_Q[first_dof:last_dof] = Q

            first_dof = last_dof


        # Generate matrices associated to Lagrange multipliers
        LM_C, LM_K, LM_Q = lagrangemultipliers.generate_lagrange_matrix(MBdict, MB_beam, MB_tstep, ts, self.num_LM_eq, self.sys_size, dt, Lambda, Lambda_dot)

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
        for ibody in range(1, len(MB_tstep)):
            # I think this is the right way to do it, but to make it match the rest I change it temporally
            if False:
                # MB_tstep[ibody].mb_quat[ibody,:] =  algebra.quaternion_product(MB_tstep[ibody].quat, MB_tstep[ibody].mb_quat[ibody,:])
                acc[0:3] = (0.5-self.beta)*np.dot(MB_beam[ibody].timestep_info.cga(),MB_beam[ibody].timestep_info.for_acc[0:3])+self.beta*np.dot(MB_tstep[ibody].cga(),MB_tstep[ibody].for_acc[0:3])
                vel[0:3] = np.dot(MB_beam[ibody].timestep_info.cga(),MB_beam[ibody].timestep_info.for_vel[0:3])
                MB_tstep[ibody].for_pos[0:3] += dt*(vel[0:3] + dt*acc[0:3])
            else:
                # print("quat: ", MB_tstep[ibody].quat)
                # print("cga: ", MB_tstep[ibody].cga())
                # print("for_vel: ", MB_tstep[ibody].for_vel)
                MB_tstep[ibody].for_pos[0:3] += dt*np.dot(MB_tstep[ibody].cga(),MB_tstep[ibody].for_vel[0:3])

        # Use next line for double pendulum (fix position of the second FoR)
        # MB_tstep[ibody].for_pos[0:3] = np.dot(algebra.quat2rotation(MB_tstep[0].quat), MB_tstep[0].pos[-1,:])
        # print("tip final pos: ", np.dot(algebra.quat2rotation(MB_tstep[0].quat), MB_tstep[0].pos[-1,:]))
        # print("FoR final pos: ", MB_tstep[ibody].for_pos[0:3])
        # print("pause")

    def extract_resultants(self):
        # TODO: code
        pass

    def run(self, structural_step=None):
        #ipdb.set_trace()

        if structural_step is None:
            structural_step = self.data.structure.timestep_info[-1]
        # Initialize varaibles
        MBdict = self.data.structure.mb_dict
        dt = self.settings['dt'].value

        # TODO: there should be a better way to do the following
        if self.data.ts < 2:
                structural_step.steady_applied_forces *= 0.
                structural_step.unsteady_applied_forces *= 0.

        # print("beg quat: ", structural_step.quat)
        # TODO: only working for constant forces
        # self.data.structure.timestep_info[-1].unsteady_applied_forces = self.data.structure.dynamic_input[1]['dynamic_forces'].astype(dtype=ct.c_double, order='F', copy=True)
        MB_beam, MB_tstep = mb.split_multibody(self.data.structure, structural_step, MBdict, self.data.ts)
        # Lagrange multipliers parameters
        num_LM_eq = self.num_LM_eq
        Lambda = np.zeros((num_LM_eq,), dtype=ct.c_double, order='F')
        Lambda_dot = np.zeros((num_LM_eq,), dtype=ct.c_double, order='F')

        # for ts in range(self.settings['num_steps'].value):
        # comment time stepping
        # ipdb.set_trace()
        # Initialize
        q = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')
        dqdt = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')
        dqddt = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')

        # Predictor step
        mb.disp2state(MB_beam, MB_tstep, q, dqdt, dqddt)

        #print("q: ", q[-num_LM_eq-10:-num_LM_eq])
        #print("dqdt: ", dqdt[-num_LM_eq-10:-num_LM_eq])
        #print("dqddt: ", dqddt[-num_LM_eq-10:-num_LM_eq])

        # print("-----  BEGINNING PREDICTOR STEP -----")
        # print("dqddt FoR0: ", dqddt[60:70])
        # print("MB_tstep[0].dqddt: ", MB_tstep[0].dqddt[60:70])
        # print("MB_tstep[0].for_acc: ", MB_tstep[0].for_acc)
        # print("dqddt FoR1: ", dqddt[130:140])
        # print("MB_tstep[1].dqddt: ", MB_tstep[1].dqddt[60:70])
        # print("MB_tstep[1].for_acc: ", MB_tstep[1].for_acc)
        # print("-----  END PREDICTOR STEP -----")

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

        converged = False
        for iter in range(self.settings['max_iterations'].value):

            # Check if the maximum of iterations has been reached
            if (iter == self.settings['max_iterations'].value):
                print('Solver did not converge in ', iter, ' iterations.')
                break

            # Update positions and velocities
            mb.state2disp(q, dqdt, dqddt, MB_beam, MB_tstep)

            # print("dqddt FoR0: ", dqddt[60:70])
            # print("MB_tstep[0].dqddt: ", MB_tstep[0].dqddt[60:70])
            # print("MB_tstep[0].for_acc: ", MB_tstep[0].for_acc)
            # print("dqddt FoR1: ", dqddt[130:140])
            # print("MB_tstep[1].dqddt: ", MB_tstep[1].dqddt[60:70])
            # print("MB_tstep[1].for_acc: ", MB_tstep[1].for_acc)

            MB_Asys, MB_Q = self.assembly_MB_eq_system(MB_beam, MB_tstep, self.data.ts, dt, Lambda, Lambda_dot)

            # Compute the correction
            Dq = np.zeros((self.sys_size+num_LM_eq,), dtype=ct.c_double, order='F')
            # if np.isnan(MB_Asys).any():
            #     print("ERROR: Nan in Asys")
            #     embed()
            # print("cond: ", np.linalg.cond(MB_Asys))
            Dq = scipy.linalg.solve(MB_Asys, -MB_Q)
            # if np.isnan(Dq).any():
            #     print("ERROR: Nan in DX")



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
                    # break
                    converged = True

            # Compute variables from previous values and increments
            # TODO:decide If I want other way of updating lambda
            # print("Dq vel and quat: ", Dq[-num_LM_eq-4-6:-num_LM_eq])
            q = q + Dq
            dqdt = dqdt + self.gamma/(self.beta*dt)*Dq
            dqddt = dqddt + 1.0/(self.beta*dt*dt)*Dq

            # mat = np.zeros((4,4),)
            # mat[0,1:4] = -1.0*dqdt[-num_LM_eq-7:-num_LM_eq-4]
            # mat[1:4,0] = 1.0*dqdt[-num_LM_eq-7:-num_LM_eq-4]
            # mat[1:4,1:4] = -1.0*algebra.skew(dqdt[-num_LM_eq-7:-num_LM_eq-4])
            # print("check quat eq: ", dqddt[-num_LM_eq-4:-num_LM_eq] - 0.5*np.dot(mat, dqdt[-num_LM_eq-4:-num_LM_eq]))

            #
            # dqdt[:(-num_LM_eq-4)] = dqdt[:(-num_LM_eq-4)] + self.gamma/(self.beta*dt)*Dq[:(-num_LM_eq-4)]
            # dqdt[(-num_LM_eq-4):-num_LM_eq] = algebra.rotate_quaternion(dqdt[(-num_LM_eq-4):-num_LM_eq], Dq[(-num_LM_eq-7):(-num_LM_eq-4)])
            # dqdt[-num_LM_eq:] = dqdt[-num_LM_eq:] + self.gamma/(self.beta*dt)*Dq[-num_LM_eq:]
            #
            # dqddt[:(-num_LM_eq-4)] = dqddt[:(-num_LM_eq-4)] + 1.0/(self.beta*dt*dt)*Dq[:(-num_LM_eq-4)]
            #
            # dqddt[-num_LM_eq:] = dqddt[-num_LM_eq:] + 1.0/(self.beta*dt*dt)*Dq[-num_LM_eq:]

            Lambda = q[-num_LM_eq:].astype(dtype=ct.c_double, copy=True, order='F')
            Lambda_dot = dqdt[-num_LM_eq:].astype(dtype=ct.c_double, copy=True, order='F')

            # MB_tstep[1].for_pos = q[self.sys_size-10:self.sys_size-4].astype(dtype=ct.c_double, copy=True, order='F')
            if converged:
                break

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

            # print("NB it: ", iter)

        mb.state2disp(q, dqdt, dqddt, MB_beam, MB_tstep)

        # Check boundary conditions
        # embed()
        # print("point vel: ", np.dot(algebra.quat2rotation(MB_tstep[0].quat),MB_tstep[0].pos_dot[-1,:] + MB_tstep[0].for_vel[0:3] + np.cross(MB_tstep[0].for_vel[3:6],MB_tstep[0].pos[-1,:])))
        # print("FoR vel: ", np.dot(algebra.quat2rotation(MB_tstep[1].quat),MB_tstep[1].for_vel[0:3]))

        # print("Dtheta from quat: ", 2.0*(np.arccos(MB_tstep[0].quat[0])-np.arccos(structural_step.quat[0])))
        # print("Dtheta wrong?: ", self.gamma*MB_tstep[0].for_vel[4]*dt)
        # print("Dtheta from velocities: ", structural_step.for_vel[4]*dt + (0.5-self.beta)*structural_step.for_acc[4]*dt**2.+self.beta*MB_tstep[0].for_acc[4]*dt**2.)

        # end: comment time stepping

            # BC Check
            # print("last_point AFoR vel: ", MB_tstep[0].pos_dot[-1,:])
            # print("FoR AFoR vel       : ", np.dot(algebra.quat2rotation(MB_tstep[1].quat),MB_tstep[1].for_vel[0:3]))
            # print("FoR acc theta_z:     ", np.dot(algebra.quat2rotation(MB_tstep[1].quat),MB_tstep[1].for_acc[0:3]))
            # print("FoR AFoR rot vel xA: ", MB_tstep[1].for_vel[3])
            # print("psi camping Aslave: ", MB_tstep[1].psi[0,0,:])
            # print("psi camping G: ",np.dot(algebra.quat2rotation(MB_tstep[1].quat), MB_tstep[1].psi[0,0,:]))


        # End of Newmark-beta iterations

        self.integrate_position(MB_beam, MB_tstep, dt)
        lagrangemultipliers.postprocess(MB_beam, MB_tstep, MBdict)
        # Force position
        # MB_tstep[1].for_pos[0:3] = MB_tstep[0].pos[-1,:] - np.zeros((3,),)
        # print("for vel: ", MB_tstep[1].for_vel)
        # print("for quat: ", MB_tstep[0].quat)
        # print("for acc: ", MB_tstep[0].for_acc)
        mb.merge_multibody(MB_tstep, MB_beam, self.data.structure, structural_step, MBdict, dt)

        # embed()
        # print("for pos: ", MB_tstep[1].for_pos[0:3])
        # print("db for pos: ", MB_tstep[1].mb_FoR_pos[1,0:3])
        # print("position", structural_step.pos)
        # tstep = len(self.data.structure.timestep_info)
        # print("for quat MB: ", MB_tstep[0].quat)
        # print("for quat structural: ", structural_step.quat)
        # print("numerical angle: ", 2.*np.arccos(structural_step.quat[0])*180./np.pi)
        # print("expected angle: ", 0.5*200*(tstep*dt)**2.*180/np.pi)
        # print("angle from rot matrix from quat: ", np.arccos(algebra.quat2rotation(structural_step.quat)[0,0])*180./np.pi)

        # print("numerical Dangle: ", 2.*(np.arccos(structural_step.quat[0])-np.arccos(self.data.structure.timestep_info[-1].quat[0]))*180./np.pi)
        # print("expected Dangle: ", 0.5*200*((tstep*dt)**2.-((tstep-1)*dt)**2.)*180/np.pi)
        # print("Dangle from integrated speed: ", (q[-6-num_LM_eq] - self.data.structure.timestep_info[-1].q[-6])*180/np.pi)

        # mat = np.zeros((4,4),)
        # mat[0,1:4] = -1.0*dqdt[-num_LM_eq-7:-num_LM_eq-4]
        # mat[1:4,0] = 1.0*dqdt[-num_LM_eq-7:-num_LM_eq-4]
        # mat[1:4,1:4] = -1.0*algebra.skew(dqdt[-num_LM_eq-7:-num_LM_eq-4])
        # print("check quat eq2: ", dqddt[-num_LM_eq-4:-num_LM_eq] - 0.5*np.dot(mat, dqdt[-num_LM_eq-4:-num_LM_eq]))


        # mat = np.zeros((4,4),)
        # mat[0,1:4] = -1.0*MB_tstep[0].for_vel[3:6]
        # mat[1:4,0] = 1.0*MB_tstep[0].for_vel[3:6]
        # mat[1:4,1:4] = -1.0*algebra.skew(MB_tstep[0].for_vel[3:6])
        # print("check quat eq3: ", dqddt[-num_LM_eq-4:-num_LM_eq] - 0.5*np.dot(mat, MB_tstep[0].quat))

        # embed()
        # print("end quat: ", structural_step.quat)

        # I do this to be able to write variables, but I want them to be in GFoR in the future
        # self.data.ts += 1
        # for ibody in range(len(MB_tstep)):
        #     self.data.structure.timestep_info[-1].mb_FoR_pos[ibody,:] = MB_tstep[ibody].for_pos.astype(dtype=ct.c_double, copy=True, order='F')
        #     self.data.structure.timestep_info[-1].mb_FoR_vel[ibody,:] = MB_tstep[ibody].for_vel.astype(dtype=ct.c_double, copy=True, order='F')
        #     self.data.structure.timestep_info[-1].mb_FoR_acc[ibody,:] = MB_tstep[ibody].for_acc.astype(dtype=ct.c_double, copy=True, order='F')
        #     self.data.structure.timestep_info[-1].mb_quat[ibody,:] = MB_tstep[ibody].quat.astype(dtype=ct.c_double, copy=True, order='F')
        #print("BC error: ", np.dot(np.transpose(algebra.crv2tan(MB_tstep[0].psi[-1,1,:])),MB_tstep[0].psi_dot[-1,1,:]) - np.dot(algebra.quat2rotation(MB_tstep[1].quat),MB_tstep[1].for_vel[3:6]))
        # print("rotation error: ", MB_tstep[0].psi[-1,1,:] - algebra.quat2crv(MB_tstep[1].quat))
        # print("tip pos: ", MB_tstep[1].pos[-1,1])
        # run postprocessors
        # if self.with_postprocessors:
        #     for postproc in self.postprocessors:
        #         self.data = self.postprocessors[postproc].run(online=True)
        #
        # print("ts: ", ts, " finished")
        # if ts == 999:
        #
        #
        #
        # self.data.ts -= 1
        return self.data
