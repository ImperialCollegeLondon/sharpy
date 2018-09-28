import ctypes as ct
import numpy as np


from sharpy.utils.settings import str2bool
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


        # Define Newmark constants
        self.gamma = 0.5 + self.settings['newmark_damp'].value
        self.beta = 0.25*(self.gamma + 0.5)*(self.gamma + 0.5)

        # Define the number of equations
        self.define_num_LM_eq()

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

    def define_num_LM_eq(self):

        MBdict = self.data.structure.mb_dict
        num_constraints = MBdict['num_constraints']
        self.num_LM_eq = 0

        # Define the number of equations that I need
        for iconstraint in range(num_constraints):

            if MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'hinge_node_FoR':
                #self.num_LM_eq += 6*(MBdict["constraint_%02d" % iconstraint]['num_nodes_involved']-1)
                self.num_LM_eq += 3
            elif MBdict["constraint_%02d" % iconstraint]['behaviour'] == 'free':
                self.num_LM_eq += 0
            else:
                print("ERROR: not recognized constraint type")


    def generate_lagrange_matrix(self, MB_beam, MB_tstep, dt):
        '''
        Generates the matrix associated to the Lagrange multipliers of a dictionary of "constraints"
        It is the matrix called "B" in Geradin and Cardona
        LM_pos_matrix: matrix associate to holonomic constraints. It should be included into K
        LM_vel_matrix: matrix associate to non-holonomic constraints. It should be included into C
        Qeq: vector of the constraints evaluated at
        call: LM_pos_matrix, LM_vel_matrix, LM_Q = generate_lagrange_matrix(self.data.structure, structural_step)
        '''

        # TODO: This is not correct, just wanted to check if it runs
        data_structure = MB_beam[0]
        structural_step = MB_tstep[0]

        MBdict = self.data.structure.mb_dict
        num_constraints = MBdict['num_constraints']
        num_eq = self.num_LM_eq
        sys_size = self.sys_size

        LM_pos_matrix = np.zeros((num_eq,sys_size), dtype=ct.c_double, order = 'F')
        LM_vel_matrix = np.zeros((num_eq,sys_size), dtype=ct.c_double, order = 'F')
        LM_Q = np.zeros((num_eq,),dtype=ct.c_double, order = 'F')


        return LM_pos_matrix, LM_vel_matrix, LM_Q

    def assembly_MB_eq_system(self, MB_beam, MB_tstep, dt, Lambda, Lambda_dot):

        #print("Lambda: ", Lambda)
        #print("LambdaDot: ", Lambda_dot)
        MBdict = self.data.structure.mb_dict
        MB_Asys = np.zeros((self.sys_size+self.num_LM_eq, self.sys_size+self.num_LM_eq),)
        MB_Q = np.zeros((self.sys_size+self.num_LM_eq, ),)

        # Lagrange multipliers parameters
        penaltyFactor = 0.0
        scalingFactor = 1.0

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
            MB_Asys[first_dof:last_dof,first_dof:last_dof] = K
            MB_Asys[first_dof:last_dof,first_dof:last_dof] += C*self.gamma/(self.beta*dt)
            MB_Asys[first_dof:last_dof,first_dof:last_dof] += M/(self.beta*dt*dt)

            #Q
            MB_Q[first_dof:last_dof] = Q

            first_dof = last_dof

        if not self.num_LM_eq == 0:
            # Generate matrices associated to Lagrange multipliers
            LM_pos_matrix, LM_vel_matrix, LM_Q = self.generate_lagrange_matrix(MB_beam, MB_tstep, dt)

            # Include the matrices associated to Lagrange Multipliers
            # MB_Asys[0:first_dof, 0:first_dof] += penaltyFactor*np.dot(np.transpose(LM_vel_matrix), LM_vel_matrix)
            MB_Asys[0:first_dof, 0:first_dof] += penaltyFactor*np.dot(np.transpose(LM_vel_matrix), LM_vel_matrix)
            MB_Asys[first_dof:, 0:first_dof] = scalingFactor*LM_vel_matrix
            MB_Asys[0:first_dof, first_dof:] = scalingFactor*np.transpose(LM_vel_matrix)

            MB_Q[0:first_dof] += np.dot(np.transpose(LM_vel_matrix), penaltyFactor*LM_Q + scalingFactor*Lambda_dot)
            MB_Q[first_dof:] = LM_Q

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
                MB_tstep[ibody].for_pos[0:3] = dt*np.dot(MB_tstep[ibody].cga(),MB_tstep[ibody].for_vel[0:3])

    def run(self):

        # Initialize varaibles
        MBdict = self.data.structure.mb_dict
        dt = self.settings['num_steps'].value

        # TODO: only working for constant forces
        self.data.structure.timestep_info[-1].unsteady_applied_forces = self.data.structure.dynamic_input[1]['dynamic_forces'].astype(dtype=ct.c_double, order='F', copy=True)
        MB_beam, MB_tstep = mb.split_multibody(self.data.structure, self.data.structure.timestep_info[-1], MBdict)

        # Lagrange multipliers parameters
        num_LM_eq = self.num_LM_eq

        for ts in range(self.settings['num_steps'].value):
            # Initialize
            q = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')
            dqdt = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')
            dqddt = np.zeros((self.sys_size + num_LM_eq,), dtype=ct.c_double, order='F')

            # Predictor step
            mb.disp2state(MB_beam, MB_tstep, q, dqdt, dqddt)

            dqddt = np.zeros_like(dqdt)
            dqdt = dqdt + (1.0 - self.gamma)*dt*dqddt
            q = q + dt*dqdt + (0.5 - self.beta)*dt*dt*dqddt
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
                if num_LM_eq == 0:
                    Lambda = np.zeros((num_LM_eq,),)
                    LambdaDot = np.zeros((num_LM_eq,),)
                else:
                    Lambda = np.zeros((num_LM_eq,),)
                    LambdaDot = q[-num_LM_eq:]

                MB_Asys, MB_Q = self.assembly_MB_eq_system(MB_beam, MB_tstep, dt, Lambda, LambdaDot)

                # Compute the correction
                Dq = np.zeros((self.sys_size+num_LM_eq,),)
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
                    print(res)
                    if not num_LM_eq == 0:
                        LM_res = np.max(np.abs(Dq[self.sys_size:self.sys_size+num_LM_eq]))/LM_old_Dq
                    else:
                        LM_res = res
                    if (res < self.settings['min_delta']) and (LM_res < self.settings['min_delta']):
                        break

                # Compute variables from previous values and increments
                # TODO:decide If I want other way of updating lambda
                q = q + Dq
                # Corrector step
                if not num_LM_eq ==0:
                    #dqdt = dqdt + self.gamma/(self.beta*dt)*Dq
                    dqdt[:-num_LM_eq] = dqdt[:-num_LM_eq] + self.gamma/(self.beta*dt)*Dq[:-num_LM_eq]
                    #print("Dq: ", Dq[-4-num_LM_eq:-num_LM_eq])
                    #dqddt = dqddt + 1.0/(self.beta*dt*dt)*Dq
                    dqddt[:-num_LM_eq] = dqddt[:-num_LM_eq] + 1.0/(self.beta*dt*dt)*Dq[:-num_LM_eq]
                else:
                    dqdt = dqdt + self.gamma/(self.beta*dt)*Dq
                    dqddt = dqddt + 1.0/(self.beta*dt*dt)*Dq

                if iter == 0:
                    old_Dq = np.max(np.abs(Dq[0:self.sys_size]))
                    if old_Dq < 1.0:
                        old_Dq = 1.0
                    if not num_LM_eq == 0:
                        LM_old_Dq = np.max(np.abs(Dq[self.sys_size:self.sys_size+num_LM_eq]))
                    else:
                        LM_old_Dq = old_Dq

                    if LM_old_Dq < 1.0:
                        LM_old_Dq = 1.0


            #ipdb.set_trace()
            mb.state2disp(q, dqdt, dqddt, MB_beam, MB_tstep)

            self.integrate_position(MB_beam, MB_tstep, dt)

            print("time step: ", ts, "pos[-1,:] 1: ", MB_tstep[1].pos[-1,:])
            # print("time step: ", ts, "for_pos 0: ", MB_tstep[0].for_pos[0:3], "for_pos 1: ", MB_tstep[1].for_pos[0:3])
            # print("time step: ", ts, "for_quat 0: ", MB_tstep[0].quat, "for_quat 1: ", MB_tstep[1].quat)
            # embed()
        return self.data
