import numpy as np
import unittest
import os
import shutil

import sharpy.sharpy_main                  # Run SHARPy inside jupyter notebooks
import sharpy.utils.generate_cases as gc
from sharpy.utils.constants import deg2rad

# Structural properties
mass_per_unit_length = 1.
mass_iner = 1e-4
EA = 1e9
GA = 1e9
GJ = 1e9
EI = 1e9

# Beam1
global nnodes1
nnodes1 = 11
l1 = 1.0
m1 = 1.0
theta_ini1 = (-60+90)*deg2rad

# Beam2
nnodes2 = nnodes1
l2 = l1*0.1
m2 = m1
theta_ini2 = (-60)*deg2rad

# Beam2
nnodes3 = nnodes1
l3 = l2
m3 = m2
theta_ini3 = (-60+90)*deg2rad


# rotation
rotation_velocity = 2.0*np.pi*0.5/60.0
dphi = 0.05*deg2rad
revs_to_simulate = 0.15

# Simulation

# numtimesteps = int(revs_to_simulate*2.*np.pi/dphi)
# set to 48 because convergence problem after! (for 0 atmosphere)
# numtimesteps = 48
# for 1.225 (uwake = 0), crashes at 73
# numtimesteps = 72
# uwake = 0.01, crashes at 52
numtimesteps = int(revs_to_simulate*2.*np.pi/dphi)
print(numtimesteps)
dt = dphi/rotation_velocity/2
print(dt)

# Create the structure
beam1 = gc.AeroelasticInformation()
# Generate an array with the location of the nodes
r1 = np.linspace(0.0, l1, nnodes1)
node_pos1 = np.zeros((nnodes1,3),)
node_pos1[:, 0] = r1*np.sin(theta_ini1)
node_pos1[:, 2] = -r1*np.cos(theta_ini1)
print(node_pos1)

beam1.StructuralInformation.generate_uniform_sym_beam(node_pos1, 
                                                      mass_per_unit_length, 
                                                      mass_iner, 
                                                      EA, 
                                                      GA, 
                                                      GJ, 
                                                      EI, 
                                                      num_node_elem = 3, 
                                                      y_BFoR = 'y_AFoR', 
                                                      num_lumped_mass=1)
beam1.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype = int)
beam1.StructuralInformation.boundary_conditions[0] = 1
beam1.StructuralInformation.boundary_conditions[-1] = -1
beam1.StructuralInformation.lumped_mass_nodes = np.array([nnodes1-1], dtype = int)
beam1.StructuralInformation.lumped_mass = np.ones((1,))*m1
beam1.StructuralInformation.lumped_mass_inertia = np.zeros((1,3,3))
beam1.StructuralInformation.lumped_mass_position = np.zeros((1,3))

beam2 = gc.AeroelasticInformation()
r2 = np.linspace(0.0, l2, nnodes2)
node_pos2 = np.zeros((nnodes2,3),)
node_pos2[:, 0] = r2*np.sin(theta_ini2) + node_pos1[-1, 0]
node_pos2[:, 2] = -r2*np.cos(theta_ini2) + node_pos1[-1, 2]+0.00000001
print(node_pos2)

beam2.StructuralInformation.generate_uniform_sym_beam(node_pos2, mass_per_unit_length, mass_iner, EA, GA, GJ, EI, num_node_elem = 3, y_BFoR = 'y_AFoR', num_lumped_mass=1)
beam2.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype = int)
beam2.StructuralInformation.boundary_conditions[0] = 1
beam2.StructuralInformation.boundary_conditions[-1] = -1
beam2.StructuralInformation.lumped_mass_nodes = np.array([nnodes2-1], dtype = int)
beam2.StructuralInformation.lumped_mass = np.ones((1,))*m2
beam2.StructuralInformation.lumped_mass_inertia = np.zeros((1,3,3))
beam2.StructuralInformation.lumped_mass_position = np.zeros((1,3))

beam3 = gc.AeroelasticInformation()
r3 = np.linspace(0.0, l3, nnodes3)
node_pos3 = np.zeros((nnodes3,3),)
node_pos3[:, 0] = r3*np.sin(theta_ini3) + node_pos2[-1, 0]
node_pos3[:, 2] = -r3*np.cos(theta_ini3) + node_pos2[-1, 2]-0.00000001
print(node_pos3)

beam3.StructuralInformation.generate_uniform_sym_beam(node_pos3, mass_per_unit_length, mass_iner, EA, GA, GJ, EI, num_node_elem = 3, y_BFoR = 'y_AFoR', num_lumped_mass=1)
beam3.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype = int)
beam3.StructuralInformation.boundary_conditions[0] = 1
beam3.StructuralInformation.boundary_conditions[-1] = -1
beam3.StructuralInformation.lumped_mass_nodes = np.array([nnodes3-1], dtype = int)
beam3.StructuralInformation.lumped_mass = np.ones((1,))*m3
beam3.StructuralInformation.lumped_mass_inertia = np.zeros((1,3,3))
beam3.StructuralInformation.lumped_mass_position = np.zeros((1,3))

# Define the coordinates of the camber line of the wing
# airfoils
airfoil = np.zeros((1,20,2),)
print(airfoil)
airfoil[0,:,0] = np.linspace(0.,1.,20)
print(airfoil)

# Generate blade aerodynamics
beam1.AerodynamicInformation.create_one_uniform_aerodynamics(
                                            beam1.StructuralInformation,
                                            chord = 0.1,
                                            twist = 0.,
                                            sweep = 0.,
                                            num_chord_panels = 4,
                                            m_distribution = 'uniform',
                                            elastic_axis = 0.25,
                                            num_points_camber = 20,
                                            airfoil = airfoil)

beam2.AerodynamicInformation.create_one_uniform_aerodynamics(
                                            beam2.StructuralInformation,
                                            chord = 0.01,
                                            twist = 0.,
                                            sweep = 0.,
                                            num_chord_panels = 4,
                                            m_distribution = 'uniform',
                                            elastic_axis = 0.25,
                                            num_points_camber = 20,
                                            airfoil = airfoil)


beam3.AerodynamicInformation.create_one_uniform_aerodynamics(
                                            beam3.StructuralInformation,
                                            chord = 0.01,
                                            twist = 0.,
                                            sweep = 0.,
                                            num_chord_panels = 4,
                                            m_distribution = 'uniform',
                                            elastic_axis = 0.25,
                                            num_points_camber = 20,
                                            airfoil = airfoil)

beam1.assembly(beam2,beam3)
print(beam1.AerodynamicInformation)
# beam2.assembly(beam3)


# Simulation details
SimInfo = gc.SimulationInformation()
SimInfo.set_default_values()

SimInfo.define_uinf(np.array([0.0,1.0,0.0]), -1.)
#if set uinf as sth like 0 0 1 will gg!

SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                'AerogridLoader',
                                'DynamicCoupled']
global name_direct_anothermodel
name_direct_anothermodel = 'dpg_direct_anothermodel'
SimInfo.solvers['SHARPy']['case'] = name_direct_anothermodel
SimInfo.solvers['SHARPy']['write_screen'] = 'on'
SimInfo.solvers['SHARPy']['route'] = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/'
SimInfo.solvers['SHARPy']['log_folder'] = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/'
SimInfo.set_variable_all_dicts('dt', dt)
SimInfo.define_num_steps(numtimesteps)
SimInfo.set_variable_all_dicts('rho', 1.225)
SimInfo.set_variable_all_dicts('velocity_field_input', SimInfo.solvers['SteadyVelocityField'])
SimInfo.set_variable_all_dicts('output', os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/')




SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
SimInfo.solvers['AerogridLoader']['mstar'] = 2
SimInfo.solvers['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] = {'u_inf':-1.,
                                                                   'u_inf_direction': np.array([0., 0., 1.0]),
                                                                   'dt': dt}

SimInfo.solvers['WriteVariablesTime']['FoR_number'] = np.array([0, 1, 2], dtype = int)
SimInfo.solvers['WriteVariablesTime']['FoR_variables'] = ['mb_quat','quat']
SimInfo.solvers['WriteVariablesTime']['structure_nodes'] = np.array([nnodes1-1, nnodes1+nnodes2-1, nnodes1+nnodes2+nnodes3-1], dtype = int)
SimInfo.solvers['WriteVariablesTime']['structure_variables'] = ['pos']

SimInfo.solvers['NonLinearDynamicMultibody']['gravity_on'] = True
SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator'] = 'NewmarkBeta'
SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator_settings'] = {'newmark_damp': 0.15,
                                                                                    'dt': dt}
SimInfo.solvers['NonLinearDynamicMultibody']['write_lm'] = True
SimInfo.solvers['NonLinearDynamicMultibody']['max_iterations'] = 50
# SimInfo.solvers['NonLinearDynamicMultibody']['relax_factor_lm'] = 0.2


SimInfo.solvers['BeamPlot']['include_FoR'] = True




SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['WriteVariablesTime', 'BeamPlot', 'AerogridPlot', 'PlotFlowField']
SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {'WriteVariablesTime': SimInfo.solvers['WriteVariablesTime'],
                                                                'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                                'AerogridPlot': SimInfo.solvers['AerogridPlot'],
                                                                'PlotFlowField': SimInfo.solvers['PlotFlowField']}

SimInfo.define_num_steps(numtimesteps)
SimInfo.with_forced_vel = True
SimInfo.for_vel = np.zeros((numtimesteps,6), dtype=float)
SimInfo.for_vel[:,5] = rotation_velocity
SimInfo.for_acc = np.zeros((numtimesteps,6), dtype=float)
SimInfo.with_dynamic_forces = True
SimInfo.dynamic_forces = np.zeros((numtimesteps,beam1.StructuralInformation.num_node,6), dtype=float)




# Create the MB and BC files
LC1 = gc.LagrangeConstraint()
LC1.behaviour = 'constant_vel_FoR'
LC1.FoR_body = 0
LC1.vel = np.array([0.0,0.0,0.0,0.0,0.0,1.0])
LC1.scalingFactor = 1e6
LC1.penaltyFactor = 1e-6

LC2 = gc.LagrangeConstraint()
LC2.behaviour = 'hinge_node_FoR'
LC2.node_in_body = nnodes1-1
LC2.body = 0
LC2.body_FoR = 1
LC2.rot_axisB = np.array([0.0,1.0,0.0])
LC2.scalingFactor = 1e6
LC2.penaltyFactor = 1e-6

LC3 = gc.LagrangeConstraint()
LC3.behaviour = 'hinge_node_FoR'
LC3.node_in_body = nnodes1-1
LC3.body = 1
LC3.body_FoR = 2
LC3.rot_axisB = np.array([0.0,1.0,0.0])
LC3.scalingFactor = 1e6
LC3.penaltyFactor = 1e-6

LC = []
LC.append(LC1)
LC.append(LC2)
LC.append(LC3)

MB1 = gc.BodyInformation()
MB1.body_number = 0
MB1.FoR_position = np.zeros((6,),)
MB1.FoR_velocity = np.zeros((6,),)
MB1.FoR_acceleration = np.zeros((6,),)
MB1.FoR_movement = 'free'
MB1.quat = np.array([1.0,0.0,0.0,0.0])

MB2 = gc.BodyInformation()
MB2.body_number = 1
MB2.FoR_position = np.array([node_pos2[0, 0], node_pos2[0, 1], node_pos2[0, 2], 0.0, 0.0, 0.0])
MB2.FoR_velocity = np.zeros((6,),)
MB2.FoR_acceleration = np.zeros((6,),)
MB2.FoR_movement = 'free'
MB2.quat = np.array([1.0,0.0,0.0,0.0])

MB3 = gc.BodyInformation()
MB3.body_number = 2
MB3.FoR_position = np.array([node_pos3[0, 0], node_pos3[0, 1], node_pos3[0, 2], 0.0, 0.0, 0.0])
MB3.FoR_velocity = np.zeros((6,),)
MB3.FoR_acceleration = np.zeros((6,),)
MB3.FoR_movement = 'free'
MB3.quat = np.array([1.0,0.0,0.0,0.0])

MB = []
MB.append(MB1)
MB.append(MB2)
MB.append(MB3)



# Write files
gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
SimInfo.generate_solver_file()
SimInfo.generate_dyn_file(numtimesteps)
beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

sharpy_output = sharpy.sharpy_main.main(['',
                                     SimInfo.solvers['SHARPy']['route'] +
                                     SimInfo.solvers['SHARPy']['case'] +
                                     '.sharpy'])

# # Same case without dissipation
# global name_nb_zero_dis
# name_nb_zero_dis = 'dpg_nb_zero_dis'
# SimInfo.solvers['SHARPy']['case'] = name_nb_zero_dis

# SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator_settings'] = {'newmark_damp': 0.,
#                                                                                 'dt': dt}

# gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
# SimInfo.generate_solver_file()
# SimInfo.generate_dyn_file(numtimesteps)
# beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
# gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

# sharpy_output = sharpy.sharpy_main.main(['',
#                                      SimInfo.solvers['SHARPy']['route'] +
#                                      SimInfo.solvers['SHARPy']['case'] +
#                                      '.sharpy'])



# # Same case with generalised alpha
# global name_ga
# name_ga = 'dpg_ga'
# SimInfo.solvers['SHARPy']['case'] = name_ga

# SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator'] = 'GeneralisedAlpha'
# SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator_settings'] = {'am': 0.5,
#                                                                                 'af': 0.5,
#                                                                                 'dt': dt}

# gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
# SimInfo.generate_solver_file()
# SimInfo.generate_dyn_file(numtimesteps)
# beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
# gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

# sharpy_output = sharpy.sharpy_main.main(['',
#                                      SimInfo.solvers['SHARPy']['route'] +
#                                      SimInfo.solvers['SHARPy']['case'] +
#                                      '.sharpy'])

# # Same case with spherical joints
# global name_spherical
# name_spherical = 'dpg_spherical'
# SimInfo.solvers['SHARPy']['case'] = name_spherical

# SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator'] = 'NewmarkBeta'
# SimInfo.solvers['NonLinearDynamicMultibody']['time_integrator_settings'] = {'newmark_damp': 0.15,
#                                                                                 'dt': dt}

# LC1 = gc.LagrangeConstraint()
# LC1.behaviour = 'spherical_FoR'
# LC1.body_FoR = 0
# LC1.scalingFactor = 1e6

# LC2 = gc.LagrangeConstraint()
# LC2.behaviour = 'spherical_node_FoR'
# LC2.node_in_body = nnodes1-1
# LC2.body = 0
# LC2.body_FoR = 1
# LC2.scalingFactor = 1e6

# gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
# SimInfo.generate_solver_file()
# SimInfo.generate_dyn_file(numtimesteps)
# beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
# gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

# sharpy_output = sharpy.sharpy_main.main(['',
#                                      SimInfo.solvers['SHARPy']['route'] +
#                                      SimInfo.solvers['SHARPy']['case'] +
#                                      '.sharpy'])