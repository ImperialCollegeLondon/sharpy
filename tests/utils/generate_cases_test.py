import sharpy.utils.algebra as algebra
import numpy as np
import unittest
import random
# from IPython import embed

# Usual SHARPy
import h5py as h5
import os
import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout  # to use color output
# Read excel files
import pandas as pd
from pandas import ExcelFile
# Generate errors during execution
import sys
# Use .isnan
import math

from copy import deepcopy


"""
Each one of the test cases can be use as a .py file to generate a case
"""

class TestGenerateCases(unittest.TestCase):
    """
    Tests the generate_cases module
    """


    def test_generate_cantilever_uniform_beam(self):
        """
        """

        # DEFINE CASE
        case_name = 'one_beam'
        route = os.path.dirname(os.path.realpath(__file__)) + '/'

        flow = ['BeamLoader',
                'AerogridLoader',
                'StaticCoupled',
                # 'SteadyHelicoidalWake',
                'DynamicPrescribedCoupled',
                # 'NonLinearDynamicMultibody',
                'AerogridPlot',
                'BeamPlot'
                ]

        gravity = 'off'
        # Number of panels on the blade (chordwise direction)
        m = 1
        # Number of panels on the wake (flow direction)
        mstar = 10

        m_distribution = 'uniform'



        ######################################################################
        ##############################  ASSEMBLY  ############################
        ######################################################################

        nnodes = 11
        length = 1.0
        mass_per_unit_length = 1.0
        mass_iner = 1.0
        EA = 1e6
        GJ = 1e6
        EI = 1e6
        tip_force = 0.0*np.array([0.0,1.0,0.0,0.0,0.0,0.0])

        # Create the structure
        beam1 = AeroelasticInformation()
        node_pos = np.zeros((nnodes,3),)
        node_pos[:, 0] = np.linspace(0.0, length, nnodes)
        beam1.StructuralInformation.generate_uniform_sym_beam(node_pos, mass_per_unit_length, mass_iner, EA, GJ, EI)
        beam1.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype = int)
        beam1.define_basic_aerodynamics(beam1.StructuralInformation)
        # beam1.AerodynamicInformation.aero_node[:] = np.ones((3,), dtype = bool)
        # beam1.AerodynamicInformation.surface_distribution = np.array([0], dtype = int)

        beam2 = AeroelasticInformation()
        node_pos = node_pos + np.array([length, 0.0, 0.0])
        beam2.StructuralInformation.generate_uniform_sym_beam(node_pos, mass_per_unit_length, mass_iner, EA, GJ, EI)
        beam2.StructuralInformation.body_number = np.ones((beam1.StructuralInformation.num_elem,),dtype = int)
        beam2.define_basic_aerodynamics(beam2.StructuralInformation)
        #
        beam1.assembly(beam2)
        # # beam1.remove_clamping()

        clean_test_files()
        beam1.write_h5_files()

        # Simulation details
        SimulationInformation = SimulationInformation()
        SimulationInformation.generate_basic(n_tstep=1000, dt=0.05)
        SimulationInformation.with_dynamic_forces = True
        SimulationInformation.dynamic_forces_time = np.zeros((SimulationInformation.n_tstep,2*nnodes,6),)
        # SimulationInformation.dynamic_forces_time[:,0,:] -= tip_force
        # SimulationInformation.dynamic_forces_time[:,nnodes-1,:] += tip_force
        # SimulationInformation.dynamic_forces_time[:,nnodes,:] -= tip_force
        # SimulationInformation.dynamic_forces_time[:,-1,:] += tip_force
        SimulationInformation.generate_dyn_file()
        SimulationInformation.generate_solver_file()

        # Create the BC file
        # LC = []
        # LC1 = LagrangeConstraint()
        # LC1.behaviour = 'hinge_node_FoR_constant_rotation'
        # LC1.node_in_body = nnodes - 1
        # LC1.body = 0
        # LC1.body_FoR = 1
        # LC1.rot_vel = 0.2
        # LC.append(LC1)
        #
        # MB = []
        # MB1 = BodyInformation()
        # MB1.body_number = 0
        # MB1.FoR_position = np.zeros((6,),)
        # MB1.FoR_velocity = np.zeros((6,),)
        # MB1.FoR_acceleration = np.zeros((6,),)
        # MB1.FoR_movement = 'prescribed'
        # MB1.quat = np.array([1.0,0.0,0.0,0.0])
        # MB.append(MB1)
        # MB2 = BodyInformation()
        # MB2.body_number = 1
        # MB2.FoR_position = np.array([length,0.0,0.0,0.0,0.0,0.0])
        # MB2.FoR_velocity = np.zeros((6,),)
        # MB2.FoR_acceleration = np.zeros((6,),)
        # MB2.FoR_movement = 'free'
        # MB2.quat = np.array([1.0,0.0,0.0,0.0])
        # MB.append(MB2)
        #
        # generate_multibody_file(LC, MB)

        # print("DONE")

if __name__=='__main__':

    T=TestGenerateCases()

    T.test_generate_cantilever_uniform_beam()
