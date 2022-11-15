import os
import shutil
import unittest
import numpy as np

import sharpy.utils.generate_cases as gc
import sharpy.sharpy_main

folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


class TestFixNodeVelocitywrtG(unittest.TestCase):
    """ """

    def setUp(self):
        nodes_per_elem = 3

        # beam1: uniform and symmetric with aerodynamic properties equal to zero
        nnodes1 = 11
        length1 = 10.0
        mass_per_unit_length = 0.1
        mass_iner = 1e-4
        EA = 1e9
        GA = 1e9
        GJ = 1e3
        EI = 1e4

        # Create beam1
        beam1 = gc.AeroelasticInformation()
        # Structural information
        beam1.StructuralInformation.num_node = nnodes1
        beam1.StructuralInformation.num_node_elem = nodes_per_elem
        beam1.StructuralInformation.compute_basic_num_elem()
        beam1.StructuralInformation.set_to_zero(
            beam1.StructuralInformation.num_node_elem,
            beam1.StructuralInformation.num_node,
            beam1.StructuralInformation.num_elem,
        )
        node_pos = np.zeros(
            (nnodes1, 3),
        )
        node_pos[:, 0] = np.linspace(0.0, length1, nnodes1)
        beam1.StructuralInformation.generate_uniform_sym_beam(
            node_pos,
            mass_per_unit_length,
            mass_iner,
            EA,
            GA,
            GJ,
            EI,
            num_node_elem=3,
            y_BFoR="y_AFoR",
            num_lumped_mass=2,
        )
        beam1.StructuralInformation.boundary_conditions[0] = 1
        beam1.StructuralInformation.boundary_conditions[-1] = -1
        beam1.StructuralInformation.lumped_mass_nodes = np.array(
            [0, nnodes1 - 1], dtype=int
        )
        beam1.StructuralInformation.lumped_mass = np.array([2.0, 1.0])
        beam1.StructuralInformation.lumped_mass_inertia = np.zeros(
            (2, 3, 3),
        )
        beam1.StructuralInformation.lumped_mass_position = np.zeros(
            (2, 3),
        )

        # Aerodynamic information
        airfoil = np.zeros(
            (1, 20, 2),
        )
        airfoil[0, :, 0] = np.linspace(0.0, 1.0, 20)
        beam1.AerodynamicInformation.create_one_uniform_aerodynamics(
            beam1.StructuralInformation,
            chord=1.0,
            twist=0.0,
            sweep=0.0,
            num_chord_panels=4,
            m_distribution="uniform",
            elastic_axis=0.5,
            num_points_camber=20,
            airfoil=airfoil,
        )

        # SOLVER CONFIGURATION
        SimInfo = gc.SimulationInformation()
        SimInfo.set_default_values()

        SimInfo.define_uinf(np.array([0.0, 1.0, 0.0]), 10.0)

        SimInfo.solvers["SHARPy"]["flow"] = [
            "BeamLoader",
            "AerogridLoader",
            "StaticCoupled",
            "DynamicCoupled",
        ]
        self.name = "fix_node_velocity_wrtG"
        SimInfo.solvers["SHARPy"]["case"] = self.name
        SimInfo.solvers["SHARPy"]["write_screen"] = "off"
        SimInfo.solvers["SHARPy"]["route"] = folder + "/"
        SimInfo.solvers["SHARPy"]["log_folder"] = folder + "/output/"
        SimInfo.set_variable_all_dicts("dt", 0.1)
        SimInfo.set_variable_all_dicts("rho", 0.0)
        SimInfo.set_variable_all_dicts(
            "velocity_field_input", SimInfo.solvers["SteadyVelocityField"]
        )
        SimInfo.set_variable_all_dicts("folder", folder + "/output/")

        SimInfo.solvers["BeamLoader"]["unsteady"] = "on"

        SimInfo.solvers["AerogridLoader"]["unsteady"] = "on"
        SimInfo.solvers["AerogridLoader"]["mstar"] = 2
        SimInfo.solvers["AerogridLoader"]["wake_shape_generator"] = "StraightWake"
        SimInfo.solvers["AerogridLoader"]["wake_shape_generator_input"] = {
            "u_inf": 10.0,
            "u_inf_direction": np.array([0.0, 1.0, 0.0]),
            "dt": 0.1,
        }

        SimInfo.solvers["NonLinearStatic"]["print_info"] = False

        SimInfo.solvers["StaticCoupled"]["structural_solver"] = "NonLinearStatic"
        SimInfo.solvers["StaticCoupled"][
            "structural_solver_settings"
        ] = SimInfo.solvers["NonLinearStatic"]
        SimInfo.solvers["StaticCoupled"]["aero_solver"] = "StaticUvlm"
        SimInfo.solvers["StaticCoupled"]["aero_solver_settings"] = SimInfo.solvers[
            "StaticUvlm"
        ]

        SimInfo.solvers["WriteVariablesTime"]["structure_nodes"] = np.array(
            [0, int((nnodes1 - 1) / 2), -1], dtype=int
        )
        SimInfo.solvers["WriteVariablesTime"]["structure_variables"] = ["pos"]

        SimInfo.solvers["BeamPlot"]["include_FoR"] = True

        SimInfo.solvers["NonLinearDynamicMultibody"]["relaxation_factor"] = 0.0
        SimInfo.solvers["NonLinearDynamicMultibody"]["min_delta"] = 1e-5
        SimInfo.solvers["NonLinearDynamicMultibody"]["max_iterations"] = 200
        # SimInfo.solvers['NonLinearDynamicMultibody']['gravity_on'] = 'off'
        SimInfo.solvers["NonLinearDynamicMultibody"]["time_integrator"] = "NewmarkBeta"
        SimInfo.solvers["NonLinearDynamicMultibody"]["time_integrator_settings"] = {
            "newmark_damp": 1e-3,
            "dt": 0.1,
        }

        SimInfo.solvers["NonLinearDynamicMultibody"]["relaxation_factor"] = 0.0

        SimInfo.solvers["DynamicCoupled"][
            "structural_solver"
        ] = "NonLinearDynamicMultibody"
        SimInfo.solvers["DynamicCoupled"][
            "structural_solver_settings"
        ] = SimInfo.solvers["NonLinearDynamicMultibody"]
        SimInfo.solvers["DynamicCoupled"]["aero_solver"] = "StepUvlm"
        SimInfo.solvers["DynamicCoupled"]["aero_solver_settings"] = SimInfo.solvers[
            "StepUvlm"
        ]
        SimInfo.solvers["DynamicCoupled"]["postprocessors"] = [
            "WriteVariablesTime",
            "BeamPlot",
            "AerogridPlot",
        ]
        SimInfo.solvers["DynamicCoupled"]["postprocessors_settings"] = {
            "WriteVariablesTime": SimInfo.solvers["WriteVariablesTime"],
            "BeamPlot": SimInfo.solvers["BeamPlot"],
            "AerogridPlot": SimInfo.solvers["AerogridPlot"],
        }

        ntimesteps = 10

        SimInfo.define_num_steps(ntimesteps)

        # Define dynamic simulation
        SimInfo.with_forced_vel = False
        SimInfo.with_dynamic_forces = False

        LC2 = gc.LagrangeConstraint()
        LC2.behaviour = "lin_vel_node_wrtG"
        LC2.velocity = np.zeros((ntimesteps, 3))
        LC2.velocity[: int(ntimesteps / 2), 1] = 0.5
        LC2.velocity[int(ntimesteps / 2) :, 1] = -0.5
        LC2.body_number = 0
        LC2.node_number = int((nnodes1 - 1) / 2)

        LC = []
        # LC.append(LC1)
        LC.append(LC2)

        # Define the multibody infromation for the tower and the rotor
        MB1 = gc.BodyInformation()
        MB1.body_number = 0
        MB1.FoR_position = np.zeros(
            (6,),
        )
        MB1.FoR_velocity = np.zeros(
            (6,),
        )
        MB1.FoR_acceleration = np.zeros(
            (6,),
        )
        MB1.FoR_movement = "free"
        MB1.quat = np.array([1.0, 0.0, 0.0, 0.0])

        MB = []
        MB.append(MB1)

        gc.clean_test_files(
            SimInfo.solvers["SHARPy"]["route"], SimInfo.solvers["SHARPy"]["case"]
        )
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(ntimesteps)
        beam1.generate_h5_files(
            SimInfo.solvers["SHARPy"]["route"], SimInfo.solvers["SHARPy"]["case"]
        )
        gc.generate_multibody_file(
            LC,
            MB,
            SimInfo.solvers["SHARPy"]["route"],
            SimInfo.solvers["SHARPy"]["case"],
        )

    def test_testfixnodevelocitywrtg(self):
        """ """
        solver_path = folder + "/fix_node_velocity_wrtG.sharpy"
        sharpy.sharpy_main.main(["", solver_path])

        # read output and compare
        output_path = folder + "/output/fix_node_velocity_wrtG/WriteVariablesTime/"
        pos_tip_data = np.loadtxt(
            "%sstruct_pos_node-1.dat" % output_path,
        )
        self.assertAlmostEqual(pos_tip_data[-1, 1], 9.999386, 4)
        self.assertAlmostEqual(pos_tip_data[-1, 2], -0.089333, 4)
        self.assertAlmostEqual(pos_tip_data[-1, 3], 0.0, 4)

    def tearDown(self):
        files_to_delete = [
            self.name + ".aero.h5",
            self.name + ".dyn.h5",
            self.name + ".fem.h5",
            self.name + ".mb.h5",
            self.name + ".sharpy",
        ]
        for f in files_to_delete:
            os.remove(folder + "/" + f)

        shutil.rmtree(folder + "/output/")
