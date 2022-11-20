"""
spar_from_excel_type04

Example of a file used to generate a floating wind turbine case
"""
# Load libraries
import sharpy.utils.generate_cases as gc
import cases.templates.template_wt as template_wt
import numpy as np
import os
import sharpy.utils.algebra as algebra
from sharpy.utils.constants import deg2rad
import sharpy.sharpy_main
import unittest
import shutil

case = "flex_wsp18_val"


class TestFloatingWindTrubine(unittest.TestCase):
    """
    Test and example to run floating wind turbines
    """

    def generate_floating_wind_turbine(self, restart=False):
        ######################################################################
        ###########################  PARAMETERS  #############################
        ######################################################################
        # Case
        route = os.path.dirname(os.path.realpath(__file__)) + "/"

        # Geometry discretization
        chord_panels = np.array([16], dtype=int)
        revs_in_wake = 1

        # Operation
        rotation_velocity = 12.1 * 2 * np.pi / 60
        pitch_deg = 14.6  # degrees
        yaw_deg = 0.0  # degrees

        # Wind
        WSP = 18.0
        air_density = 1.225
        gravity_on = True
        grav_value = 9.80665

        # Simulation
        dphi = 4.0 * deg2rad
        revs_to_simulate = 120
        structural_substeps = 0

        ######################################################################
        ##########################  GENERATE WT  #############################
        ######################################################################
        dt = dphi / rotation_velocity
        # time_steps = int(revs_to_simulate*2.*np.pi/dphi)
        if not restart:
            time_steps = 1
        else:
            time_steps = 2
        struct_dt = dt / (structural_substeps + 1)

        op_params = {}
        op_params["rotation_velocity"] = rotation_velocity
        op_params["pitch_deg"] = pitch_deg
        op_params["wsp"] = WSP
        op_params["dt"] = dt

        geom_params = {}
        geom_params["chord_panels"] = chord_panels
        geom_params["tol_remove_points"] = 1e-8
        geom_params["n_points_camber"] = 100
        geom_params["h5_cross_sec_prop"] = None
        geom_params["m_distribution"] = "uniform"

        options = {}
        options["camber_effect_on_twist"] = False
        options["user_defined_m_distribution_type"] = None
        options["include_polars"] = True
        options["separate_blades"] = True
        options["concentrate_spar"] = True
        options["twist_in_aero"] = False

        excel_description = {}
        excel_description[
            "excel_file_name"
        ] = "../../../../docs/source/content/example_notebooks/source/type04_db_nrel5mw_oc3_v06.xlsx"
        excel_description["excel_sheet_parameters"] = "parameters"
        excel_description["excel_sheet_structural_tower"] = "structural_tower"
        excel_description["excel_sheet_structural_spar"] = "structural_spar"
        excel_description["excel_sheet_structural_blade"] = "structural_blade"
        excel_description["excel_sheet_discretization_blade"] = "discretization_blade"
        excel_description["excel_sheet_aero_blade"] = "aero_blade"
        excel_description["excel_sheet_airfoil_info"] = "airfoil_info"
        excel_description["excel_sheet_airfoil_chord"] = "airfoil_coord"

        spar, LC, MB = template_wt.spar_from_excel_type04(
            op_params, geom_params, excel_description, options
        )

        tower_top_node = 24
        hub_node = 25
        blade_tip_node = 51

        for ilc in range(len(LC)):
            LC[ilc].behaviour = "hinge_node_FoR_pitch"
            LC[ilc].rotor_vel = rotation_velocity
            del LC[ilc].rot_vect
            LC[ilc].scalingFactor = 1e6
            LC[ilc].penaltyFactor = 0.0

        MB[0].FoR_movement = "free"

        ######################################################################
        ######################  DEFINE SIMULATION  ###########################
        ######################################################################
        SimInfo = gc.SimulationInformation()
        SimInfo.set_default_values()

        if not restart:
            SimInfo.solvers["SHARPy"]["flow"] = [
                "BeamLoader",
                "AerogridLoader",
                "StaticUvlm",
                "BeamPlot",
                "AerogridPlot",
                "DynamicCoupled",
                "PickleData",
            ]

            SimInfo.solvers["SHARPy"]["case"] = case
        else:
            SimInfo.solvers["SHARPy"]["flow"] = [
                "DynamicCoupled",
                "PickleData",
            ]

            SimInfo.solvers["SHARPy"]["case"] = "restart_%s" % case

        SimInfo.solvers["SHARPy"]["route"] = route
        SimInfo.solvers["SHARPy"]["write_log"] = True
        SimInfo.solvers["SHARPy"]["write_screen"] = False
        SimInfo.solvers["SHARPy"]["log_file"] = (
            "log_%s" % SimInfo.solvers["SHARPy"]["case"]
        )
        SimInfo.set_variable_all_dicts("dt", dt)
        SimInfo.set_variable_all_dicts("rho", air_density)

        SimInfo.solvers["SteadyVelocityField"]["u_inf"] = WSP
        SimInfo.solvers["SteadyVelocityField"]["u_inf_direction"] = np.array(
            [0.0, -np.sin(yaw_deg * deg2rad), np.cos(yaw_deg * deg2rad)]
        )

        SimInfo.solvers["BeamLoader"]["unsteady"] = "on"

        centre_rot_ini = np.array([87.6, 0.0, -5.0191])
        rbm_vel_g_ini = np.zeros((6))
        rbm_vel_g_ini[3:6] = np.array([0.0, 0.0, rotation_velocity])

        quat0 = algebra.euler2quat(np.array([0.0, -2.28 * deg2rad, 0.0]))
        for_pos0 = np.array([0.0, 0.0, 11.6])
        SimInfo.solvers["BeamLoader"]["orientation"] = quat0.copy()
        SimInfo.solvers["BeamLoader"]["for_pos"] = for_pos0.copy()

        rot_mat0 = algebra.quat2rotation(quat0)
        centre_rot = np.dot(rot_mat0, centre_rot_ini) + for_pos0
        rbm_vel_g = np.zeros((6))
        rbm_vel_g[3:6] = np.dot(rot_mat0, rbm_vel_g_ini[3:6])

        MB[0].FoR_position[0:3] = for_pos0.copy()
        MB[0].quat = quat0.copy()

        for ibody in range(1, len(MB)):
            MB[ibody].FoR_position[0:3] = centre_rot.copy()
            MB[ibody].FoR_velocity[3:6] = rbm_vel_g[3:6]
            az = (360.0 / 3) * (ibody - 1)
            rot_mat_az = algebra.rotation3d_z(az * deg2rad)
            quat = algebra.rotation2quat(np.dot(rot_mat_az, rot_mat0))
            MB[ibody].quat[0:4] = quat.copy()

        # Compute mstar
        SimInfo.solvers["AerogridLoader"]["wake_shape_generator"] = "HelicoidalWake"
        SimInfo.solvers["AerogridLoader"]["wake_shape_generator_input"] = {
            "u_inf": WSP,
            "u_inf_direction": SimInfo.solvers["SteadyVelocityField"][
                "u_inf_direction"
            ],
            "rotation_velocity": rbm_vel_g[3:6],
            "h_ref": centre_rot[0],
            "h_corr": 0.0,
            "dt": dt,
            "dphi1": dphi,
            "ndphi1": int(10),
            "r": 1.05,
            "dphimax": 10 * deg2rad,
        }

        import sharpy.utils.generator_interface as gi

        gi.dictionary_of_generators(print_info=False)
        hw = gi.dict_of_generators["HelicoidalWake"]
        wsg_in = SimInfo.solvers["AerogridLoader"][
            "wake_shape_generator_input"
        ]  # for simplicity
        angle = 0
        mstar = 0
        while angle < (revs_in_wake * 2 * np.pi):
            mstar += 1
            angle += hw.get_dphi(
                mstar, wsg_in["dphi1"], wsg_in["ndphi1"], wsg_in["r"], wsg_in["dphimax"]
            )

        SimInfo.solvers["AerogridLoader"]["unsteady"] = "on"
        SimInfo.solvers["AerogridLoader"]["mstar"] = mstar
        SimInfo.solvers["AerogridLoader"]["freestream_dir"] = np.array([0.0, 0, 0])

        struct_static_solver = "NonLinearStatic"
        SimInfo.solvers[struct_static_solver]["gravity_on"] = gravity_on
        SimInfo.solvers[struct_static_solver]["gravity"] = grav_value
        SimInfo.solvers[struct_static_solver]["gravity_dir"] = np.array([1.0, 0.0, 0.0])
        SimInfo.solvers[struct_static_solver]["max_iterations"] = 100
        SimInfo.solvers[struct_static_solver]["num_load_steps"] = 1
        SimInfo.solvers[struct_static_solver]["min_delta"] = 1e-5
        SimInfo.solvers[struct_static_solver]["newmark_damp"] = 1e-1
        SimInfo.solvers[struct_static_solver]["dt"] = dt

        SimInfo.solvers["StaticCoupled"]["structural_solver"] = struct_static_solver
        SimInfo.solvers["StaticCoupled"][
            "structural_solver_settings"
        ] = SimInfo.solvers[struct_static_solver]
        SimInfo.solvers["StaticCoupled"]["aero_solver"] = "StaticUvlm"
        SimInfo.solvers["StaticCoupled"]["aero_solver_settings"] = SimInfo.solvers[
            "StaticUvlm"
        ]

        SimInfo.solvers["StaticCoupled"]["tolerance"] = 1e-8
        SimInfo.solvers["StaticCoupled"]["n_load_steps"] = 0
        SimInfo.solvers["StaticCoupled"]["relaxation_factor"] = 0.0
        SimInfo.solvers["StaticCoupled"]["max_iter"] = 100

        SimInfo.solvers["StaticUvlm"]["horseshoe"] = False
        SimInfo.solvers["StaticUvlm"]["num_cores"] = 8
        SimInfo.solvers["StaticUvlm"]["n_rollup"] = 0
        SimInfo.solvers["StaticUvlm"]["rollup_dt"] = dt
        SimInfo.solvers["StaticUvlm"]["rollup_aic_refresh"] = 1
        SimInfo.solvers["StaticUvlm"]["rollup_tolerance"] = 1e-8
        SimInfo.solvers["StaticUvlm"]["rbm_vel_g"] = rbm_vel_g
        SimInfo.solvers["StaticUvlm"]["centre_rot_g"] = centre_rot
        SimInfo.solvers["StaticUvlm"]["cfl1"] = False
        SimInfo.solvers["StaticUvlm"]["vortex_radius"] = 1e-6
        SimInfo.solvers["StaticUvlm"]["vortex_radius_wake_ind"] = 1e-3
        SimInfo.solvers["StaticUvlm"][
            "velocity_field_generator"
        ] = "SteadyVelocityField"
        SimInfo.solvers["StaticUvlm"]["velocity_field_input"] = SimInfo.solvers[
            "SteadyVelocityField"
        ]
        SimInfo.solvers["StaticUvlm"]["map_forces_on_struct"] = True

        SimInfo.solvers["FloatingForces"]["n_time_steps"] = time_steps
        SimInfo.solvers["FloatingForces"]["dt"] = dt
        SimInfo.solvers["FloatingForces"]["water_density"] = 1025  # kg/m3
        SimInfo.solvers["FloatingForces"]["gravity"] = grav_value
        SimInfo.solvers["FloatingForces"]["gravity_dir"] = [1.0, 0.0, 0.0]
        SimInfo.solvers["FloatingForces"][
            "floating_file_name"
        ] = "oc3_cs_v07.floating.h5"
        SimInfo.solvers["FloatingForces"]["concentrate_spar"] = True
        SimInfo.solvers["FloatingForces"]["method_matrices_freq"] = "rational_function"
        SimInfo.solvers["FloatingForces"]["matrices_freq"] = 2 * np.pi / 120.0
        SimInfo.solvers["FloatingForces"]["steps_constant_matrices"] = 0
        SimInfo.solvers["FloatingForces"]["method_wave"] = "jonswap"
        SimInfo.solvers["FloatingForces"]["wave_amplitude"] = 0.0
        SimInfo.solvers["FloatingForces"]["wave_freq"] = 0.0
        SimInfo.solvers["FloatingForces"]["wave_Tp"] = 10.0
        SimInfo.solvers["FloatingForces"]["wave_Hs"] = 6.0
        SimInfo.solvers["FloatingForces"]["wave_incidence"] = 0.0 * deg2rad
        SimInfo.solvers["FloatingForces"]["added_mass_in_mass_matrix"] = True
        SimInfo.solvers["FloatingForces"]["write_output"] = True

        SimInfo.solvers["StepUvlm"]["convection_scheme"] = 2
        SimInfo.solvers["StepUvlm"]["num_cores"] = 8
        SimInfo.solvers["StepUvlm"]["velocity_field_generator"] = "SteadyVelocityField"
        SimInfo.solvers["StepUvlm"]["velocity_field_input"] = SimInfo.solvers[
            "SteadyVelocityField"
        ]
        SimInfo.solvers["StepUvlm"]["cfl1"] = False
        SimInfo.solvers["StepUvlm"]["interp_coords"] = 0
        SimInfo.solvers["StepUvlm"]["filter_method"] = 0
        SimInfo.solvers["StepUvlm"]["interp_method"] = 3
        SimInfo.solvers["StepUvlm"]["centre_rot"] = centre_rot
        SimInfo.solvers["StepUvlm"]["vortex_radius"] = 1e-6
        SimInfo.solvers["StepUvlm"]["vortex_radius_wake_ind"] = 1e-3

        struct_dyn_solver = "NonLinearDynamicMultibody"
        SimInfo.solvers[struct_dyn_solver]["gravity_on"] = gravity_on
        SimInfo.solvers[struct_dyn_solver]["gravity"] = grav_value
        SimInfo.solvers[struct_dyn_solver]["gravity_dir"] = np.array([1.0, 0.0, 0.0])
        SimInfo.solvers[struct_dyn_solver]["max_iterations"] = 300
        SimInfo.solvers[struct_dyn_solver]["min_delta"] = 1e-6
        SimInfo.solvers[struct_dyn_solver]["newmark_damp"] = 1e-1
        SimInfo.solvers[struct_dyn_solver]["dt"] = struct_dt
        SimInfo.solvers[struct_dyn_solver]["write_lm"] = True
        SimInfo.solvers[struct_dyn_solver]["allow_skip_step"] = False
        SimInfo.solvers[struct_dyn_solver]["relax_factor_lm"] = 0.0
        SimInfo.solvers[struct_dyn_solver]["rigid_bodies"] = False
        SimInfo.solvers[struct_dyn_solver]["zero_ini_dot_ddot"] = True
        SimInfo.solvers[struct_dyn_solver]["time_integrator"] = "NewmarkBeta"
        SimInfo.solvers[struct_dyn_solver]["time_integrator_settings"] = {
            "newmark_damp": 1e-1,
            "dt": struct_dt,
        }

        SimInfo.solvers["SaveData"]["compress_float"] = True
        SimInfo.solvers["SaveData"]["save_wake"] = False

        SimInfo.solvers["WriteVariablesTime"]["FoR_variables"] = [
            "for_pos",
            "for_vel",
            "quat",
        ]
        SimInfo.solvers["WriteVariablesTime"]["structure_variables"] = ["pos"]
        SimInfo.solvers["WriteVariablesTime"]["structure_nodes"] = [
            tower_top_node,
            blade_tip_node,
        ]

        SimInfo.solvers["PickleData"]["stride"] = 180
        SimInfo.solvers["AerogridPlot"]["stride"] = 180

        SimInfo.solvers["DynamicCoupled"]["structural_solver"] = struct_dyn_solver
        SimInfo.solvers["DynamicCoupled"][
            "structural_solver_settings"
        ] = SimInfo.solvers[struct_dyn_solver]
        SimInfo.solvers["DynamicCoupled"]["aero_solver"] = "StepUvlm"
        SimInfo.solvers["DynamicCoupled"]["aero_solver_settings"] = SimInfo.solvers[
            "StepUvlm"
        ]
        SimInfo.solvers["DynamicCoupled"]["postprocessors"] = [
            "Cleanup",
            "BeamPlot",
            "AerogridPlot",
            "PickleData",
            "SaveData",
            "WriteVariablesTime",
        ]
        SimInfo.solvers["DynamicCoupled"]["postprocessors_settings"] = {
            "Cleanup": SimInfo.solvers["Cleanup"],
            "BeamPlot": SimInfo.solvers["BeamPlot"],
            "AerogridPlot": SimInfo.solvers["AerogridPlot"],
            "PickleData": SimInfo.solvers["PickleData"],
            "SaveData": SimInfo.solvers["SaveData"],
            "WriteVariablesTime": SimInfo.solvers["WriteVariablesTime"],
        }

        SimInfo.solvers["DynamicCoupled"]["minimum_steps"] = 0
        SimInfo.solvers["DynamicCoupled"]["include_unsteady_force_contribution"] = True
        SimInfo.solvers["DynamicCoupled"]["relaxation_factor"] = 0.0
        SimInfo.solvers["DynamicCoupled"]["final_relaxation_factor"] = 0.0
        SimInfo.solvers["DynamicCoupled"]["dynamic_relaxation"] = False
        SimInfo.solvers["DynamicCoupled"]["relaxation_steps"] = 0
        SimInfo.solvers["DynamicCoupled"]["fsi_tolerance"] = 1e-6
        SimInfo.solvers["DynamicCoupled"]["structural_substeps"] = structural_substeps
        SimInfo.solvers["DynamicCoupled"]["runtime_generators"] = {
            "FloatingForces": SimInfo.solvers["FloatingForces"]
        }

        power = 5296609.984
        GBR = 97.0
        PID0 = [0.006275604, 0.0008965149, 0.0]  # Offshore

        # Gain scheduler
        pitch_deg_doubled_sens = 6.302336  # deg
        gk = 1.0 / (1 + pitch_deg / pitch_deg_doubled_sens)
        PID = np.zeros((3))
        for i in range(3):
            PID[i] = PID0[i] * gk

        SimInfo.solvers["DynamicCoupled"]["controller_id"][
            "colective_pitch"
        ] = "BladePitchPid"
        SimInfo.solvers["DynamicCoupled"]["controller_settings"][
            "colective_pitch"
        ] = dict()
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "P"
        ] = PID[0]
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "I"
        ] = PID[1]
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "D"
        ] = PID[2]
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "sp_type"
        ] = "gen_vel"
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "sp_source"
        ] = "const"
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "sp_const"
        ] = (rotation_velocity * GBR)
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "gen_model_const_var"
        ] = "torque"
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "gen_model_const_value"
        ] = (power / rotation_velocity / GBR)
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "GBR"
        ] = GBR
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "inertia_dt"
        ] = 43702538.05
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "dt"
        ] = dt
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "ntime_steps"
        ] = time_steps
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "blade_num_body"
        ] = [1, 2, 3]
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "min_pitch"
        ] = 0.0
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "max_pitch"
        ] = (90.0 * deg2rad)
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "initial_pitch"
        ] = (pitch_deg * deg2rad)
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "pitch_sp"
        ] = (pitch_deg * deg2rad)
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "initial_rotor_vel"
        ] = rotation_velocity
        SimInfo.solvers["DynamicCoupled"]["controller_settings"]["colective_pitch"][
            "nocontrol_steps"
        ] = 0

        SimInfo.define_num_steps(time_steps)

        # Define dynamic simulation
        SimInfo.with_forced_vel = False
        SimInfo.with_dynamic_forces = False

        ######################################################################
        #######################  GENERATE FILES  #############################
        ######################################################################
        gc.clean_test_files(
            SimInfo.solvers["SHARPy"]["route"], SimInfo.solvers["SHARPy"]["case"]
        )
        spar.generate_h5_files(
            SimInfo.solvers["SHARPy"]["route"], SimInfo.solvers["SHARPy"]["case"]
        )
        SimInfo.generate_solver_file()
        gc.generate_multibody_file(
            LC,
            MB,
            SimInfo.solvers["SHARPy"]["route"],
            SimInfo.solvers["SHARPy"]["case"],
        )

        return SimInfo.solvers["SHARPy"]["case"]

    def test_floating_wind_turbine(self):
        name = self.generate_floating_wind_turbine(restart=False)
        solver_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)) + "/" + name + ".sharpy"
        )
        sharpy.sharpy_main.main(["", solver_path])

        name_restart = self.generate_floating_wind_turbine(restart=True)
        solver_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__)) + "/" + name_restart + ".sharpy"
        )
        restart_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/output/"
            + name
            + "/"
            + name
            + ".pkl"
        )
        sharpy.sharpy_main.main(["", "-r" + restart_path, solver_path])

        self.clean_files(name)

    def clean_files(self, case):
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        solver_path += "/"
        files_to_delete = [
            case + ".aero.h5",
            case + ".fem.h5",
            case + ".mb.h5",
            case + ".sharpy",
            "restart_" + case + ".aero.h5",
            "restart_" + case + ".fem.h5",
            "restart_" + case + ".mb.h5",
            "restart_" + case + ".sharpy",
        ]
        for f in files_to_delete:
            os.remove(solver_path + f)

        shutil.rmtree(solver_path + "output/")
