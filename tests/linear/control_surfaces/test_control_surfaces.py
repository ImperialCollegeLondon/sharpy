import numpy as np
import os
import unittest
import sharpy.cases.templates.flying_wings as wings
import sharpy.sharpy_main
import pickle


# @unittest.skip('Control Surface Test for visual inspection')
class TestGolandControlSurface(unittest.TestCase):
    def setup(self):
        self.deflection_degrees = 5
        # Problem Set up
        u_inf = 1.0
        alpha_deg = 0.0
        rho = 1.02
        num_modes = 2

        # Lattice Discretisation
        M = 4
        N = 16
        M_star_fact = 1

        # Linear UVLM settings
        integration_order = 2
        remove_predictor = False
        use_sparse = True

        # Case Admin - Create results folders
        case_name = "goland_cs"
        case_nlin_info = "M%dN%dMs%d_nmodes%d" % (M, N, M_star_fact, num_modes)

        case_name += case_nlin_info

        self.route_test_dir = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
        )
        fig_folder = self.route_test_dir + "/figures/"
        os.makedirs(fig_folder, exist_ok=True)

        # SHARPy nonlinear reference solution
        ws = wings.GolandControlSurface(
            M=M,
            N=N,
            Mstar_fact=M_star_fact,
            u_inf=u_inf,
            alpha=alpha_deg,
            cs_deflection=[0, 0],
            rho=rho,
            sweep=0,
            physical_time=2,
            n_surfaces=2,
            route=self.route_test_dir + "/cases",
            case_name=case_name,
        )

        ws.gust_intensity = 0.01
        ws.sigma = 1

        ws.clean_test_files()
        ws.update_derived_params()
        ws.set_default_config_dict()

        # Full moving control surface on the right wing
        ws.control_surface_chord[0] = M
        ws.control_surface_hinge_coord[0] = 0.5
        ws.control_surface[4, :] = 1

        ws.generate_aero_file()
        ws.generate_fem_file()

        x0 = np.zeros(256)
        lin_tsteps = 10
        u_vec = np.zeros((lin_tsteps, 8))
        # elevator
        u_vec[:, 4] = self.deflection_degrees * np.pi / 180
        np.savetxt(self.route_test_dir + "/cases/elevator.txt", u_vec[:, 4])
        ws.create_linear_files(x0, u_vec)

        ws.config["SHARPy"] = {
            "flow": [
                "BeamLoader",
                "AerogridLoader",
                "StaticCoupled",
                "AerogridPlot",
                "BeamPlot",
                "Modal",
                "LinearAssembler",
                # 'FrequencyResponse',
                "LinDynamicSim",
                "PickleData",
            ],
            "case": ws.case_name,
            "route": ws.route,
            "write_screen": "off",
            "write_log": "on",
            "log_folder": self.route_test_dir + "/output/" + ws.case_name + "/",
            "log_file": ws.case_name + ".log",
        }

        ws.config["BeamLoader"] = {"unsteady": "off", "orientation": ws.quat}

        ws.config["AerogridLoader"] = {
            "unsteady": "off",
            "aligned_grid": "on",
            "mstar": ws.Mstar_fact * ws.M,
            "freestream_dir": ws.u_inf_direction,
            "wake_shape_generator": "StraightWake",
            "wake_shape_generator_input": {
                "u_inf": ws.u_inf,
                "u_inf_direction": ws.u_inf_direction,
                "dt": ws.dt,
            },
        }

        ws.config["StaticUvlm"] = {
            "rho": ws.rho,
            "velocity_field_generator": "SteadyVelocityField",
            "velocity_field_input": {
                "u_inf": ws.u_inf,
                "u_inf_direction": ws.u_inf_direction,
            },
            "rollup_dt": ws.dt,
            "print_info": "on",
            "horseshoe": "off",
            "num_cores": 4,
            "n_rollup": 0,
            "rollup_aic_refresh": 0,
            "rollup_tolerance": 1e-4,
        }

        ws.config["StaticCoupled"] = {
            "print_info": "on",
            "max_iter": 200,
            "n_load_steps": 1,
            "tolerance": 1e-10,
            "relaxation_factor": 0.0,
            "aero_solver": "StaticUvlm",
            "aero_solver_settings": {
                "rho": ws.rho,
                "print_info": "off",
                "horseshoe": "off",
                "num_cores": 4,
                "n_rollup": 0,
                "rollup_dt": ws.dt,
                "rollup_aic_refresh": 1,
                "rollup_tolerance": 1e-4,
                "velocity_field_generator": "SteadyVelocityField",
                "velocity_field_input": {
                    "u_inf": ws.u_inf,
                    "u_inf_direction": ws.u_inf_direction,
                },
            },
            "structural_solver": "NonLinearStatic",
            "structural_solver_settings": {
                "print_info": "off",
                "max_iterations": 150,
                "num_load_steps": 4,
                "delta_curved": 1e-1,
                "min_delta": 1e-10,
                "gravity_on": "on",
                "gravity": 9.81,
            },
        }

        ws.config["AerogridPlot"] = {
            "include_rbm": "off",
            "include_applied_forces": "on",
            "minus_m_star": 0,
        }

        ws.config["AeroForcesCalculator"] = {
            "write_text_file": "on",
            "text_file_name": ws.case_name + "_aeroforces.csv",
            "screen_output": "on",
            "unsteady": "off",
        }

        ws.config["BeamPlot"] = {"include_rbm": "off", "include_applied_forces": "on"}

        ws.config["Modal"] = {
            "NumLambda": 20,
            "rigid_body_modes": "off",
            "print_matrices": "on",
            "save_data": "off",
            "rigid_modes_cg": "off",
            "continuous_eigenvalues": "off",
            "dt": 0,
            "plot_eigenvalues": False,
            "max_rotation_deg": 15.0,
            "max_displacement": 0.15,
            "write_modes_vtk": True,
            "use_undamped_modes": True,
        }

        ws.config["LinearAssembler"] = {
            "linear_system": "LinearAeroelastic",
            "linear_system_settings": {
                "beam_settings": {
                    "modal_projection": "on",
                    "inout_coords": "modes",
                    "discrete_time": "on",
                    "newmark_damp": 0.5e-4,
                    "discr_method": "newmark",
                    "dt": ws.dt,
                    "proj_modes": "undamped",
                    "use_euler": "off",
                    "num_modes": num_modes,
                    "print_info": "on",
                    "gravity": "on",
                    "remove_sym_modes": "on",
                    "remove_dofs": [],
                },
                "aero_settings": {
                    "dt": ws.dt,
                    "integr_order": integration_order,
                    "density": ws.rho,
                    "remove_predictor": remove_predictor,
                    "use_sparse": use_sparse,
                    "remove_inputs": ["u_gust"],
                },
            },
        }

        ws.config["LinDynamicSim"] = {
            "n_tsteps": lin_tsteps,
            "dt": ws.dt,
            "input_generators": [
                {
                    "name": "control_surface_deflection",
                    "index": 0,
                    "file_path": self.route_test_dir + "/cases/elevator.txt",
                },
                {
                    "name": "control_surface_deflection",
                    "index": 1,
                    "file_path": self.route_test_dir + "/cases/elevator.txt",
                },
            ],
            "postprocessors": ["AerogridPlot"],
            "postprocessors_settings": {
                "AerogridPlot": {
                    "include_rbm": "on",
                    "include_applied_forces": "on",
                    "minus_m_star": 0,
                },
            },
        }

        ws.config["PickleData"] = {}

        ws.config.write()

        restart = False  # useful for debugging to load pickle
        if not restart:
            self.data = sharpy.sharpy_main.main(
                ["", ws.route + ws.case_name + ".sharpy"]
            )
        else:
            with open(
                self.route_test_dir + "/output/" + ws.case_name + ".pkl", "rb"
            ) as f:
                self.data = pickle.load(f)

    def test_control_surface(self):
        self.setup()

        for i_surf in range(2):
            with self.subTest(i_surf=i_surf):
                zeta = self.data.aero.timestep_info[-1].zeta[i_surf]
                zeta0 = self.data.linear.tsaero0.zeta[i_surf]

                # zeta indices [(xyz), chord, span]
                if i_surf == 0:
                    # Full moving control surface with hinge at c=0.5
                    zeta_elev = (
                        zeta[:, -1, -1] - zeta[:, 0, -1]
                    )  # elevator starts at chordwise node 0
                    zeta_0elev = zeta0[:, -1, -1] - zeta0[:, 0, -1]
                elif i_surf == 1:
                    # mirrored surface. span index 0 is the wing tip
                    zeta_elev = (
                        zeta[:, -1, 0] - zeta[:, 2, 0]
                    )  # elevator starts at chordwise node 2
                    zeta_0elev = zeta0[:, -1, 0] - zeta0[:, 2, 0]

                deflection = np.arccos(
                    (zeta_elev.dot(zeta_0elev))
                    / np.linalg.norm(zeta_elev)
                    / np.linalg.norm(zeta_0elev)
                )

                deflection_actual_deg = deflection * 180 / np.pi
                np.testing.assert_array_almost_equal(
                    self.deflection_degrees, deflection_actual_deg, decimal=2
                )

    def tearDown(self):
        import shutil

        folders = ["cases", "figures", "output"]
        for folder in folders:
            shutil.rmtree(self.route_test_dir + "/" + folder)


if __name__ == "__main__":
    unittest.main()
