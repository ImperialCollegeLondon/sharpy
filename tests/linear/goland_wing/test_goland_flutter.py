import numpy as np
import os
import unittest
import sharpy.cases.templates.flying_wings as wings
import sharpy.sharpy_main


class TestGolandFlutter(unittest.TestCase):
    def setup(self):
        # Problem Set up
        u_inf = 1.0
        alpha_deg = 0.0
        rho = 1.02
        num_modes = 4

        # Lattice Discretisation
        M = 16
        N = 32
        M_star_fact = 10

        # Linear UVLM settings
        integration_order = 2
        remove_predictor = False
        use_sparse = True

        # ROM Properties
        rom_settings = dict()
        rom_settings["algorithm"] = "mimo_rational_arnoldi"
        rom_settings["r"] = 6
        rom_settings["single_side"] = "observability"
        frequency_continuous_k = np.array([0.0])

        # Case Admin - Create results folders
        case_name = "goland_cs"
        case_nlin_info = "M%dN%dMs%d_nmodes%d" % (M, N, M_star_fact, num_modes)
        case_rom_info = "rom_MIMORA_r%d_sig%04d_%04dj" % (
            rom_settings["r"],
            frequency_continuous_k[-1].real * 100,
            frequency_continuous_k[-1].imag * 100,
        )

        case_name += case_nlin_info + case_rom_info

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

        ws.generate_aero_file()
        ws.generate_fem_file()

        frequency_continuous_w = 2 * u_inf * frequency_continuous_k / ws.c_ref
        rom_settings["frequency"] = frequency_continuous_w
        rom_settings["tangent_input_file"] = ws.route + "/" + ws.case_name + ".rom.h5"

        ws.config["SHARPy"] = {
            "flow": [
                "BeamLoader",
                "AerogridLoader",
                "StaticCoupled",
                "AerogridPlot",
                "BeamPlot",
                "Modal",
                "LinearAssembler",
                "FrequencyResponse",
                "AsymptoticStability",
                "SaveData",
            ],
            "case": ws.case_name,
            "route": ws.route,
            "write_screen": "off",
            "write_log": "on",
            "log_folder": self.route_test_dir + "/output/",
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
                    "ScalingDict": {
                        "length": 0.5 * ws.c_ref,
                        "speed": u_inf,
                        "density": rho,
                    },
                    "integr_order": integration_order,
                    "density": ws.rho,
                    "remove_predictor": remove_predictor,
                    "use_sparse": use_sparse,
                    "remove_inputs": ["u_gust"],
                    "rom_method": ["Krylov"],
                    "rom_method_settings": {"Krylov": rom_settings},
                },
            },
        }

        ws.config["AsymptoticStability"] = {
            "print_info": True,
            "velocity_analysis": [160, 180, 20],
        }

        ws.config["LinDynamicSim"] = {
            "dt": ws.dt,
            "n_tsteps": ws.n_tstep,
            "sys_id": "LinearAeroelastic",
            "postprocessors": ["BeamPlot", "AerogridPlot"],
            "postprocessors_settings": {
                "AerogridPlot": {
                    "u_inf": ws.u_inf,
                    "include_rbm": "on",
                    "include_applied_forces": "on",
                    "minus_m_star": 0,
                },
                "BeamPlot": {"include_rbm": "on", "include_applied_forces": "on"},
            },
        }

        ws.config["FrequencyResponse"] = {
            "quick_plot": "off",
            "frequency_unit": "k",
            "frequency_bounds": [0.0001, 1.0],
            "num_freqs": 100,
            "frequency_spacing": "log",
            "target_system": ["aeroelastic"],
        }

        ws.config["SaveData"] = {
            "save_aero": "off",
            "save_struct": "off",
            "save_rom": "on",
        }

        ws.config.write()

        self.data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])

    def run_rom_stable(self):
        ssrom = self.data.linear.linear_system.uvlm.rom["Krylov"].ssrom
        eigs_rom = np.linalg.eigvals(ssrom.A)

        assert all(np.abs(eigs_rom) <= 1.0), (
            "UVLM Krylov ROM is unstable - flutter speed may not be correct. Change"
            "ROM settings to achieve stability"
        )
        # print('ROM is stable')

    def run_flutter(self):
        flutter_ref_speed = 166  # at current discretisation

        # load results file - variables below determined by ``velocity_analysis`` setting in AsymptoticStability
        ulb = 160  # velocity lower bound
        uub = 180  # velocity upper bound
        num_u = 20  # n_speeds
        res = np.loadtxt(
            self.route_test_dir
            + "/output/%s/stability/" % self.data.settings["SHARPy"]["case"]
            + "/velocity_analysis_min%04d_max%04d_nvel%04d.dat"
            % (ulb * 10, uub * 10, num_u),
        )

        u_inf = res[:, 0]
        eval_real = res[:, 1]
        eval_imag = res[:, 2]

        # Flutter onset
        ind_zero_real = np.where(eval_real >= 0)[0][0]
        assert ind_zero_real > 0, "Flutter speed not below 165.00 m/s"
        flutter_speed = 0.5 * (u_inf[ind_zero_real] + u_inf[ind_zero_real - 1])
        flutter_frequency = np.sqrt(
            eval_real[ind_zero_real] ** 2 + eval_imag[ind_zero_real] ** 2
        )

        # print('Flutter speed = %.1f m/s' % flutter_speed)
        # print('Flutter frequency = %.2f rad/s' % flutter_frequency)
        assert np.abs(flutter_speed - flutter_ref_speed) / flutter_ref_speed < 1e-2, (
            " Flutter speed error greater than "
            "1 percent. \nFlutter speed: %.2f m/s\n"
            "Frequency: %.2f rad/s" % (flutter_speed, flutter_frequency)
        )

    def test_flutter(self):
        self.setup()
        self.run_rom_stable()
        self.run_flutter()

    def tearDown(self):
        import shutil

        folders = ["cases", "figures", "output"]
        for folder in folders:
            shutil.rmtree(self.route_test_dir + "/" + folder)


if __name__ == "__main__":
    unittest.main()
