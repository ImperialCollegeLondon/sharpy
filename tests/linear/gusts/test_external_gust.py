import numpy as np
import sharpy.utils.algebra as algebra
import sharpy.sharpy_main as smain
import unittest
import sharpy.cases.templates.flying_wings as wings
import os
import shutil


def generate_sharpy(alpha=0.0, case_name="hale_static", case_route="./", **kwargs):
    output_route = kwargs.get("output_route", "./output/")
    m = kwargs.get("M", 4)
    rho = kwargs.get("rho", 1.225)
    tolerance = kwargs.get("tolerance", 1e-5)
    u_inf = kwargs.get("u_inf", 10)
    tsteps = kwargs.get("tsteps", 100)

    ws = wings.QuasiInfinite(
        M=m,
        aspect_ratio=10,
        N=8,
        Mstar_fact=10,
        u_inf=u_inf,
        alpha=alpha,
        rho=rho,
        sweep=0,
        n_surfaces=2,
        route=case_route,
        case_name=case_name,
    )

    ws.gust_intensity = 0.01
    ws.sigma = 1e-1

    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()

    ws.generate_aero_file()
    ws.generate_fem_file()

    settings = dict()

    settings["SHARPy"] = {
        "case": case_name,
        "route": case_route,
        "flow": kwargs.get("flow", []),
        "write_screen": "off",
        "write_log": "on",
        "log_folder": output_route,
        "log_file": case_name + ".log",
    }

    settings["BeamLoader"] = {
        "unsteady": "on",
        "orientation": algebra.euler2quat(np.array([0.0, alpha, 0.0])),
    }

    settings["AerogridLoader"] = {
        "unsteady": "on",
        "aligned_grid": "on",
        "mstar": int(kwargs.get("wake_length", 10) * m),
        "control_surface_deflection": ["", ""],
        "control_surface_deflection_generator_settings": {"0": {}, "1": {}},
        "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {
            "u_inf": u_inf,
            "u_inf_direction": [1.0, 0.0, 0.0],
            "dt": ws.dt,
        },
    }

    settings["NonLinearStatic"] = {
        "print_info": "off",
        "max_iterations": 150,
        "num_load_steps": 1,
        "delta_curved": 1e-1,
        "min_delta": tolerance,
        "gravity_on": kwargs.get("gravity", "on"),
        "gravity": 9.81,
        "initial_position": [0.0, 0.0, 0.0],
    }

    settings["StaticUvlm"] = {
        "print_info": "on",
        "horseshoe": kwargs.get("horseshoe", "off"),
        "num_cores": 4,
        "velocity_field_generator": "SteadyVelocityField",
        "velocity_field_input": {"u_inf": u_inf, "u_inf_direction": [1.0, 0, 0]},
        "rho": rho,
    }

    settings["StaticCoupled"] = {
        "print_info": "off",
        "structural_solver": "NonLinearStatic",
        "structural_solver_settings": settings["NonLinearStatic"],
        "aero_solver": "StaticUvlm",
        "aero_solver_settings": settings["StaticUvlm"],
        "max_iter": 100,
        "n_load_steps": kwargs.get("n_load_steps", 1),
        "tolerance": kwargs.get("fsi_tolerance", 1e-5),
        "relaxation_factor": kwargs.get("relaxation_factor", 0.2),
    }

    settings["BeamPlot"] = {"include_FoR": "on"}

    settings["AerogridPlot"] = {
        "include_rbm": "off",
        "include_applied_forces": "on",
        "minus_m_star": 0,
        "u_inf": u_inf,
    }

    settings["AeroForcesCalculator"] = {
        "write_text_file": "on",
        "text_file_name": "aeroforces.txt",
        "screen_output": "on",
        "coefficients": True,
        "q_ref": 0.5 * rho * u_inf**2,
        "S_ref": 12.809,
    }

    settings["BeamPlot"] = {
        "include_rbm": "on",
        "include_applied_forces": "on",
        "include_FoR": "on",
    }

    struct_solver_settings = {
        "print_info": "off",
        "max_iterations": 950,
        "delta_curved": 1e-6,
        "min_delta": tolerance,
        "newmark_damp": 5e-3,
        "gravity_on": kwargs.get("gravity", "on"),
        "gravity": 9.81,
        "num_steps": 1,
        "dt": ws.dt,
    }

    gust_vec = kwargs.get("nl_gust", None)
    if gust_vec is not None:
        t_dom = np.linspace(0, ws.dt * (tsteps - 1), tsteps)
        np.savetxt(
            ws.route + "/gust.txt",
            np.column_stack(
                (t_dom, np.zeros_like(t_dom), np.zeros_like(t_dom), gust_vec)
            ),
        )

    step_uvlm_settings = {
        "print_info": "on",
        "num_cores": 4,
        "convection_scheme": 0,  # ws.wake_type,
        "vortex_radius": 1e-7,
        "velocity_field_generator": "GustVelocityField",
        "velocity_field_input": {
            "u_inf": u_inf * 1,
            "u_inf_direction": [1.0, 0.0, 0.0],
            "relative_motion": "on",
            "gust_shape": "time varying",
            "gust_parameters": {
                "file": ws.route + "/gust.txt",
            },
        },
        "rho": rho,
        "n_time_steps": tsteps,
        "dt": ws.dt,
        "gamma_dot_filtering": 3,
    }

    settings["DynamicCoupled"] = {
        "print_info": "on",
        "structural_solver": "NonLinearDynamicPrescribedStep",
        "structural_solver_settings": struct_solver_settings,
        "aero_solver": "StepUvlm",
        "aero_solver_settings": step_uvlm_settings,
        "fsi_substeps": 200,
        "fsi_tolerance": tolerance,
        "relaxation_factor": 0.3,
        "minimum_steps": 1,
        "relaxation_steps": 150,
        "final_relaxation_factor": 0.5,
        "n_time_steps": tsteps,  # ws.n_tstep,
        "dt": ws.dt,
        "include_unsteady_force_contribution": "on",
        "steps_without_unsteady_force": 5,
        "postprocessors": ["BeamPlot", "AerogridPlot", "WriteVariablesTime"],
        "postprocessors_settings": {
            "BeamLoads": {"csv_output": "off"},
            "BeamPlot": {"include_rbm": "on", "include_applied_forces": "on"},
            "AerogridPlot": {
                "u_inf": u_inf,
                "include_rbm": "on",
                "include_applied_forces": "on",
                "minus_m_star": 0,
            },
            "WriteVariablesTime": {
                "cleanup_old_solution": "on",
                "delimiter": ",",
                "FoR_variables": [
                    "total_forces",
                    "total_gravity_forces",
                    "for_pos",
                    "quat",
                ],
            },
        },
    }

    settings["Modal"] = {
        "print_info": True,
        "use_undamped_modes": True,
        "NumLambda": 50,
        "rigid_body_modes": "off",
        "write_modes_vtk": "on",
        "print_matrices": "on",
        "save_data": "on",
        "continuous_eigenvalues": "off",
        "dt": ws.dt,
        "plot_eigenvalues": False,
        "rigid_modes_ppal_axes": "on",
    }
    # ROM settings
    rom_settings = dict()
    rom_settings["algorithm"] = "mimo_rational_arnoldi"
    rom_settings["r"] = 10
    rom_settings["frequency"] = np.array([0], dtype=float)
    rom_settings["single_side"] = "observability"

    settings["LinearAssembler"] = {
        "linear_system": "LinearAeroelastic",
        "linearisation_tstep": -1,
        "linear_system_settings": {
            "beam_settings": {
                "modal_projection": "off",
                "inout_coords": "modes",
                "discrete_time": "on",
                "newmark_damp": 0.5e-4,
                "discr_method": "newmark",
                "dt": ws.dt,
                "proj_modes": "undamped",
                "use_euler": "on",
                "num_modes": 20,
                "print_info": "on",
                "gravity": kwargs.get("gravity", "on"),
                "remove_dofs": [],
                "remove_rigid_states": "off",
            },
            "aero_settings": {
                "dt": ws.dt,
                "integr_order": 2,
                "density": rho,
                "remove_predictor": "off",
                "use_sparse": False,
                "vortex_radius": 1e-7,
                "convert_to_ct": "off",
                "gust_assembler": kwargs.get("gust_name", ""),
                "gust_assembler_inputs": kwargs.get("gust_settings", {}),
            },
            "track_body": "on",
            "use_euler": "on",
        },
    }

    settings["AsymptoticStability"] = {
        "print_info": "on",
        "modes_to_plot": [],
        # 'velocity_analysis': [27, 29, 3],
        "display_root_locus": "off",
        "frequency_cutoff": 0,
        "export_eigenvalues": "on",
        "num_evals": 100,
    }

    settings["FrequencyResponse"] = {
        "target_system": ["aeroelastic", "aerodynamic", "structural"],
        "quick_plot": "off",
        "frequency_spacing": "log",
        "frequency_unit": "w",
        "frequency_bounds": [1e-3, 1e3],
        "num_freqs": 200,
        "print_info": "on",
    }

    settings["PickleData"] = {}

    settings["LinDynamicSim"] = {
        "n_tsteps": tsteps,
        "dt": ws.dt,
        "write_dat": ["x", "y", "u"],
        "input_generators": kwargs.get("linear_input_generators", []),
        "postprocessors": ["AerogridPlot"],
        "postprocessors_settings": {
            "AerogridPlot": {
                "include_rbm": "on",
                "include_applied_forces": "on",
                "minus_m_star": 0,
            },
        },
    }

    case_file = ws.settings_to_config(settings)
    return case_file


def run_case(case_file, pickle_file=None):
    if pickle_file is not None:
        restart = True
    else:
        restart = False
    if restart:
        data = smain.main(["", case_file, "-r", pickle_file])
    else:
        data = smain.main(["", case_file])

    return data


class TestGustAssembly(unittest.TestCase):
    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    directories = ["cases", "output", "inputs"]

    time_steps = 80
    gust_intensity = 0.01
    gust_ramp = 5

    M = 4
    u_inf = 1

    gust_vec = np.zeros(time_steps)

    @classmethod
    def setUpClass(cls):
        for folder in cls.directories:
            os.makedirs(cls.route_test_dir + "/" + folder, exist_ok=True)

        # Gust input profile
        cls.gust_vec[5 : cls.gust_ramp + 5] = np.linspace(
            0, cls.gust_intensity, cls.gust_ramp
        )
        cls.gust_vec[5 + cls.gust_ramp :] = cls.gust_intensity

    @classmethod
    def tearDownClass(cls):
        for folder in cls.directories:
            if os.path.isdir(cls.route_test_dir + "/" + folder):
                shutil.rmtree(cls.route_test_dir + "/" + folder)

    def linear_response(self):
        np.savetxt(self.route_test_dir + "/inputs/gust_linear.txt", self.gust_vec)

        input_generators = [
            {
                "name": "u_gust",
                "index": 0,
                "file_path": self.route_test_dir + "/inputs/gust_linear.txt",
            }
        ]

        data = self.run_sharpy_linear(
            gust_name="LeadingEdge", linear_input_generators=input_generators
        )  #

        return data

    def nonlinear_response(self):
        data = self.run_sharpy_nonlinear(nl_gust=self.gust_vec)

        return data

    def test_gust(self):
        linear = self.linear_response()
        nonlinear = self.nonlinear_response()
        final_linear = linear.aero.timestep_info[-1]
        final_nonlinear = nonlinear.aero.timestep_info[-1]
        with self.subTest("zeta"):
            np.testing.assert_almost_equal(
                final_linear.zeta[0][2, 0, -1],
                final_nonlinear.zeta[0][2, 0, -1],
                decimal=4,
            )
        with self.subTest("forces"):
            np.testing.assert_almost_equal(
                np.sum(final_linear.forces[0][2, :, 0]),
                np.sum(final_nonlinear.forces[0][2, :, 0]),
                decimal=4,
            )

        with self.subTest("gamma"):
            np.testing.assert_almost_equal(
                final_linear.gamma[0][2, 2], final_nonlinear.gamma[0][2, 2], decimal=4
            )

    def wingtip_timeseries(self, data):
        nsteps = len(data.aero.timestep_info)

        wingtip = np.zeros(nsteps)

        for i in range(nsteps):
            wingtip[i] = data.aero.timestep_info[i].zeta[0][2, 0, -1]

        return wingtip

    def gamma_timeseries(self, data):
        nsteps = len(data.aero.timestep_info)

        wingtip = np.zeros(nsteps)

        for i in range(nsteps):
            wingtip[i] = data.aero.timestep_info[i].gamma[0][0, -1]

        return wingtip

    def run_sharpy_nonlinear(self, **kwargs):
        flow = [
            "BeamLoader",
            "AerogridLoader",
            "StaticUvlm",
            "DynamicCoupled",
            # 'AeroForcesCalculator',
            "PickleData",
        ]

        alpha = 0  # 2 * np.pi / 180

        # Discretisation
        n_elem_multiplier = 1.5
        wake_length = 10
        horseshoe = False

        case_file = generate_sharpy(
            alpha=alpha,
            case_name="nonlinear",
            case_route=self.route_test_dir + "/cases/",
            output_route=self.route_test_dir + "/output/",
            flow=flow,
            u_inf=self.u_inf,
            M=self.M,
            n_elem_multiplier=n_elem_multiplier,
            horseshoe=horseshoe,
            wake_length=wake_length,
            relaxation_factor=0.6,
            tolerance=1e-5,
            gravity="off",
            fsi_tolerance=1e-5,
            tsteps=self.time_steps,
            **kwargs,
        )

        data = run_case(case_file)

        return data

    def run_sharpy_linear(self, **kwargs):
        flow = [
            "BeamLoader",
            "AerogridLoader",
            "StaticCoupled",
            "Modal",
            "AeroForcesCalculator",
            "LinearAssembler",
            "LinDynamicSim",
            "PickleData",
        ]

        restart = False
        if restart:
            flow = ["LinDynamicSim"]
            pickle_file = self.route_test_dir + "/output/linear.pkl"
        else:
            pickle_file = None
        alpha = 0  # 2 * np.pi / 180
        elevator = 0  # -0.5 * np.pi / 180
        thrust = 0  # 5

        # Discretisation
        M = 4
        n_elem_multiplier = 1.5
        wake_length = 10
        horseshoe = False

        case_file = generate_sharpy(
            alpha=alpha,
            case_name="linear",
            case_route=self.route_test_dir + "/cases/",
            output_route=self.route_test_dir + "/output",
            flow=flow,
            u_inf=self.u_inf,
            M=self.M,
            n_elem_multiplier=n_elem_multiplier,
            horseshoe=horseshoe,
            wake_length=wake_length,
            relaxation_factor=0.6,
            tolerance=1e-5,
            gravity="off",
            fsi_tolerance=1e-5,
            tsteps=self.time_steps,
            **kwargs,
        )

        data = run_case(case_file, pickle_file)

        return data

    def assert_gust_propagation(self, path_to_input, output_route):
        x_sharpy = np.loadtxt(output_route + "/lindynamicsim/x_out.dat")
        u_in = np.loadtxt(path_to_input)

        np.testing.assert_array_almost_equal(x_sharpy[1:, 0], u_in[:-1])

    def tearDown(self):
        folders = ["cases", "output"]
        import shutil

        for folder in folders:
            path = self.route_test_dir + "/" + folder
            if os.path.isdir(path):
                shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()
