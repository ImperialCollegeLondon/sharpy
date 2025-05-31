import unittest
import numpy as np
import os
import shutil

import sharpy.utils.generate_cases as gc
import sharpy.cases.templates.template_wt as template_wt
from sharpy.utils.constants import deg2rad


class TestGenerateCases(unittest.TestCase):
    """
    Tests the generate_cases module
    Based on the NREL-5MW wind turbine rotor
    """

    def setUp(self):
        # remove_terminal_output = True
        ######################################################################
        ###########################  PARAMETERS  #############################
        ######################################################################
        # Case
        global case
        case = 'rotor'
        route = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/'

        # Geometry discretization
        chord_panels = np.array([4], dtype=int)
        revs_in_wake = 5

        # Operation
        rotation_velocity = 1.366190
        pitch_deg = 0. #degrees

        # Wind
        WSP = 11.4
        air_density = 1.225

        # Simulation
        dphi = 4.*deg2rad
        revs_to_simulate = 5

        ######################################################################
        ##########################  GENERATE WT  #############################
        ######################################################################
        dt = dphi/rotation_velocity
        time_steps = int(revs_to_simulate*2.*np.pi/dphi)
        time_steps = 1 # For the test cases
        mstar = int(revs_in_wake*2.*np.pi/dphi)
        mstar = 2 # For the test cases

        # Remove screen output
        # if remove_terminal_output:
        #     sys.stdout = open(os.devnull, "w")

        op_params = {'rotation_velocity': rotation_velocity,
                     'pitch_deg': pitch_deg,
                     'wsp': WSP,
                     'dt': dt}

        geom_params = {'chord_panels':chord_panels,
                    'tol_remove_points': 1e-8,
                    'n_points_camber': 100,
                    'm_distribution': 'uniform'}

        excel_description = {'excel_file_name': route + '../../docs/source/content/example_notebooks/source/type04_db_nrel5mw_oc3_v06.xlsx',
                            'excel_sheet_parameters': 'parameters',
                            'excel_sheet_structural_blade': 'structural_blade',
                            'excel_sheet_discretization_blade': 'discretization_blade',
                            'excel_sheet_aero_blade': 'aero_blade',
                            'excel_sheet_airfoil_info': 'airfoil_info',
                            'excel_sheet_airfoil_chord': 'airfoil_coord'}

        options = {'camber_effect_on_twist': False,
                   'user_defined_m_distribution_type': None,
                   'include_polars': False,
                   'separate_blades': False}

        rotor, hub_nodes = template_wt.rotor_from_excel_type03(op_params,
                                                    geom_params,
                                                    excel_description,
                                                    options)

        # Return the standard output to the terminal
        # if remove_terminal_output:
        #     sys.stdout.close()
        #     sys.stdout = sys.__stdout__

        ######################################################################
        ######################  DEFINE SIMULATION  ###########################
        ######################################################################
        SimInfo = gc.SimulationInformation()
        SimInfo.set_default_values()

        SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                'AerogridLoader',
                                'StaticCoupled',
                                'DynamicCoupled',
                                'SaveData']

        SimInfo.solvers['SHARPy']['case'] = case
        SimInfo.solvers['SHARPy']['write_screen'] = 'off'
        SimInfo.solvers['SHARPy']['route'] = route
        SimInfo.solvers['SHARPy']['write_log'] = True
        SimInfo.solvers['SHARPy']['log_folder'] = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/'
        SimInfo.set_variable_all_dicts('dt', dt)
        SimInfo.set_variable_all_dicts('rho', air_density)

        SimInfo.solvers['SteadyVelocityField']['u_inf'] = WSP
        SimInfo.solvers['SteadyVelocityField']['u_inf_direction'] = np.array([0., 0., 1.])
        SimInfo.set_variable_all_dicts('velocity_field_input', SimInfo.solvers['SteadyVelocityField'])
        SimInfo.set_variable_all_dicts('output', os.path.abspath(os.path.dirname(os.path.realpath(__file__))))

        SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

        SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
        SimInfo.solvers['AerogridLoader']['mstar'] = mstar
        SimInfo.solvers['AerogridLoader']['freestream_dir'] = np.array([0.,0.,0.])
        SimInfo.solvers['AerogridLoader']['wake_shape_generator'] = 'HelicoidalWake'
        SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': WSP,
                                                                           'u_inf_direction': np.array([0., 0., 1.]),
                                                                           'dt': dt,
                                                                           'rotation_velocity': rotation_velocity*np.array([0., 0., 1.])}

        SimInfo.solvers['StaticCoupled']['structural_solver'] = 'RigidDynamicPrescribedStep'
        SimInfo.solvers['StaticCoupled']['structural_solver_settings'] = SimInfo.solvers['RigidDynamicPrescribedStep']
        # SimInfo.solvers['StaticCoupled']['structural_solver'] = 'NonLinearDynamicPrescribedStep'
        # SimInfo.solvers['StaticCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicPrescribedStep']
        SimInfo.solvers['StaticCoupled']['aero_solver'] = 'StaticUvlm'
        SimInfo.solvers['StaticCoupled']['aero_solver_settings'] = SimInfo.solvers['StaticUvlm']

        SimInfo.solvers['StaticCoupled']['tolerance'] = 1e-6
        SimInfo.solvers['StaticCoupled']['n_load_steps'] = 0
        SimInfo.solvers['StaticCoupled']['relaxation_factor'] = 0.

        SimInfo.solvers['StaticUvlm']['horseshoe'] = False
        SimInfo.solvers['StaticUvlm']['num_cores'] = 8
        SimInfo.solvers['StaticUvlm']['n_rollup'] = 0
        SimInfo.solvers['StaticUvlm']['rollup_dt'] = dt
        SimInfo.solvers['StaticUvlm']['rollup_aic_refresh'] = 1
        SimInfo.solvers['StaticUvlm']['rollup_tolerance'] = 1e-8
        SimInfo.solvers['StaticUvlm']['rbm_vel_g'] = np.array([0., 0., 0.,
                                                               0., 0., rotation_velocity])

        SimInfo.solvers['StepUvlm']['convection_scheme'] = 2
        SimInfo.solvers['StepUvlm']['num_cores'] = 1
        SimInfo.solvers['StepUvlm']['cfl1'] = False

        # SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
        # SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
        SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'RigidDynamicPrescribedStep'
        SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['RigidDynamicPrescribedStep']
        SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
        SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
        SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['BeamPlot', 'AerogridPlot', 'Cleanup', 'SaveData']
        SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                                     'AerogridPlot': SimInfo.solvers['AerogridPlot'],
                                                                     'Cleanup': SimInfo.solvers['Cleanup'],
                                                                     'SaveData': SimInfo.solvers['SaveData']}
        SimInfo.solvers['DynamicCoupled']['minimum_steps'] = 0
        SimInfo.solvers['DynamicCoupled']['include_unsteady_force_contribution'] = True
        SimInfo.solvers['DynamicCoupled']['relaxation_factor'] = 0.
        SimInfo.solvers['DynamicCoupled']['final_relaxation_factor'] = 0.
        SimInfo.solvers['DynamicCoupled']['dynamic_relaxation'] = False
        SimInfo.solvers['DynamicCoupled']['relaxation_steps'] = 0

        SimInfo.define_num_steps(time_steps)

        # Define dynamic simulation
        SimInfo.with_forced_vel = True
        SimInfo.for_vel = np.zeros((time_steps,6), dtype=float)
        SimInfo.for_vel[:,5] = rotation_velocity
        SimInfo.for_acc = np.zeros((time_steps,6), dtype=float)
        SimInfo.with_dynamic_forces = True
        SimInfo.dynamic_forces = np.zeros((time_steps,rotor.StructuralInformation.num_node,6), dtype=float)


        ######################################################################
        #######################  GENERATE FILES  #############################
        ######################################################################
        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        rotor.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(time_steps)

    def test_generatecases(self):

        import sharpy.sharpy_main

        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/' + case +'.sharpy')
        sharpy.sharpy_main.main(['', solver_path])

    def tearDown(self):
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        solver_path += '/'
        files_to_delete = [case + '.aero.h5',
                           case + '.dyn.h5',
                           case + '.fem.h5',
                           case + '.sharpy',
                           '/output/' + case + '/log']

        for f in files_to_delete:
            os.remove(solver_path + f)

        output_path = solver_path + 'output/'
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
