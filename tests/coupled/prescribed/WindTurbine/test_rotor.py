import numpy as np
import os
import unittest
import shutil
import glob


folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


class TestRotor(unittest.TestCase):
    """
    NREL 5MW case
    J. Jonkman et al. "Definition of a 5-MW Reference Wind Turbine for Offshore System Development", NREL/TP-500-38060, 2009
    In this report, the whole system natural frequencies are provided. This implies that rotor frequencies
    are provided and influenced by nacelle characterisitcs
    """

    def setUp(self):
        import sharpy.utils.generate_cases as gc
        import cases.templates.template_wt as template_wt
        from sharpy.utils.constants import deg2rad

        ######################################################################
        ###########################  PARAMETERS  #############################
        ######################################################################
        # Case
        global case
        route = folder + '/'
        case = 'rotor'

        # Geometry discretization
        chord_panels = np.array([8], dtype=int)
        revs_in_wake = 1

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
        # time_steps = int(revs_to_simulate*2.*np.pi/dphi)
        time_steps = 2 # For the test cases

        mstar = int(revs_in_wake*2.*np.pi/dphi)

        op_params = {'rotation_velocity': rotation_velocity,
                     'pitch_deg': pitch_deg,
                     'wsp': WSP,
                     'dt': dt}

        geom_params = {'chord_panels':chord_panels,
                    'tol_remove_points': 1e-8,
                    'n_points_camber': 100,
                    'm_distribution': 'uniform'}

        excel_description = {'excel_file_name': route + '../../../../docs/source/content/example_notebooks/source/type02_db_NREL5MW_v02.xlsx',
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

        ######################################################################
        ######################  DEFINE SIMULATION  ###########################
        ######################################################################
        SimInfo = gc.SimulationInformation()
        SimInfo.set_default_values()

        SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                'Modal']
        SimInfo.solvers['SHARPy']['case'] = case
        SimInfo.solvers['SHARPy']['route'] = route
        SimInfo.solvers['SHARPy']['write_log'] = True
        SimInfo.solvers['SHARPy']['write_screen'] = 'off'
        SimInfo.solvers['SHARPy']['log_folder'] = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/output/'
        SimInfo.set_variable_all_dicts('dt', dt)
        SimInfo.set_variable_all_dicts('rho', air_density)

        SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

        SimInfo.solvers['Modal']['write_modes_vtk'] = False
        SimInfo.solvers['Modal']['save_data'] = True

        ######################################################################
        #######################  GENERATE FILES  #############################
        ######################################################################
        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        rotor.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()

    def test_rotor(self):
        import sharpy.sharpy_main

        solver_path = folder + '/' + case + '.sharpy'
        sharpy.sharpy_main.main(['', solver_path])

        # read output and compare
        output_path = folder + '/output/' + case + '/beam_modal_analysis/'
        freq_data = np.atleast_2d(np.genfromtxt(output_path + "frequencies.dat"))

        # Data from reference. Several values are provided, the average is used
        flap_1 = np.average([0.6664, 0.6296, 0.6675, 0.6686, 0.6993, 0.7019])*2*np.pi
        edge_1 = np.average([1.0793, 1.0740, 1.0898, 1.0877])*2*np.pi
        flap_2 = np.average([1.9337, 1.6507, 1.9223, 1.8558, 2.0205, 1.9601])*2*np.pi

        self.assertAlmostEqual(freq_data[0, 0], flap_1, 0) # 1st flapwise
        self.assertAlmostEqual(freq_data[0, 3], edge_1, 0) # 1st edgewise
        self.assertAlmostEqual(freq_data[0, 6], flap_2, 0) # 2nd flapwise

    def tearDown(self):
        files_to_delete = [case + '.aero.h5',
                           case + '.fem.h5',
                           case + '.sharpy',]

        for f in files_to_delete:
            os.remove(folder +'/' + f)

        shutil.rmtree(folder + '/output/')
