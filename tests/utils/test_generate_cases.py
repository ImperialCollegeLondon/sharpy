import sharpy.utils.generate_cases as gc
import cases.templates.template_wt_excel_type01 as template_wt_excel_type01
import unittest
import numpy as np
import os


class TestGenerateCases(unittest.TestCase):
    """
    Tests the generate_cases module
    """
    SimInfo = list()

    def test_01(self):
        print('Test not being run, needs fixing!')
        return
        nodes_per_elem = 3

        # beam1: uniform and symmetric with aerodynamic properties equal to zero
        nnodes1 = 11
        length1  = 10.
        mass_per_unit_length = 1.
        mass_iner = 1e-4
        EA = 1e9
        GA = 1e9
        GJ = 1e9
        EI = 1e9

        # Create beam1
        beam1 = gc.AeroelasticInformation()
        # Structural information
        beam1.StructuralInformation.num_node = nnodes1
        beam1.StructuralInformation.num_node_elem = nodes_per_elem
        beam1.StructuralInformation.compute_basic_num_elem()
        beam1.StructuralInformation.set_to_zero(beam1.StructuralInformation.num_node_elem, beam1.StructuralInformation.num_node, beam1.StructuralInformation.num_elem)
        node_pos = np.zeros((nnodes1, 3), )
        node_pos[:, 2] = np.linspace(0.0, length1, nnodes1)
        beam1.StructuralInformation.generate_uniform_sym_beam(node_pos, mass_per_unit_length, mass_iner, EA, GA, GJ, EI, num_node_elem = 3, y_BFoR = 'y_AFoR', num_lumped_mass=1)
        beam1.StructuralInformation.lumped_mass_nodes[0] = 5
        beam1.StructuralInformation.lumped_mass[0] = 0.5
        beam1.StructuralInformation.lumped_mass_inertia[0] = np.eye(3)
        beam1.StructuralInformation.lumped_mass_position[0,:] = np.zeros((3,),)
        # Aerodynamic information
        beam1.AerodynamicInformation.set_to_zero(beam1.StructuralInformation.num_node_elem, beam1.StructuralInformation.num_node, beam1.StructuralInformation.num_elem)

        # beam2
        beam2 = beam1.copy()
        beam2.StructuralInformation.rotate_around_origin(np.array([0.,1.,0.]), 90*np.pi/180.)
        beam2.StructuralInformation.coordinates[:,2] += length1
        beam2.StructuralInformation.lumped_mass_nodes[0] = 0
        airfoil = np.zeros((1,20,2),)
        airfoil[0,:,0] = np.linspace(0.,1.,20)
        beam2.AerodynamicInformation.create_one_uniform_aerodynamics(
                                            beam2.StructuralInformation,
                                            chord = 1.,
                                            twist = 0.,
                                            sweep = 0.,
                                            num_chord_panels = 4,
                                            m_distribution = 'uniform',
                                            elastic_axis = 0.5,
                                            num_points_camber = 20,
                                            airfoil = airfoil)

        # beam3
        nnodes3 = 9
        beam3 = gc.AeroelasticInformation()
        # Structural information
        beam3.StructuralInformation.num_node = nnodes3
        beam3.StructuralInformation.num_node_elem = nodes_per_elem
        beam3.StructuralInformation.compute_basic_num_elem()
        beam3.StructuralInformation.set_to_zero(beam3.StructuralInformation.num_node_elem, beam3.StructuralInformation.num_node, beam3.StructuralInformation.num_elem)
        node_pos = np.zeros((nnodes3, 3), )
        node_pos[:,0] = length1
        node_pos[:, 2] = np.linspace(length1, 0.0, nnodes3)

        beam3.StructuralInformation.generate_uniform_beam(node_pos, mass_per_unit_length, mass_iner, 2.*mass_iner, 3.*mass_iner, np.zeros((3,),), EA, GA, 2.0*GA, GJ, EI, 4.*EI, num_node_elem = 3, y_BFoR = 'y_AFoR', num_lumped_mass=1)
        beam3.StructuralInformation.lumped_mass_nodes[0] = nnodes3-1
        beam3.StructuralInformation.lumped_mass[0] = 0.25
        beam3.StructuralInformation.lumped_mass_inertia[0] = 2.*np.eye(3)
        beam3.StructuralInformation.lumped_mass_position[0,:] = np.zeros((3,),)

        # Aerodynamic information
        airfoils = np.zeros((1, 20, 2), )
        airfoils[0, :, 0] = np.linspace(0., 1., 20)
        beam3.AerodynamicInformation.create_aerodynamics_from_vec(StructuralInformation = beam3.StructuralInformation,
                                     vec_aero_node = np.ones((nnodes3), dtype = bool),
                                     vec_chord = np.linspace(1.,0.1,nnodes3),
                                     vec_twist = 0.1*np.ones((nnodes3,),),
                                     vec_sweep = 0.2*np.ones((nnodes3,), ),
                                     vec_surface_m = 4*np.ones((1,), dtype = int),
                                     vec_surface_distribution = np.zeros((beam3.StructuralInformation.num_elem,), dtype=int),
                                     vec_m_distribution = np.array(['uniform']),
                                     vec_elastic_axis = 0.5*np.ones((nnodes3,), ),
                                     vec_airfoil_distribution = np.zeros((nnodes3,), dtype=int),
                                     airfoils = airfoils)

        beam1.assembly(beam2, beam3)
        beam1.StructuralInformation.boundary_conditions[0] = 1
        beam1.StructuralInformation.boundary_conditions[-1] = -1
        beam1.remove_duplicated_points(1e-3)

        beam1.StructuralInformation.check_StructuralInformation()
        beam1.AerodynamicInformation.check_AerodynamicInformation(beam1.StructuralInformation)

        # SOLVER CONFIGURATION
        SimInfo = gc.SimulationInformation()
        self.SimInfo.append(SimInfo)
        SimInfo.set_default_values()

        SimInfo.define_uinf(np.array([0.0,1.0,0.0]), 10.)

        SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                'AerogridLoader',
                                'StaticCoupled',
                                'DynamicCoupled',
                                'AerogridPlot',
                                'BeamPlot'
                                ]
        SimInfo.solvers['SHARPy']['case'] = 'test_01'
        SimInfo.solvers['SHARPy']['route'] = os.path.dirname(os.path.realpath(__file__)) + '/'
        SimInfo.set_variable_all_dicts('dt', 0.05)

        SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

        SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
        SimInfo.solvers['AerogridLoader']['mstar'] = 13

        # Default values for NonLinearStatic
        # Default values for StaticUvlm
        # Default values for NonLinearDynamicCoupledStep
        # Default values for StepUvlm

        SimInfo.solvers['StaticCoupled']['structural_solver'] = 'NonLinearStatic'
        SimInfo.solvers['StaticCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearStatic']
        SimInfo.solvers['StaticCoupled']['aero_solver'] = 'StaticUvlm'
        SimInfo.solvers['StaticCoupled']['aero_solver_settings'] = SimInfo.solvers['StaticUvlm']

        SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicCoupledStep'
        SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicCoupledStep']
        SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
        SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
        SimInfo.solvers['DynamicCoupled']['post_processors'] = ['BeamPlot', 'AerogridPlot']
        SimInfo.solvers['DynamicCoupled']['post_processor_settings'] = {'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                                     'AerogridPlot': SimInfo.solvers['AerogridPlot']}
        SimInfo.define_num_steps(20)

        # Define dynamic simulation
        SimInfo.with_forced_vel = True
        SimInfo.for_vel = np.zeros((20,6), dtype=float)
        SimInfo.for_acc = np.zeros((20,6), dtype=float)
        SimInfo.with_dynamic_forces = True
        SimInfo.dynamic_forces = np.zeros((20,beam1.StructuralInformation.num_node,6), dtype=float)

        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(20)
        beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

    def test_generate_wt_excel_type01(self):
        print('Test not being run, needs fixing!')
        return
        deg2rad = np.pi/180.
        ######################################################################
        ###########################  PARAMETERS  #############################
        ######################################################################
        """
        generate_wt_excel_type01

        Example of a file used to generate a wind turbine case
        """
        # Case
        case = 'wt'
        route = os.path.dirname(os.path.realpath(__file__)) + '/'

        # Geometry discretization
        chord_panels = np.array([4], dtype=int)
        mstar = 13

        # Operation
        rotation_velocity = 1.27
        pitch = 0. #degrees

        # Wind
        WSP = 11.4
        air_density = 1.225

        # Simulation
        dt = 0.01
        time_steps = 800

        ######################################################################
        ##########################  GENERATE WT  #############################
        ######################################################################
        wt, LC, MB = template_wt_excel_type01.generate_wt_from_excel_type01(
                                          chord_panels,
                                          rotation_velocity,
                                          pitch,
                                          excel_file_name= 'database_type01.xlsx',
                                          excel_sheet_structural_blade = 'structural_blade',
                                          excel_sheet_aero_blade = 'aero_blade',
                                          excel_sheet_airfoil_coord = 'airfoil_coord',
                                          excel_sheet_rotor = 'rotor_parameters',
                                          excel_sheet_structural_tower = 'structural_tower',
                                          excel_sheet_nacelle = 'structural_nacelle',
                                          m_distribution = 'uniform',
                                          n_points_camber = 100,
                                          tol_remove_points = 1e-3)

        ######################################################################
        ######################  DEFINE SIMULATION  ###########################
        ######################################################################
        SimInfo = gc.SimulationInformation()
        self.SimInfo.append(SimInfo)
        SimInfo.set_default_values()

        SimInfo.define_uinf(np.array([0.0,0.0,1.0]), WSP)

        SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                'AerogridLoader',
                                'BeamPlot',
                                'AerogridPlot',
                                'InitializeMultibody',
                                'DynamicPrescribedCoupled',
                                'AerogridPlot',
                                'BeamPlot'
                                ]
        SimInfo.solvers['SHARPy']['case'] = case
        SimInfo.solvers['SHARPy']['route'] = route
        SimInfo.set_variable_all_dicts('dt', dt)
        SimInfo.set_variable_all_dicts('rho', air_density)

        SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

        SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
        SimInfo.solvers['AerogridLoader']['mstar'] = mstar
        SimInfo.solvers['AerogridLoader']['freestream_dir'] = np.array([0.,0.,0.])

        SimInfo.solvers['StaticUvlm']['rho'] = 0.0
        # Default values for NonLinearStatic
        # Default values for NonLinearDynamicCoupledStep
        SimInfo.solvers['StepUvlm']['convection_scheme'] = 2

        SimInfo.solvers['InitializeMultibody']['structural_solver'] = 'NonLinearStatic'
        SimInfo.solvers['InitializeMultibody']['structural_solver_settings'] = SimInfo.solvers['NonLinearStatic']
        SimInfo.solvers['InitializeMultibody']['aero_solver'] = 'StaticUvlm'
        SimInfo.solvers['InitializeMultibody']['aero_solver_settings'] = SimInfo.solvers['StaticUvlm']

        SimInfo.solvers['DynamicPrescribedCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
        SimInfo.solvers['DynamicPrescribedCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
        SimInfo.solvers['DynamicPrescribedCoupled']['aero_solver'] = 'StepUvlm'
        SimInfo.solvers['DynamicPrescribedCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
        SimInfo.solvers['DynamicPrescribedCoupled']['postprocessors'] = ['BeamPlot', 'AerogridPlot']
        SimInfo.solvers['DynamicPrescribedCoupled']['postprocessors_settings'] = {'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                                     'AerogridPlot': SimInfo.solvers['AerogridPlot']}
        SimInfo.solvers['DynamicPrescribedCoupled']['minimum_steps'] = 0

        SimInfo.define_num_steps(time_steps)

        # Define dynamic simulation
        SimInfo.with_forced_vel = True
        SimInfo.for_vel = np.zeros((time_steps,6), dtype=float)
        SimInfo.for_acc = np.zeros((time_steps,6), dtype=float)
        SimInfo.with_dynamic_forces = True
        SimInfo.dynamic_forces = np.zeros((time_steps,wt.StructuralInformation.num_node,6), dtype=float)


        ######################################################################
        #######################  GENERATE FILES  #############################
        ######################################################################
        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        wt.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(time_steps)
        gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

    def TearDown(self):
        for si in self.SimInfo:
            print(si.solvers['SHARPy']['route'])
            gc.clean_test_files(si.solvers['SHARPy']['route'], si.solvers['SHARPy']['case'])

