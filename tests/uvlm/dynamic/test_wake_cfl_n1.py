import numpy as np
import os
import sys
import unittest
# import shutil

import sharpy.utils.generate_cases as gc
import sharpy.sharpy_main

# maths
deg2rad = np.pi/180.

# Generate the file containing the gust information
gust = np.array([[-1e12, 0, 0, 0],
[-1e-12, 0, 0, 0],
[0, -1.523086e-6, 0, 1.745328e-3],
[1e12, -1.523086e-6, 0, 1.745328e-3]])
np.savetxt("0p1_step_gust.txt", gust, header="Time[s] DeltaUx[m/s] DeltaUy[m/s] DeltaUz[m/s]")

# Analytical kussner function as a function of the non-dimensional time
kussner_function = np.array([[2.50000000e-01, 2.24788834e-01],
       [5.00000000e-01, 3.08322030e-01],
       [7.50000000e-01, 3.69409233e-01],
       [1.00000000e+00, 4.18591199e-01],
       [1.25000000e+00, 4.59608283e-01],
       [1.50000000e+00, 4.94597861e-01],
       [1.75000000e+00, 5.24998850e-01],
       [2.00000000e+00, 5.51822632e-01],
       [2.25000000e+00, 5.75792734e-01],
       [2.50000000e+00, 5.97434419e-01],
       [3.25000000e+00, 6.51811966e-01],
       [4.00000000e+00, 6.94721054e-01],
       [5.00000000e+00, 7.39586015e-01],
       [7.50000000e+00, 8.13230851e-01],
       [1.00000000e+01, 8.56395643e-01],
       [1.25000000e+01, 8.84426328e-01],
       [1.50000000e+01, 9.04257362e-01],
       [1.75000000e+01, 9.19155200e-01],
       [2.00000000e+01, 9.30758783e-01],
       [0.00000000e+00, 6.43293001e-02],
       [2.50000000e+01, 9.47358507e-01],
       [3.00000000e+01, 9.58126128e-01],
       [3.50000000e+01, 9.65279097e-01],
       [4.00000000e+01, 9.70275111e-01],
       [4.50000000e+01, 9.74053377e-01],
       [5.00000000e+01, 9.77137592e-01],
       [6.00000000e+01, 9.81903866e-01],
       [7.00000000e+01, 9.84820831e-01],
       [8.00000000e+01, 9.86297902e-01],
       [9.00000000e+01, 9.87557134e-01],
       [9.97500000e+01, 9.89298872e-01]])


class TestWakeCFLn1(unittest.TestCase):
    """
    Validate an airfoil response to a Kussner gust
    Wake discretisation not complying with CFL=1
    Non-linear and linear responses
    """

    chord = 1. # 
    AR = 1e7 # Wing aspect ratio
    
    nodes_AR = 4 # even number of nodes in the spanwise direction
    
    num_chord_panels = 8 # Number of chord panels
    
    uinf = 1. # Flow velocity
    uinf_dir = np.array([1., 0., 0.]) # Flow velocity direction
    air_density = 1.225 # Flow density
    
    aoa_ini_deg = 1. # Initial angle of attack
    
    wake_chords = 100 # Length of the wake in airfoil chords units
    
    # Time discretization
    final_ndtime = 12.0 # Final non-dimensional time
    time_steps_per_chord = 8 # Number of time steps required by the flow to cover the airfoil chord
    dt = chord/time_steps_per_chord/uinf # time step
    nd_dt = dt*uinf/chord*2 # non-dimensional time step
    offset_time_steps = 10 # Number of time steps before the gust gets to the airfoil
    
    offset = 0.

    def generate_geometry(self):
        # offset is just the distance between the gust edge and the first colocation point
        # It is just used for plotting purposes
        len_first_panel = (self.chord - self.dt*self.uinf)/(self.num_chord_panels - 1)
        self.offset =  self.offset_time_steps*self.dt*self.uinf + (-0.25*self.chord + 0.75*len_first_panel)*np.cos(self.aoa_ini_deg*deg2rad) # should be 0.5*len_first_panel but to make sure it happens I increase it a bit
        offset_LE = self.offset - 0.25*self.chord*np.cos(self.aoa_ini_deg*deg2rad)
        
        # Compute span
        span = self.chord*self.AR
        assert self.nodes_AR%2 == 0, "nodes_AR must be even"
        
        # Compute number of nodes
        num_node = self.nodes_AR + 1
        num_node_semispan = int((num_node - 1)/2 + 1)
        nodes_y_semispan = np.zeros((num_node_semispan,))
        
        # Number of seminodes
        n2 = int(self.nodes_AR/2 + 1)
        
        # Nodes coordinates in the spanwise direction
        nodes_y_semispan[0:n2] = np.linspace(0, self.AR/2*self.chord, n2)
        
        # Nodes coordinates
        nodes_y = np.zeros((num_node,))
        nodes_y[:num_node_semispan] = -1*nodes_y_semispan[::-1]
        nodes_y[num_node_semispan - 1:] = nodes_y_semispan
        
        assert not (np.diff(nodes_y) == 0).any(), "Repeated nodes"
        
        nodes = np.zeros((len(nodes_y), 3))
        nodes[:, 1] = nodes_y
        
        # Irrelevant structural properties
        mass_per_unit_length = 1.
        mass_iner = 1e-4
        EA = 1e9
        GA = 1e9
        GJ = 1e9
        EI = 1e9
        
        # Airfoil camber
        airfoil_camber = np.zeros((1, 100, 2))
        airfoil_camber[0,:,0] = np.linspace(0.,1.,100)
        
        # Structure
        airfoil = gc.AeroelasticInformation()
        airfoil.StructuralInformation.num_node = len(nodes)
        airfoil.StructuralInformation.num_node_elem = 3
        airfoil.StructuralInformation.compute_basic_num_elem()
        
        airfoil.StructuralInformation.generate_uniform_sym_beam(nodes,
                             mass_per_unit_length,
                             mass_iner,
                             EA,
                             GA,
                             GJ,
                             EI,
                             num_node_elem = airfoil.StructuralInformation.num_node_elem,
                             y_BFoR = 'x_AFoR',
                             num_lumped_mass=0)
        
        airfoil.StructuralInformation.structural_twist = 0.*deg2rad*np.ones_like(airfoil.StructuralInformation.structural_twist)
        airfoil.StructuralInformation.boundary_conditions = np.zeros((airfoil.StructuralInformation.num_node), dtype = int)
        airfoil.StructuralInformation.boundary_conditions[0] = 1 # Clamped end
        airfoil.StructuralInformation.boundary_conditions[-1] = -1 # Free end
        
        # Generate blade aerodynamics
        airfoil.AerodynamicInformation.create_one_uniform_aerodynamics(airfoil.StructuralInformation,
                                         chord = self.chord,
                                         twist = self.aoa_ini_deg*deg2rad,
                                         sweep = 0.,
                                         num_chord_panels = self.num_chord_panels,
                                         m_distribution = 'uniform',
                                         elastic_axis = 0.25,
                                         num_points_camber = 100,
                                         airfoil = airfoil_camber)

        return airfoil


    def generate_files(self, case_header, airfoil):

        # Define the simulation
        case = ('%s_aoa%.2f' % (case_header, self.aoa_ini_deg)).replace(".", "p")
        route = os.path.dirname(os.path.realpath(__file__)) + '/'
        
        SimInfo = gc.SimulationInformation()
        SimInfo.set_default_values()
        
        SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                'AerogridLoader',
                                'StaticCoupled',
                                'DynamicCoupled']
        
        SimInfo.solvers['SHARPy']['case'] = case
        SimInfo.solvers['SHARPy']['route'] = route
        SimInfo.solvers['SHARPy']['log_folder'] = "./output/%s" % case
        SimInfo.solvers['SHARPy']['write_log'] = False
        SimInfo.solvers['SHARPy']['write_screen'] = False
        SimInfo.set_variable_all_dicts('rho', self.air_density)
        SimInfo.set_variable_all_dicts('dt', self.dt)
       
        SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

        import sharpy.utils.generator_interface as gi
        if case_header == 'traditional' or case_header == 'statespace_cfl1':

            SimInfo.solvers['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
            SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': self.uinf,
                                                'u_inf_direction': self.uinf_dir,
                                                'dt': self.dt,
                                                'dx1': self.chord/self.num_chord_panels,
                                                'ndx1': int((self.wake_chords*self.chord)/(self.chord/self.num_chord_panels)),
                                                'r':1.,
                                                'dxmax':10*self.chord}
        
            gi.dictionary_of_generators(print_info=False)
            hw = gi.dict_of_generators['StraightWake']
            wsg_in = SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] # for simplicity
            length = 0
            mstar = 0
            while length < (self.wake_chords*self.chord):
                mstar += 1
                length += hw.get_deltax(mstar, wsg_in['dx1'],
                                        wsg_in['ndx1'],
                                        wsg_in['r'],
                                        wsg_in['dxmax'])
        
            SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
            SimInfo.solvers['AerogridLoader']['mstar'] = mstar
            SimInfo.solvers['AerogridLoader']['freestream_dir'] = np.array([0.,0.,0.])
        
        elif case_header in ['new_nonlinear', 'new_linear', 'statespace']:

            SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] = {'u_inf': self.uinf,
                                                'u_inf_direction': self.uinf_dir,
                                                'dt': self.dt,
                                                'dx1': self.chord/self.num_chord_panels,
                                                'ndx1': int((1*self.chord)/(self.chord/self.num_chord_panels)),
                                                'r':2.,
                                                'dxmax':10*self.chord}

            gi.dictionary_of_generators(print_info=False)
            hw = gi.dict_of_generators['StraightWake']
            wsg_in = SimInfo.solvers['AerogridLoader']['wake_shape_generator_input'] # for simplicity
            length = 0
            mstar = 0
            while length < (self.wake_chords*self.chord):
                mstar += 1
                length += hw.get_deltax(mstar, wsg_in['dx1'],
                                wsg_in['ndx1'],
                                wsg_in['r'],
                                wsg_in['dxmax'])

            SimInfo.solvers['AerogridLoader']['mstar'] = mstar


        SimInfo.solvers['SteadyVelocityField']['u_inf'] = self.uinf
        SimInfo.solvers['SteadyVelocityField']['u_inf_direction'] = self.uinf_dir
        
        SimInfo.solvers['StaticUvlm']['horseshoe'] = False
        SimInfo.solvers['StaticUvlm']['num_cores'] = 8
        SimInfo.solvers['StaticUvlm']['n_rollup'] = 0
        SimInfo.solvers['StaticUvlm']['rollup_dt'] = self.dt
        SimInfo.solvers['StaticUvlm']['velocity_field_generator'] = 'SteadyVelocityField'
        SimInfo.solvers['StaticUvlm']['velocity_field_input'] = SimInfo.solvers['SteadyVelocityField']
        
        SimInfo.solvers['StaticCoupled']['structural_solver'] = 'RigidDynamicPrescribedStep'
        SimInfo.solvers['StaticCoupled']['structural_solver_settings'] = SimInfo.solvers['RigidDynamicPrescribedStep']
        SimInfo.solvers['StaticCoupled']['aero_solver'] = 'StaticUvlm'
        SimInfo.solvers['StaticCoupled']['aero_solver_settings'] = SimInfo.solvers['StaticUvlm']

        if case_header in ['traditional', 'new_nonlinear']:            
            SimInfo.solvers['StepUvlm']['convection_scheme'] = 0
            SimInfo.solvers['StepUvlm']['num_cores'] = 8
            SimInfo.solvers['StepUvlm']['velocity_field_generator'] = 'GustVelocityField'
            SimInfo.solvers['StepUvlm']['velocity_field_input'] = {'u_inf' : self.uinf,
                                                               'u_inf_direction': np.array([1.,0.,0.]),
                                                               'gust_shape': 'time varying',
                                                               'offset': self.offset,
                                                               'relative_motion': True,
                                                               'gust_parameters': {'file' : '0p1_step_gust.txt',
                                                                                   'gust_length': 0.,
                                                                                   'gust_intensity': 0.}}
        
            SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
            SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
        elif case_header == 'new_linear':

            SimInfo.solvers['StepLinearUVLM']['dt'] = self.dt
            SimInfo.solvers['StepLinearUVLM']['integr_order'] = 2
            SimInfo.solvers['StepLinearUVLM']['remove_predictor'] = 'off'
            SimInfo.solvers['StepLinearUVLM']['use_sparse'] = True
            SimInfo.solvers['StepLinearUVLM']['density'] = self.air_density
            SimInfo.solvers['StepLinearUVLM']['vortex_radius'] = 1e-6
            SimInfo.solvers['StepLinearUVLM']['vortex_radius_wake_ind'] = 1e-3
            SimInfo.solvers['StepLinearUVLM']['velocity_field_generator'] = 'GustVelocityField'
            SimInfo.solvers['StepLinearUVLM']['velocity_field_input'] = {'u_inf' : self.uinf,
                                                   'u_inf_direction': self.uinf_dir,
                                                   'gust_shape': 'time varying',
                                                   'offset': self.offset,
                                                   'relative_motion': True,
                                                   'gust_parameters': {'file' : '0p1_step_gust.txt',
                                                                       'gust_length': 0.,
                                                                       'gust_intensity': 0.}}

            SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepLinearUVLM'
            SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepLinearUVLM']

        if case_header == 'new_nonlinear':
            SimInfo.solvers['StaticUvlm']['cfl1'] = False
            SimInfo.solvers['StepUvlm']['cfl1'] = False

        elif case_header == 'new_linear':
            SimInfo.solvers['StaticUvlm']['cfl1'] = False
            SimInfo.solvers['StepLinearUVLM']['cfl1'] = False
        
        elif case_header == 'traditional':
            SimInfo.solvers['StaticUvlm']['cfl1'] = True
            SimInfo.solvers['StepUvlm']['cfl1'] = True

        SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'RigidDynamicPrescribedStep'
        SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['RigidDynamicPrescribedStep']
        SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['Cleanup']
        SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {'Cleanup': SimInfo.solvers['Cleanup']}
        SimInfo.solvers['DynamicCoupled']['minimum_steps'] = 0
        SimInfo.solvers['DynamicCoupled']['include_unsteady_force_contribution'] = True
        SimInfo.solvers['DynamicCoupled']['relaxation_factor'] = 0.
        SimInfo.solvers['DynamicCoupled']['final_relaxation_factor'] = 0.
        SimInfo.solvers['DynamicCoupled']['dynamic_relaxation'] = False
        SimInfo.solvers['DynamicCoupled']['relaxation_steps'] = 0
        SimInfo.solvers['DynamicCoupled']['fsi_tolerance'] = 1e-6

        # Time discretization
        time_steps = int(self.final_ndtime / self.nd_dt - 1)
        SimInfo.define_num_steps(time_steps)

        SimInfo.with_forced_vel = True
        SimInfo.for_vel = np.zeros((time_steps,6), dtype=float)
        SimInfo.for_acc = np.zeros((time_steps,6), dtype=float)
        SimInfo.with_dynamic_forces = True
        SimInfo.dynamic_forces = np.zeros((time_steps,airfoil.StructuralInformation.num_node,6), dtype=float)

        SimInfo.solvers['PickleData'] = {'folder': route + '/output/'}
        if 'statespace' in case_header:
            SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                                                 'AerogridLoader',
                                                 'StaticCoupled',
                                                 'Modal',
                                                 'LinearAssembler',
                                                 'AsymptoticStability',
                                                 ]

            SimInfo.solvers['SHARPy']['write_screen'] = 'on'
            SimInfo.solvers['Modal'] = {'save_data': 'off',
                                        'NumLambda': 50,
                                        'rigid_body_modes': 'off'}

            SimInfo.solvers['LinearAssembler'] = {'linear_system': 'LinearAeroelastic',
                                   'linear_system_settings': {
                                       'beam_settings': {'modal_projection': 'off',
                                                         'inout_coords': 'nodes',
                                                         'discrete_time': 'on',
                                                         'newmark_damp': 0.5e-4,
                                                         'discr_method': 'newmark',
                                                         'dt': self.dt,
                                                         'proj_modes': 'undamped',
                                                         'use_euler': 'on',
                                                         'num_modes': 2,
                                                         'print_info': 'on',
                                                         'gravity': 'on',
                                                         'remove_dofs': [],
                                                         'remove_rigid_states': 'off'},
                                       'aero_settings': {'dt': self.dt,
                                                         'integr_order': 2,
                                                         'density': self.air_density,
                                                         'remove_predictor': 'off',
                                                         'use_sparse': 'off',
                                                         'rigid_body_motion': 'off',
                                                         'use_euler': 'on',
                                                         'vortex_radius': 1e-6,
                                                         'convert_to_ct': 'off',
                                                         'gust_assembler': 'LeadingEdge',
                                                         'gust_assembler_inputs': {},
                                                         },
                                       'rigid_body_motion': 'off',
                                       'track_body': 'on',
                                       'use_euler': 'on',
                                       'linearisation_tstep': -1
                                   },
                                                  }
            SimInfo.solvers['AsymptoticStability'] = {'num_evals': 100}

        gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        airfoil.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
        SimInfo.generate_solver_file()
        SimInfo.generate_dyn_file(time_steps)
        
        return case, route

    def run_test(self, case_header):

        airfoil = self.generate_geometry()
        
        case, route = self.generate_files(case_header, airfoil)
        sharpy_file = route + case + '.sharpy'
        sharpy_output = sharpy.sharpy_main.main(['', sharpy_file])
        
        lift = sharpy_output.structure.timestep_info[-1].steady_applied_forces[self.nodes_AR//2, 2]
        lift += sharpy_output.structure.timestep_info[-1].unsteady_applied_forces[self.nodes_AR//2, 2]

        nd_time = sharpy_output.ts*self.nd_dt
        daoa = 0.1
        clsteady = 2*np.pi*(self.aoa_ini_deg + daoa)*deg2rad
        offset_plot = 0.84375
        factor = 0.5*self.air_density*self.uinf**2*self.chord*2.5e6
        sharpy_cl_clss = -lift/factor/clsteady
        kussner_cl_clss = 2*np.pi*self.aoa_ini_deg*deg2rad
        kussner_cl_clss += 2 * np.pi * daoa * deg2rad * np.interp(nd_time,
                                                                  kussner_function[:, 0],
                                                                  kussner_function[:, 1])
        kussner_cl_clss /= clsteady

        self.assertAlmostEqual(sharpy_cl_clss, kussner_cl_clss, 1)

    def test_traditional(self):
        self.run_test('traditional')


    def test_new_nonlinear(self):
        self.run_test('new_nonlinear')


    def test_new_linear(self):
        self.run_test('new_linear')

    def test_statespace(self):
        results = {}
        for case in ['statespace', 'statespace_cfl1']:
            with self.subTest(case):
                results[case] = self.run_statespace(case)

    def run_statespace(self, case_header):
        airfoil = self.generate_geometry()

        case, route = self.generate_files(case_header, airfoil)
        sharpy_file = route + case + '.sharpy'
        return sharpy.sharpy_main.main(['', sharpy_file])

    @classmethod
    def tearDownClass(cls):
        solver_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        solver_path += '/'
        files_to_delete = []
        for name in ['traditional_aoa1p00', 'new_nonlinear_aoa1p00', 'new_linear_aoa1p00']:
            files_to_delete.extend((name + '.aero.h5',
                           name + '.dyn.h5',
                           name + '.fem.h5',
                           name + '.sharpy'))
        files_to_delete.append("0p1_step_gust.txt")
        
        for f in files_to_delete:
            os.remove(solver_path + f)

        # shutil.rmtree(solver_path + 'output/')

