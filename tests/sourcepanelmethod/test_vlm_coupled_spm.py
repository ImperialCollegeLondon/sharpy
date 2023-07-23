import unittest
import os
import numpy as np
from sharpy.cases.templates.fuselage_wing_configuration.fuselage_wing_configuration import Fuselage_Wing_Configuration
from define_simulation_settings import define_simulation_settings
import json

class TestVlmCoupledWithSourcePanelMethod(unittest.TestCase):

 
    def test_phantom_panels(self):
        """
            The lift distribution over a rectangular high-aspect ratio
            wing is computed. First a wing_only configuration is considered,
            while second, we activate the phantom panels created within the 
            fuselage although, the effect of the source panels is omitted. 
            With the defined interpolation scheme, the same lift distribution 
            must be obtained.
        """
        self.define_folder()
        

        model = 'phantom_wing'
        fuselage_length = 10
        dict_geometry_parameters = self.get_geometry_parameters(model,
                                                                fuselage_length=fuselage_length)

        # Freestream Conditions
        alpha_deg = 5
        u_inf = 10

        # Discretization
        dict_discretization = {
            'n_elem_per_wing': 20,
            'n_elem_fuselage': 10,
            'num_chordwise_panels': 8
        }

        # Simulation settings
        horseshoe = True
        list_phantom_test = [False, True]
        # Simlation Solver Flow
        flow = ['BeamLoader',
                'AerogridLoader',
                'NonliftingbodygridLoader',
                'StaticUvlm',
                'BeamLoads',
                'LiftDistribution',
                'AerogridPlot',
                    ]
        list_results_lift_distribution = []
        # define model variables
        for icase in range(len(list_phantom_test)):
            phantom_test = list_phantom_test[icase]
            lifting_only = not phantom_test
            case_name = model + '_coupled_{}'.format(int(phantom_test))
            
            # generate ellipsoid model
            phantom_wing = self.generate_model(case_name, 
                                               dict_geometry_parameters,
                                               dict_discretization, 
                                               lifting_only)
            # Adjust flow for case
            flow_case = flow.copy()

            if lifting_only:
                flow_case.remove('NonliftingbodygridLoader')
                
            self.generate_simulation_settings(flow_case, 
                                              phantom_wing, 
                                              alpha_deg, 
                                              u_inf, 
                                              lifting_only,
                                              horseshoe=horseshoe,
                                              phantom_test=phantom_test)
            # run simulation
            phantom_wing.run()

            # get results
            list_results_lift_distribution.append(self.load_lift_distribution(
                self.output_route + '/' + case_name,
                phantom_wing.structure.n_node_right_wing
            )) 

        # check results
        with self.subTest('lift distribution'):
            np.testing.assert_array_almost_equal(list_results_lift_distribution[0][3:, 1], list_results_lift_distribution[1][3:, 1], decimal=3)


    def test_fuselage_wing_configuration(self):
        """
            Lift distribution on a low wing configuration is computed. The final 
            results are compared to a previous solution (backward compatibility)
            that matches the experimental lift distribution for this case. 
        """
        self.define_folder()
        model = 'low_wing'
        fuselage_length = 10
        dict_geometry_parameters = self.get_geometry_parameters(model,
                                                                fuselage_length=fuselage_length)

        # Freestream Conditions
        alpha_deg = 2.9
        u_inf = 10

        # Discretization
        dict_discretization = {
            'n_elem_per_wing': 10,
            'n_elem_fuselage': 30,
            'num_chordwise_panels': 8,
            'num_radial_panels': 36
        }

        # Simulation settings
        horseshoe = True
        phantom_test = False
        lifting_only = False
        # Simlation Solver Flow
        flow = ['BeamLoader',
                'AerogridLoader',
                'NonliftingbodygridLoader',
                'StaticUvlm',
                'StaticCoupled',
                'BeamLoads',
                'LiftDistribution',
                'AerogridPlot',
                    ]
        for static_coupled_solver in [False, True]:
            case_name = '{}_coupled_{}'.format(model, int(static_coupled_solver))
            wing_fuselage_model = self.generate_model(case_name, 
                                                dict_geometry_parameters,
                                                dict_discretization, 
                                                lifting_only)

            flow_case = flow.copy()
            if static_coupled_solver:
                flow_case.remove('StaticUvlm')
            else:
                flow_case.remove('StaticCoupled')
            print(flow_case)
            self.generate_simulation_settings(flow_case, 
                                                wing_fuselage_model, 
                                                alpha_deg, 
                                                u_inf, 
                                                lifting_only,
                                                horseshoe=horseshoe,
                                                phantom_test=phantom_test)
            # run simulation
            wing_fuselage_model.run()
            # get results
            lift_distribution = self.load_lift_distribution(
                self.output_route + '/' + case_name,
                wing_fuselage_model.structure.n_node_right_wing
                )

            # check results
            lift_distribution_test = np.loadtxt(self.route_test_dir + "/test_data/results_{}.csv".format(case_name))
            with self.subTest('lift distribution and spanwise wing deformation'):        
                np.testing.assert_array_almost_equal(lift_distribution_test, lift_distribution, decimal=3)


    def get_geometry_parameters(self, model_name,fuselage_length=10):
        """
            Geometry parameters are loaded from json init file for the specified model. 
            Next, final geoemtry parameteres, depending on the fuselage length are 
            calculated and return within a dict.
        """
        with open(self.route_test_dir + '/geometry_parameter_models.json', 'r') as fp:
            parameter_models = json.load(fp)[model_name]

        geometry_parameters = {
            'fuselage_length': fuselage_length,         
            'max_radius': fuselage_length/parameter_models['length_radius_ratio'],
            'fuselage_shape': parameter_models['fuselage_shape'],
        }
        geometry_parameters['chord']=geometry_parameters['max_radius']/parameter_models['radius_chord_ratio']
        geometry_parameters['half_wingspan'] = geometry_parameters['max_radius']/parameter_models['radius_half_wingspan_ratio']
        geometry_parameters['offset_nose_wing'] = parameter_models['length_offset_nose_to_wing_ratio'] * fuselage_length
        geometry_parameters['vertical_wing_position'] = parameter_models['vertical_wing_position'] * geometry_parameters['max_radius']
        return geometry_parameters

    def generate_model(self, 
                       case_name, 
                       dict_geometry_parameters,
                       dict_discretisation, 
                       lifting_only):
        """
            Aircraft model object is generated and structural and aerodynamic (lifting and nonlifting)
            input files are generated.
        """
        aircraft_model = Fuselage_Wing_Configuration(case_name, self.case_route, self.output_route)
        aircraft_model.init_aeroelastic(lifting_only=lifting_only,
                                    **dict_discretisation,
                                    **dict_geometry_parameters)
        aircraft_model.generate()
        return aircraft_model
    
    def generate_simulation_settings(self, 
                                     flow, 
                                     aircraft_model, 
                                     alpha_deg, 
                                     u_inf, 
                                     lifting_only,
                                     horseshoe=True,
                                     nonlifting_only=False,
                                     phantom_test=False):
        """
            Simulation settings are defined and written to the ".sharpy" input file.
        """
        settings = define_simulation_settings(flow, 
                                                aircraft_model, 
                                                alpha_deg, 
                                                u_inf, 
                                                lifting_only=lifting_only, 
                                                phantom_test=phantom_test, 
                                                nonlifting_only=nonlifting_only, 
                                                horseshoe=horseshoe)
        aircraft_model.create_settings(settings)

    def define_folder(self):        
        """
            Initializes all folder path needed and creates case folder.
        """
        self.route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        self.case_route = self.route_test_dir + '/cases/'
        self.output_route = self.route_test_dir + '/output/'
    
        if not os.path.exists(self.case_route):
            os.makedirs(self.case_route)
        
    def load_lift_distribution(self, output_folder, n_node_wing):
        """
            Loads the resulting pressure coefficients saved in txt-files.
        """
        lift_distribution = np.loadtxt(output_folder + '/lift_distribution.txt', delimiter=',')
        return lift_distribution[:n_node_wing,[1,-1]]

    def tearDown(self):
        """
            Removes all created files within this test.
        """
        import shutil
        folders = ['cases', 'output']
        for folder in folders:
            shutil.rmtree(self.route_test_dir + '/' + folder)


if __name__ == '__main__':
    import unittest

    unittest.main()
