import unittest
import os
import numpy as np
from fuselage_wing_configuration.fuselage_wing_configuration import Fuselage_Wing_Configuration
from define_simulation_settings import define_simulation_settings
import json

class TestVlmCoupledWithSourcePanelMethod(unittest.TestCase):

    def test_phantom_panels(self):
        """
            run with and without phantom panels (should give same results).
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
            print(flow_case)
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


    def get_geometry_parameters(self, model_name,fuselage_length=10):

        with open(self.route_test_dir + '/geometry_parameter_models.json', 'r') as fp:
            parameter_models = json.load(fp)[model_name]
            
        geometry_parameters = {
            'fuselage_length': fuselage_length,         
            'max_radius': fuselage_length/parameter_models['length_radius_ratio'],
            'fuselage_shape': parameter_models['fuselage_shape'],
        }
        geometry_parameters['chord']=geometry_parameters['max_radius']/parameter_models['radius_chord_ratio']
        geometry_parameters['half_wingspan'] = geometry_parameters['max_radius']/parameter_models['radius_half_wingspan_ratio']
        geometry_parameters['offset_nose_wing'] = parameter_models['length_offset_nose_to_wing_ratio'] / fuselage_length

        return geometry_parameters

    def generate_model(self, 
                       case_name, 
                       dict_geometry_parameters,
                       dict_discretisation, 
                       lifting_only):
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
