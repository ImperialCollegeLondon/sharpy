import unittest
import os
import numpy as np
from fuselage_wing_configuration.fuselage_wing_configuration import Fuselage_Wing_Configuration
from define_simulation_settings import define_simulation_settings



class TestPhantomPanels(unittest.TestCase):
    
    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


    def test_phantom(self):
        """
            run with and without phantom panels (should give same results).
        """

        # define model variables
        half_wingspan = 10
        fuselage_junction = 0.2
        fuselage_length= 10
        offset_nose_wing = fuselage_length/2 - 0.25
        lifting_only = False
        nonlifting_only = False
        phantom_test = True
        horseshoe = True
        chord = 1.
        u_inf = 10 
        alpha_deg = 5
        n_elem_per_wing = 20
        n_elem_fuselage = 10
        ea_main = 0.5
        num_chordwise_panels  = 8
        fuselage_shape = 'cylindrical'

        # define case name and folders
        case_route = self.route_test_dir + '/cases/'
        output_route = self.route_test_dir + '/output/'
        if not os.path.exists(case_route):
            os.makedirs(case_route)
        
        list_case_name = ['wing_only', 'phantom_wing']
        list_phantom_test = [False, True]
        list_lifting_only = [True, False]
        for icase in range(len(list_case_name)):
            case_name = list_case_name[icase] 
            phantom_test = list_phantom_test[icase]
            lifting_only = list_lifting_only[icase]
        # generate ellipsoid model
            phantom_wing = Fuselage_Wing_Configuration(case_name, case_route, output_route)
            phantom_wing.init_aeroelastic(lifting_only=lifting_only,
                                        elastic_axis=ea_main,
                                        num_chordwise_panels=num_chordwise_panels,
                                        max_radius=fuselage_junction,
                                        half_wingspan=half_wingspan,
                                        fuselage_length=fuselage_length,
                                        offset_nose_wing=offset_nose_wing,
                                        n_elem_per_wing=n_elem_per_wing,
                                        n_elem_fuselage=n_elem_fuselage,
                                        chord=chord,
                                        fuselage_shape=fuselage_shape)
            phantom_wing.generate()
            # define settings
            flow = ['BeamLoader',
                    'AerogridLoader',
                    'NonliftingbodygridLoader',
                    'AerogridPlot',
                    'StaticUvlm',
                    'BeamLoads',
                    'AerogridPlot',
                    'BeamPlot',
                    'LiftDistribution',
                    ]
            if lifting_only:
                flow.remove('NonliftingbodygridLoader')
            settings = define_simulation_settings(flow, 
                                                phantom_wing, 
                                                alpha_deg, 
                                                u_inf, 
                                                lifting_only=lifting_only, 
                                                phantom_test=phantom_test, 
                                                nonlifting_only=nonlifting_only, 
                                                horseshoe=horseshoe)
            phantom_wing.create_settings(settings)

            # run simulation
            phantom_wing.run()

        # # postprocess
        lift_distribution_wing_only = self.load_lift_distribution(output_route + '/' + list_case_name[0])[:phantom_wing.structure.n_node_right_wing,:]
        lift_distribution_wing_phantom = self.load_lift_distribution(output_route + '/' + list_case_name[1])[:phantom_wing.structure.n_node_right_wing,:]

        # # check results
        with self.subTest('lift distribution'):
            # 
            np.testing.assert_array_almost_equal(lift_distribution_wing_only[3:], lift_distribution_wing_phantom[3:], decimal=3)
    def load_lift_distribution(self, output_folder):
        """
            Loads the resulting pressure coefficients saved in txt-files.
        """
        lift_distribution = np.loadtxt(output_folder + '/lift_distribution.txt', delimiter=',')
        y_coordinate = lift_distribution[:,1]
        cl_distribution = lift_distribution[:,-1]

        return np.column_stack((y_coordinate, cl_distribution))

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
