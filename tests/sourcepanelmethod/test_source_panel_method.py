import unittest
import os
import numpy as np
from sharpy.cases.templates.fuselage_wing_configuration.fuselage_wing_configuration import Fuselage_Wing_Configuration
from sharpy.cases.templates.fuselage_wing_configuration.fwc_get_settings import define_simulation_settings



class TestSourcePanelMethod(unittest.TestCase):

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    def test_ellipsoid(self):
        """
        Computes the pressure distribution over an ellipsoid. The results should
        match the analyitcal solution according to potential flow theory (see Chapter 5
        Katz, J., and Plotkin, A., Low-speed aerodynamics, Vol. 13, Cambridge University 
        Press, 2001).
        """

        # define model variables
        radius_ellipsoid = 0.2
        length_ellipsoid = 2.
        u_inf = 10 
        alpha_deg = 0
        n_elem = 30
        num_radial_panels  = 24
        lifting_only= False
        fuselage_shape = 'ellipsoid'
        fuselage_discretisation = 'uniform'
        # define case name and folders
        case_route = self.route_test_dir + '/cases/'
        output_route = self.route_test_dir + '/output/'
        case_name = 'ellipsoid'
        enforce_uniform_fuselage_discretisation = True

        # generate ellipsoid model
        ellipsoidal_body = Fuselage_Wing_Configuration(case_name, case_route, output_route) 
        ellipsoidal_body.init_aeroelastic(lifting_only=lifting_only,
                                          max_radius=radius_ellipsoid,
                                      fuselage_length=length_ellipsoid,
                                      offset_nose_wing=length_ellipsoid/2,
                                      n_elem_fuselage=n_elem,
                                      num_radial_panels=num_radial_panels,
                                      fuselage_shape=fuselage_shape,
                                      enforce_uniform_fuselage_discretisation=enforce_uniform_fuselage_discretisation,
                                      fuselage_discretisation=fuselage_discretisation)
        ellipsoidal_body.generate()
        
        # define settings
        flow = ['BeamLoader',
                'NonliftingbodygridLoader',
                'StaticUvlm',
                'WriteVariablesTime'
                ]
        settings = define_simulation_settings(flow, 
                                              ellipsoidal_body, 
                                              alpha_deg, 
                                              u_inf, 
                                              lifting_only=False, 
                                              nonlifting_only=True, 
                                              horseshoe=True,
                                              writeCpVariables=True)
        ellipsoidal_body.create_settings(settings)

        # run simulation
        ellipsoidal_body.run()

        # postprocess
        cp_distribution_SHARPy = self.load_pressure_distribution(output_route + '/' + case_name + '/WriteVariablesTime/', 
                                                                 ellipsoidal_body.structure.n_node_fuselage)
        dx = length_ellipsoid/(ellipsoidal_body.structure.n_node_fuselage-1)
        x_collocation_points = np.linspace(-length_ellipsoid/2+dx/2, length_ellipsoid/2-dx/2, ellipsoidal_body.structure.n_node_fuselage)
        cp_distribution_analytcal = self.get_analytical_pressure_distribution(radius_ellipsoid, x_collocation_points)

        # check results
        with self.subTest('pressure_coefficient'):
            # Higher deviations near the nose and tail due to local coarser discretisation. Check only mid-body values!
            np.testing.assert_array_almost_equal(cp_distribution_SHARPy[4:-4], cp_distribution_analytcal[4:-4], decimal=3)

    def load_pressure_distribution(self, output_folder, n_collocation_points):
        """
            Loads the resulting pressure coefficients saved in txt-files.
        """
        cp_distribution = np.zeros((n_collocation_points,))
        for i_collocation_point in range(n_collocation_points):
            cp_distribution[i_collocation_point] = np.loadtxt(output_folder + 'nonlifting_pressure_coefficients_panel_isurf0_im0_in{}.dat'.format(i_collocation_point))[1]
        return cp_distribution

    def get_analytical_pressure_distribution(self, radius, x_coordinates):
        """
            Computes the analytical solution of the pressure distribution over
            an ellipsoid in potential flow for the previous specified ellipsoid
            model. Equations used are taken from
                https://www.symscape.com/examples/panel/potential_flow_ellipsoid
        """
        a = np.sqrt(1 - radius**2)
        b = 2 * ((1-a**2)/a**3) * (np.arctanh(a)-a)
        u = 2./(2.-b) * np.sqrt((1-x_coordinates**2)/(1-x_coordinates**2 * a**2))
        return 1-u**2
    
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
