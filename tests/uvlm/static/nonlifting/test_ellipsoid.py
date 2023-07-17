import unittest
import os
import numpy as np
from fuselage import Fuselage
from define_simulation_settings import define_simulation_settings



class TestFuselage(unittest.TestCase):
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
        n_elem = 21
        num_radial_panels  = 12
        
        # define case name and folders
        case_route = self.route_test_dir + '/cases/'
        output_route = self.route_test_dir + '/output/'
        if not os.path.exists(case_route):
            os.makedirs(case_route)
        case_name = 'ellipsoid'

        # generate ellipsoid model
        ellipsoidal_body = Fuselage(case_name, case_route, output_route)
        ellipsoidal_body.generate(
            num_radial_panels = num_radial_panels,
            max_radius = radius_ellipsoid,
            fuselage_shape = 'ellipsoid',
            length = length_ellipsoid,
            n_elem = n_elem)
        
        # define settings
        flow = ['BeamLoader',
                'NonliftingbodygridLoader',
                'StaticUvlm',
                'WriteVariablesTime'
                ]
        settings = define_simulation_settings(flow, ellipsoidal_body, alpha_deg, u_inf, lifting_only=False, nonlifting_only=True, horseshoe=True)
        ellipsoidal_body.create_settings(settings)

        # run simulation
        ellipsoidal_body.run()

        # postprocess
        cp_distribution_SHARPy = self.load_pressure_distribution(output_route + '/' + case_name + '/WriteVariablesTime/', ellipsoidal_body.n_node - 1)
        x_collocation_points = ellipsoidal_body.x[:-1] + np.diff(ellipsoidal_body.x[:2])/2
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
