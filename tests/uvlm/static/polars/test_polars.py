import unittest
import tests.uvlm.static.polars.generate_wing as gw
import os
import glob
import configobj
import numpy as np

from sharpy.aero.utils.utils import local_stability_axes
import sharpy.utils.algebra as algebra
import h5py
import sharpy.utils.h5utils as h5utils
import sharpy.aero.utils.airfoilpolars as airfoilpolars


class InfiniteWing:
    area = 90000000.0
    chord = 3

    def force_coef(self, rho, uinf):
        return 0.5 * rho * uinf ** 2 * self.area

    def moment_coef(self, rho, uinf):
        return self.force_coef(rho, uinf) * self.chord


class TestAirfoilPolars(unittest.TestCase):
    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    polar_data = np.loadtxt(route_test_dir + '/xf-naca0018-il-50000.txt', skiprows=12)

    polar = airfoilpolars.Polar()
    polar.initialise(np.column_stack((polar_data[:, 0] * np.pi / 180, polar_data[:, 1], polar_data[:, 2],
                                      polar_data[:, 4])))

    def test_infinite_wing(self):
        """
        Infinite wing should yield same results as airfoil polar
        """

        cases_route = self.route_test_dir + '/cases/'
        output_route = self.route_test_dir + '/output/'

        wing = InfiniteWing()

        gw.run(compute_uind=False,
               infinite_wing=True,
               main_ea=0.25,
               high_re=False,
               case_route_root=cases_route,
               output_route_root=output_route,
               use2pi=True,
               polar_file=self.route_test_dir + '/xf-naca0018-il-50000.txt')

        case_header = gw.get_case_header(polar=True,
                                         infinite_wing=True,
                                         compute_uind=False,
                                         high_re=False,
                                         main_ea=0.25,
                                         use2pi=True)

        results = postprocess(output_route + '/' + case_header + '/')

        results[:, 1:3] /= wing.force_coef(1.225, 1)
        results[:, -1] /= wing.moment_coef(1.225, 1)

        with self.subTest('lift'):
            cl_polar = np.interp(results[:, 0], self.polar_data[:, 0], self.polar_data[:, 1])
            np.testing.assert_array_almost_equal(cl_polar, results[:, 1], decimal=3)

        with self.subTest('drag'):
            cd_polar = np.interp(results[:, 0], self.polar_data[:, 0], self.polar_data[:, 2])
            np.testing.assert_array_almost_equal(cd_polar, results[:, 2], decimal=3)

        with self.subTest('moment'):
            cm_polar = np.interp(results[:, 0], self.polar_data[:, 0], self.polar_data[:, 4])
            np.testing.assert_array_almost_equal(cm_polar, results[:, 3], decimal=3)

    def test_linear_infinite_wing(self):

        cases_route = self.route_test_dir + '/cases/'
        output_route = self.route_test_dir + '/output/'

        wing = InfiniteWing()

        # put inside function and test diff alpha
        for alpha in [0, 2.5, 5]:
            with self.subTest(alpha):
                self.run_linear_test(alpha, cases_route, output_route)

    def run_linear_test(self, alpha, cases_route, output_route):
        case_name = 'linear_alpha{:04g}'.format(alpha * 100).replace('-', 'M')

        flow = ['BeamLoader',
                'AerogridLoader',
                'StaticCoupled',
                'AeroForcesCalculator',
                'Modal',
                'LinearAssembler',
                'StabilityDerivatives',
                'SaveParametricCase']

        gw.generate_infinite_wing(case_name,
                                  alpha=alpha,
                                  flow=flow,
                                  case_route=cases_route,
                                  polar_file=self.route_test_dir + '/xf-naca0018-il-50000.txt',
                                  aspect_ratio=1e7,
                                  main_ea=0.25,
                                  output_route=output_route)

        derivatives = self.postprocess_linear(case_name)

        with self.subTest(msg='CL_alpha at {:f}'.format(alpha)):
            cla_sharpy = derivatives['force_angle_velocity'][2, 1]
            cla_polar = self.polar.get_derivatives_at_aoa(alpha * np.pi / 180)[0]
            delta = np.abs(cla_sharpy - cla_polar) / cla_polar
            np.testing.assert_array_less(delta, 0.15, f'Difference in polar {cla_polar:0.3f} and '
                                                      f'SHARPy CL_alpha {cla_sharpy:0.3f} '
                                                      f'greater than 15% ({delta * 100:0.3f} %)')

    def postprocess_linear(self, case_name):
        with h5py.File(self.route_test_dir + '/output/' + case_name +
                       '/derivatives/aerodynamic_stability_derivatives.h5', 'r') as f:
            aerodynamic_ders = h5utils.load_h5_in_dict(f)
        return aerodynamic_ders

    def tearDown(self):
        import shutil
        folders = ['cases', 'output']
        for folder in folders:
            shutil.rmtree(self.route_test_dir + '/' + folder)


def postprocess(output_folder):
    cases = glob.glob(output_folder + '/*')

    n_cases = 0
    for case in cases:
        alpha, lift, drag, moment = process_case(case)
        if n_cases == 0:
            results = np.array([alpha, lift, drag, moment], dtype=float)
        else:
            results = np.vstack((results, np.array([alpha, lift, drag, moment])))
        n_cases += 1
    results = results.astype(float)
    results = results[results[:, 0].argsort()]
    return results


def process_case(path_to_case):
    case_name = path_to_case.split('/')[-1]
    pmor = configobj.ConfigObj(path_to_case + f'/{case_name}.pmor.sharpy')
    alpha = pmor['parameters']['alpha']
    inertial_forces = np.loadtxt(f'{path_to_case}/forces/forces_aeroforces.txt',
                                 skiprows=1, delimiter=',', dtype=float)[1:4]
    inertial_moments = np.loadtxt(f'{path_to_case}/forces/moments_aeroforces.txt',
                                  skiprows=1, delimiter=',', dtype=float)[1:4]

    return alpha, inertial_forces[2], inertial_forces[0], inertial_moments[1]


class TestStab(unittest.TestCase):

    def test_stability(self):
        """
        |
        |  / free stream
        | /
        |/
        +------------ chord ax (const)


        """
        dir_urel = np.array([1, 0, 0])
        dir_chord = np.array([1, 0, 0])

        alpha = 4 * np.pi / 180

        # rotate the freestream
        dir_urel = algebra.rotation3d_y(alpha).T.dot(dir_urel)

        # Test free stream rotation
        lab = ['x', 'z']
        with self.subTest(msg='Freestream test'):
            assert dir_urel[2] == np.sin(alpha), 'z component of freestream not properly rotated'
        with self.subTest(msg='Freestream test'):
            assert dir_urel[0] == np.cos(alpha), 'x component of freestream not properly rotated'

        # Stability axes
        c_bs = local_stability_axes(dir_urel, dir_chord)

        # Checking X_s
        for ax in [0, 2]:
            with self.subTest(msg='X_s', ax=ax):
                assert c_bs.dot(np.eye(3)[0])[ax] > 0, f'{ax}_b component of X_s not correct'

        # Checking Z_s
        ax = 0
        with self.subTest(msg='Z_s', ax=ax):
            assert c_bs.dot(np.eye(3)[2])[ax] < 0, f'{ax}_b component of Z_s not correct'

        ax = 2
        with self.subTest(msg='Z_s', ax=ax):
            assert c_bs.dot(np.eye(3)[2])[ax] > 0, f'{ax}_b component of Z_s not correct'


if __name__ == '__main__':
    import unittest

    unittest.main()
