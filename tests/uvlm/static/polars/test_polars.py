import unittest
import tests.uvlm.static.polars.generate_wing as gw
import os
import glob
import configobj
import numpy as np


class InfiniteWing:
    area = 90000000.0
    chord = 3

    def force_coef(self, rho, uinf):
        return 0.5 * rho * uinf ** 2 * self.area

    def moment_coef(self, rho, uinf):
        return self.force_coef(rho, uinf) * self.chord


class TestAirfoilPolars(unittest.TestCase):

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    polar_data = np.loadtxt('xf-naca0018-il-50000.txt', skiprows=12)

    def test_infinite_wing(self):

        cases_route = self.route_test_dir + '/cases/'
        output_route = self.route_test_dir + '/output/'

        wing = InfiniteWing()

        gw.run(polar=True,
               compute_uind=False,
               infinite_wing=True,
               main_ea=0.25,
               high_re=False,
               case_route_root=cases_route,
               output_route_root=output_route)

        case_header = gw.get_case_header(polar=True,
                                         infinite_wing=True,
                                         compute_uind=False,
                                         high_re=False,
                                         main_ea=0.25,
                                         use2pi=False)

        results = self.postprocess(output_route + '/' + case_header + '/')

        import pdb; pdb.set_trace()
        print(type(results))
        results[:, 1] /= wing.force_coef(1.225, 1)
        results[:, -1] /= wing.moment_coef(1.225, 1)

        with self.subTest('lift'):
            cl_polar = np.interp(results[:, 0], self.polar_data[:, 0], self.polar_data[:, 1])
            print(cl_polar)
            print('sh', results[:, 1])
            # np.testing.assert_array_almost_equal(cl_polar, results[:, 1], decimal=1)

    def postprocess(self, output_folder):
        cases = glob.glob(output_folder + '/*')

        n_cases = 0
        for case in cases:
            alpha, lift, drag, moment = self.process_case(case)
            if n_cases == 0:
                results = np.array([alpha, lift, drag, moment], dtype=float)
            else:
                results = np.vstack((results, np.array([alpha, lift, drag, moment])))
            n_cases += 1

        order = np.argsort(results[:, 0])
        results = np.array([results[i, :] for i in order], dtype=float)

        return results

    def process_case(self, path_to_case):
        case_name = path_to_case.split('/')[-1]
        pmor = configobj.ConfigObj(path_to_case + f'/{case_name}.pmor.sharpy')
        alpha = pmor['parameters']['alpha']
        inertial_forces = np.loadtxt(f'{path_to_case}/forces/forces_aeroforces.txt',
                                     skiprows=1, delimiter=',', dtype=float)[1:4]
        inertial_moments = np.loadtxt(f'{path_to_case}/forces/moments_aeroforces.txt',
                                      skiprows=1, delimiter=',', dtype=float)[1:4]

        return alpha, inertial_forces[2], inertial_forces[0], inertial_moments[1]
