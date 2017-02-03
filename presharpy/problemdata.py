"""The main class for preSHARPy.

Contains the necessary code for reading, processing and storing the input data. It implements
the ``ProblemData`` class, which is the base for the preSHARPy module.

Examples:
    This example assumes that ``generate_test_data`` has been already executed and ``pwd`` is set
    at ``sharpy/bin``.
        >>> from presharpy.problemdata import ProblemData
        >>> problem = ProblemData('test', '../presharpy/test/')
"""

import configparser

import h5py as h5
import matplotlib.pyplot as plt

import presharpy.aerogrid.aerogrid as aerogrid
import presharpy.beam.beam as beam
import presharpy.utils.h5utils as h5utils
from sharpy.utils.solver_interface import solver, solver_types, dict_of_solvers
import sharpy.utils.plotutils as plotutils


@solver
class ProblemData(object):
    """Main class for preSHARPy.

    Args:
        settings (dict): dictionary from DictConfigParser

    Note:
        As a summary, the case data files have to be located in ``<in_case_route>/<in_case_name>+ext``

        The ``fem_handle`` and ``aero_handle`` attributes will still be open until the destruction of
        the ``ProblemData`` instance.
    """
    solver_id = 'preSHARPy'
    solver_type = 'general'

    def __init__(self, settings):
        settings['SHARPy']['flow'] = settings['SHARPy']['flow'][0].split(',')
        # getting rid of leading and trailing spaces
        settings['SHARPy']['flow'] = list(map(lambda x: x.strip(), settings['SHARPy']['flow']))

        self.settings = settings
        self.solver_config = settings
        self.case_route = settings['SHARPy']['route'] + '/'
        self.case_name = settings['SHARPy']['case']


        self.only_structural = True
        for solver_name in settings['SHARPy']['flow']:
            if (not solver_types[solver_name] == 'general' and
                not solver_types[solver_name] == 'structural' and
                not solver_types[solver_name] == 'postproc'):
                self.only_structural = False

        if self.only_structural:
            print('Running a structural case only')

        # Check if the problem is steady or not
        self.steady_problem = True
        for solver_name in settings['SHARPy']['flow']:
            try:
                if dict_of_solvers[solver_name].solver_unsteady:
                    self.steady_problem = False
                    break
            except AttributeError:
                pass

        self.initialise()

    def initialise(self):
        fem_file_name = self.case_route + '/' + self.case_name + '.fem.h5'
        if not self.only_structural:
            aero_file_name = self.case_route + '/' + self.case_name + '.aero.h5'

        if not self.steady_problem:
            dyn_file_name = self.case_route + '/' + self.case_name + '.dyn.h5'

        # Check if files exist
        h5utils.check_file_exists(fem_file_name)
        print('\tThe FEM file exists, it will be loaded.')
        if not self.only_structural:
            h5utils.check_file_exists(aero_file_name)
            print('\tThe aero file exists, it will be loaded.')
        if not self.steady_problem:
            h5utils.check_file_exists(dyn_file_name)
            print('\tThe dynamic forces file exists, it will be loaded.')

        # Assign handles
        # !!!
        # These handles are stored in the class.
        # That means that they will stay open until
        # we manually close them or the instance of
        # ProblemData is destroyed
        self.fem_handle = h5.File(fem_file_name, 'r')
        """h5py.File: .fem.h5 file handle"""
        if not self.only_structural:
            self.aero_handle = h5.File(aero_file_name, 'r')
            """h5py.File: .aero.h5 file handle"""
        if not self.steady_problem:
            self.dyn_handle = h5.File(dyn_file_name, 'r')

        # Store h5 info in dictionaries
        self.fem_data_dict = (
            h5utils.load_h5_in_dict(self.fem_handle))
        """dict: contains all the input data of the ``FEM`` file stored in a dictionary"""
        h5utils.check_fem_dict(self.fem_data_dict)

        if not self.only_structural:
            self.aero_data_dict = (
                h5utils.load_h5_in_dict(self.aero_handle))
            """dict: contains all the input data of the ``aero`` file stored in a dictionary"""
            # h5utils.check_aero_dict(self.aero_data_dict)   #TODO

            # FLIGHT CONDITIONS settings file input
            flightcon_file_name = (self.case_route + '/' +
                                   self.case_name + '.flightcon.txt')
            # Check if flightcon file exists
            h5utils.check_file_exists(flightcon_file_name)
            print('\tThe FLIGHTCON file exist, it will be loaded.')
            self.flightcon_config = self.load_config_file(flightcon_file_name)

        if not self.steady_problem:
            self.dyn_data_dict = (
                h5utils.load_h5_in_dict(self.dyn_handle))

        print('\tDONE')
        print('--------------------------------------------------------------')
        print('Processing fem input and generating beam model...')
        if not self.steady_problem:
            self.beam = beam.Beam(self.fem_data_dict, self.dyn_data_dict)
        else:
            self.beam = beam.Beam(self.fem_data_dict)

        if not self.only_structural:
            print('Processing aero input and generating grid...')
            ProblemData.grid = aerogrid.AeroGrid(self.aero_data_dict,
                                                 self.solver_config,
                                                 self.flightcon_config,
                                                 self.beam)

    @staticmethod
    def load_config_file(file_name):
        """This function reads the flight condition and solver input files.

        Args:
            file_name (str): contains the path and file name of the file to be read by the ``configparser``
                reader.

        Returns:
            config (dict): a ``ConfigParser`` object that behaves like a dictionary
        """
        config = configparser.ConfigParser()
        config.read(file_name)
        return config

    def plot_configuration(self, plot_beam=True, plot_grid=True, persp_correction=True, defor=True):
        """Main wrapper for case plotting in 3D using matplotlib.

        Args:
            plot_beam (bool, optional): if ``True`` the beam is plotted
            plot_grid (bool, optional): if ``True`` the aero grid is plotted
            persp_correction (bool, optional): if ``True``, the perspective is disable to try to
                simulate an orthogonal perspective.
                (see http://stackoverflow.com/questions/23840756/how-to-disable-perspective-in-mplot3d)
            defor (bool, optional): if ``True``, the deformed configuration is also plotted

        Returns:
            None

        Notes:
            A new set of axes is created using:

               >>> fig = plt.figure()
               >>> ax = fig.add_subplot(111, projection='3d')
               >>> plt.title('Case: %s -- structure plot' % self.case_name)
               >>> ax.set_xlabel('x (m)')
               >>> ax.set_ylabel('y (m)')
               >>> ax.set_zlabel('z (m)')

        """
        if self.settings['SHARPy']['plot']:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D, proj3d
            import numpy as np
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.title('Case: %s -- structure plot' % self.case_name)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')

            if plot_beam:
                self.beam.plot(fig, ax, plot_triad=True, defor=defor, ini=True)
            if plot_grid:
                self.grid.plot(fig, ax)

            if persp_correction:
                # correction of perspective
                def orthogonal_projection(zfront, zback):
                    a = (zfront + zback) / (zfront - zback)
                    b = -2 * (zfront * zback) / (zfront - zback)
                    return np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, a, b],
                                     [0, 0, -1e-5, zback]])

                proj3d.persp_transformation = orthogonal_projection
            plt.axis('equal')
            plotutils.set_axes_equal(ax)
            plt.show()

