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


class ProblemData(object):
    """Main class for preSHARPy.

    Args:
        in_case_name (str): name for the current case. The input files will be named
            ``<in_case_name>.[fem.h5/aero.h5]``
        in_case_route (str, optional): relative route to the case files. Defaults to ``./``.

    Note:
        As a summary, the case data files have to be located in ``<in_case_route>/<in_case_name>+ext``

        The ``fem_handle`` and ``aero_handle`` attributes will still be open until the destruction of
        the ``ProblemData`` instance.
    """
    def __init__(self, in_case_name, in_case_route = './'):
        print('Reading input info for the case: %s in %s...' % (in_case_name,
                                                                in_case_route))
        self.case_route = in_case_route
        """str: instance copy of in_case_route"""
        self.case_name = in_case_name
        """str: instance copy of in_case_name"""
        fem_file_name = in_case_route + '/' + in_case_name + '.fem.h5'
        aero_file_name = in_case_route + '/' + in_case_name + '.aero.h5'

        # Check if files exist
        h5utils.check_file_exists(fem_file_name)
        h5utils.check_file_exists(aero_file_name)
        print('\tThe FEM and aero files exist, they will be loaded.')

        # Assign handles
        # !!!
        # These handles are stored in the class.
        # That means that they will stay open until
        # we manually close them or the instance of
        # ProblemData is destroyed
        self.fem_handle = h5.File(fem_file_name, 'r')
        """h5py.File: .fem.h5 file handle"""
        self.aero_handle = h5.File(aero_file_name, 'r')
        """h5py.File: .aero.h5 file handle"""

        # Store h5 info in dictionaries
        self.fem_data_dict = (
            h5utils.load_h5_in_dict(self.fem_handle))
        """dict: contains all the input data of the ``FEM`` file stored in a dictionary"""
        h5utils.check_fem_dict(self.fem_data_dict)

        self.aero_data_dict = (
            h5utils.load_h5_in_dict(self.aero_handle))
        """dict: contains all the input data of the ``aero`` file stored in a dictionary"""
        # h5utils.check_aero_dict(self.aero_data_dict)   #TODO

        # FLIGHT CONDITIONS and SOLVER settings files input
        flightcon_file_name = (in_case_route + '/' +
                               in_case_name + '.flightcon.txt')
        solver_file_name = in_case_route + '/' + in_case_name + '.solver.txt'
        # Check if files exist
        h5utils.check_file_exists(flightcon_file_name)
        h5utils.check_file_exists(solver_file_name)
        print('\tThe FLIGHTCON and SOLVER files exist, they will be loaded.')
        self.flightcon_config = self.load_config_file(flightcon_file_name)
        self.solver_config = self.load_config_file(solver_file_name)

        # import pdb;pdb.set_trace()
        print('\tDONE')
        print('--------------------------------------------------------------')
        print('Processing fem input and generating beam model...')
        self.beam = beam.Beam(self.fem_data_dict)
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

        Retuns:
            config (dict): a ``ConfigParser`` object that behaves like a dictionary
        """
        config = configparser.ConfigParser()
        config.read(file_name)
        return config

    def plot_configuration(self, plot_beam=True, plot_grid=True, persp_correction=True):
        """Main wrapper for case plotting in 3D using matplotlib.

        Args:
            plot_beam (bool, optional): if ``True`` the beam is plotted
            plot_grid (bool, optional): if ``True`` the aero grid is plotted
            persp_correction (bool, optional): if ``True``, the perspective is disable to try to
                simulate an orthogonal perspective.
                (see http://stackoverflow.com/questions/23840756/how-to-disable-perspective-in-mplot3d)

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
            self.beam.plot(fig, ax, plot_triad=True)
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
        plt.show()

