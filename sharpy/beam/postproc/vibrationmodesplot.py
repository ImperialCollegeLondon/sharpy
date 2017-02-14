import ctypes as ct
import numpy as np
import matplotlib.pyplot as plt

import sharpy.beam.utils.beamlib as beamlib
from presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class VibrationModesPlot(BaseSolver):
    solver_id = 'VibrationModesPlot'
    solver_type = 'postproc'
    solver_unsteady = False

    def __init__(self):
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()

    def run(self):
        print('Plotting modes...')
        self.process_eigenvalues()
        self.plot_modes()
        print('...Finished')
        return self.data

    def convert_settings(self):
        try:
            self.settings['print_info'] = (str2bool(self.settings['print_info']))
        except KeyError:
            pass
        try:
            self.settings['save_figures'] = (str2bool(self.settings['save_figures']))
        except KeyError:
            pass

        try:
            self.settings['figures_location'] = (str2bool(self.settings['figures_location']))
        except KeyError:
            pass

        try:
            self.settings['number_modes_plot'] = (str2bool(self.settings['number_modes_plot']))
        except KeyError:
            pass

        try:
            self.settings['save_frequency_list'] = (str2bool(self.settings['save_frequency_list']))
        except KeyError:
            pass

        try:
            self.settings['save_modes_list'] = (str2bool(self.settings['save_modes_list']))
        except KeyError:
            pass

        try:
            self.settings['save_number_modes'] = (str2bool(self.settings['save_number_modes']))
        except KeyError:
            pass

        try:
            self.settings['files_location'] = (str(self.settings['files_location']))
        except KeyError:
            pass

        try:
            self.settings['files_format'] = (str(self.settings['files_format']))
        except KeyError:
            pass

    def plot_modes(self):
        a = 1

    def process_eigenvalues(self):
        self.data.beam.freq = np.sqrt(self.data.beam.w)/(2*np.pi)
        pos_def = np.copy(self.data.beam.pos_ini, order='F')
        psi_def = np.copy(self.data.beam.psi_ini, order='F')
        beamlib.cbeam3_solv_update_static_python(self.data.beam,
                                                 self.data.beam.v[:, 0],
                                                 pos_def,
                                                 psi_def)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('mode 1')
        plt.plot(pos_def[:,0], pos_def[:, 1], pos_def[:, 2])
        plt.show()


        # 2nd mode
        pos_def = np.copy(self.data.beam.pos_ini, order='F')
        psi_def = np.copy(self.data.beam.psi_ini, order='F')
        beamlib.cbeam3_solv_update_static_python(self.data.beam,
                                                 self.data.beam.v[:, 1],
                                                 pos_def,
                                                 psi_def)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('mode 2')
        plt.plot(pos_def[:,0], pos_def[:, 1], pos_def[:, 2])
        plt.show()

        # 3rd mode
        pos_def = np.copy(self.data.beam.pos_ini, order='F')
        psi_def = np.copy(self.data.beam.psi_ini, order='F')
        beamlib.cbeam3_solv_update_static_python(self.data.beam,
                                                 self.data.beam.v[:, 2],
                                                 pos_def,
                                                 psi_def)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('mode 3')
        plt.plot(pos_def[:,0], pos_def[:, 1], pos_def[:, 2])
        plt.show()
        a = 1
