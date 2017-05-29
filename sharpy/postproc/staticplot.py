import sharpy.utils.cout_utils as cout
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver


@solver
class StaticPlot(BaseSolver):
    solver_id = 'StaticPlot'
    solver_type = 'postproc'
    solver_unsteady = False

    def __init__(self):
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()

    def run(self):
        if self.settings['plot_shape']:
            print('Plotting the structure...')
            self.plot_shape()
        if self.settings['print_info']:
            self.print_info()
        print('...Finished')
        return self.data

    def convert_settings(self):
        try:
            self.settings['print_info'] = (str2bool(self.settings['print_info']))
        except KeyError:
            self.settings['print_info'] = True

        try:
            self.settings['plot_shape'] = (str2bool(self.settings['plot_shape']))
        except KeyError:
            self.settings['plot_shape'] = True

        try:
            self.settings['figures_location'] = (str2bool(self.settings['figures_location']))
        except KeyError:
            cout.cout_wrap('StaticPlot: no location for figures defined, defaulting to ./')
            self.settings['figures_location'] = './'

    def plot_shape(self):
        pos_def = self.data.beam.timestep_info[self.data.beam.it].pos_def
        pos_ini = self.data.beam.pos_ini


        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D, proj3d
        self.shape_fig = plt.figure()
        self.shape_ax = self.shape_fig.add_subplot(111, projection='3d')
        plt.title('Structure plot')
        self.shape_ax.set_xlabel('x (m)')
        self.shape_ax.set_ylabel('y (m)')
        self.shape_ax.set_zlabel('z (m)')
        plt.axis('equal')
        # plt.hold('on')
        for elem in self.data.beam.elements:
            elem.plot(self.shape_fig, self.shape_ax, plot_triad=True, defor=False)
        # nodes
        nodes = self.shape_ax.scatter(pos_ini[:, 0],
                                      pos_ini[:, 1],
                                      pos_ini[:, 2], color='k')

        for elem in self.data.beam.elements:
            elem.plot(self.shape_fig, self.shape_ax, plot_triad=True, defor=True)
        # nodes
        nodes = self.shape_ax.scatter(pos_def[:, 0],
                                      pos_def[:, 1],
                                      pos_def[:, 2], color='b')

        plt.show()


    def print_info(self):
        psi_def = self.data.beam.psi_def
        psi_ini = self.data.beam.psi_ini
        pos_def = self.data.beam.pos_def
        pos_ini = self.data.beam.pos_ini
        print('Tip:')
        print('\tPos_def:')
        print('\t\t' + '%8.6f '*3 % (pos_def[-1, 0],
                                     pos_def[-1, 1],
                                     pos_def[-1, 2]))
        print('\tPsi_def:')
        print('\t\t' + '%8.6f '*3 % (psi_def[-1, 1, 0],
                                     psi_def[-1, 1, 1],
                                     psi_def[-1, 1, 2]))
