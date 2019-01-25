import ctypes as ct
import matplotlib.pyplot as plt
import numpy as np
import os
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.h5utils as h5

from sharpy.linear.src.lin_aeroelastic import LinAeroEla

@solver
class AsymptoticStabilityAnalysis(BaseSolver):
    """
    Calculates the asymptotic stability properties of aeroelastic systems by creating linearised systems and computing
    the corresponding eigenvalues

    Todo:
        Better integration of the linear system settings (create a loader and check that the system has not been
        previously assembled.

    Warnings:
        Currently under development. Support offered for clamped structures only.

    """
    solver_id = 'AsymptoticStabilityAnalysis'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = False

        self.settings_types['LinearUvlm'] = 'dict'
        self.settings_default['LinearUvlm'] = None

        # self.settings_types['dt'] = 'float'
        # self.settings_default['dt'] = 0.05
        #
        # self.settings_types['density'] = 'float'
        # self.settings_default['density'] = 1.225
        #
        # self.settings_types['integr_order'] = 'int'
        # self.settings_default['integr_order'] = 1
        #
        # self.settings_types['remove_predictor'] = 'bool'
        # self.settings_default['remove_predictor'] = False
        #
        # self.settings_types['ScalingDict'] = 'dict'
        # self.settings_default['ScalingDict'] = {'length': 1,
        #                                         'speed': 1,
        #                                         'density': 1}
        #
        # self.settings_types['use_sparse'] = 'bool'
        # self.settings_default['use_sparse'] = False

        self.settings_types['frequency_cutoff'] = 'float'
        self.settings_default['frequency_cutoff'] = 100

        self.settings_types['export_eigenvalues'] = 'bool'
        self.settings_default['export_eigenvalues'] = True

        self.settings = None
        self.data = None
        self.dt = None

        self.aeroelastic = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequency_cutoff = np.inf

    def initialise(self, data, custom_settings=None):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def run(self):
        """
        Assembles a linearised system in discrete-time state-space form and computes the eigenvalues

        Returns:
            eigenvalues (np.ndarray): Eigenvalues sorted and frequency truncated
            eigenvectors (np.ndarray): Corresponding mode shapes

        """
        try:
            self.frequency_cutoff = self.settings['frequency_cutoff'].value
        except AttributeError:
            self.frequency_cutoff = float(self.settings['frequency_cutoff'])

        try:
            self.dt = self.settings['LinearUvlm']['dt'].value
        except AttributeError:
            self.dt = float(self.settings['LinearUvlm']['dt'])

        # Assemble linear system
        self.aeroelastic = LinAeroEla(self.data, self.settings)
        self.aeroelastic.assemble_ss()

        # Calculate eigenvectors and eigenvalues of the full system
        eigenvalues, eigenvectors = np.linalg.eig(self.aeroelastic.SS.A)

        # Convert DT eigenvalues into CT
        eigenvalues = np.log(eigenvalues) / self.dt

        self.eigenvalues, self.eigenvectors = self.sort_eigenvalues(eigenvalues, eigenvectors, self.frequency_cutoff)

        if self.settings['export_eigenvalues'].value:
            pass

        return self.eigenvalues, self.eigenvectors

    def export_eigenvalues(self):
        pass

    def display_root_locus(self):
        """
        Displays root locus diagrams.

        Returns the ``fig`` and ``ax`` handles for further editing.

        Returns:
            fig:
            ax:
        """

        # Title
        predictor = self.settings['LinearUvlm']['remove_predictor']
        integr_order = self.settings['LinearUvlm']['integr_order']
        sparse = self.settings['LinearUvlm']['use_sparse']



        fig, ax = plt.subplots()

        ax.scatter(np.real(self.eigenvalues), np.imag(self.eigenvalues),
                   s=6,
                   color='k',
                   marker='s')
        ax.set_xlim([-30, 10])
        ax.set_title('Predictor Removed = %r, Int order = %g, Sparse = %r' %(predictor, integr_order, sparse))
        ax.set_xlabel('Real, $\mathbb{R}(\lambda_i)$ [rad/s]')
        ax.set_ylabel('Imag, $\mathbb{I}(\lambda_i)$ [rad/s]')
        ax.set_ylim([0, self.frequency_cutoff])
        ax.grid(True)
        fig.show()

        return fig, ax

    @staticmethod
    def sort_eigenvalues(eigenvalues, eigenvectors, frequency_cutoff=None):
        """
        Sort continuous-time eigenvalues.

        The conjugate of complex eigenvalues is removed, then if specified, high frequency modes are truncated.
        Finally, the eigenvalues are sorted by largest to smallest real part.

        Args:
            eigenvalues (np.ndarray): Continuous-time eigenvalues
            eigenvectors (np.ndarray): Corresponding right eigenvectors
            frequency_cutoff (float): Cutoff frequency for truncation ``[rad/s]``

        Returns:

        """

        if frequency_cutoff is None:
            frequency_cutoff = np.inf

        # Remove poles in the negative imaginary plane (Im(\lambda)<0)
        criteria_a = np.imag(eigenvalues) <= frequency_cutoff
        criteria_b = np.imag(eigenvalues) > -1e-2
        eigenvalues_truncated = eigenvalues[criteria_a * criteria_b].copy()
        eigenvectors_truncated = eigenvectors[:, criteria_a * criteria_b].copy()

        order = np.argsort(-np.real(eigenvalues_truncated))

        return eigenvalues_truncated[order], eigenvectors_truncated[:, order]

if __name__ == '__main__':
    u_inf = 140
    try:
        data = h5.readh5('/home/ng213/code/sharpy/tests/linear/goland_wing/cases/output/goland_u%04g.data.h5' %u_inf).data
    except FileNotFoundError:
        raise FileNotFoundError('Unable to find test case')


    integr_order = 2
    predictor = True
    sparse = False

    aeroelastic_settings = {'LinearUvlm':{
        'dt': data.settings['LinearUvlm']['dt'],
        'integr_order': integr_order,
        'density': 1.020,
        'remove_predictor': predictor,
        'use_sparse': sparse,
        'ScalingDict': {'length': 1,
                        'speed': 1,
                        'density': 1},
        'frequency_cutoff': 100,
        'export_eigenvalues': True
    }}

    analysis = AsymptoticStabilityAnalysis()

    analysis.initialise(data, aeroelastic_settings)
    eigenvalues, eigenvectors = analysis.run()

    analysis.display_root_locus()