import os
#import warnings as warn
import numpy as np
import scipy.linalg as sclalg
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.cout_utils as cout
import sharpy.structure.utils.modalutils as modalutils


@solver
class DynamicLoads(BaseSolver):
    """
    Calculates a series of figures of merit for the assessment of dynamic_loads around 
    a static equilibrium. 

    """
    solver_id = 'DynamicLoads'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = False
    settings_description['print_info'] = 'Print information and table of eigenvalues'

    settings_types['reference_velocity'] = 'float'
    settings_default['reference_velocity'] = 1.
    settings_description['reference_velocity'] = 'Reference velocity at which to compute \
    eigenvalues for scaled systems'

    settings_types['velocity_increment'] = 'float'
    settings_default['velocity_increment'] = 1. 
    settings_description['velocity_increment'] = 'Increment or decrement of the free-stream velocity \
    in the flutter computation until an interval is found where damping of the flutter mode \
    has opposite signs (the secant method is then employed to refine the search)'

    settings_types['flutter_error'] = 'float'
    settings_default['flutter_error'] = 0.1
    settings_description['flutter_error'] = 'Accepted error in the flutter speed'

    settings_types['damping_tolerance'] = 'float'
    settings_default['damping_tolerance'] = 1e-6
    settings_description['damping_tolerance'] = 'Determine the flutter speed when damping is above \
    this value instead of 0. (useful for some ROMs where stability might not preserved and some \
    eigenvalues are slightly above 0. but do not determine flutter)'

    settings_types['root_method'] = 'str'
    settings_default['root_method'] = 'secant'
    settings_description['root_method'] = 'Method to find the damping of the aeroelastic system \
    crossing the x-axis'
    settings_options['root_method'] = ['secant', 'bisection']
    
    settings_types['calculate_flutter'] = 'bool'
    settings_default['calculate_flutter'] = True
    settings_description['calculate_flutter'] = 'Launch the computation of the flutter speed \
    at the reference velocity'        
    
    settings_types['frequency_cutoff'] = 'float'
    settings_default['frequency_cutoff'] = 0
    settings_description['frequency_cutoff'] = 'Truncate higher frequency modes. \
    If zero none are truncated'

    settings_types['save_eigenvalues'] = 'bool'
    settings_default['save_eigenvalues'] = False
    settings_description['save_eigenvalues'] = 'Save eigenvalues to file. '


    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None
        self.folder = None

        self.save_eigenvalues = False
        self.frequency_cutoff = 0
        self.u_flutter = 1.
        self.dt = 0.1
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 no_ctype=True)

        self.save_eigenvalues = self.settings['save_eigenvalues']
        self.frequency_cutoff = self.settings['frequency_cutoff']

        self.folder = data.output_folder + '//'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Output dict
        self.data.linear.dynamic_loads = dict()
        self.data.linear.dynamic_loads['flutter_results'] = dict()
        self.data.linear.dynamic_loads['turbulenceloads_results'] = dict()

        self.caller = caller

    def run(self, online=False):
        """
        Computes the 

        Returns:
             (np.ndarray): Eigenvalues sorted and frequency truncated
            eigenvectors (np.ndarray): Corresponding mode shapes

        """
        
        if not self.frequency_cutoff:
            self.frequency_cutoff = np.inf

        try:
            ss = self.data.linear.linear_system.update(self.settings['reference_velocity'])
        except:
            ss = self.data.linear.ss
            
        # Convert DT eigenvalues into CT
        if ss.dt:
            # Obtain dimensional time step
            try:
                ScalingFacts = self.data.linear.linear_system.uvlm.sys.ScalingFacts
                if ScalingFacts['length'] != 1.0 and ScalingFacts['time'] != 1.0:
                    self.dt = ScalingFacts['length'] / self.settings['reference_velocity'] * ss.dt
                else:
                    self.dt = ss.dt
            except AttributeError:
                self.dt = ss.dt

        if self.settings['calculate_flutter']:
            self.get_flutter_speed()
               
        return self.data

    def get_flutter_speed(self):
        """
        Calculates the flutter speed

        """
        
        u_inf = self.settings['reference_velocity']
        h = self.settings['velocity_increment']
        epsilon = self.settings['flutter_error']
        damping_tolerance = self.settings['damping_tolerance']

        if self.save_eigenvalues:
            eigs_r_series = [] 
            eigs_i_series = []
            u_inf_series = []
        
        u_new = u_inf
        ss_aeroelastic = self.data.linear.linear_system.update(u_new)
        eigs, eigenvectors = sclalg.eig(ss_aeroelastic.A)
        # Obtain dimensional time
        if self.dt:
            eigs = np.log(eigs) / self.dt
            
        eigs, eigenvectors = self.sort_eigenvalues(eigs, eigenvectors, self.frequency_cutoff)
        damping_vector = eigs.real/np.abs(eigs)
        if damping_tolerance: # set damping as the maximum of eigenvalues above a tolerance
            damping_condition = np.abs(damping_vector) > damping_tolerance
            damping_old = np.max(damping_vector[damping_condition])
        else:
            damping_old = np.max(damping_vector)
        damping_new = damping_old
        if self.save_eigenvalues:
            eigs_r_series.append(eigs.real)
            eigs_i_series.append(eigs.imag)
            u_inf_series.append(np.ones_like(eigs.real)*u_new)
        
        while damping_old*damping_new > 0.:   # Find bounds (+- h) to the flutter speed 
            if damping_new > damping_tolerance: # Decrease the velocity by h
                u_old = u_new
                u_new-=h                
            elif damping_new < damping_tolerance: # Increase the velocity by h 
                u_old = u_new
                u_new+=h                
            else:  # Singularity: only possible if damping_tolerance=0 and damping_new=0 too
                u_old = u_new
                break
            #print('Increment velocity: %s'%u_new)
            ss_aeroelastic = self.data.linear.linear_system.update(u_new) #Build new aeroelastic system
            eigs, eigenvectors = sclalg.eig(ss_aeroelastic.A)
            if self.dt:
                eigs = np.log(eigs) / self.dt
            eigs, eigenvectors = self.sort_eigenvalues(eigs, eigenvectors, self.frequency_cutoff)

            damping_vector = eigs.real/np.abs(eigs)
            damping_old = damping_new
            if damping_tolerance:
                damping_condition = np.abs(damping_vector) > damping_tolerance
                damping_new = np.max(damping_vector[damping_condition])
            else:
                damping_new = np.max(damping_vector)
            if self.save_eigenvalues:
                eigs_r_series.append(eigs.real)
                eigs_i_series.append(eigs.imag)
                u_inf_series.append(np.ones_like(eigs.real)*u_new)

        # Secant method (x-axis=speed, y-axis=damping)
        # self.u_flutter = u_new
        while np.abs(u_new - u_old) > epsilon: # Stop searching when interval is smaller than set error
            if self.settings['root_method'] == 'secant':
                ddamping = (damping_new-damping_old)/(u_new-u_old)  # Slope in secant method               
                du = -damping_old/ddamping
                u_secant = u_old - damping_old/ddamping # Calculated speed to set damping to 0: \
                                                    # damping_old + ddamping*(u_secant-u_old) = 0
            elif self.settings['root_method'] == 'bisection':
                u_secant = (u_new + u_old)/2
            ss_aeroelastic = self.data.linear.linear_system.update(u_secant)
            #print('Secant velocity new: %s'%u_new)
            #print('Secant velocity: %s'%u_secant)
            eigs, eigenvectors = sclalg.eig(ss_aeroelastic.A)
            if self.dt:
                eigs = np.log(eigs) / self.dt
            eigs, eigenvectors = self.sort_eigenvalues(eigs, eigenvectors, self.frequency_cutoff)
            damping_vector = eigs.real/np.abs(eigs)
            # Store eigenvalues for plot
            eigs_r_series.append(eigs.real)
            eigs_i_series.append(eigs.imag)
            u_inf_series.append(np.ones_like(eigs.real)*u_secant)
            if damping_tolerance:
                damping_condition = np.abs(damping_vector) > damping_tolerance
                damping_secant = np.max(damping_vector[damping_condition])
            else:
                damping_secant = np.max(damping_vector)
            if damping_secant > 0.:
                if damping_new > 0.:   # Damping_new same sign as damping_secant so it is updated 
                    u_new = u_secant
                    damping_new = damping_secant
                elif damping_new < 0.: # Damping_old updated with damping_secant
                    u_old = u_secant
                    damping_old = damping_secant
            elif damping_secant < 0.:
                if damping_new < 0.:   # Damping_new updated with damping_secant
                    u_new = u_secant
                    damping_new = damping_secant
                elif damping_new > 0.: # Damping_old updated with damping_secant 
                    u_old = u_secant
                    damping_old = damping_secant
            else:
                u_new = u_old = u_secant  # break the loop, damping = 0.
        self.u_flutter = u_new
        
        if self.settings['print_info']:
            cout.cout_wrap('Calculated flutter speed: %.2f m/s' %self.u_flutter, 1)
        if self.save_eigenvalues:
            eigs_r_series = np.hstack(eigs_r_series)
            eigs_i_series = np.hstack(eigs_i_series)
            u_inf_series = np.hstack(u_inf_series)
            cout.cout_wrap('Saving flutter eigenvalues...')
            np.savetxt(self.folder + '/flutter_results.dat',
                   np.concatenate((u_inf_series, eigs_r_series, eigs_i_series)).reshape((-1, 3), order='F'))
            cout.cout_wrap('\tSuccessful', 1)

        self.data.linear.dynamic_loads['flutter_results']['u_flutter'] = np.array([self.u_flutter])

    @staticmethod
    def sort_eigenvalues(eigenvalues, eigenvectors, frequency_cutoff=0):
        """
        Sort continuous-time eigenvalues by order of magnitude.

        The conjugate of complex eigenvalues is removed, then if specified, high frequency modes are truncated.
        Finally, the eigenvalues are sorted by largest to smallest real part.

        Args:
            eigenvalues (np.ndarray): Continuous-time eigenvalues
            eigenvectors (np.ndarray): Corresponding right eigenvectors
            frequency_cutoff (float): Cutoff frequency for truncation ``[rad/s]``

        Returns:

        """

        if frequency_cutoff == 0:
            frequency_cutoff = np.inf

        # Remove poles in the negative imaginary plane (Im(\lambda)<0)
        criteria_a = np.abs(np.imag(eigenvalues)) <= frequency_cutoff
        # criteria_b = np.imag(eigenvalues) > -1e-2
        eigenvalues_truncated = eigenvalues[criteria_a].copy()
        eigenvectors_truncated = eigenvectors[:, criteria_a].copy()

        order = np.argsort(eigenvalues_truncated.real)[::-1]

        return eigenvalues_truncated[order], eigenvectors_truncated[:, order]
