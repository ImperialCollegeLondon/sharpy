import os
#import warnings as warn
import numpy as np
import scipy.linalg as sclalg
##########
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.cout_utils as cout
import sharpy.structure.utils.modalutils as modalutils
import sharpy.linear.src.libss as libss
import sharpy.linear.assembler.lineargustassembler as lineargustassembler
import sharpy.utils.cs25 as cs25
import sharpy.utils.stochastic as stochastic
##########

@solver
class DynamicLoads(BaseSolver):
    """
    Calculates flutter and a series of figures of merit for 
    the assessment of dynamic loads around a static equilibrium. 

    """
    solver_id = 'DynamicLoads'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Print information and table of eigenvalues'

    settings_types['calculate_flutter'] = 'bool'
    settings_default['calculate_flutter'] = True
    settings_description['calculate_flutter'] = 'Launch the computation of the flutter speed \
    at the reference velocity'

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
    settings_description['damping_tolerance'] = 'Determine the flutter speed when \
    damping is above this value instead of 0. (useful for some ROMs where \
    stability might not preserved and some eigenvalues are slightly above 0. \
    but do not determine flutter)'

    settings_types['root_method'] = 'str'
    settings_default['root_method'] = 'secant'
    settings_description['root_method'] = 'Method to find the damping of the aeroelastic system \
    crossing the x-axis'
    settings_options['root_method'] = ['secant', 'bisection']

    settings_types['secant_max_calls'] = 'int'
    settings_default['secant_max_calls'] = 0
    settings_description['secant_max_calls'] = 'Maximum number of calls in secant algorithm, \
    after which bisection is employed (secant is usually faster but convergence is not guaranteed)'
    settings_types['mach_number'] = 'float'
    settings_default['mach_number'] = 0.
    settings_description['mach_number'] = 'Scale results with Mach number'

    settings_types['flutter_upperbound'] = 'float'
    settings_default['flutter_upperbound'] = 0.
    settings_description['flutter_upperbound'] = 'Set an upper velocity bound (> reference_velocity) after \
    which flutter is not calculated (useful for optimization problems where flutter is a constraint)'

    settings_types['flutter_lowerbound'] = 'float'
    settings_default['flutter_lowerbound'] = 0.
    settings_description['flutter_lowerbound'] = 'Set a lower velocity bound (< reference_velocity)'

    settings_types['frequency_cutoff'] = 'float'
    settings_default['frequency_cutoff'] = 0
    settings_description['frequency_cutoff'] = 'Truncate higher frequency modes. \
    If zero none are truncated'

    settings_types['save_eigenvalues'] = 'bool'
    settings_default['save_eigenvalues'] = False
    settings_description['save_eigenvalues'] = 'Save eigenvalues to file. '

    settings_types['calculate_rootloads'] = 'bool'
    settings_default['calculate_rootloads'] = False
    settings_description['calculate_rootloads'] = ''

    settings_types['flight_conditions'] = 'dict'
    settings_default['flight_conditions'] = {}
    settings_description['flight_conditions'] = ''

    settings_types['gust_regulation'] = 'str'
    settings_default['gust_regulation'] = 'Continuous_gust'
    settings_description['gust_regulation'] = ''

    settings_types['white_noise_covariance'] = 'list(float)'
    settings_default['white_noise_covariance'] = []
    settings_description['white_noise_covariance'] = ''


    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    flight_conditions_settings_types = dict()
    flight_conditions_settings_default = dict()
    flight_conditions_settings_description = dict()

    flight_conditions_settings_types['U_inf'] = 'float'
    flight_conditions_settings_default['U_inf'] = 1.0
    flight_conditions_settings_description['U_inf'] = 'Flying speed'

    flight_conditions_settings_types['altitude'] = 'float'
    flight_conditions_settings_default['altitude'] = 1.0
    flight_conditions_settings_description['altitude'] = 'Flying altitude'
    
    __doc__ += settings_table.generate(flight_conditions_settings_types,
                                       flight_conditions_settings_default,
                                       flight_conditions_settings_description)

    def __init__(self):
        self.settings = None
        self.data = None
        self.folder = None

        self.save_eigenvalues = False
        self.frequency_cutoff = None
        self.u_flutter = None
        self.dt = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 no_ctype=True)
        settings.to_custom_types(self.settings['flight_conditions'],
                                 self.flight_conditions_settings_types,
                                 self.flight_conditions_settings_default, no_ctype=True)

        self.save_eigenvalues = self.settings['save_eigenvalues']
        self.frequency_cutoff = self.settings['frequency_cutoff']
        self.white_noise_covariance = self.settings['white_noise_covariance']
        self.flight_conditions = self.settings['flight_conditions']
        self.gust_regulation = self.settings['gust_regulation']

        self.folder = data.output_folder + '/dynamicloads_analysis'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Output dict
        self.data.linear.dynamic_loads = dict()
        self.data.linear.dynamic_loads['flutter_results'] = dict()
        self.data.linear.dynamic_loads['loads_results'] = dict()

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

        # try:
        #     ss = self.data.linear.linear_system.update(self.settings['reference_velocity'])
        # except:
        #     ss = self.data.linear.ss
            
        # # Convert DT eigenvalues into CT
        # if ss.dt:
        #     # Obtain dimensional time step
        #     try:
        #         ScalingFacts = self.data.linear.linear_system.uvlm.sys.ScalingFacts
        #         if ScalingFacts['length'] != 1.0 and ScalingFacts['time'] != 1.0:
        #             self.dt = ScalingFacts['length'] / self.settings['reference_velocity'] * ss.dt
        #         else:
        #             self.dt = ss.dt
        #     except AttributeError:
        #         self.dt = ss.dt
        # import pdb; pdb.set_trace();
                                
        if self.settings['calculate_flutter']:
            self.get_flutter_speed()
        if self.settings['calculate_rootloads']:
            self.get_max_rootloads()
               
        return self.data

    def get_rootloads(self):

        Max_x = dict() ; Max_y = dict()
        key_factory = [ki for ki in self.data.linear.statespaces.keys()
                       if 'ss_factory_' in ki]
        for ki in key_factory:
            Sigma_x, Sigma_y = stochastic.system_covariance(
                self.data.linear.statespaces[ki].A,
                self.data.linear.statespaces[ki].B,
                self.data.linear.statespaces[ki].C,
                self.white_noise_covariance)
            sigma, rho = stochastic.correlation_coeff(Sigma_y)
            rho = rho[0,1]
            gust = getattr(cs25, self.gust_regulation)(
                              **self.flight_conditions)
            U_sigma = gust.U_sigma
            points_xaxis, points_yaxis = cs25.bivariate_ellipse_design(sigma,
                                                                       rho,
                                                                       U_sigma,
                                                                       P_1g=[0., 0.])
            max_x = max(points_xaxis)
            max_y = max(points_yaxis)
            Max_x[ki] = max_x
            Max_y[ki] = max_y
        return Max_x, Max_y

    def get_max_rootloads(self):

        self.root_x, self.root_y = self.get_rootloads()
        self.root_max_x = max([vi for ki,vi in self.root_x.items()])
        self.root_max_y = max([vi for ki,vi in self.root_y.items()])
        self.data.linear.dynamic_loads['loads_results']['root_max_x'] = \
            self.root_max_x
        self.data.linear.dynamic_loads['loads_results']['root_max_y'] = \
            self.root_max_y
        
    def get_flutter_speed(self):
        """
        Calculates the flutter speed by updating the beam DLTI system with new velocities
        until the velocity of 0 damping is found

        """
        
        u_inf = self.settings['reference_velocity']
        h = self.settings['velocity_increment']
        epsilon = self.settings['flutter_error']
        damping_tolerance = self.settings['damping_tolerance']
        if self.settings['secant_max_calls'] == 0:
            secant_max_calls = np.inf
        else:
            secant_max_calls = self.settings['secant_max_calls']
            
        if self.settings['flutter_upperbound'] == 0:
            flutter_upperbound = np.inf
        else:
            flutter_upperbound = self.settings['flutter_upperbound']
        if self.settings['flutter_lowerbound'] == 0:
            flutter_lowerbound = 0.
        else:
            flutter_lowerbound = self.settings['flutter_lowerbound']
            
        flutter_calculation = 1 # find flutter speed unless an upper bound is
                                # defined and the speed is beyond that bound   
        if self.save_eigenvalues:
            eigs_r_series = []
            eigs_i_series = []
            damping_series = []
            u_inf_series = []
        length_ref = self.data.linear.linear_system.uvlm.sys.ScalingFacts['length']
        u_new = u_inf
        ss_aeroelastic = self.data.linear.linear_system.update(u_new)
        self.dt = ss_aeroelastic.dt
        dt_dimensional = (length_ref / u_new) * self.dt
        eigs, eigenvectors = sclalg.eig(ss_aeroelastic.A)
        # Obtain dimensional time
        if self.dt:
            eigs = np.log(eigs) / dt_dimensional
            
        eigs, eigenvectors = self.sort_eigenvalues(eigs, eigenvectors, self.frequency_cutoff)
        damping_vector = eigs.real / np.abs(eigs)
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
            damping_series.append(damping_vector)
        ###########################################
        # Find bounds (+- h) to the flutter speed #
        ###########################################
        while damping_old*damping_new > 0.:
            if damping_new > damping_tolerance: # Decrease the velocity by h
                u_old = u_new
                u_new-=h                
            elif damping_new < damping_tolerance: # Increase the velocity by h 
                u_old = u_new
                u_new+=h                
            else:  # Singularity: only possible if damping_tolerance=0 and damping_new=0 too
                u_old = u_new
                break

            dt_dimensional = (length_ref / u_new) * self.dt
            ss_aeroelastic = self.data.linear.linear_system.update(u_new) #Build new aeroelastic system
            eigs, eigenvectors = sclalg.eig(ss_aeroelastic.A)
            if self.dt:
                eigs = np.log(eigs) / dt_dimensional

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
                damping_series.append(damping_vector)
            if u_new >= flutter_upperbound:                
                flutter_calculation = 0
                break
            elif u_new <= flutter_lowerbound:
                flutter_calculation = 0
                break

        ##############################################################################
        # root finding via secant or bisection method (x-axis=speed, y-axis=damping) #
        ##############################################################################
        self.flutter_root_calls = 0 # counter for number of calls inside next while loop
        while np.abs(u_new - u_old) > epsilon and flutter_calculation:
            # Stop searching when interval is smaller than set error
            if (self.settings['root_method'] == 'secant' and
                self.flutter_root_calls <= secant_max_calls):
                ddamping = (damping_new-damping_old)/(u_new-u_old)  # Slope in secant method               
                du = -damping_old/ddamping
                u_secant = u_old - damping_old/ddamping # Calculated speed to set damping to 0: \
                                                    # damping_old + ddamping*(u_secant-u_old) = 0
            elif (self.settings['root_method'] == 'bisection' or
                  self.flutter_root_calls > secant_max_calls):
                u_secant = (u_new + u_old)/2
                
            dt_dimensional = (length_ref / u_secant) * self.dt
            ss_aeroelastic = self.data.linear.linear_system.update(u_secant) #Build new aeroelastic system
            eigs, eigenvectors = sclalg.eig(ss_aeroelastic.A)
            if self.dt:
                eigs = np.log(eigs) / dt_dimensional
            eigs, eigenvectors = self.sort_eigenvalues(eigs, eigenvectors, self.frequency_cutoff)
            damping_vector = eigs.real/np.abs(eigs)
            # Store eigenvalues
            if self.save_eigenvalues:
                eigs_r_series.append(eigs.real)
                eigs_i_series.append(eigs.imag)
                u_inf_series.append(np.ones_like(eigs.real)*u_secant)
                damping_series.append(damping_vector)
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

            self.flutter_root_calls += 1
            
        self.u_flutter = u_new
        if self.settings['mach_number'] > 0.:
            self.u_flutter *= np.sqrt(1 - self.settings['mach_number']**2)
        if self.settings['print_info']:
            cout.cout_wrap('Calculated flutter speed: %.2f m/s' %self.u_flutter, 1)
        if self.save_eigenvalues:
            eigs_r_series = np.hstack(eigs_r_series)
            eigs_i_series = np.hstack(eigs_i_series)
            u_inf_series = np.hstack(u_inf_series)
            damping_series = np.hstack(damping_series)
            cout.cout_wrap('Saving flutter eigenvalues...')
            np.savetxt(self.folder + '/flutter_eigs.dat',
                       np.concatenate((u_inf_series, eigs_r_series, eigs_i_series,
                                       damping_series)).reshape((-1, 4),
                                                                order='F'),
                       fmt='%.8e', header='u_inf eigs_r eigs_i damping')
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

        #order = np.argsort(eigenvalues_truncated.real)[::-1]
        order = (np.argsort(eigenvalues_truncated.real / np.abs(eigenvalues_truncated)))[::-1]

        return eigenvalues_truncated[order], eigenvectors_truncated[:, order]
