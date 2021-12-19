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
    flight_conditions_settings_description['U_inf'] = ''

    flight_conditions_settings_types['altitude'] = 'float'
    flight_conditions_settings_default['altitude'] = 1.0
    flight_conditions_settings_description['altitude'] = ''
    
    __doc__ += settings_table.generate(flight_conditions_settings_types,
                                       flight_conditions_settings_default,
                                       flight_conditions_settings_description)

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
        #import pdb; pdb.set_trace();

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

        self.folder = data.output_folder + '//'
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


class LoadPaths(object):
    """Documentation for ClassName

    """
    def __init__(self,
                 component_nodes,
                 father_components=None):
        
        self.component_nodes = component_nodes
        self.component_names = list(component_nodes.keys())
        self.components = self.get_components(**component_nodes)
        if father_components == None: 
            self.father_components = [self.component_names[0]]
        else:
            self.father_components = father_components
        self.num_father_components = len(self.father_components)
        self.node2component = dict() # dictionary that relates any node in the model to
        # the component it belongs to
        self.get_node2component()
        self.connection_nodes = collections.defaultdict(list)
        self.connection_nodes_upstream = collections.defaultdict(list)
        self.get_connection_nodes()
        self.check_connections()
        
    def get_components(self, **kwargs):
        
        components = dict()
        for ci, vi in kwargs.items():
            if (type(vi[0]) is list or type(vi[0]) is range or type(vi[0]) is np.ndarray) \
               and len(vi)==1: # [[0,1,2...]] 
                components[ci] = list(vi[0])
            elif type(vi[0]) is int and len(vi)==2: 
                components[ci] = list(range(vi[0],vi[1]+1))
            else:
                raise ValueError('Components (%s) values (%s) incorrect in get_components'
                                 %(ci, vi))
        return components

    def get_node2component(self):

        for ci in self.component_names:
            if ci in self.father_components:
                for ni in self.components[ci]:
                    self.node2component[ni] = ci
            else:
                for ni in self.components[ci][1:]:
                    self.node2component[ni] = ci

    def get_connection_nodes(self):

        for ci in self.component_names[self.num_father_components:]:
            ni0 = self.components[ci][0]  # first node in a non-father component must be shared
            self.connection_nodes[ni0] += [ci]
            
        for ni in self.connection_nodes.keys():
            for ci in self.component_names:
                if ci not in self.connection_nodes[ni] and \
                   ni in self.components[ci]:
                    self.connection_nodes_upstream[ni] = ci
                    self.connection_nodes[ni].insert(0, ci)
                                    
    def get_loadpaths(self, node_i, forward=True):

        loadpath = dict()
        component0 = self.node2component[node_i]
        index = self.components[component0].index(node_i)
        if forward:
            nodes = self.components[component0][index:]
        else:
            if component0 in self.father_components:
                nodes = self.components[component0][:index]
            else:
                nodes = self.components[component0][1:index]
                
        self.load_components(loadpath, component0, nodes)
        self.loadpath = loadpath
        path_nodes = []
        for ci, vi in loadpath.items():
            path_nodes += vi
        assert len(set(path_nodes)) == len(path_nodes), "Incorrect definition of load path: repeated \
        nodes in different component"
        path_nodes.sort()
        return path_nodes

    def load_components(self, loadpath, component_name, nodes):

        for ni in nodes: # cycle through nodes in component_name
            if ni in self.connection_nodes.keys(): # branches of components coming out
                # of this node: cycle through them 
                for component_i in self.connection_nodes[ni]:
                    if component_name != component_i:
                        if component_i == self.connection_nodes_upstream[ni]: # not a forward path
                            index = self.components[component_i].index(ni)
                            nodes_i = self.components[component_i][:index]
                        else:
                            nodes_i = self.components[component_i][1:]
                        self.load_components(loadpath, component_i, nodes_i)
        loadpath[component_name] = nodes
        
    def check_connections(self):
        
        for ni in self.connection_nodes.keys():
            assert self.node2component[ni] == self.connection_nodes_upstream[ni], \
            "The upstream component of node %s (%s) is not the same as the output of \
            node to component, %s" %(ni, self.connection_nodes_upstream[ni],
                                     self.node2component[ni])
            
    def check_boundary_conditions(self, nodes, boundary_conditions):
        
        for ni in nodes:
            if boundary_conditions[ni] == 1:
                raise NameError('Boundary condition (%d) of A-frame node (%d) in the path!' \
                                % (boundary_conditions[ni], ni))
            elif boundary_conditions[ni] != -1 or \
                 boundary_conditions[ni] != 0:
                raise NameError('Invalid boundary condition (%d) at node %d!' \
                                % (boundary_conditions[ni], ni))


if (__name__ == '__main__'):

    import unittest

    class Test_LoadPaths(unittest.TestCase):
        """ Test methods into this module """

        def __init__(self, *args, **kwargs):
            super(Test_LoadPaths, self).__init__(*args, **kwargs)
            self.path1 = LoadPaths({'c1':[1,4],'c2':[[3,5]],'c3':[[3,6,7,8]],'c4':[8,11]})

        def test_loadpath0(self):
            """
            to do: add check on moments gain
            """
            path_nodes = self.path1.get_loadpaths(1,forward=False)
            assert self.path1.loadpath['c1'] == [], 'Error in test_loadpath0 c1'
        def test_loadpath0f(self):
            """
            to do: add check on moments gain
            """
            path_nodes = self.path1.get_loadpaths(1,forward=True)
            assert self.path1.loadpath['c1'] == [1, 2, 3, 4], 'Error in test_loadpath0f c1'
            assert self.path1.loadpath['c2'] == [5], 'Error in test_loadpath0f c2'
            assert self.path1.loadpath['c3'] == [6, 7, 8], 'Error in test_loadpath0f c3'
            assert self.path1.loadpath['c4'] == [9, 10, 11], 'Error in test_loadpath0f c4'
            
        def test_loadpath1(self):
            """
            to do: add check on moments gain
            """
            path_nodes = self.path1.get_loadpaths(3,forward=False)
            assert self.path1.loadpath['c1'] == [1, 2], 'Error in test_loadpath1 c1'
        def test_loadpath1f(self):
            """
            to do: add check on moments gain
            """
            path_nodes = self.path1.get_loadpaths(3,forward=True)
            assert self.path1.loadpath['c1'] == [3, 4], 'Error in test_loadpath1f c1'
            assert self.path1.loadpath['c2'] == [5], 'Error in test_loadpath1f c2'
            assert self.path1.loadpath['c3'] == [6, 7, 8], 'Error in test_loadpath1f c3'
            assert self.path1.loadpath['c4'] == [9, 10, 11], 'Error in test_loadpath1f c4'
            
        def test_loadpath2(self):
            """
            to do: add check on moments gain
            """
            path_nodes = self.path1.get_loadpaths(8,forward=False)
            assert self.path1.loadpath['c3'] == [6, 7], 'Error in test_loadpath2 c1'
        def test_loadpath2f(self):
            """
            to do: add check on moments gain
            """
            path_nodes = self.path1.get_loadpaths(8,forward=True)
            assert self.path1.loadpath['c3'] == [8], 'Error in test_loadpath2f c3'
            assert self.path1.loadpath['c4'] == [9, 10, 11], 'Error in test_loadpath2f c4'

    unittest.main()

