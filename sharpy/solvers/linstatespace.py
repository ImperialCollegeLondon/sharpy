"""
sadfsad
"""
import os
import copy
#import warnings as warn
import numpy as np
import scipy.linalg as sclalg
##########
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.cout_utils as cout
import sharpy.structure.utils.modalutils as modalutils
import sharpy.linear.src.libss as libss
from sharpy.utils.stochastic import Iterations, LoadPaths
import sharpy.linear.utils.ss_interface as ssi
import sharpy.utils.algebra as algebra
import sharpy.linear.assembler.lincontrolsurfacedeflector \
as lincontrolsurfacedeflector
import sharpy.linear.assembler.lineargustassembler as lineargust

##########


@solver
class LinStateSpace(BaseSolver):
    """
    Calculates a series of figures of merit for the assessment of dynamic_loads around 
    a static equilibrium.

    """
    solver_id = 'LinStateSpace'
    solver_classification = 'coupled'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = False
    settings_description['print_info'] = 'Print information'

    settings_types['ss_workflow'] = 'list(str)'
    settings_default['ss_workflow'] = []
    settings_description['ss_workflow'] = 'List of systems to be added as built in \
    the functions below'
    settings_options['ss_workflow'] = ['', 'ss_gust', 'ss_aeroelastic' '...']

    settings_types['ss_gust'] = 'list(dict)'
    settings_default['ss_gust'] = []
    settings_description['ss_gust'] = ''

    settings_types['ss_aeroelastic'] = 'list(dict)'
    settings_default['ss_aeroelastic'] = {}
    settings_description['ss_aeroelastic'] = ''
    
    settings_types['gain_controlsurfaces'] = 'list(dict)'
    settings_default['gain_controlsurfaces'] = {}
    settings_description['gain_controlsurfaces'] = ''

    settings_types['ss_turbulence'] = 'list(dict)'
    settings_default['ss_turbulence'] = {}
    settings_description['ss_turbulence'] = ''

    settings_types['gain_loads'] = 'list(dict)'
    settings_default['gain_loads'] = {}
    settings_description['gain_loads'] = ''

    settings_types['gain_internalloads'] = 'list(dict)'
    settings_default['gain_internalloads'] = {}
    settings_description['gain_internalloads'] = ''

    settings_types['combination_type'] = 'str'
    settings_default['combination_type'] = 'Full_Factorial'
    settings_description['combination_type'] = ''
    settings_options['combination_type'] = ['', 'ss_gust', 'ss_aeroelastic']

    settings_types['save_all'] = 'bool'
    settings_default['save_all'] = False
    settings_description['save_all'] = ''

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
        self.SS_ABBREVIATION = {'ss_gust':'gt',
                                'gain_controlsurfaces':'cs',
                                'ss_turbulence':'te',
                                'ss_aeroelastic':'ac',
                                'gain_loads':'ls',
                                'gain_internalloads':'is'}

    def initialise(self, data, custom_settings=None, caller=None):
        
        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                no_ctype=True)

        self.caller = caller
        
        # Output dict
        self.data.linear.statespaces = dict()
        self.save_all =  self.settings['save_all']
        self.ss_workflow = self.settings['ss_workflow']
        self.uvlm = copy.deepcopy(self.data.linear.linear_system.uvlm)
        self.beam = copy.deepcopy(self.data.linear.linear_system.beam)
        self.aeroelastic = copy.deepcopy(self.data.linear.linear_system)
        
    def run(self, online=False):
        """
        Computes the 

        Returns:
             (np.ndarray): Eigenvalues sorted and frequency truncated
            eigenvectors (np.ndarray): Corresponding mode shapes

        """

        # try:
        #     ss = self.data.linear.linear_system.update(self.settings['reference_velocity'])
        # except:
        #     ss = self.data.linear.ss
        self.get_combinations()            
        self.ss_factory()
                               
        return self.data

    def get_combinations(self):

        combinations = dict()
        for i_ss in self.ss_workflow:
            if type(self.settings[i_ss]) != list:
                combinations[self.SS_ABBREVIATION[i_ss]] = [self.settings[i_ss]]
            else:
                combinations[self.SS_ABBREVIATION[i_ss]] = []
                for si in self.settings[i_ss]:
                    combinations[self.SS_ABBREVIATION[i_ss]].append(si)
            # else:
            #     raise TypeError("type(%s) = %s => Incorrect type in the \
            #     input settings of linspace solver (must be list or dict)")

        iterations = Iterations(combinations, self.settings['combination_type'])
        self.combination_settings = iterations.get_combinations_dict()
        self.combination_labels = iterations.label_number()
        
    def save_system(self, ss, label):
        self.data.linear.statespaces[label] = ss
        
    def ss_factory(self):
        import pdb; pdb.set_trace();
        
        for i_c, v_c in enumerate(self.combination_settings):
            system = dict()
            system['ss_uvlm'] = copy.deepcopy(self.uvlm.ss0)
            system['ss_beam'] = copy.deepcopy(self.beam.ss)
            system['ss_aeroelas'] = copy.deepcopy(self.aeroelastic.ss)
            label = self.combination_labels[i_c]
            for i_ss in self.ss_workflow:
                build_ss = getattr(self, 'build_'+i_ss)
                build_ss(system,
                         label=label,
                         **v_c[self.SS_ABBREVIATION[i_ss]])
            self.save_system(system['ss_aeroelas'], 'ss_factory_'+label)
                    
    def get_ss_turbulence(self,
                          turbulence_filter,
                          turbulence_filter_in,
                          label='',
                          save=None):

        turbulence_filter = getattr(lineargust, turbulence_filter)
        ss_turbulence = turbulence_filter(**turbulence_filter_in)
        if save or self.save_all:
            self.save_system(ss_turbulence, label='ss_turbulence'+'_'+label)

        return ss_turbulence
    
    def build_ss_gust(self,
                      system,
                      gust_assembler,
                      gust_assembler_in,
                      turbulence_filter='',
                      turbulence_filter_in={},
                      label='',
                      save=None):
        
        if turbulence_filter:
            ss_turbulence = self.get_ss_turbulence(turbulence_filter,
                                                   turbulence_filter_in,
                                                   label,
                                                   save)
        else:
            ss_turbulence = None
        
        self.gust_assembler = lineargust.gust_from_string(gust_assembler)
        self.gust_assembler.initialise(self.data.aero,
                                       self.uvlm.sys,
                                       self.uvlm.tsaero0,
                                       custom_settings=gust_assembler_in
                                       )

        system['ss_uvlm'] = self.gust_assembler.apply(system['ss_uvlm'], ss_turbulence)
        if save or self.save_all:
            self.save_system(system['ss_uvlm'], label='ss_gust'+'_'+label)
            
    def build_gain_controlsurfaces(self,
                                   system,
                                   label='',
                                   save=None):
        
        self.control_surface = lincontrolsurfacedeflector.LinControlSurfaceDeflector()
        self.control_surface.initialise(self.data, self.uvlm.sys)
        #ss2 = self.control_surface.apply(system['ss_uvlm'])
        gain_cs = self.control_surface.gain_cs
        system['ss_uvlm'].addGain(gain_cs, where='in')

    def build_ss_aeroelastic(self,
                             system,
                             u_inf,
                             label='',
                             save=None):
        ##### Normalise beam system with u_inf #####
        t_ref = self.uvlm.sys.ScalingFacts['length'] / u_inf
        self.beam.sys.update_matrices_time_scale(t_ref)
        self.beam.sys.assemble()
        if self.beam.sys.SSdisc is not None:
            self.beam.ss = self.beam.sys.SSdisc
        elif self.beam.sys.SScont is not None:
            self.beam.ss = self.beam.sys.SScont
        else:
            raise AttributeError('Could not find either a continuous \
            or discrete system in Beam')
        ##########

        self.uvlm.ss = system['ss_uvlm']
        #self.data.linear.linear_system.assemble(self.ss_aerogust, beam)
        system['ss_aeroelas'] = self.aeroelastic.assemble(self.uvlm,
                                                          self.beam)
        if save or self.save_all:
            self.save_system(system['ss_aeroelas'], label='ss_aeroelastic'+'_'+label)

    def build_gain_loads(self,
                         system,
                         label='',
                         save=None):
        
        """
        Modify a fully modal SHARPy state space to output the root shear and bending loads.

        Args:
            data (sharpy.PreSharpy):

        """

        ss = system['ss_aeroelas']
        # stiffness matrix at linearisation
        Kstr = self.beam.sys.Kstr  
        structure = self.data.structure
        tsstruct0 = self.data.linear.tsstruct0
        boundary_conditions = structure.boundary_conditions
        loads_transform = np.zeros_like(Kstr)

        if self.beam.sys.modal:

            phi = self.beam.sys.U
            ss.remove_outputs('Q', 'q_dot')
            ss.remove_inputs('q', 'q_dot', 'Q')
        else:
            ss.remove_outputs('forces_n', 'eta_dot')
            ss.remove_inputs('eta', 'eta_dot', 'forces_n')

        for i_global_node in range(structure.num_node): # cycle through all nodes in structure
            # ra = tsstruct0.pos[i_global_node, :]
            i_elem, i_local_node = structure.node_master_elem[i_global_node, :]
            psi = tsstruct0.psi[i_elem, i_local_node]
            Tan = algebra.crv2tan(psi)
            Cab = algebra.crv2rotation(psi)
            if boundary_conditions[i_global_node] == 1:  # A-frame node
                continue
            elif boundary_conditions[i_global_node] == -1 or \
                 boundary_conditions[i_global_node] == 0:  # free-end or internal node
                jj_tra = 6 * structure.vdof[i_global_node] + np.array([0, 1, 2], dtype=int)
                jj_rot = 6 * structure.vdof[i_global_node] + np.array([3, 4, 5], dtype=int)
            else:
                raise NameError('Invalid boundary condition (%d) at node %d!' \
                                % (boundary_conditions[i_global_node], i_global_node))

            # Force-component of loads will be just the stiffness matrix multiplied by displacements  
            loads_transform[np.ix_(jj_tra, jj_tra)] += np.eye(3)
            # loads_transform[np.ix_(jj_rot, jj_tra)] += algebra.skew(ra)
            # Moment-component of loads need to be brought back to the A-frame \
            # (see Ch. 6 of Geradin's book)
            loads_transform[np.ix_(jj_rot, jj_rot)] += Cab.dot(np.linalg.inv(Tan.T)) 
        loads_var = ssi.OutputVariable('loads', size=Kstr.shape[0], index=0)

        if self.beam.sys.modal:
            gain_matrix = loads_transform.dot(Kstr.dot(phi))
        else:
            gain_matrix = loads_transform.dot(Kstr)

        loads_gain = libss.Gain(gain_matrix,
                                input_vars=ssi.LinearVector.transform(ss.output_variables,
                                                                      to_type=ssi.InputVariable),
                                output_vars=ssi.LinearVector([loads_var]))
        
        system['ss_aeroelas'].addGain(loads_gain, where='out')
        if save or self.save_all:
            self.save_system(loads_gain, label='gain_loads'+'_'+label)

    def build_gain_internalloads(self,
                                 system,
                                 monitoring_stations,
                                 components=None,
                                 remove_dof=[],
                                 father_components=None,
                                 label='',
                                 save=None):
                                 
        ss = system['ss_aeroelas']
        structure = self.data.structure
        tsstruct0 = self.data.linear.tsstruct0
        if not components:
            components = {'c1':range(1, structure.num_node)}
        path = LoadPaths(components, father_components, monitoring_stations)
        nodes_ms = path.monitor.nodes
        internal_loads = np.zeros((6*len(nodes_ms), ss.outputs))

        for i_globalnode, v_globalnode in enumerate(nodes_ms):

            node_nexto = path.monitor.node_nexto[v_globalnode]
            ra1 = tsstruct0.pos[v_globalnode, :]
            if node_nexto != 0:
                ra2 = tsstruct0.pos[node_nexto, :]
            else:
                ra2 = np.array([0., 0., 0.])
            ra = (ra1 + ra2)/2
            ii_tra = 6*i_globalnode
            ii_rot = 6*i_globalnode+3
            path_nodes = path.monitor.loadpath[v_globalnode]
            for i_current_node in path_nodes:

                jj_curr_tra = 6 * structure.vdof[i_current_node]
                jj_curr_rot = 6 * structure.vdof[i_current_node] + 3
                delta_r = tsstruct0.pos[i_current_node] - ra
                internal_loads[ii_tra:ii_tra+3,
                               jj_curr_tra:jj_curr_tra+3] += np.eye(3)
                internal_loads[ii_rot:ii_rot+3,
                               jj_curr_tra:jj_curr_tra+3] += algebra.skew(delta_r)
                internal_loads[ii_rot:ii_rot+3,
                               jj_curr_rot:jj_curr_rot+3] += np.eye(3)

            if path.monitor.direction[i_globalnode] is not 'forward':
                #all_nodes = path.get_all_nodes()
                pass
                
            # for i_current_node in all_nodes:

            #     jj_curr_tra = 6 * structure.vdof[i_current_node] + np.array([0, 1, 2], dtype=int)
            #     jj_curr_rot = 6 * structure.vdof[i_current_node] + np.array([3, 4, 5], dtype=int)
            #     delta_r = tsstruct0.pos[i_current_node] - ra
            #     internal_loads[ii_tra, jj_curr_tra] -= np.eye(3)
            #     internal_loads[ii_rot, jj_curr_tra] -= algebra.skew(delta_r)
            #     internal_loads[ii_rot, jj_curr_rot] -= np.eye(3)

        if remove_dof:

            remove_rows = [6*ni+dof for ni in range(len(nodes_ms)) for dof in remove_dof]
            internal_loads = np.delete(internal_loads, remove_rows, axis=0)

            # sum of shear forces (kronecker delta distributed loads -> integral is the sum)
            # internal_loads[-6:-3, jj_tra] += np.eye(3)

            # sum of moments
            # internal_loads[-3:, jj_tra] += algebra.skew(ra)
            # internal_loads[-3:, jj_rot] += np.eye(3)

        internal_loads_var = ssi.OutputVariable('internal_loads',
                                                size=len(internal_loads),
                                                index=0)
        gain_internal_loads = libss.Gain(internal_loads,
                                     input_vars=ssi.LinearVector.transform(ss.output_variables,
                                                                           to_type=ssi.InputVariable),
                                     output_vars=ssi.LinearVector([internal_loads_var]))
        
        system['ss_aeroelas'].addGain(gain_internal_loads, where='out')
        if save or self.save_all:
            self.save_system(gain_internal_loads, label='gain_internalloads'+'_'+label)
