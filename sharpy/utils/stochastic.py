"""
Module to include tools regarding combinatorial problems, probability theory, 
stochastic analysis and path-trees constructures
"""

import numpy as np
import collections
import scipy.stats as stats
import scipy.linalg as linalg

class Iterations:

    """
    Class that manages iterations from the variables to iterate, vars2iterate,
    the type of iteration, full factorial and design of experiments implemented so far,
    to the labels and values corresponding to each iteration
    """
    
    def __init__(self,var2iterate,iter_type='Full_Factorial'):

        self.iter_type = iter_type
        if isinstance(var2iterate, list):
            self.varl = var2iterate
        elif isinstance(var2iterate, dict):
            self.vard = var2iterate
            self.varl = [var2iterate[k] for k in var2iterate.keys()]
            self.var_names = [k for k in var2iterate.keys()]
        self.num_var = len(self.varl)
        self.shape_var = [len(i) for i in self.varl]
        if self.iter_type == 'Full_Factorial':
            self.num_combinations = np.prod(self.shape_var)
        elif self.iter_type == 'DoE':
            self.num_combinations = self.shape_var[0]
            
    def get_combinations_FF(self):
        
        """
        List with all combinations from full factorial experiment
        """
        
        self.varl_combinations = []
        self.varl_index = []
        for i in range(self.num_combinations):
            comb_i = []
            for j in range(self.num_var):
                if j==0:
                    n=1
                else:
                    n = np.prod(self.shape_var[:j])
                l = int(i/n)
                comb_i.append(l%self.shape_var[j])    # indices of the combinations

            self.varl_combinations.append([self.varl[k][comb_i[k]] for k in range(self.num_var)])
            self.varl_index.append(comb_i)
        #self.num_combinations = len(self.varl_combinations)
        return self.varl_combinations

    def get_combinations_DoE(self):
        
        """
        List with all combinations from full factorial experiment
        """
        
        self.varl_combinations = []
        self.varl_index = []
        for k in range(self.num_combinations):
            self.varl_combinations.append([self.varl[j][k] for j in range(self.num_var)])
            self.varl_index.append([k for j in range(self.num_var)])
        #self.num_combinations = len(self.varl_combinations)
        return self.varl_combinations

    def get_combinations_dict(self):

        """
        List of all the combinations in the iteration and
        formed of dictionaries with keys the name of the variables
        """

        if self.iter_type == 'Full_Factorial':
            varl_combinations = self.get_combinations_FF()
        elif self.iter_type == 'DoE':
            varl_combinations = self.get_combinations_DoE()
        self.vard_combinations = []
        for i in range(self.num_combinations):
            dic = dict()
            for j in range(self.num_var):
                dic[self.var_names[j]] = varl_combinations[i][j]
            self.vard_combinations.append(dic)
        return self.vard_combinations
    
    def label_name(self, num2keep=3, space='_', print_name_var=1):

        """
        Labels for the iteration with the name of the variable followed its value or index
        """
        
        dic1 = self.get_combinations_dict()
        label = []
        for ni in range(self.num_combinations):
            name = ''
            for k, i in dic1[ni].items():
                if not print_name_var:
                    k = ''
                name += k +str(i)[0:num2keep]
                name+= space
            if space:
                label.append(name[:-1])
            else:
                label.append(name)
        return label

    def label_number(self, num2keep=3, space='_', print_name_var=1):

        dic1 = self.get_combinations_dict()
        #import pdb;pdb.set_trace()
        label = []
        for ni in range(self.num_combinations):
            name = ''
            j=0
            for k,i in dic1[ni].items():
                if not print_name_var:
                    k = ''
                #for j in 
                name += k +str(self.varl_index[ni][j])
                name+= space
                j+=1
            if space:
                label.append(name[:-1])
            else:
                label.append(name)
        return label

    def labels(self,label_type='name', **kwargs):

        if label_type == 'name':
            return self.label_name(**kwargs)
        elif label_type == 'number':
            return self.label_number(**kwargs)

    def write_vars2text(self, to_file,
                        vars_dict=None,
                        labels_list=None, **kwargs):

        import pandas as pd

        if vars_dict == None:
            vars_dict = self.get_combinations_dict()
        if labels_list == None:
            labels_list = self.labels()
            
        df = pd.DataFrame(vars_dict)
        df.insert(0, "Labels", labels_list, True)
        with open(to_file, 'w') as f:
            df_str = df.to_string(**kwargs)
            f.write(df_str)
        
class LoadPaths(object):
    """
    Define the load-paths in an aircraft geometry

    """
    def __init__(self,
                 component_nodes,
                 father_components=None,
                 monitoring_stations=None):
        """
        """
        self.component_nodes = component_nodes
        self.component_names = list(component_nodes.keys())
        self.components = self.get_components(**component_nodes)
        if father_components == None: 
            self.father_components = [self.component_names[0]]
        else:
            name_mismatch = [name for name in father_components
                             if name not in self.component_names]
            assert len(name_mismatch) == 0, "Names %s from father_components not \
            defined in component_nodes" % name_mismatch
            self.father_components = father_components
        self.num_father_components = len(self.father_components)
        self.monitoring_stations = monitoring_stations
        
        self.node2component = dict() # dictionary that relates any node in the model to
        # the component it belongs to
        self.connection_nodes = collections.defaultdict(list)
        self.connection_nodes_upstream = collections.defaultdict(list)
        self.monitor = None
        
        self.get_node2component()
        self.get_connection_nodes()
        self.check_connections()
        if monitoring_stations is not None:
            self.get_monitoring_stations()
            
    def get_monitoring_stations(self):

        self.monitor = type('monitoring', (object,), {'nodes': [],
                                                      'direction': [],
                                                      'loadpath': {},
                                                      'node_nexto': {}})
        if self.monitoring_stations == 'all':
            self.monitoring_stations = self.all_nodes
        if type(self.monitoring_stations) == list:
            self.monitor.nodes = self.monitoring_stations
            for ni in self.monitor.nodes:
                direction = 'forward'
                self.monitor.direction.append(direction)
                self.monitor.loadpath[ni], self.monitor.node_nexto[ni] = \
                    self.get_loadpaths(ni, forward=True)
        elif type(self.monitoring_stations) == dict:
            self.monitor.nodes = self.monitoring_stations.keys()
            for ni in self.monitor.nodes:
                direction = self.monitoring_stations[ni]
                self.monitor.direction.append(direction)
                if direction == 'forward':
                    self.monitor.loadpath[ni], self.monitor.node_nexto[ni] = \
                        self.get_loadpaths(ni, forward=True)
                else:
                    self.monitor.loadpath[ni], self.monitor.node_nexto[ni] = \
                        self.get_loadpaths(ni, forward=False)
            
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
        self.all_nodes = self.node2component.keys()

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
        if index != 0:
            node_nexto = self.components[component0][index-1]
        elif index == 0 and component0 in self.father_components:
            node_nexto = 0
        else:
            raise ValueError("Definition problem")
        self.load_components(loadpath, component0, nodes)
        self.loadpath = loadpath
        path_nodes = []
        for ci, vi in loadpath.items():
            path_nodes += vi
        assert len(set(path_nodes)) == len(path_nodes), "Incorrect definition \
        of load path: repeated nodes in different component"
        path_nodes.sort()
        return path_nodes, node_nexto

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


def system_covariance(A, B, C, Sigma_n=[]):

    if len(Sigma_n) > 0:
        Sigma_n = np.diag(Sigma_n)
    else:
        Sigma_n = np.eye(np.shape(B)[1])
    import pdb; pdb.set_trace()
    Sigma_x = linalg.solve_discrete_lyapunov(A, B.dot(Sigma_n.dot(B.T)))
    #Sigma_x = linalg.solve_continuous_lyapunov(A,-B.dot(Sigma_n.dot(B.T)))
    Sigma_y = C.dot(Sigma_x.dot(C.T))
    return Sigma_x, Sigma_y

def system_covariance2(A, B, C, Sigma_n=[]):

    if not Sigma_n:
        Sigma_n = np.eye(np.shape(B)[1])
    else:
        Sigma_n = np.diag(Sigma_n)
    Sigma_x = linalg.solve_continuous_lyapunov(A,-B.dot(Sigma_n.dot(B.T)))
    Sigma_y = C.dot(Sigma_x.dot(C.T))
    return Sigma_x, Sigma_y


def correlation_coeff(covariance):

    dim = len(covariance)
    rho = np.zeros((dim, dim))
    sigma = np.zeros(dim)
    st1 = set(range(dim))
    for i in st1:
        sigma[i] = np.sqrt(covariance[i,i])
    for i in st1:
        for j in (st1-set([i])):
            rho[i, j] = covariance[i, j]/(sigma[i]*sigma[j])
    return sigma, rho


def multivariate_normal_pdf(X, mu, cov):
    rv = stats.multivariate_normal(mean=mu, cov=cov)
    return rv.pdf(X)


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
            path_nodes, node_nexto = self.path1.get_loadpaths(1,forward=False)
            assert self.path1.loadpath['c1'] == [], 'Error in test_loadpath0 c1'
            
        def test_loadpath0f(self):
            """
            to do: add check on moments gain
            """
            path_nodes, node_nexto = self.path1.get_loadpaths(1,forward=True)
            assert self.path1.loadpath['c1'] == [1, 2, 3, 4], 'Error in test_loadpath0f c1'
            assert self.path1.loadpath['c2'] == [5], 'Error in test_loadpath0f c2'
            assert self.path1.loadpath['c3'] == [6, 7, 8], 'Error in test_loadpath0f c3'
            assert self.path1.loadpath['c4'] == [9, 10, 11], 'Error in test_loadpath0f c4'
            
        def test_loadpath1(self):
            """
            to do: add check on moments gain
            """
            path_nodes, node_nexto = self.path1.get_loadpaths(3,forward=False)
            assert self.path1.loadpath['c1'] == [1, 2], 'Error in test_loadpath1 c1'
        def test_loadpath1f(self):
            """
            to do: add check on moments gain
            """
            path_nodes, node_nexto = self.path1.get_loadpaths(3,forward=True)
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
            path_nodes, node_nexto = self.path1.get_loadpaths(8,forward=True)
            assert self.path1.loadpath['c3'] == [8], 'Error in test_loadpath2f c3'
            assert self.path1.loadpath['c4'] == [9, 10, 11], 'Error in test_loadpath2f c4'

    #unittest.main()

