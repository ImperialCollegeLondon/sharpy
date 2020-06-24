import struct
import yaml
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=20)
logger = logging.getLogger(__name__)

class Variable:
    num_vars = 0

    def __init__(self, name, inout, **kwargs):

        self.name = name  # str: should be the same as in timestep info
        self.xplane_name = kwargs.get('xplane_name', None)  # equivalent xplane name
        self.inout = inout  # str: (in, out, inout)

        self.index = kwargs.get('index', None)  # if variable is a vector
        position = kwargs.get('position', None)  # int for node, (i_surf, m, n) for panel
        if type(position) is int:
            self.node = position
            self.panel = None
        elif len(position) == 3:
            self.panel = position
            self.node = None
        else:
            raise TypeError('Position should be either an integer for nodes or a tuple for panels')

        self.dref_name = None
        self.set_dref_name()
        print(self.dref_name)

        # update variable counter
        self.variable_index = Variable.num_vars
        Variable.num_vars += 1

        self.value = None
        logger.info('Loaded variable {}'.format(self.dref_name))


    def encode(self, data):
        value = self.get_variable_value(data)
        dref_name = self.dref_name

        return value
        # TODO: add variable index to keep track of how many have been created and sent that as well

    def get_variable_value(self, data):
        if self.node is not None:
            # structural variables for now
            variable = getattr(data.structure.timestep_info[-1], self.name)
            try:
                value = variable[self.node, self.index]
            except IndexError:
                logger.error('Node {} and/or Index {} are out of index of variable {}, '
                             'which is of size ({})'.format(self.node, self.index, self.dref_name,
                                                            variable.shape))
                raise IndexError
        else:  # aero variables
            raise NotImplementedError('Aero variables not yet implemented')

        self.value = value
        return value

    def set_dref_name(self):
        divider = '_'
        dref_name = self.name + divider

        if self.node is not None:
            dref_name += 'node{}'.format(self.node)
        else:
            dref_name += 'paneli{}m{}n{}'.format(*self.panel)

        if self.index is not None:
            dref_name += divider + 'index{}'.format(self.index)

        self.dref_name = dref_name

def encode_dref(value, dref_name):
    pass


class SetOfVariables:
    def __init__(self):
        self.variables = []  # list of Variables()
        self.out_variables = []  #indices
        self.in_variables = []

    def load_variables_from_yaml(self, path_to_yaml):
        with open(path_to_yaml, 'r') as yaml_file:
            variables_in_yaml = yaml.load(yaml_file, Loader=yaml.Loader)
        for var in variables_in_yaml:
            new_var = Variable(**var)
            self.variables.append(new_var)
            if new_var.inout == 'out' or new_var.inout == 'inout':
                self.out_variables.append(new_var.variable_index)
            elif new_var.inout == 'in' or new_var.inout == 'inout':
                self.in_variables.append(new_var.variable_index)
            logger.info('Number of tracked variables {}'.format(Variable.num_vars))

    def __iter__(self):
        return VariableIterator(self)

    def __getitem__(self, item):
        return self.variables[item]

    def __len__(self):
        return len(self.variables)


class VariableIterator:

    def __init__(self, set_of_variables):
        self._set_variables = set_of_variables
        self._index = 0

    def __next__(self):
        if self._index < len(self._set_variables):
            res = self._set_variables(self._index)
            self._index += 1
            return res

        raise StopIteration
