import struct
import yaml
import logging
import numpy as np

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
        # if type(position) is int:
        #     self.node = position
        #     self.panel = None
        # elif len(position) == 3:
        #     self.panel = position
        #     self.node = None
        # else:
        #     raise TypeError('Position should be either an integer for nodes or a tuple for panels')

        self.node = None
        self.panel = None
        self.cs_index = None

        var_type = kwargs.get('var_type', None)
        if  var_type == 'node':
            self.node = position
        elif var_type == 'panel':
            self.panel = position
        elif var_type == 'control_surface':
            self.cs_index = position
        else:
            raise Exception('Unknown variable type')

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
        logger.info('Getting value {} for variable {}'.format(self.value, self.dref_name))
        return value

    def set_variable_value(self, value):
        """
        Set the value of input variables

        Args:
            value: value of variable
        """
        if self.inout == 'in' or self.inout == 'inout':
            self.value = value
        else:
            logger.warning('Trying to set the value of {} which is only an output variable'.format(self.dref_name))

    def set_in_timestep(self, data):
        """
        Set the variable value in the time step

        Args:
            data:

        """
        if self.node is not None: # structural variable then
            variable = getattr(data.structure.timestep_info[-1], self.name)
            try:
                variable[self.node, self.index] = self.value
            except IndexError:
                logger.warning('Unable to set node {}, index {} of variable {}'.format(
                    self.node, self.index, self.dref_name
                ))
            else:
                setattr(data.structure.timestep_info[-1], self.name, variable)
                logger.info('Updated timestep')

        if self.cs_index is not None:
            variable = getattr(data.aero.timestep_info[-1], self.name)
            if len(variable) == 0:
                variable = np.hstack((variable, np.array([self.value])))
            else:
                variable[self.cs_index] = self.value

            setattr(data.aero.timestep_info[-1], self.name, variable)
            logger.info('Updated control surface deflection')

    def set_dref_name(self):
        divider = '_'
        dref_name = self.name + divider

        if self.node is not None:
            dref_name += 'node{}'.format(self.node)
        elif self.panel is not None:
            dref_name += 'paneli{}m{}n{}'.format(*self.panel)
        elif self.cs_index is not None:
            dref_name += 'idx{}'.format(self.cs_index)
        else:
            raise Exception('Unknown variable')

        if self.index is not None:
            dref_name += divider + 'index{}'.format(self.index)

        self.dref_name = dref_name


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
            if new_var.inout == 'in' or new_var.inout == 'inout':
                self.in_variables.append(new_var.variable_index)
            logger.info('Number of tracked variables {}'.format(Variable.num_vars))

    def __iter__(self):
        return VariableIterator(self)

    def __getitem__(self, item):
        return self.variables[item]

    def __len__(self):
        return len(self.variables)

    def encode(self):
        """
        Encode output variables in binary format with little-endian byte ordering.

        The signal consists of a 5-byte header ``RREF0`` followed by 8 bytes per variable.
        Of those 8 bytes allocated to each variable, the first 4 are the integer value of the variable index
        and the last 4 are the single precision float value.

        Returns:
            bytes: Encoded message of length ``5 + num_var * 8``.
        """
        msg = struct.pack('<5s', b'RREF0')
        for var_idx in self.out_variables:
            variable = self.variables[var_idx]
            logger.info('Encoding variable {}'.format(variable.dref_name))
            msg += struct.pack('<if', variable.variable_index, variable.value)

        return msg

    def get_value(self, data):
        """
        Sets the value from the data structure for output variables
        """

        for out_idx in self.out_variables:
            self.variables[out_idx].get_variable_value(data)

    def set_value(self, values):
        """
        Sets the values of the input variables.

        Args:
            values (list(tuple)): List of tuples containing the index and value of the respective input variables.
        """

        for idx, value in values:
            self.variables[idx].set_variable_value(value)
            logger.info('Set the input variable {} to {}'.format(self.variables[idx].dref_name,
                                                                 self.variables[idx].value))

    def update_timestep(self, data, values):

        logger.info('Update time step routine')
        self.set_value(values)
        for idx in self.in_variables:
            self.variables[idx].set_in_timestep(data)


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
