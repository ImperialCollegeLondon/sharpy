import struct
import yaml
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Variable:
    num_vars = 0

    def __init__(self, name, inout, **kwargs):

        self.name = name  # str: should be the same as in timestep info
        self.xplane_name = kwargs.get('xplane_name', None)  # equivalent xplane name
        self.inout = inout  # str: (in, out, inout)

        self.index = kwargs.get('index', None)  # if variable is a vector
        position = kwargs.get('position', None)  # int for node, (i_surf, m, n, idx) for panel

        self.node = None
        self.panel = None
        self.cs_index = None
        self.var_type = kwargs.get('var_type', None)

        if self.var_type == 'node':
            self.node = position
        elif self.var_type == 'panel':
            self.panel = position
        elif self.var_type == 'control_surface':
            self.cs_index = position
        elif self.name == 'dt' or self.name == 'nt':
            pass
        else:
            raise Exception('Unknown variable type')

        self.dref_name = None
        self.set_dref_name()

        # update variable counter
        self.variable_index = Variable.num_vars
        Variable.num_vars += 1

        self.value = None
        logger.info('Loaded variable {}'.format(self.dref_name))

    def get_variable_value(self, data, timestep_index=-1):
        """
        Get the variables value at the selected timestep (last one by default)

        Args:
            data (sharpy.presharpy.PreSharpy): the standard SHARPy class
            timestep_index (int (optional)): Integer representing the time step value. Defaults to ``-1`` i.e. the
              last one available.

        Returns:
            float: value of the variable
        """
        if self.node is not None:
            # structural variables for now
            try:
                #Look for the variables in time_step_info
                variable = getattr(data.structure.timestep_info[timestep_index], self.name)
            except AttributeError:
                try:
                    #First get the dict postproc_cell and the try to find the variable in it.
                    get_postproc_cell = getattr(data.structure.timestep_info[timestep_index], 'postproc_cell')
                    variable = get_postproc_cell[self.name]
                except (KeyError, AttributeError):
                    msg = ('Node {} is neither in timestep_info nor in postproc_cell.'.format(self.node))
                    logger.error(msg)
                    raise IndexError(msg)

            #Needed for for_pos and for_vel since they are arrays.
            if len(variable.shape) == 1:
                try:
                    value = variable[self.node, self.index]
                except IndexError:
                    msg = 'Node {} and/or Index {} are out of index of variable {}, ' \
                          'which is of size ({})'.format(self.node, self.index, self.dref_name,
                                                         variable.shape)
                    logger.error(msg)
                    raise IndexError(msg)

            elif len(variable.shape) == 2:
                try:
                    value = variable[self.node, self.index]
                except IndexError:
                    msg = 'Node {} and/or Index {} are out of index of variable {}, ' \
                          'which is of size ({})'.format(self.node, self.index, self.dref_name,
                                                         variable.shape)
                    logger.error(msg)
                    raise IndexError(msg)
            elif len(variable.shape) == 3:
                try:
                    ielem, inode_in_elem = data.structure.node_master_elem[self.node]
                    value = variable[ielem, inode_in_elem, self.index]
                except IndexError:
                    msg = 'Node {} and/or Index {} are out of index of variable {}, ' \
                          'which is of size ({})'.format(self.node, self.index, self.dref_name,
                                                         variable.shape)
                    logger.error(msg)
                    raise IndexError(msg)
            else:
                msg = f'Variable {self.name} is neither a node variable nor an element variable. The ' \
                      f'variable {self.name} is stored as a {variable.shape} array.'
                logger.error(msg)
                raise IndexError(msg)

        elif self.name == 'dt':
            value = data.settings['DynamicCoupled']['dt']
        elif self.name == 'nt':
            value = len(data.structure.timestep_info[:timestep_index]) - 1  # (-1) needed since first time step is idx 0
        elif self.panel is not None:
            variable = getattr(data.aero.timestep_info[timestep_index], self.name)[self.panel[0]]  # surface index
            i_m = self.panel[1]
            i_n = self.panel[2]

            try:
                i_idx = self.panel[3]
            except IndexError:
                value = variable[i_m, i_n]
            else:
                value = variable[i_m, i_n, i_idx]
        elif self.cs_index is not None:
            try:
                value = data.aero.timestep_info[timestep_index].control_surface_deflection[self.cs_index]
            except AttributeError:
                logger.error('Model not equipped with dynamic control surfaces')
                raise AttributeError
            except IndexError:
                logger.error('Requested index {} for control surface is out of range (size {})'.format(
                    self.cs_index, len(data.aero.timestep_info[timestep_index].control_surface_deflection)))
        else:
            raise NotImplementedError('Unable to get value for {} variable'.format(self.name))

        self.value = value
        logger.debug('Getting value {} for variable {}'.format(self.value, self.dref_name))
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
            data (sharpy.presharpy.PreSharpy): Simulation data object

        """
        if self.node is not None:
            # Check if the node is an app_forces (f.e. Thrust)
            if self.name == 'app_forces':
                logger.debug('Setting thrust variable')
                variable = data.structure.ini_info.steady_applied_forces
                try:
                    variable[self.node, self.index] = self.value
                except IndexError:
                    logger.warning('Unable to set node {}, index {} of variable {}'.format(
                        self.node, self.index, self.dref_name
                    ))

                data.structure.ini_info.steady_applied_forces = variable
                logger.debug('Updated timestep')

            # else it is a structural variable
            else:
                variable = getattr(data.structure.timestep_info[-1], self.name)
                try:
                    variable[self.node, self.index] = self.value
                except IndexError:
                    logger.warning('Unable to set node {}, index {} of variable {}'.format(
                        self.node, self.index, self.dref_name
                    ))

                setattr(data.structure.timestep_info[-1], self.name, variable)
                logger.debug('Updated timestep')

        if self.cs_index is not None:
            variable = getattr(data.aero.timestep_info[-1], self.name)

            # Creates an array as long as needed. Not required Cs_deflections will be set to zero. If the CS_type in the
            # aero.h5 file is 0 this shouldnt have a influence on them.

            while len(variable) <= self.cs_index:
                # Adds an element in the array for the new control surface.
                variable = np.hstack((variable, np.array(0)))
            try:
                variable[self.cs_index] = self.value
            except IndexError:
                logger.warning('Unable to set control surface deflection {}. Check the order of '
                               'you control surfaces.'.format(self.cs_index))

            setattr(data.aero.timestep_info[-1], self.name, variable)
            logger.debug('Updated control surface deflection')

    def set_dref_name(self):
        divider = '_'
        dref_name = self.name

        if self.node is not None:
            dref_name += divider +'node{}'.format(self.node)
        elif self.panel is not None:
            dref_name += divider + 'paneli{}m{}n{}'.format(*self.panel)
        elif self.cs_index is not None:
            dref_name += divider + 'idx{}'.format(self.cs_index)
        elif self.name == 'dt' or self.name == 'nt':
            pass
        else:
            raise Exception('Unknown variable')

        if self.index is not None:
            dref_name += divider + 'index{}'.format(self.index)

        self.dref_name = dref_name


class SetOfVariables:
    """
    Iterable class containing the input and output variables

    Attributes:
        variables (list(Variable)): List of :class:`Variable`
    """
    def __init__(self):
        self.variables = []  # list of Variables()
        self.out_variables = []  # indices
        self.in_variables = []

        self._byte_ordering = '<'

        self.file_name = None  # for input variables

    def set_byte_ordering(self, value):
        self._byte_ordering = value

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
            logger.debug('Number of tracked variables {}'.format(Variable.num_vars))

    def set_input_file(self, filename):
        self.file_name = filename

        with open(self.file_name, 'w') as f:
            header = ''
            for var_idx in self.in_variables:
                header += '{},\t'.format(self.variables[var_idx].dref_name)
            header += '\n'

            f.write(header)

    @property
    def input_msg_len(self):
        msg_len = 5 + 8 * len(self.in_variables)  # 5 bytes header + 8 for each channel
        return msg_len

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
        msg = struct.pack('{}5s'.format(self._byte_ordering), b'RREF0')
        for var_idx in self.out_variables:
            variable = self.variables[var_idx]
            logger.debug('Encoding variable {}'.format(variable.dref_name))
            msg += struct.pack('{}if'.format(self._byte_ordering), variable.variable_index, variable.value)

        return msg

    def get_value(self, data, timestep_index=-1):
        """
        Gets the value from the data structure for output variables

        Args:
            data (sharpy.presharpy.PreSharpy): the standard SHARPy class
            timestep_index (int (optional)): Integer representing the time step value. Defaults to ``-1`` i.e. the
              last one available.
        """

        for out_idx in self.out_variables:
            self.variables[out_idx].get_variable_value(data, timestep_index=timestep_index)

    def set_value(self, values):
        """
        Sets the values of the input variables.

        Args:
            values (list(tuple)): List of tuples containing the index and value of the respective input variables.
        """

        for idx, value in values:
            self.variables[idx].set_variable_value(value)
            logger.info('Set the input variable {} to {:.4f}'.format(self.variables[idx].dref_name,
                                                                     self.variables[idx].value))
        # save to file:
        self.save_to_file(values)

    def update_timestep(self, data, values):

        logger.debug('Update time step routine')
        self.set_value(values)
        for idx in self.in_variables:
            self.variables[idx].set_in_timestep(data)

    def save_to_file(self, input_variables):
        if self.file_name is not None:
            input_values = [value for idx, value in input_variables]
            with open(self.file_name, 'a') as f:
                out_msg = ''
                for value in input_values:
                    out_msg += '{:10.6f},\t'.format(value)
                out_msg += '\n'
                f.write(out_msg)


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
