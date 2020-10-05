"""State-space modules loading utilities"""
import os
import sharpy.utils.cout_utils as cout
from abc import ABCMeta, abstractmethod
import numpy as np
import copy

dict_of_systems = dict()
systems_dict_import = dict()

# Define the system decorator
def linear_system(arg):
    global dict_of_systems

    try:
        arg.sys_id
    except AttributeError:
        raise AttributeError('Class defined as linear_system with no sys_id')

    dict_of_systems[arg.sys_id] = arg
    return arg


class BaseElement(metaclass=ABCMeta):

    @property
    def sys_id(self):
        raise NotImplementedError

    @abstractmethod
    def initialise(self, data, custom_settings=None):
        pass

    @abstractmethod
    def assemble(self):
        pass

    # Some method to connect
    # Input (from where) - (i.e. generator, system etc)
    # Output (to display, to system)


def sys_list_from_path(cwd):
    """
    Returns the files containing linear system state space elements

    Args:
        cwd (str): Current working directory

    Returns:

    """
    onlyfiles = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

    for i_file in range(len(onlyfiles)):
        if ".py" in onlyfiles[i_file]:
            if onlyfiles[i_file] == "__init__.py":
                onlyfiles[i_file] = ""
                continue
            onlyfiles[i_file] = onlyfiles[i_file].replace('.py', '')
        else:
            onlyfiles[i_file] = ""

    files = [file for file in onlyfiles if not file == ""]

    return files


def sys_from_string(string):
    return dict_of_systems[string]


def initialise_system(sys_id):
    cout.cout_wrap('Generating an instance of %s' %sys_id, 2)
    cls_type = sys_from_string(sys_id)
    sys = cls_type()
    return sys


def dictionary_of_systems():
    dictionary = dict()
    for linear_system in dict_of_systems:
        init_sys = initialise_system(linear_system)
        dictionary[linear_system] = init_sys.settins_default

    return dictionary


class VectorVariable:

    def __init__(self, name, size, index, var_system=None):

        self.name = name
        self.var_system = var_system
        self.size = size
        self._index = index

        self._first_position = 0

        self._rows_loc = None

    @property
    def cols_loc(self):
        return np.arange(self.first_position, self.end_position, dtype=int)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def first_position(self):
        return self._first_position

    @first_position.setter
    def first_position(self, position):
        self._first_position = position
        self.rows_loc = position

    @property
    def end_position(self):
        return self.first_position + self.size

    @property
    def rows_loc(self):
        return self._rows_loc

    @rows_loc.setter
    def rows_loc(self, value=None):
        if self._rows_loc is None:
            self._rows_loc = np.arange(self.first_position, self.end_position, dtype=int) # Original location, should not update

    def __repr__(self):
        return '(VectorVariable: {:s}, size: {:g}, index: {:g}, starting at: {:g}, finishing: at {:g})'.format(
            self.name,
            self.size,
            self.index,
            self.first_position,
            self.end_position)


class LinearVector:

    def __init__(self, list_of_vector_variables):
        self.vector_variables = list_of_vector_variables

        # initialise position of variables
        self.update_indices()
        self.update_column_locations()

    @property
    def num_variables(self):
        return len(self.vector_variables)

    @property
    def size(self):
        return sum([variable.size for variable in self.vector_variables])

    def remove(self, *variable_name_list):

        for variable_name in variable_name_list:
            list_of_variable_names = [variable.name for variable in self.vector_variables]
            try:
                remove_variable_index = list_of_variable_names.index(variable_name)
            except ValueError:
                cout.cout_wrap('Trying to remove non-existent {:s} variable'.format(variable_name))
            else:
                self.__remove_variable(remove_variable_index)

    def __remove_variable(self, index):
        self.vector_variables.pop(index)
        self.update_indices()

    def modify(self, variable_name, **kwargs):
        """
        Modifies the attributes of the desired variable. The new attributes are passed as **kwargs. If an attribute is
        not recognised a ``NameError`` is returned.

        Note:
            If changing the size of the variable, you should then update the linear vector.

        Args:
            variable_name (str): Name of the variable to modify
            **kwargs: Key-word arguments containing the attributes of the variable as keys and the new values as values.

        """
        variable = self.get_variable_from_name(variable_name)
        for var_attribute, new_var_value in kwargs.items():
            if hasattr(variable, var_attribute):
                variable.__setattr__(var_attribute, new_var_value)
            else:
                raise NameError('Unknown variable attribute {:s}'.format(var_attribute))

    def add(self, vector_variable, **kwargs):

        if type(vector_variable) is VectorVariable:
            self.__add_vector_variable(vector_variable)
        elif type(vector_variable) is str:
            new_variable = VectorVariable(name=vector_variable, **kwargs)
            self.__add_vector_variable(new_variable)
        else:
            raise TypeError('Only can add a variable from either a VectorVariable object or with name and kwargs')

    def append(self, vector_variable, **kwargs):
        self.update_indices()
        appending_index = self.num_variables
        if type(vector_variable) is VectorVariable:
            vector_variable.index = appending_index
        elif type(vector_variable) is str:
            kwargs['index'] = appending_index
        else:
            raise TypeError('Only can add a variable from either a VectorVariable object or with name and kwargs')

        self.add(vector_variable, **kwargs)

    def __add_vector_variable(self, vector_variable):
        list_of_indices = [variable.index for variable in self.vector_variables]

        if vector_variable.index in list_of_indices:
            raise IndexError('New variable index is already in use')
        self.vector_variables.append(vector_variable)
        self.update_indices()

    def update_indices(self):

        list_of_indices = [variable.index for variable in self.vector_variables]
        ordered_list = sorted(list_of_indices)

        index_variable_dict = {key_index: self.vector_variables[i] for i, key_index in enumerate(list_of_indices)}
        updated_list = []
        for i_var in range(self.num_variables):

            current_variable_index = ordered_list[i_var]
            current_variable = index_variable_dict[current_variable_index]
            current_variable.index = i_var

            updated_list.append(current_variable)

        self.vector_variables = updated_list

    def update_column_locations(self):
        for i_var in range(self.num_variables):
            if i_var == 0:
                self.vector_variables[i_var].first_position = 0
            elif i_var > 0:
                # since end item is not included in range methods, set the first position to last variable end position
                self.vector_variables[i_var].first_position = self.vector_variables[i_var - 1].end_position

    def update_row_locations(self):
        pass

    @classmethod
    def merge(cls, vec1, vec2):
        """
        Merges two instances of LinearVectors

        Args:
            vec1 (LinearVector): Vector 1
            vec2 (LinearVector): Vector 2

        Returns:
            LinearVector: Merged vectors 1 and 2
        """
        vec1.update()
        vec2.update()

        for variable in vec2:
            variable.index += vec1.num_variables

        list_of_variables_1 = [variable for variable in vec1]
        list_of_variables_2 = [variable for variable in vec2]

        merged_vector = cls(list_of_variables_1 + list_of_variables_2)
        return merged_vector

    def differentiate(self):
        pass
        # going from second order to first order systems

    def get_variable_from_name(self, name):
        list_of_variable_names = [variable.name for variable in self.vector_variables]
        try:
            variable_index = list_of_variable_names.index(name)
        except ValueError:
            raise ValueError('Variable {:s} is non existent'.format(name))
        else:
            return self.vector_variables[variable_index]

    def update(self):
        self.update_indices()
        self.update_column_locations()
        self.update_row_locations()

    def copy(self):
        return copy.deepcopy(self)

    def __iter__(self):
        return SetIterator(self)

    def __getitem__(self, item):
        return self.vector_variables[item]

    def __repr__(self):
        string_out = ''
        for var in self.vector_variables:
            try:
                out_var = str(repr(var))
            except TypeError:
                print('Error printing var {:s}'.format(var.name))
            else:
                string_out += out_var + '\n'
        return string_out


class SetIterator:

    def __init__(self, linear_vector):
        self._set_cases = linear_vector
        self._index = 0

    def __next__(self):
        if self._index < self._set_cases.num_variables:
            res = self._set_cases.vector_variables[self._index]
            self._index += 1
            return res

        raise StopIteration



import unittest

class TestVariables(unittest.TestCase):

    def setUp(self):
        # initialising with index out of order
        Kzeta = 4
        input_variables_list = [VectorVariable('zeta', size=3 * Kzeta, index=0),
                                VectorVariable('zeta_dot', size=3 * Kzeta, index=1), # this should be one
                                VectorVariable('u_gust', size=3 * Kzeta, index=2)]

        self.input_variables = LinearVector(input_variables_list)

    def test_initialisation(self):
        Kzeta = 4
        input_variables_list = [VectorVariable('zeta', size=3 * Kzeta, index=0),
                                VectorVariable('u_gust', size=3 * Kzeta, index=2),
                                VectorVariable('zeta_dot', size=3 * Kzeta, index=1)]

        input_variables = LinearVector(input_variables_list)

        sorted_indices = [variable.index for variable in input_variables]
        variable_names = [variable.name for variable in input_variables]
        assert sorted_indices == [0, 1, 2], 'Error sorting indices in initialisation'
        assert variable_names == ['zeta', 'zeta_dot', 'u_gust'], 'Error sorting indices in initialisation'

    def test_remove(self):

        self.input_variables.remove('zeta_dot')

        sorted_indices = [variable.index for variable in self.input_variables]
        assert sorted_indices == [0, 1], 'Error removing variable'

    def test_modify(self):

        new_values = {'name': 'new_u_gust',
                      'size': 5,
                      }

        self.input_variables.modify('u_gust', **new_values)

        assert self.input_variables.get_variable_from_name('new_u_gust').size == 5, 'Variable not modified correctly'

    def test_add(self):

        new_variable = VectorVariable('control_surface', size=2, index=3)

        self.input_variables.add(new_variable)
        self.input_variables.update_column_locations()

        # test it's properly included - it will raise an error internally if not
        included_var = self.input_variables.get_variable_from_name('control_surface')

        new_variable_repeated_index = VectorVariable('control_surface_2', size=2, index=3)
        try:
            self.input_variables.add(new_variable_repeated_index)
        except IndexError:
            # correct behaviour
            pass
        else:
            raise IndexError('Variable was added when it should have been rejected because of repeated index')

        new_variable_inbetween = VectorVariable('first_variable', size=2, index=-1)
        self.input_variables.add(new_variable_inbetween)
        self.input_variables.update_column_locations()

        assert self.input_variables[0].name == new_variable_inbetween.name, 'New variable not added at the start of' \
                                                                            ' the list'

        self.input_variables.add('var_from_string', size=3, index=10)
        assert self.input_variables[-1].name == 'var_from_string', 'Variable from string not added correctly'

        self.input_variables.update()

    def test_append(self):
        self.input_variables.append('last_var', size=3)
        self.input_variables.update()

        assert self.input_variables[-1].name == 'last_var', 'Variable not properly appended'

    def test_merge(self):
        second_variables_list = [VectorVariable('eta', size=3, index=0),
                                 VectorVariable('eta_dot', size=3, index=1)]

        second_vector = LinearVector(second_variables_list)

        merged_vector = LinearVector.merge(self.input_variables, second_vector)

        assert merged_vector[-1].name == 'eta_dot', 'Vectors not coupled properly'
        assert merged_vector[2].name == 'u_gust', 'Vectors not coupled properly'

    def test_copy(self):
        vec1 = self.input_variables
        vec2 = vec1.copy()

        assert vec1 is not vec2, 'Object not deep copied'

if __name__ == '__main__':
    unittest.main()

