"""State-space modules loading utilities"""
import os
import sharpy.utils.cout_utils as cout
from abc import ABCMeta, abstractmethod
import numpy as np

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
        self.update()

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
                self._remove_variable(remove_variable_index)

    def _remove_variable(self, index):
        self.vector_variables.pop(index)
        self.update()

    def modify(self, variable_name, new_length):
        pass
        # either modify or condense (i..e new length)
        # replace i.e. when projecting

    def add(self, vector_variable):
        pass

    def update(self):
        list_of_indices = [variable.index for variable in self.vector_variables]
        ordered_list = sorted(list_of_indices)

        index_variable_dict = {key_index: self.vector_variables[i] for i, key_index in enumerate(list_of_indices)}
        updated_list = []
        for i_var in range(self.num_variables):

            current_variable_index = ordered_list[i_var]
            current_variable = index_variable_dict[current_variable_index]
            current_variable.index = i_var
            if i_var == 0:
                current_variable.first_position = 0
            elif i_var > 0:
                # since end item is not included in range methods, set the first position to last variable end position
                current_variable.first_position = updated_list[i_var - 1].end_position
            updated_list.append(current_variable)

        self.vector_variables = updated_list

    def update_column_locations(self):
        for i_var in range(self.num_variables):
            if i_var == 0:
                self.vector_variables[i_var].first_position = 0
            elif i_var > 0:
                # since end item is not included in range methods, set the first position to last variable end position
                self.vector_variables[i_var].first_position = self.vector_variables[i_var - 1].end_position

    def merge(self):
        pass

    def __iter__(self):
        return SetIterator(self)

    def __getitem__(self, item):
        return self.vector_variables[item]


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

    @classmethod
    def setUpClass(cls):
        # initialising with index out of order
        Kzeta = 4
        input_variables_list = [VectorVariable('zeta', size=3 * Kzeta, index=0),
                                VectorVariable('zeta_dot', size=3 * Kzeta, index=4), # this should be one
                                VectorVariable('u_gust', size=3 * Kzeta, index=2)]

        cls.input_variables = LinearVector(input_variables_list)

    def test_initialisation(self):

        sorted_indices = [variable.index for variable in self.input_variables]
        assert sorted_indices == [0, 1, 2], 'Error sorting indices in initialisation'

    def test_remove(self):

        self.input_variables.remove('zeta_dot')

        sorted_indices = [variable.index for variable in self.input_variables]
        assert sorted_indices == [0, 1], 'Error removing variable'


if __name__ == '__main__':
    unittest.main()

