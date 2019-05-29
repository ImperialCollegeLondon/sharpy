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


class VectorVariable(object):

    def __init__(self, name, pos_list, var_system):

        self.name = name
        self.var_system = var_system

        self.first_pos = pos_list[0]
        self.end_pos = pos_list[1]
        self.rows_loc = np.arange(self.first_pos, self.end_pos, dtype=int) # Original location, should not update


    # add methods to reorganise into SHARPy method?

    @property
    def cols_loc(self):
        return np.arange(self.first_pos, self.end_pos, dtype=int)

    @property
    def size(self):
        return self.end_pos - self.first_pos


class LinearVector():

    def __init__(self, dof_db, sys_id):
        self.vector_vars = dict()

        vec_db = dict()
        for item in dof_db:
            vector_var = VectorVariable(item, dof_db[item], sys_id)
            vec_db[item] = vector_var

        self.vector_vars = vec_db

    def remove(self, trim_list):
        vec_db = self.vector_vars
        used_vars_db = self.vector_vars.copy()

        # Variables to remove
        removed_dofs = 0
        removed_db = dict()
        for item in trim_list:
            removed_db[item] = vec_db[item]
            removed_dofs += vec_db[item].size
            del used_vars_db[item]

        # Update variables position
        for rem_item in removed_db:
            for item in used_vars_db:
                if used_vars_db[item].first_pos < removed_db[rem_item].first_pos:
                    pass
                else:
                    # Update position
                    used_vars_db[item].first_pos -= removed_db[rem_item].size
                    used_vars_db[item].end_pos -= removed_db[rem_item].size

        self.vector_vars = used_vars_db

        return removed_dofs

def remove_variables(trim_list, dof_db, sys_id):
    # Remove should be a method of class
    # Create class of variables
    # All variables
    vec_db = dict()
    for item in dof_db:
        vector_var = VectorVariable(item, dof_db[item], sys_id)
        vec_db[item] = vector_var

    used_vars_db = vec_db.copy()

    # Variables to remove
    removed_dofs = 0
    removed_db = dict()
    for item in trim_list:
        removed_db[item] = vec_db[item]
        removed_dofs += vec_db[item].size
        del used_vars_db[item]

    # Update variables position
    for rem_item in removed_db:
        for item in used_vars_db:
            if used_vars_db[item].first_pos < removed_db[rem_item].first_pos:
                pass
            else:
                # Update order and position
                used_vars_db[item].first_pos -= removed_db[rem_item].size
                used_vars_db[item].end_pos -= removed_db[rem_item].size
