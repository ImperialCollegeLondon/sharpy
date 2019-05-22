import os
import sharpy.utils.cout_utils as cout
from abc import ABCMeta, abstractmethod

dict_of_systems = dict()
systems_dict_import = dict()

# Define the system decorator
def linear_system(arg):
    global dict_of_systems

    try:
        arg.sys_id
    except AttributeError:
        raise AttributeError('Class defined as linear_system as no sys_id')

    dict_of_systems[arg.sys_id] = arg
    return arg


class BaseElement(metaclass=ABCMeta):

    @property
    def sys_id(self):
        raise NotImplementedError

    @abstractmethod
    def initialise(self, data):
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
