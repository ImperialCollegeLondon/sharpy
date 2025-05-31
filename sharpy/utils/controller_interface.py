from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os

dict_of_controllers = {}
controllers = {}  # for internal working


# decorator
def controller(arg):
    # global available_solvers
    global dict_of_controllers
    try:
        arg.controller_id
    except AttributeError:
        raise AttributeError('Class defined as controller has no controller_id attribute')
    dict_of_controllers[arg.controller_id] = arg
    return arg


def print_available_controllers():
    cout.cout_wrap('The available controllers in this session are:', 2)
    for name, i_controller in dict_of_controllers.items():
        cout.cout_wrap('%s ' % i_controller.controller_id, 2)


class BaseController(metaclass=ABCMeta):
    
    def teardown(self):
        pass


def controller_from_string(string):
    return dict_of_controllers[string]


def controller_list_from_path(cwd):
    onlyfiles = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

    for i_file in range(len(onlyfiles)):
        if onlyfiles[i_file].split('.')[-1] == 'py': # support autosaved files in the folder
            if onlyfiles[i_file] == "__init__.py":
                onlyfiles[i_file] = ""
                continue
            onlyfiles[i_file] = onlyfiles[i_file].replace('.py', '')
        else:
            onlyfiles[i_file] = ""

    files = [file for file in onlyfiles if not file == ""]
    return files


def initialise_controller(controller_name, print_info=True):
    if print_info:
        cout.cout_wrap('Generating an instance of %s' % controller_name, 2)
    cls_type = controller_from_string(controller_name)
    controller = cls_type()
    return controller

def dictionary_of_controllers(print_info=True):
    import sharpy.controllers
    dictionary = dict()
    for controller in dict_of_controllers:
        init_controller = initialise_controller(controller, print_info=print_info)
        dictionary[controller] = init_controller.settings_default

    return dictionary
