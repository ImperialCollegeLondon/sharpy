from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os

dict_of_roms = {}
roms = {}  # for internal working


# decorator
def rom(arg):
    # global available_solvers
    global dict_of_roms
    try:
        arg.rom_id
    except AttributeError:
        raise AttributeError('Class defined as ROM has no rom_id attribute')
    dict_of_roms[arg.rom_id] = arg
    return arg


def print_available_solvers():
    cout.cout_wrap('The available ROMs on this session are:', 2)
    for name, i_solver in dict_of_roms.items():
        cout.cout_wrap('%s ' % i_solver.solver_id, 2)


class BaseRom(metaclass=ABCMeta):

    # Solver id for populating available_roms[]
    @property
    def rom_id(self):
        raise NotImplementedError

    # The input is a ProblemData class structure
    @abstractmethod
    def initialise(self):
        pass

    # This executes the solver
    @abstractmethod
    def run(self, ss):
        pass


def rom_from_string(string):
    return dict_of_roms[string]


def rom_list_from_path(cwd):
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


def initialise_rom(rom_name):
    cout.cout_wrap('Generating an instance of %s' % rom_name, 2)
    cls_type = rom_from_string(rom_name)
    solver = cls_type()
    return solver

def dictionary_of_solvers():
    # import sharpy.rom
    dictionary = dict()
    for solver in dict_of_roms:
        if not solver.lower() == 'SaveData'.lower():
            # TODO: why it does not work for savedata?
            init_solver = initialise_rom(solver)
            dictionary[solver] = init_solver.settings_default

    return dictionary
