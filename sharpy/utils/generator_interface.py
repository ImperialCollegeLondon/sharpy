from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os

dict_of_generators = {}
generators = {}  # for internal working


# decorator
def generator(arg):
    # global available_solvers
    global dict_of_generators
    try:
        arg.generator_id
    except AttributeError:
        raise AttributeError('Class defined as generator has no generator_id attribute')
    dict_of_generators[arg.generator_id] = arg
    return arg


def print_available_generators():
    cout.cout_wrap('The available generators on this session are:', 2)
    for name, i_generator in dict_of_generators.items():
        cout.cout_wrap('%s ' % i_generator.generator_id, 2)


class BaseGenerator(metaclass=ABCMeta):
    pass

def generator_from_string(string):
    return dict_of_generators[string]


def generator_list_from_path(cwd):
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


def initialise_generator(generator_name):
    cout.cout_wrap('Generating an instance of %s' % generator_name, 2)
    cls_type = generator_from_string(generator_name)
    gen = cls_type()
    return gen

def dictionary_of_generators():

    import sharpy.generators
    dictionary = dict()
    for gen in dict_of_generators:
        init_gen = initialise_generator(gen)
        dictionary[gen] = init_gen.settings_default

    return dictionary