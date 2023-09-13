"""Generator Interface
"""
from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os
import shutil

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
        if onlyfiles[i_file].split('.')[-1] == 'py': # support autosaved files in the folder
            if onlyfiles[i_file] == "__init__.py":
                onlyfiles[i_file] = ""
                continue
            onlyfiles[i_file] = onlyfiles[i_file].replace('.py', '')
        else:
            onlyfiles[i_file] = ""

    files = [file for file in onlyfiles if not file == ""]
    return files


def initialise_generator(generator_name, print_info=True):
    if print_info:
        cout.cout_wrap('Generating an instance of %s' % generator_name, 2)
    cls_type = generator_from_string(generator_name)
    gen = cls_type()
    return gen

def dictionary_of_generators(print_info=True):

    import sharpy.generators
    dictionary = dict()
    for gen in dict_of_generators:
        init_gen = initialise_generator(gen, print_info)
        dictionary[gen] = init_gen.settings_default

    return dictionary

def output_documentation(route=None):
    """
    Creates the ``.rst`` files for the generators that have a docstring such that they can be parsed to Sphinx

    Args:
        route (str): Path to folder where generator files are to be created.

    """
    import sharpy.utils.sharpydir as sharpydir
    if route is None:
        route = sharpydir.SharpyDir + '/docs/source/includes/generators/'
        if os.path.exists(route):
            print('Cleaning %s' %route)
            shutil.rmtree(route)
    print('Creating documentation files for generators in %s' % route)

    created_generators = dict()

    for k, v in dict_of_generators.items():
        if k[0] == '_':
            continue
        generator_class = v()
        created_generators[k] = generator_class

        filename = k + '.rst'
        # try:
        #     solver_folder = solver.solver_classification.lower()
        # except AttributeError:
        #     solver_folder = 'other'
        #
        # if solver_folder not in solver_types:
        #     solver_types.append(solver_folder)

        os.makedirs(route + '/', exist_ok=True)
        title = k + '\n'
        title += len(k)*'-' + 2*'\n'
        if generator_class.__doc__ is not None:

            print('\tCreating %s' % (route + '/' + filename))
            autodoc_string = '\n\n.. autoclass:: sharpy.generators.' + k.lower() + '.'+ k + '\n\t:members:'
            with open(route + '/' + filename, "w") as out_file:
                out_file.write(title + autodoc_string)

    # # Creates index files depending on the type of solver
    # filename = 'generators.rst'
    # title = 'Velocity Field Generators'
    # title += '\n' + len(title)*'+' + 2*'\n'
    # with open(route + '/' + filename, "w") as out_file:
    #     out_file.write(title)
    #     out_file.write('.. toctree::' + '\n')
    #     for k in dict_of_generators.keys():
    #         if k[0] == '_':
    #             continue
    #         out_file.write('    ./' + k + '\n')
