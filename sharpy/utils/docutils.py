import sharpy.utils.sharpydir as sharpydir
# import sharpy.utils.algebra as algebra
import os
import shutil
import sharpy.utils.cout_utils as cout
import inspect
import importlib.util
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.generator_interface as generator_interface


def generate_documentation():
    solver_interface.output_documentation()
    generator_interface.output_documentation()

    # WIP: will eventually source which docs from file etc
    output_documentation_module_page(sharpydir.SharpyDir + '/utils/algebra', 'algebra')
    output_documentation(sharpydir.SharpyDir + '/sharpy/rom', 'rom')


def output_documentation_module_page(path_to_module, docs_folder_name):
    """
    Generates the documentation for a package with a single page per module in the desired folder
    Returns:

    """

    # import_path = os.path.dirname(path_to_module)
    import_path = path_to_module
    import_path = import_path.replace(sharpydir.SharpyDir, "")
    if import_path[0] == "/": import_path = import_path[1:]
    import_path = import_path.replace("/", ".")
    module = importlib.import_module('sharpy.' + import_path)

    print('Generating documentation files')
    path_to_folder = sharpydir.SharpyDir + '/docs/source/includes/' + docs_folder_name

    if os.path.exists(path_to_folder):
        print('Cleaning directory %s' %path_to_folder)
        shutil.rmtree(path_to_folder)
    print('Creating directory %s' %path_to_folder)
    os.makedirs(path_to_folder, exist_ok=True)

    module_content = inspect.getmembers(module)

    for item in module_content:
        if not inspect.isfunction(item[1]):
            continue

        if not item[1].__doc__:
            continue

        docs = ''
        filename = item[1].__name__ + '.rst'
        with open(path_to_folder + '/' + filename, 'w') as outfile:
            title = item[1].__name__

            python_method_path = 'sharpy.' + import_path + '.' + title

            docs += title + '\n' + len(title)*'-' + '\n\n'
            docs += '.. automodule:: ' + python_method_path

            outfile.write(docs)
            print('\tCreated %s' % path_to_folder + '/' + filename)


def output_documentation(package_path, docs_folder_name):
    print('Generating documentation files')

    docs_folder = sharpydir.SharpyDir + '/docs/source/includes/' + docs_folder_name
    if os.path.exists(docs_folder):
        print('Cleaning directory %s' % docs_folder)
        shutil.rmtree(docs_folder)
    print('Creating directory %s' % docs_folder)
    os.makedirs(docs_folder, exist_ok=True)

    files = solver_interface.solver_list_from_path(package_path)
    for file in files:
        module, module_path = module_from_path(package_path, file)
        content = inspect.getmembers(module)

        for member in content:
            if member[0].lower() == file:
                title = member[0]

                if not member[1].__doc__:
                    continue

                docs = ''
                docs += title + '\n' + len(title)*'-' + '\n\n'

                docs += '.. autoclass:: ' + module_path + '.' + member[1].__name__
                docs += '\n\t:members:'

                with open(docs_folder + '/' + file + '.rst', 'w') as outfile:
                    outfile.write(docs)

                print('\tCreated %s' % docs_folder + '/' + file + '.rst')
            else:
                continue


def module_from_path(package_path, filename):
    name = inspect.getmodulename(package_path + '/' + filename + '.py')
    python_path = package_path.replace(sharpydir.SharpyDir, "")
    if python_path[0] == '/':
        python_path = python_path[1:]
    python_path = python_path.replace("/", ".")

    module_path = python_path + '.' + name
    module = importlib.import_module(module_path)

    return module, module_path


if __name__ == '__main__':

    # mod1 = sharpydir.SharpyDir + '/utils/algebra'
    # output_documentation_algebra(mod1, 'algebra')

    # mod1 = sharpydir.SharpyDir + 'utils/algebra'
    # solver_interface.solver_list_from_path(mod1)
    # output_documentation(mod1)
    # output_documentation_module_page(mod1, 'algebra')
    # mod1 = sharpydir.SharpyDir + '/sharpy/aero/models'
    # output_documentation(mod1, 'aero')
    generate_documentation()