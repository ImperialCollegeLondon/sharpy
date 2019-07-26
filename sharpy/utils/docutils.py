import sharpy.utils.sharpydir as sharpydir
# import sharpy.utils.algebra as algebra
import os
import shutil
import sharpy.utils.cout_utils as cout
import inspect
import importlib.util
import sharpy.utils.solver_interface as solver_interface

def output_documentation_module_page(path_to_module, docs_folder_name):
    """
    Generates the algebra package documentation files
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

# def output_documentation(package):
#     files = solver_interface.solver_list_from_path(package)
#     for file in files:
#         pass
#         output_documentation_algebra(package + '/' + file, package)




if __name__ == '__main__':

    # mod1 = sharpydir.SharpyDir + '/utils/algebra'
    # output_documentation_algebra(mod1, 'algebra')

    mod1 = sharpydir.SharpyDir + 'utils/algebra'
    # solver_interface.solver_list_from_path(mod1)
    # output_documentation(mod1)
    output_documentation_module_page(mod1, 'algebra')
