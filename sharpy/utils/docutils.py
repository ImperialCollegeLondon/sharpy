"""Documentation Generator

Functions to automatically document the code.

Comments and complaints: N. Goizueta
"""
import os
import shutil
import inspect
import glob
import warnings
import importlib.util
import yaml
import sharpy.utils.sharpydir as sharpydir
import sharpy.utils.exceptions as exceptions
import sharpy.utils.solver_interface as solver_interface


def generate_documentation():
    """
    Main routine that generates the documentation in ``./docs/source/includes``

    """
    print('Cleaning docs/source/includes')
    shutil.rmtree(sharpydir.SharpyDir + '/docs/source/includes/')
    solver_interface.output_documentation()  # Solvers and generators have a slightly different generation method
    # generator_interface.output_documentation()

    # Main sharpy source code
    sharpy_folders = get_sharpy_folders()
    ignore_modules = yaml.load(open(sharpydir.SharpyDir + '/docs/docignore.yml', 'r'), Loader=yaml.Loader)

    for folder in sharpy_folders:
        folder_name = folder.replace(sharpydir.SharpyDir, '')
        folder_name = folder_name[1:]
        if check_folder_in_ignore(folder, ignore_modules['modules']):
            continue
        mtitle, mbody = write_folder(folder, ignore_modules['modules'])
        create_index_files(folder, mtitle, mbody)

    main_msg = 'The core SHARPy documentation is found herein.\n\n' \
               '.. note::\n\n' \
               '\tThe docs are still a work in progress and therefore, ' \
               'most functions/classes with which there is not much user interaction are not fully documented. ' \
               'We would appreciate any help by means of you contributing to our growing documentation!\n\n\n' \
               'If you feel that a function/class is not well documented and, hence, you cannot use it, feel free ' \
               'to raise an issue so that we can improve it.\n\n'

    create_index_files('./', 'SHARPy Source Code', main_msg)


def write_folder(folder, ignore_list):
    """
    Creates the documentation for the contents in a folder.

    It checks that the file folder is not in the ``ignore_list``. If there is a subfolder in the folder, this gets
    opened, written and an index file is created.

    Args:
        folder (str): Absolute path to folder
        ignore_list (list): List with filenames and folders to ignore and skip

    Returns:
        tuple: Tuple containing the title and body of the docstring found for it to be added to the index of the
          current folder.
    """
    files, mtitle, mbody = open_folder(folder)
    for file in files:
        if os.path.isfile(file) and not check_folder_in_ignore(file, ignore_list):
            if file[-3:] != '.py':
                continue
            write_file(file)
        elif os.path.isdir(file) and not check_folder_in_ignore(file, ignore_list):
            mtitlesub, mbodysub = write_folder(file, ignore_list)
            create_index_files(file, mtitlesub, mbodysub)
    return mtitle, mbody


def write_file(file):
    """
    Writes the contents of a python file with one module per page.

    Warnings:
        If the function to be written does not have a docstring no output will be produced and a warning will be given.

    Args:
        file (str): Absolute path to file

    """
    file_name = file.replace(sharpydir.SharpyDir, '')
    source = file_name.replace('.py', '')
    outfile = source.replace('sharpy/', '')
    try:
        output_documentation_module_page(source,
                                         outfile)
    except exceptions.DocumentationError:
        # Future feature- remove try except so it raises the error that no title has been given
        warnings.warn('Module %s not written - no title given' %source)


def check_folder_in_ignore(folder, ignore_list):
    """
    Checks whether a folder is in the ``ignore_list``.

    Args:
        folder (str): Absolute path to folder
        ignore_list (list): Ignore list

    Returns:
        bool: Bool whether file/folder is in ignore list.
    """
    file_name_check = folder.replace(sharpydir.SharpyDir, '')
    file_name_check = file_name_check[1:]
    if file_name_check in ignore_list:
        return True
    else:
        return False


def output_documentation_module_page(path_to_module, docs_folder_name):
    """
    Generates the documentation for a package with a single page per module in the desired folder
    Returns:

    """

    import_path = path_to_module
    import_path = import_path.replace(sharpydir.SharpyDir, "")
    if import_path[0] == "/": import_path = import_path[1:]
    import_path = import_path.replace("/", ".")
    module = importlib.import_module(import_path)

    print('Generating documentation files')
    path_to_folder = sharpydir.SharpyDir + '/docs/source/includes/' + docs_folder_name

    if os.path.exists(path_to_folder):
        print('Cleaning directory %s' %path_to_folder)
        shutil.rmtree(path_to_folder)

    module_content = inspect.getmembers(module)

    index_file_content = []

    for item in module_content:

        if not inspect.isfunction(item[1]) and not inspect.isclass(item[1]):
            continue

        md_path = item[1].__module__.split('.')
        if md_path[0] != 'sharpy':
            continue

        isclass = False
        if inspect.isclass(item[1]):
            isclass = True

        if not item[1].__doc__:
            continue
        if not os.path.exists(path_to_folder):
            print('Creating directory %s' %path_to_folder)
            os.makedirs(path_to_folder, exist_ok=True)

        docs = ''
        filename = item[1].__name__ + '.rst'
        index_file_content.append(item[1].__name__)

        with open(path_to_folder + '/' + filename, 'w') as outfile:
            title = item[1].__name__

            python_method_path = import_path + '.' + title

            docs += title + '\n' + len(title)*'-' + '\n\n'
            if isclass:
                docs += '.. autoclass:: ' + python_method_path
                docs += '\n\t:members:'
            else:
                docs += '.. automodule:: ' + python_method_path

            outfile.write(docs)
            print('\tCreated %s' % path_to_folder + '/' + filename)

    # create index file
    if index_file_content != []:
        with open(path_to_folder + '/index.rst', 'w') as outfile:
            index_title, body = get_module_title_and_body(module)

            if module.__doc__ is not None:
                outfile.write(index_title + '\n' + len(index_title)*'+' + '\n\n')
                outfile.write(body + '\n\n')

            outfile.write('.. toctree::\n\t:glob:\n\n')

            for item in index_file_content:
                outfile.write('\t./' + item + '\n')


def output_documentation(package_path, docs_folder_name):
    docs_folder = sharpydir.SharpyDir + '/docs/source/includes/' + docs_folder_name
    if os.path.exists(docs_folder):
        print('Cleaning directory %s' % docs_folder)
        shutil.rmtree(docs_folder)
    print('Creating directory %s' % docs_folder)
    os.makedirs(docs_folder, exist_ok=True)

    files = solver_interface.solver_list_from_path(package_path)

    index_file_content = []

    for file in files:
        module, module_path = module_from_path(package_path, file)
        content = inspect.getmembers(module)

        for member in content:
            if member[0].lower() == file:
                title = member[0]

                if not member[1].__doc__:
                    continue

                index_file_content.append(title.lower())
                docs = ''
                docs += title + '\n' + len(title)*'-' + '\n\n'

                docs += '.. autoclass:: ' + module_path + '.' + member[1].__name__
                docs += '\n\t:members:'

                with open(docs_folder + '/' + file + '.rst', 'w') as outfile:
                    outfile.write(docs)

                print('\tCreated %s' % docs_folder + '/' + file + '.rst')
            else:
                continue

    # create index file
    with open(docs_folder + '/index.rst', 'w') as outfile:
        index_title = docs_folder_name.capitalize()
        outfile.write(index_title + '\n' + len(index_title)*'+' + '\n\n')

        outfile.write('.. toctree::\n\t:glob:\n\n')

        for item in index_file_content:
            outfile.write('\t./' + item + '\n')


def module_from_path(package_path, filename):
    if filename is None:
        name = inspect.getmodulename(package_path)
    else:
        name = inspect.getmodulename(package_path + '/' + filename + '.py')
    python_path = package_path.replace(sharpydir.SharpyDir, "")
    if python_path[0] == '/':
        python_path = python_path[1:]
    python_path = python_path.replace("/", ".")
    python_path = python_path.replace('.__init__.py', '')

    if name == '__init__':
        module_path = python_path
        # module_path = module_path.replace('..py', '')
    else:
        module_path = python_path + '.' + name
    module = importlib.import_module(module_path)

    return module, module_path


def create_index_files(docs_folder, folder_title=None, folder_body=None):

    file_name = docs_folder.replace(sharpydir.SharpyDir, '')
    source = file_name.replace('.py', '')
    outfile = source.replace('sharpy/', '')

    docs_path = sharpydir.SharpyDir + '/docs/source/includes/' + outfile

    autodocindexfilename = docs_path + '/index.rst'

    rst_files = glob.glob('%s/*/*index.rst' % docs_path)
    # Sort files alphabetically
    rst_files.sort(key=lambda x: x.replace(docs_path, ''))

    if folder_title is None:
        folder_title = docs_path.split('/')[-1].capitalize()

    if rst_files:
        ordered_list = []
        with open(autodocindexfilename, 'w') as outfile:
            outfile.write(folder_title + '\n' + len(folder_title)*'-' + '\n\n')
            if folder_body is not None:
                outfile.write(folder_body + '\n')
            outfile.write('.. toctree::\n\t:maxdepth: 1\n\n')
            for item in rst_files:
                # index_file = item.replace(sharpydir.SharpyDir, '')
                index_file = item.replace(docs_path, '')
                index_file = index_file.replace('.rst', '')
                if index_file[0] != '/':
                    outfile.write('\t./' + index_file + '\n')
                else:
                    outfile.write('\t.' + index_file + '\n')


def get_module_title_and_body(module):
    docstring = module.__doc__
    title = None
    body = None
    if docstring is not None:
        docstring = docstring.split('\n')
        title = docstring[0]
        if title == '':
            try:
                title = docstring[1]
            except IndexError:
                raise exceptions.DocumentationError('Module %s has been given no title in neither the 1st or 2nd '
                                                    'lines of its docstring'
                                                    % module.__name__)

        body = '\n'.join(docstring[1:])
    else:
        # Had some issues with complete folders not being written if no docs present... need to verify
        # raise exceptions.DocumentationError('Module %s has been given no title in neither the 1st or 2nd lines of '
        #                                     'its docstring'
        #                 % module.__name__)
        pass
    return title, body


def get_sharpy_folders():
    sharpy_directory = sharpydir.SharpyDir + '/sharpy/'
    files = glob.glob('%s/*' % sharpy_directory)
    for item in files:
        if not os.path.isdir(item):
            files.remove(item)
        elif item.replace(sharpy_directory, '')[0] == '_':
            files.remove(item)
    return files


def open_folder(folder_path):
    files = glob.glob(folder_path + '/*')
    outfiles = []
    mtitle = None
    mbody = None
    for file in files:
        if file.replace(folder_path, '')[1:] == '__init__.py':
            mtitle, mbody = module_title(file)
        elif file.replace(folder_path, '')[1] != '_':
            outfiles.append(file)
    return outfiles, mtitle, mbody


def module_title(file):
    module, module_path = module_from_path(file, None)

    title, body = get_module_title_and_body(module)

    return title, body


if __name__ == '__main__':
    generate_documentation()
