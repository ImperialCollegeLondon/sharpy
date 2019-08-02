import sharpy.utils.sharpydir as sharpydir
import sharpy.utils.exceptions as exceptions
# import sharpy.utils.algebra as algebra
import os
import shutil
import sharpy.utils.cout_utils as cout
import inspect
import importlib.util
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.generator_interface as generator_interface
import glob
import yaml
import warnings


def generate_documentation():
    print('cleaning includes')
    shutil.rmtree(sharpydir.SharpyDir + '/docs/source/includes/')
    solver_interface.output_documentation()
    generator_interface.output_documentation()

    # changing below here
    data_file = yaml.load(open(sharpydir.SharpyDir + '/docs/docinclude.yml', 'r'), Loader=yaml.Loader)
    #
    # for item in data_file['packages']:
    #     output_documentation(sharpydir.SharpyDir + '/' + item['folder'], item['docs_folder'])

  # for item in data_file['modules']:
    #     try:
    #         output_documentation_module_page(sharpydir.SharpyDir + item['folder'], item['docs_folder'],
    #                                          item.get('docs_title', None))
    #     except ModuleNotFoundError:
    #         warnings.warn('Unable to load %s to create %s' % (item['folder'], item['docs_folder']))


    sharpy_folders = get_sharpy_folders()
    ignore_modules = yaml.load(open(sharpydir.SharpyDir + '/docs/docignore.yml', 'r'), Loader=yaml.Loader)
    # print(ignore_modules)
    for folder in sharpy_folders:
        folder_name = folder.replace(sharpydir.SharpyDir, '')
        folder_name = folder_name[1:]
        if check_folder_in_ignore(folder, ignore_modules['modules']):
            continue
        mtitle, mbody = write_folder(folder, ignore_modules['modules'])
        # mtitle, mbody = module_title(folder)
        create_index_files(folder, mtitle, mbody)
        # files = open_folder(folder)
        # for file in files:
        #     if os.path.isfile(file) and not check_folder_in_ignore(file, ignore_modules['modules']):
        #         write_file(file)
    create_index_files('./')

def write_folder(folder, ignore_list):
    files, mtitle, mbody = open_folder(folder)
    for file in files:
        if os.path.isfile(file) and not check_folder_in_ignore(file, ignore_list):
            write_file(file)
        elif os.path.isdir(file) and not check_folder_in_ignore(file, ignore_list):
            mtitlesub, mbodysub = write_folder(file, ignore_list)
            create_index_files(file, mtitlesub, mbodysub)
    return mtitle, mbody

def write_file(file):
    file_name = file.replace(sharpydir.SharpyDir, '')
    source = file_name.replace('.py', '')
    outfile = source.replace('sharpy/', '')
    # print(source)
    # print(outfile)
    try:
        output_documentation_module_page(source,
                                         outfile,
                                         None)
    except exceptions.DocumentationError:
        # Future - remove try except so it raises the error that no title has been given
        warnings.warn('Module %s not written - no title given' %source)

def check_folder_in_ignore(folder, ignore_list):
    file_name_check = folder.replace(sharpydir.SharpyDir, '')
    file_name_check = file_name_check[1:]
    if file_name_check in ignore_list:
        return True
    else:
        return False

def output_documentation_module_page(path_to_module, docs_folder_name, docs_title=None):
    """
    Generates the documentation for a package with a single page per module in the desired folder
    Returns:

    """

    # import_path = os.path.dirname(path_to_module)
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
            # if docs_title is None:
            #     index_title = docs_folder_name.capitalize()
            # else:
            #     index_title = docs_title
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

    if folder_title is None:
        folder_title = docs_path.split('/')[-1].capitalize()

    if rst_files:
        with open(autodocindexfilename, 'w') as outfile:
            outfile.write(folder_title + '\n' + len(folder_title)*'-' + '\n\n')
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

    # mod1 = sharpydir.SharpyDir + '/utils/algebra'
    # output_documentation_algebra(mod1, 'algebra')

    # mod1 = sharpydir.SharpyDir + 'utils/algebra'
    # solver_interface.solver_list_from_path(mod1)
    # output_documentation(mod1)
    # output_documentation_module_page(mod1, 'algebra')
    # mod1 = sharpydir.SharpyDir + '/sharpy/aero/models'
    # output_documentation(mod1, 'aero')
    generate_documentation()
    # a = get_sharpy_folders()
    # aa = get_files(a[1])
    # print(aa)
    # create_index_files()