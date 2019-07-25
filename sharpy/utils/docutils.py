import sharpy.utils.sharpydir as sharpydir
import sharpy.utils.algebra as algebra
import os
import shutil
import sharpy.utils.cout_utils as cout
import inspect


def output_documentation_algebra():
    """
    Generates the algebra package documentation files
    Returns:

    """

    print('Generating algebra documentation files')
    path_to_folder = sharpydir.SharpyDir + '/docs/source/includes/algebra'

    if os.path.exists(path_to_folder):
        print('Cleaning directory %s' %path_to_folder)
        shutil.rmtree(path_to_folder)
    else:
        print('Creating directory %s' %path_to_folder)
        os.makedirs(path_to_folder, exist_ok=True)

    algebra_content = inspect.getmembers(algebra)

    for item in algebra_content:
        func_name = item[0]
        if not inspect.isfunction(item[1]):
            continue

        if not item[1].__doc__:
            continue

        docs = ''
        filename = item[1].__name__ + '.rst'
        with open(path_to_folder + '/' + filename, 'w') as outfile:
            title = item[1].__name__

            python_method_path = 'sharpy.utils.algebra.' + title

            docs += title + '\n' + len(title)*'-' + '\n\n'
            docs += '.. automodule:: ' + python_method_path

            outfile.write(docs)
            print('\tCreated %s' %path_to_folder + '/' + filename)


