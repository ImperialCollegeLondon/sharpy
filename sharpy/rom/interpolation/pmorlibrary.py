"""
ROM Interpolation library interface
"""
import glob
import pickle
import configobj
import sharpy.utils.cout_utils as coututils

coututils.start_writer()
coututils.cout_wrap.cout_talk()


class ROMLibrary:
    """
    Create, load, save, append ROMs to a library

    """

    def __init__(self):
        self.library_name = None
        self.folder = None
        self.library = None

    def interface(self):

        while True:
            user_input = self.main_menu()

            if user_input == 0:
                break

            elif user_input == 1:
                self.create()

            elif user_input == 2:
                self.load_case(input('Path to SHARPy output folder: '))

            elif user_input == 3:
                self.load_library()

            elif user_input == 4:
                self.save_library()

            elif user_input == 5:
                self.display_library()

            else:
                coututils.cout_wrap('Input not recognised', 3)

    def create(self, settings=None):

        if settings is None:
            self.get_library_name()
            # look for pickles and configs
            pickle_source_path = input('Enter path to folder containing ROMs (in .pkl):')
        else:
            self.library_name = settings.get('library_name', None)
            self.folder = settings.get('folder', None)
            pickle_source_path = settings['pickle_source_path']

        self.library = []

        self.load_files(pickle_source_path)

    def get_library_name(self):
        self.library_name = input('Enter library name: ')
        self.folder = input('Enter path to folder: ')

    def load_files(self, path):

        sharpy_cases = glob.glob(path + '/*')

        for case in sharpy_cases:
            self.load_case(case)

    def load_case(self, case_path):

        coututils.cout_wrap('Loading %s' % case_path)
        pickle_path = glob.glob(case_path + '/*.pkl')
        if len(pickle_path) != 0:
            data = pickle.load(open(pickle_path[0], "rb"))

            # load pmor
            pmor_path = glob.glob(case_path + '/*.pmor')
            if len(pmor_path) != 0:
                params = configobj.ConfigObj(pmor_path[0])

                dict_params = dict()
                for item in params['parameters']:
                    dict_params[item] = float(params['parameters'][item])

                self.library.append((data, dict_params))
            else:
                coututils.cout_wrap('Unable to locate .pmor config file with parameter information', 4)
        else:
            coututils.cout_wrap('Unable to locate pickle file', 4)

    def save_library(self):
        path = self.folder + '/' + self.library_name + '.pkl'
        pickle.dump(self.library, open(path, 'wb'))
        coututils.cout_wrap('Saved library to %s' % self.folder, 2)

    def load_library(self, path=None):
        if path is None:
            self.get_library_name()
            path = self.folder + '/' + self.library_name + '.pkl'
        try:
            self.library = pickle.load(open(path, 'rb'))
            coututils.cout_wrap('Successfully loaded library from %s' % path, 2)
        except FileNotFoundError:
            coututils.cout_wrap('Unable to find library at %s' % path, 3)
            self.interface()

    def display_library(self):

        params = self.library[0][1].keys()

        library_table = coututils.TablePrinter(n_fields=len(params) + 1,
                                               field_types=['g'] * len(params) + ['s'],
                                               field_length=len(params) * [12] + [90])
        library_table.print_header(field_names=list(params) + ['Case Name'])
        [library_table.print_line(list(entry[1].values()) + [entry[0].settings['SHARPy']['case']])
         for entry in self.library]
        library_table.print_divider_line()

    @staticmethod
    def main_menu():
        coututils.cout_wrap("\nPMOR Library Interface Menu\n\n"
                            "[1] - Create library\n"
                            "[2] - Add case to library\n"
                            "[3] - Load library\n"
                            "[4] - Save library\n"
                            "[5] - Display library\n"
                            "\n\n[0] - Quit")
        try:
            user_input = int(input('\nSelect option: '))
        except ValueError:
            coututils.cout_wrap('Unrecognised input choice', 3)
            user_input = -1
        return user_input
