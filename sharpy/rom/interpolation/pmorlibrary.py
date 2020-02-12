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

        self.reference_case = None

    def interface(self):

        while True:
            user_input = self.main_menu()

            if user_input == 0:
                break

            elif user_input == 1:
                self.create()

            elif user_input == 2:
                self.load_library()

            elif user_input == 3:
                self.save_library()

            elif user_input == 4:
                self.load_case(input('Path to SHARPy output folder: '))

            elif user_input == 5:
                self.display_library()

            elif user_input == 6:
                self.delete_case()

            elif user_input == 7:
                self.set_reference_case()

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
            pmor_path = glob.glob(case_path + '/*.pmor*')
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
        if self.library is not None:
            path = self.folder + '/' + self.library_name + '.pkl'
            pickle.dump((self.library, self.reference_case), open(path, 'wb'))
            coututils.cout_wrap('Saved library to %s' % self.folder, 2)

    def load_library(self, path=None):
        if path is None:
            self.get_library_name()
            path = self.folder + '/' + self.library_name + '.pkl'
        try:
            self.library, self.reference_case = pickle.load(open(path, 'rb'))
            coututils.cout_wrap('Successfully loaded library from %s' % path, 2)
        except FileNotFoundError:
            coututils.cout_wrap('Unable to find library at %s' % path, 3)
            self.interface()

    def display_library(self):

        params = self.library[0][1].keys()

        library_table = coututils.TablePrinter(n_fields=len(params) + 2,
                                               field_types=['g'] + ['g'] * len(params) + ['s'],
                                               field_length=[4] + len(params) * [12] + [90])
        library_table.print_header(field_names=['no'] + list(params) + ['Case Name'])
        [library_table.print_line([ith] + list(entry[1].values()) + [entry[0].settings['SHARPy']['case']])
         for ith, entry in enumerate(self.library)]
        library_table.print_divider_line()

        coututils.cout_wrap('Reference case: %s' % str(self.reference_case))

    def delete_case(self):
        self.display_library()

        del_index = self.select_from_menu()

        try:
            del self.library[del_index]
            coututils.cout_wrap('Deleted case successfully', 2)
        except IndexError:
            coututils.cout_wrap('Index out of range. Unable to remove', 3)

    def set_reference_case(self, reference_case=None):

        if reference_case is None:
            self.display_library()
            reference_case = self.select_from_menu(input_message='Select reference case: ')

        if reference_case in range(len(self.library)):
            self.reference_case = reference_case
        else:
            coututils.cout_wrap('Index Error. Unable to set reference case to desired index', 4)

    def main_menu(self):
        coututils.cout_wrap("\n-----------------------------\n"
                            "PMOR Library Interface Menu\n"
                            "-------------------------------\n\n"
                            "[1] - Create library\n"
                            "[2] - Load library\n"
                            "[3] - Save library\n"
                            "[4] - Add case to library\n"
                            "[5] - Display library\n"
                            "[6] - Delete case")

        if self.reference_case is None and self.library is not None:
            ref_color = 1
        else:
            ref_color = 0

        coututils.cout_wrap('[7] - Set reference case\n', ref_color)

        coututils.cout_wrap("\n[0] - Quit")

        return self.select_from_menu()

    @staticmethod
    def select_from_menu(input_message='Select option: ',
                         unrecognised_message='Unrecognised input choice'):

        try:
            user_input = int(input('\n' + input_message))
        except ValueError:
            coututils.cout_wrap(unrecognised_message, 3)
            user_input = -1
        return user_input


if __name__ == '__main__':
    ROMLibrary().interface()
