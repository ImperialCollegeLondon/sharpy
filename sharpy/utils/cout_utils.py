import textwrap
import colorama
import os
import numpy as np
import subprocess
import sharpy.utils.sharpydir as sharpydir

cwd = os.getcwd()


class Writer(object):
    fore_colours = ['', colorama.Fore.BLUE, colorama.Fore.CYAN, colorama.Fore.YELLOW, colorama.Fore.RED]
    reset = colorama.Style.RESET_ALL

    output_columns = 80
    separator = '-'*output_columns
    sharpy_ascii = \
"""--------------------------------------------------------------------------------
            ######  ##     ##    ###    ########  ########  ##    ##
           ##    ## ##     ##   ## ##   ##     ## ##     ##  ##  ##
           ##       ##     ##  ##   ##  ##     ## ##     ##   ####
            ######  ######### ##     ## ########  ########     ##
                 ## ##     ## ######### ##   ##   ##           ##
           ##    ## ##     ## ##     ## ##    ##  ##           ##
            ######  ##     ## ##     ## ##     ## ##           ##
--------------------------------------------------------------------------------"""

    sharpy_license = \
        '''Aeroelastics Lab, Aeronautics Department.
    Copyright (c), Imperial College London.
    All rights reserved.
    License available at https://github.com/imperialcollegelondon/sharpy'''

    wrapper = textwrap.TextWrapper(width=output_columns, break_long_words=False)

    def __init__(self):
        self.print_screen = False
        self.print_file = False
        self.file = None
        self.file_route = ''
        self.file_name = ''

    def initialise(self, print_screen, print_file, file_route=None, file_name=None):
        # copy settings
        self.print_screen = print_screen
        self.print_file = print_file

        if self.print_file:
            self.file_route = file_route
            self.file_name = file_name
            # create folder if necessary
            if not os.path.exists(self.file_route):
                try:
                    os.makedirs(self.file_route)
                except FileExistsError:
                    pass

            self.file = open(self.file_route + '/' + self.file_name, 'w')

        self.print_welcome_message()

    def print_welcome_message(self):
        self.__call__(self.sharpy_ascii)
        self.__call__(self.sharpy_license)
        self.__call__('Running SHARPy from ' + cwd, 2)
        self.__call__('SHARPy being run is in ' + sharpydir.SharpyDir, 2)
        try:
            self.__call__(print_git_status(), 2)
        except subprocess.CalledProcessError:
            pass
        import sharpy.utils.solver_interface as solver_interface
        # solver_interface.print_available_solvers()

    def cout_quiet(self):
        self.print_screen = False

    def cout_talk(self):
        self.print_screen = True

    def print_separator(self, level=0):
        self.__call__(self.separator, level)

    def __call__(self, in_line, level=0):
        if self.print_screen:
            line = in_line
            lines = line.split("\n")
            if level > 4:
                raise AttributeError('Output level cannot be > 4')
            if len(lines) == 1:
                print(self.fore_colours[level] + line + self.reset)
            else:
                newline = ''
                for line in lines:
                    if len(line) > self.output_columns:
                        line = '\n'.join(self.wrapper.wrap(line))

                    print(self.fore_colours[level] + line + self.reset)
                #     newline += line + "\n"
                # print(self.fore_colours[level] + newline + self.reset)
        if self.print_file:
            line = in_line
            lines = line.split("\n")
            if len(lines) == 1:
                self.file.write(line + '\n')
            else:
                newline = ''
                for line in lines:
                    if len(line) > self.output_columns:
                        line = '\n'.join(self.wrapper.wrap(line))

                    newline += line + "\n"
                self.file.write(newline)

    def close(self):
        if self.file is not None:
            if not self.file.closed:
                self.file.close()

    def __del__(self):
        self.close()


cout_wrap = Writer()


def start_writer():
    global cout_wrap
    cout_wrap = Writer()
    # pass


def finish_writer():
    global cout_wrap
    if cout_wrap is not None:
        cout_wrap.close()


# table output for residuals
class TablePrinter(object):
    global cout_wrap

    divider_char = '|'
    line_char = '='

    def __init__(self, n_fields=3, field_length=12, field_types=[['g']]*100, filename=None):
        self.n_fields = n_fields
        self.file = None
        self.divider_line = None
        try:
            field_length[0]
        except TypeError:
            self.field_length = np.full((self.n_fields, ), field_length, dtype=int)
        else:
            if len(field_length) == n_fields:
                self.field_length = field_length
            else:
                raise Exception('len(field_length /= n_fields')
        self.field_names = None
        self.field_types = field_types

        if cout_wrap is None:
            start_writer()

        # if filename is not None:
            # self.file = open(filename, 'w')
        self.file = filename

    def print_header(self, field_names):
        self.field_names = field_names
        if not len(self.field_names) == self.n_fields:
            raise Exception('len(field_names) /= n_fields')
        for i_name in range(self.n_fields):
            name = self.field_names[i_name]
            if len(name) >= self.field_length[i_name]:
                name = name[0:self.field_length[i_name]]

        string = ''
        divider_line = ''
        # for i_field in range(self.n_fields):
            # string += '|{0[' + str(i_field) + ']:^' + str(self.field_length[i_field]) + '}'
            # divider_line += '-'*(self.field_length[i_field]) + '|'

        for i_field in range(self.n_fields):
            field_length = self.field_length[i_field]
            string += self.divider_char + '{' + str(i_field) + ':^' + str(field_length + 2) + '}'
            # string += '|{0[' + str(i_field) + ']:^' + str(self.field_length[i_field]) + '}'
            divider_line += self.divider_char + (field_length + 2)*self.line_char

        string += self.divider_char
        divider_line += self.divider_char
        self.divider_line = divider_line
        cout_wrap('\n\n')
        cout_wrap(divider_line)
        cout_wrap(string.format(*(self.field_names)))
        cout_wrap(divider_line)

        if self.file is not None:
            with open(self.file, 'a+') as f:
                f.write(divider_line)
                f.write('\n' + string.format(*(self.field_names)))
                f.write('\n' + divider_line)

    def print_line(self, line_data):
        string = ''
        for i_field in range(self.n_fields):
            string += (self.divider_char +
                       '{0[' +
                       str(i_field) +
                       ']:^' +
                       str(self.field_length[i_field] + 2) +
                       '.' +
                       str(max(int(self.field_length[i_field]/2), 4)) +
                       self.field_types[i_field] +
                       '}')

        string += self.divider_char
        cout_wrap(string.format(line_data))
        if self.file is not None:
            with open(self.file, 'a') as f:
                f.write('\n'+string.format(line_data))

    def close_file(self):
        if self.file is not None:
            with open(self.file, 'a+') as f:
                f.write('\n' + self.divider_line)

    def print_divider_line(self):
        cout_wrap(self.divider_line)
        if self.file is not None:
            with open(self.file, 'a+') as f:
                f.write('\n' + self.divider_line)

    def character_return(self, n_lines=1):
        cout_wrap(n_lines * '\n')
        if self.file is not None:
            with open(self.file, 'a+') as f:
                f.write(n_lines * '\n')


# version tracker and output
def get_git_revision_hash(di=sharpydir.SharpyDir):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=di).strip().decode('utf-8')


def get_git_revision_short_hash(di=sharpydir.SharpyDir):
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=di).strip().decode('utf-8')


def get_git_revision_branch(di=sharpydir.SharpyDir):
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=di).strip().decode('utf-8')


def get_git_tag(di=sharpydir.SharpyDir):
    return subprocess.check_output(['git', 'describe'], cwd=di).strip().decode('utf-8')


def print_git_status():
    return ('The branch being run is ' + get_git_revision_branch() + '\n'\
            'The version and commit hash are: ' + get_git_tag() + '-' + get_git_revision_short_hash())

def check_running_unittest():
    import sys
    # Define if the script is being run in unittest
    running_unittest = False
    for arg in sys.argv:
        if "unittest" in arg:
            print("running unittest")
            running_unittest = True

    return running_unittest
