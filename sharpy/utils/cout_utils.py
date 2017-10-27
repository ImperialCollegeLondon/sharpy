import textwrap
import colorama
import os


class Writer(object):
    fore_colours = ['', colorama.Fore.BLUE, colorama.Fore.CYAN, colorama.Fore.RED]
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
    All rights reserved. '''

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
                os.makedirs(self.file_route)

            self.file = open(self.file_route + '/' + self.file_name, 'w')

        self.print_welcome_message()

    def print_welcome_message(self):
        self.__call__(self.sharpy_ascii)
        self.__call__(self.sharpy_license)
        import sharpy.utils.sharpydir as sharpydir
        cwd = os.getcwd()
        self.__call__('Running SHARPy from ' + cwd, 2)
        self.__call__('SHARPy version being run is in ' + sharpydir.SharpyDir, 2)
        import sharpy.utils.solver_interface as solver_interface
        solver_interface.print_available_solvers()

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
            if level > 3:
                raise AttributeError('Output level cannot be > 3')
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
        self.file.close()


cout_wrap = None
def start_writer():
    global cout_wrap
    cout_wrap = Writer()

def finish_writer():
    global cout_wrap
    cout_wrap.close()
    cout_wrap = None


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    import sys
    """
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    
    From https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a#file-print_progress-py
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()