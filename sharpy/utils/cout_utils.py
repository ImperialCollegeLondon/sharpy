import textwrap
import colorama

fore_colours = ['', colorama.Fore.BLUE, colorama.Fore.CYAN, colorama.Fore.RED]
reset = colorama.Style.RESET_ALL


output_columns = 80
separator = '-'*output_columns
sharpy_ascii = \
    ''' --------------------------------------------------------------------------------
     ___           ___           ___           ___           ___
    /  /\         /__/\         /  /\         /  /\         /  /\        ___
   /  /:/_        \  \:\       /  /::\       /  /::\       /  /::\      /__/|
  /  /:/ /\        \__\:\     /  /:/\:\     /  /:/\:\     /  /:/\:\    |  |:|
 /  /:/ /::\   ___ /  /::\   /  /:/~/::\   /  /:/~/:/    /  /:/~/:/    |  |:|
/__/:/ /:/\:\ /__/\  /:/\:\ /__/:/ /:/\:\ /__/:/ /:/___ /__/:/ /:/   __|__|:|
\  \:\/:/~/:/ \  \:\/:/__\/ \  \:\/:/__\/ \  \:\/:::::/ \  \:\/:/   /__/::::\\
 \  \::/ /:/   \  \::/       \  \::/       \  \::/~~~~   \  \::/       ~\~~\:\\
  \__\/ /:/     \  \:\        \  \:\        \  \:\        \  \:\         \  \:\\
    /__/:/       \  \:\        \  \:\        \  \:\        \  \:\         \__\/
    \__\/         \__\/         \__\/         \__\/         \__\/
--------------------------------------------------------------------------------'''

sharpy_license = \
    '''Aeroelastics Lab, Aeronautics Department.
Copyright (c), Imperial College London.
All rights reserved. '''

wrapper = textwrap.TextWrapper(width=output_columns, break_long_words=False)

suppress_output = False


def cout_quiet():
    global suppress_output
    suppress_output = True


def cout_talk():
    global suppress_output
    suppress_output = False


def cout_wrap(line, level=0):
    lines = line.split("\n")
    if level > 3: raise AttributeError('Output level cannot be > 3')
    if len(lines) == 1:
        print(fore_colours[level] + line + reset)
    else:
        newline = ''
        for line in lines:
            if len(line) > output_columns:
                line = '\n'.join(wrapper.wrap(line))

            newline += line + "\n"
        print(fore_colours[level] + newline + reset)


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