import textwrap


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


def cout_wrap(line):
    lines = line.split("\n")
    if len(lines) == 1:
        print(line)
    else:
        newline = ''
        for line in lines:
            if len(line) > output_columns:
                line = '\n'.join(wrapper.wrap(line))

            newline += line + "\n"
        print(newline)
