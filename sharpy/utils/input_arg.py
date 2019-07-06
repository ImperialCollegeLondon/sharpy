import sys
import argparse
import sharpy.utils.exceptions as exceptions
import sharpy.utils.cout_utils as cout


def read_settings(args):
    case_settings = args.input_filename
    cout.cout_wrap('Running SHARPy using the settings file: %s' % case_settings)

    settings = parse_settings(case_settings)
    return settings


def parse_settings(file):
    import os
    from sharpy.utils.settings import load_config_file
    settings = load_config_file(os.path.realpath(file))
    try:
        settings['SHARPy']['flow']
    except KeyError:
        raise exceptions.NotValidInputFile('The solver file does not contain a SHARPy header.')
    return settings
