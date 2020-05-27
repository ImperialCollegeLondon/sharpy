import os
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
    from sharpy.utils.settings import load_config_file
    settings = load_config_file(os.path.realpath(file))
    try:
        settings['SHARPy']['flow']
    except KeyError:
        raise exceptions.NotValidInputFile('The solver file does not contain a SHARPy header.')

    from sharpy.utils.solver_interface import dict_of_solvers

    for solver in settings['SHARPy']['flow']:
        # Check that the solvers in the flow exist and that they have a valid set of settings
        try:
            dict_of_solvers[solver]
        except KeyError:
            raise exceptions.SolverNotFound(solver)

        try:
            settings[solver]
        except KeyError:
            raise exceptions.NotValidInputFile('The settings for the solver %s have not been given.' % solver)
    return settings
