#! /usr/bin/env python3
"""
This script is an example of the use of sharpy as a "black box"
for optimisation.

It works like this:

*DRIVER
    ->
*PARSE_INPUTS
    -> parser.input_file
*READ_YAML
    -> yaml_dict
OPTIMISER
(
    -> yaml_dict, x
    *WRAPPER
    (
        -> yaml_dict, x
        *UNFOLD_X
        -> yaml_dict, x_dict
        *EVALUATE
        (
            *CASE_ID
            -> yaml_dict, case_name, x_dict
            SET_CASE
            -> file_names
            RUN_CASE
            -> data, x_dict, cost_dict
            *COST_FUNCTION
            (
                # GROUND CLEARANCE CONTRIBUTION
                *GET_GROUND_CLEARANCE
                *COST_SIGMOID
            )
            -> cost
            CLEAN_CASE
            ->
        )
    )
)



"""

import os
import sys
import glob
import shutil
import argparse
import importlib
import pprint
import numpy as np
import yaml

import sharpy

cases = list()


def driver():
    # print information

    # parse args
    parser = parse_inputs()

    # read yaml
    yaml_dict = read_yaml(parser.input_file)
    pprint.pprint(yaml_dict)
    # form cost function call

    # call optimiser
    optimiser(yaml_dict)

    # postprocess output

    return 0


def parse_inputs():
    parser = argparse.ArgumentParser(description=
        """The optimiser.py script is an example of the use of SHARPy as a
black box for an optimiser. """)

    parser.add_argument('input_file', help='input file in YAML format')
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="increase output verbosity")
    parser = parser.parse_args()

    return parser


def read_yaml(file_name):
    """read_yaml

    """
    with open(file_name, 'r') as ifile:
        try:
            yaml_dict = yaml.safe_load(ifile)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def optimiser(in_dict):
    import scipy.optimize

    # create bounds and initial state vector
    x0 = np.zeros((len(in_dict['optimiser']['parameters']), ))
    bounds = [None]*len(x0)
    for k, v in in_dict['optimiser']['parameters_initial'].items():
        x0[k] = v
        bounds[k] = (in_dict['optimiser']['parameters_bounds'][k][0],
                     in_dict['optimiser']['parameters_bounds'][k][1])
    tol = in_dict['optimiser']['numerics']['tolerance']

    args = in_dict

    # call the optimiser
    res = scipy.optimize.minimize(
        fun=wrapper,
        x0=x0,
        args=args,
        method='BFGS',
        jac='2-point',
        bounds=bounds,
        tol=tol)

    print(res)


def case_id():
    global cases

    case_name = '{0:04d}'.format(len(cases))
    cases.append(case_name)
    return case_name



def evaluate(x_dict, yaml_dict):
    case_name = case_id()

    files = set_case(case_name, yaml_dict['base'], x_dict, yaml_dict['settings'])

    data = run_case(case_name, files)

    cost = cost_function(data, x_dict, yaml_dict['optimiser']['cost'])

    if yaml_dict['settings']['delete_case_folders']:
        clean_case(case_name)

    return cost


def wrapper(x, yaml_dict):
    x_dict = unfold_x(x, yaml_dict['optimiser']['parameters'])
    cost = evaluate(x_dict, yaml_dict)
    print(x)
    print(cost)
    return cost


def set_case(case_name, base_dict, x_dict, settings_dict):
    """set_case: takes care of the setup of the case

    This function copies the original case, given by route_base, then
    adds the folder to the python path, runs the generate.py file in there
    and removes the folder from the path.

    Args:
        case_name (str): name of the new case.
        base_dict(dict): dictionary with the base case info
        x_dict(dict): dictionary of state variables
    """
    # create folder for cases if it doesn't exist
    try:
        os.mkdir(settings_dict['cases_folder'])
    except FileExistsError:
        pass

    # clean folder for the new case to be run
    case_route = settings_dict['cases_folder'] + '/' + case_name
    if os.path.exists(case_route):
        # cleanup folder
        shutil.rmtree(case_route)

    # copy case
    try:
        shutil.copytree(base_dict['route'], case_route)
    except IOError as error:
        print('Problem copying the case')
        print('Original error was: {}'.format(error))
        return -1

    # add the case folder to the python path to run generate with it
    sys.path.append(case_route)

    # runs the generate.py
    import generate
    file_names = generate.generate(x_dict, case_name)

    return file_names

# TODO
def run_case(case_name, files):
    return data


def cost_function(data, x_dict, cost_dict):
    """
    x_dict is here to potentially impose constraints on the optimised
    parameters
    """
    cost = 0.0
    # ground clearance cost contribution
    try:
        cost_dict['ground_clearance']
        clearance, ts_clearance = get_ground_clearance(data)
        cost += cost_sigmoid(clearance, **cost_dict['ground_clearance'])
    except KeyError:
        pass

    # loads cost contribution
    try:
        cost_dict['loads']
        raise NotImplementedError('Loads cost contribution not implemented yet')
    except KeyError:
        pass

    return cost


def get_ground_clearance(data):
    """
    Extracts the minimum value of for_pos[2] and returns that value and the
    timestep it happened at.
    """
    structure = data.structure
    min_clear = np.PINF
    ts_min_clear = None
    for ts, tstep in enumerate(structure.timestep_info):
        if tstep.for_pos[2] < min_clear:
            min_clear = tstep.for_pos[2]
            ts_min_clear = ts
    return min_clear, ts_min_clear


def cost_sigmoid(z, z_min, z_0, x_offset=0.5, offset=0.0, scale=1.):
    # I need the input to f to be between 0 and 1 for relevant values
    def sigmoid_mod(x, c=4, x_offset=0.0, offset=0.0, scale=1.):
        return scale/(1. + np.exp(c*(x - x_offset))) + offset

    val = sigmoid_mod((z - z_min)/(z_0 - z_min),
                      x_offset=x_offset,
                      offset=offset,
                      c=5,
                      scale=scale)
    return val

def unfold_x(x, parameters_dict):
    x_dict = dict()
    for k, v in parameters_dict.items():
        x_dict[v] = x[k]

    return x_dict







































def run_sharpy(file_names, route, name):
    """

    """
    print('Running case {}...'.format(name))
    sharpy_data = sharpy.sharpy_main.main([None, file_names['solver']])
    print('\tfinished')

    return sharpy_data








# parser = argpase





































driver()



