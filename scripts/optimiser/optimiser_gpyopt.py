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
import warnings
import random
import pprint
import numpy as np
import yaml
import dill as pickle
import GPyOpt

import sharpy.sharpy_main
import sharpy.utils.exceptions as exc

cases = list()

loads_cost_array = None
prev_result = None


def driver():
    # print information

    # parse args
    parser = parse_inputs()

    # read yaml
    yaml_dict = read_yaml(parser.input_file)
    pprint.pprint(yaml_dict)

    # get previous cases
    previous_x, previous_y = process_previous_cases(yaml_dict)

    # call optimiser
    optimiser(yaml_dict, previous_x, previous_y)

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


def optimiser(in_dict, previous_x, previous_y):
    settings_dict = in_dict['settings']
    case_dict = in_dict['case']
    base_dict = in_dict['base']
    # create folder for cases if it doesn't exist
    try:
        os.mkdir(settings_dict['cases_folder'])
    except FileExistsError:
        pass
    # create folder for cases if it doesn't exist
    try:
        os.mkdir((settings_dict['cases_folder'] + '/' + case_dict['name'] + '/').replace('//', '/'))
        print('Folder made')
    except FileExistsError:
        print('cases_folder already exists')

    # clean folder for the new case to be run
    case_route = (settings_dict['cases_folder'] + '/' + case_dict['name'] + '/').replace('//', '/')

    # copy case
    try:
        print(base_dict['route'] + '/generate.py', case_route + 'generate.py')
        shutil.copyfile(base_dict['route'] + '/generate.py', case_route + 'generate.py')
    except IOError as error:
        print('Problem copying the case')
        print('Original error was: {}'.format(error))

    # add the case folder to the python path to run generate with it
    sys.path.append(case_route)
    # create folder for output if doesnt exist
    try:
        os.mkdir(in_dict['case']['output_folder'])
    except FileExistsError:
        pass

    output_route = in_dict['case']['output_folder'] + '/' + in_dict['case']['name'] + '/'
    if os.path.exists(output_route):
        warnings.warn('The folder ' + output_route + ' exists, cleaning it.')
        # cleanup folder
        try:
            shutil.rmtree(output_route)
        except:
            pass

    os.mkdir(output_route)

    n_params = len(in_dict['optimiser']['parameters'])
    bounds = []
    for k, v in in_dict['optimiser']['parameters_initial'].items():
        bounds.append({'name': in_dict['optimiser']['parameters'][k],
                       'type': 'continuous',
                       'domain': in_dict['optimiser']['parameters_bounds'][k]})
        pprint.pprint(bounds)

    constraints = list()
    try:
        length = in_dict['optimiser']['constraints']['ramp_length']
        acc_var_i = None
        release_vel_var_i = None
        for k, v in in_dict['optimiser']['parameters'].items():
            if v == 'acceleration':
                acc_var_i = k

            if v == 'release_velocity':
                release_vel_var_i = k

        # ramp_length = release_vel**2 / acceleration
        constraints.append({'name': 'length',
                            'constraint': '0.5*(x[:, ' + str(release_vel_var_i) + ']**2' +
                                          '/x[:, ' + str(acc_var_i) + '])' +
                                          ' - ' + str(length)})
    except KeyError:
        pass

    try:
        limit = in_dict['optimiser']['constraints']['incidence_angle']['limit']
        base_aoa = in_dict['optimiser']['constraints']['incidence_angle']['base_aoa']

        dAoA_var_i = None
        ramp_angle_var_i = None
        for k, v in in_dict['optimiser']['parameters'].items():
            if v == 'dAoA':
                dAoA_var_i = k
            if v == 'ramp_angle':
                ramp_angle_var_i = k

        # base_aoa + dAoA - ramp_angle < limit
        constraint_string = ''
        constraint_string += str(base_aoa) + ' + '
        constraint_string += 'x[:, ' + str(dAoA_var_i) + '] - '
        constraint_string += 'x[:, ' + str(ramp_angle_var_i) + '] - '
        constraint_string += str(limit)
        constraints.append({'name': 'angle',
                            'constraint': constraint_string})
    except KeyError:
        pass

    print(constraints)

    gpyopt_wrapper = lambda x: wrapper(x, in_dict)
    batch_size = in_dict['optimiser']['numerics']['batch_size']
    num_cores = in_dict['optimiser']['numerics']['n_cores']
    opt = GPyOpt.methods.BayesianOptimization(
        f=gpyopt_wrapper,
        domain=bounds,
        exact_feval=True,
        model_type='GP',
        acquisition_type='EI',
        normalize_y=True,
        initial_design_numdata=in_dict['optimiser']['numerics']['initial_design_numdata'],
        evaluator_type='local_penalization',
        batch_size=batch_size,
        num_cores=num_cores,
        acquisition_jitter=0,
        de_duplication=True,
        constraints=constraints,
        X=previous_x,
        Y=previous_y)

    opt.run_optimization(in_dict['optimiser']['numerics']['n_iter'],
                         report_file=output_route + 'report.log',
                         evaluations_file=output_route + 'evaluations.log',
                         models_file=output_route + 'models.log',
                         verbosity=True
                        )

    print('*'*60)
    print('Best one cost: ', opt.fx_opt)
    print('\tParameters: ', opt.x_opt)
    print('*'*60)
    with open(output_route + 'optimiser.pkl', 'wb') as f:
        pickle.dump(opt, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('FINISHED')

    import pdb; pdb.set_trace()
    # skopt_wrapper = lambda x: wrapper(x, in_dict)
    # res = skopt.gp_minimize(func=skopt_wrapper,
                            # dimensions=bounds,
                            # base_estimator=None,
                            # n_calls=100,
                            # n_random_starts=10,
                            # acq_func='gp_hedge',
                            # acq_optimizer='auto',
                            # base_estimator=None,
                            # n_calls=100,
                            # n_random_starts=10,
                            # acq_func='gp_hedge',
                            # acq_optimizer='auto',
                            # x0=None,
                            # y0=None,
                            # random_state=None,
                            # verbose=True,
                            # callback=None,
                            # n_points=10000,
                            # xi=0.01,
                            # kappa=1.96,
                            # noise=1e-8,
                            # n_jobs=4)


def case_id():
    case_name = '{0:04d}'.format(random.randint(0, 9999+1))
    return case_name


def evaluate(x_dict, yaml_dict):
    case_name = case_id()

    print('Running ' + case_name)
    files, case_name = set_case(case_name,
                                yaml_dict['base'],
                                x_dict,
                                yaml_dict['settings'],
                                yaml_dict['case'])
    data = run_case(files)
    cost = cost_function(data, x_dict, yaml_dict['optimiser']['cost'])
    print('   Case: ' + str(case_name) + '; cost = ', cost)

    if data is not None:
        data.cost = cost

        if yaml_dict['settings']['delete_case_folders']:
            raise NotImplementedError('delete_case_folders not supported yet')
        if yaml_dict['settings']['save_data']:
            with open(yaml_dict['settings']['cases_folder'] +
                      '/' +
                      case_name +
                      'data.pkl', 'wb') as data_file:
                pickle.dump(data, data_file, -1)

    return cost


def wrapper(x, yaml_dict):
    x_dict = unfold_x(x, yaml_dict['optimiser']['parameters'])
    cost = evaluate(x_dict, yaml_dict)
    return cost


def set_case(case_name, base_dict, x_dict, settings_dict, case_dict):
    """set_case: takes care of the setup of the case

    This function copies the original case, given by route_base, then
    adds the folder to the python path, runs the generate.py file in there
    and removes the folder from the path.

    Args:
        case_name (str): name of the new case.
        base_dict(dict): dictionary with the base case info
        x_dict(dict): dictionary of state variables
    """

    # runs the generate.py
    import generate
    file_names = generate.generate(x_dict, case_name)

    return file_names, case_name

def run_case(files):
    try:
        warnings.filterwarnings('ignore')
        data = sharpy.sharpy_main.main(args=['', files['sharpy']])
    except exc.NotConvergedSolver:
        print('The solver is not converged in this simulation with inputs')
        print('Returning None as data')
        data = None
    return data


def cost_function(data,
                  x_dict,
                  cost_dict,
                  insight=False):
    """
    x_dict is here to potentially impose constraints on the optimised
    parameters
    """
    cost = 0.0
    clearance_cost = 0.0
    loading_cost = 0.0
    output_dict = dict()
    # check for data == None:
    if data is None:
        cost = 15.  # need a better way
        return cost
    # ground clearance cost contribution
    try:
        cost_dict['ground_clearance']
        clearance, ts_clearance = get_ground_clearance(data)
        output_dict['ground_clearance'] = dict()
        clearance_cost = cost_sigmoid(clearance,
                                      **cost_dict['ground_clearance'])
        output_dict['ground_clearance']['clearance'] = clearance
        output_dict['ground_clearance']['cost'] = clearance_cost
        cost += clearance_cost
    except KeyError:
        pass

    # loads cost contribution
    try:
        cost_dict['loads']
        loading_cost = loads_cost(data, cost_dict['loads'])
        cost += loading_cost
        output_dict['loads'] = {'cost': loading_cost}
    except KeyError:
        pass

    if insight:
        return cost, output_dict
    else:
        return cost


def loads_cost(data, cost_loads_dict):
    index2load = {0: 'Torsion',
                  1: 'OOP',
                  2: 'IP'}
    try:
        loads_array = np.loadtxt(
            cost_loads_dict['reference_loads'],
            skiprows=1,
            delimiter=',')
    except OSError:
        try:
            warnings.warn('Not found reference_loads file, trying parent folder')
            loads_array = np.loadtxt(
                '../' + cost_loads_dict['reference_loads'],
                skiprows=1,
                delimiter=',')
        except OSError:
            warnings.warn('Not found reference_loads file, anywhere. Filling up with ones instead')
            loads_array = np.ones((data.structure.ini_info.psi.shape[0], 4))

    separate_cost = np.zeros((3,))

    loads_array = np.abs(loads_array)
    loads_array_norm = np.linalg.norm(loads_array, axis=0)
    for row in range(loads_array.shape[0]):
        for col in range(loads_array.shape[1]):
            if loads_array[row, col] < loads_array_norm[col]:
                loads_array[row, col] = loads_array_norm[col]

    max_cost = np.zeros((3,))
    for it, tstep in enumerate(data.structure.timestep_info):
        temp = np.abs(tstep.postproc_cell['loads'][:, 3:])
        max_vals = np.max(temp/loads_array[:, 1:] - 1.0, axis=0)
        for i_dim in range(3):
            max_cost[i_dim] = max(max_cost[i_dim], max_vals[i_dim])
    separate_cost = max_cost

    for k, v in index2load.items():
        separate_cost[k] *= cost_loads_dict[v]['scale']
    return np.sum(separate_cost)


def get_ground_clearance(data):
    """
    Extracts the minimum value of for_pos[2] and returns that value and the
    timestep it happened at.
    """
    structure = data.structure
    min_clear = np.PINF
    ts_min_clear = None
    for ts, tstep in enumerate(structure.timestep_info):
        try:
            tstep.mb_dict['constraint_00']
            continue
        except KeyError:
            pass
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
        x_dict[v] = x.flatten()[k]

    return x_dict


def process_previous_cases(yaml_dict):
    try:
        previous_cases_string = yaml_dict['previous_data']['cases']
    except KeyError:
        return None, None

    n_cases = len(glob.glob(previous_cases_string))
    x_out = np.zeros((n_cases, len(yaml_dict['optimiser']['parameters'])))
    y_out = np.zeros((n_cases, 1))

    for i, f in enumerate(glob.glob(previous_cases_string)):
        print('Loading ', f)
        with open(f, 'rb') as fhandle:
            data = pickle.load(fhandle)

        x_vec, x_dict = x_vec_from_data(data, yaml_dict['optimiser']['parameters'])
        x_out[i, :] = x_vec

        cost = cost_function(data, x_dict, yaml_dict['optimiser']['cost'])
        y_out[i, 0] = cost

    return x_out, y_out


def x_vec_from_data(data, param_dict):
    input_dict = eval(data.settings['Notes']['note'])

    x_vec = np.zeros((len(param_dict),))
    for k, v in param_dict.items():
        x_vec[k] = input_dict[v]
    return x_vec, input_dict


if __name__ == '__main__':
    driver()



