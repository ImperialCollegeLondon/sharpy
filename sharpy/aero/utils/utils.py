import numpy as np
import sharpy.utils.algebra as algebra


def flightcon_file_parser(fc_dict):
    fc = fc_dict['FlightCon']
    fc['u_inf'] = float(fc['u_inf'])
    fc['alpha'] = float(fc['alpha'])*np.pi/180.0
    fc['beta'] = float(fc['beta'])*np.pi/180.0
    fc['rho_inf'] = float(fc['rho_inf'])
    fc['c_ref'] = float(fc['c_ref'])
    fc['b_ref'] = float(fc['b_ref'])


def alpha_beta_to_direction(alpha, beta):
    direction = np.array([1, 0, 0])
    alpha_rot = algebra.rotation3d_y(alpha)
    beta_rot = algebra.rotation3d_z(beta)
    direction = np.dot(beta_rot, np.dot(alpha_rot, direction))
    return direction


def find_aerodynamic_solver(settings):
    """
    Retrieves the name and settings of the first aerodynamic solver used in the solution ``flow``.

    Args:
        settings (dict): SHARPy settings (usually found in ``data.settings`` )

    Returns:
        tuple: Aerodynamic solver name and solver settings
    """
    flow = settings['SHARPy']['flow']
    # Look for the aerodynamic solver
    if 'StaticUvlm' in flow:
        aero_solver_name = 'StaticUvlm'
        aero_solver_settings = settings['StaticUvlm']
    elif 'StaticCoupled' in flow:
        aero_solver_name = settings['StaticCoupled']['aero_solver']
        aero_solver_settings = settings['StaticCoupled']['aero_solver_settings']
    elif 'StaticCoupledRBM' in flow:
        aero_solver_name = settings['StaticCoupledRBM']['aero_solver']
        aero_solver_settings = settings['StaticCoupledRBM']['aero_solver_settings']
    elif 'DynamicCoupled' in flow:
        aero_solver_name = settings['DynamicCoupled']['aero_solver']
        aero_solver_settings = settings['DynamicCoupled']['aero_solver_settings']
    elif 'StepUvlm' in flow:
        aero_solver_name = 'StepUvlm'
        aero_solver_settings = settings['StepUvlm']
    else:
        raise KeyError("ERROR: aerodynamic solver not found")

    return aero_solver_name, aero_solver_settings


def find_velocity_generator(settings):
    """
    Retrieves the name and settings of the fluid velocity generator in the first aerodynamic solver used in the
    solution ``flow``.

    Args:
        settings (dict): SHARPy settings (usually found in ``data.settings`` )

    Returns:
        tuple: velocity generator name and velocity generator settings
    """
    aero_solver_name, aero_solver_settings = find_aerodynamic_solver(settings)

    vel_gen_name = aero_solver_settings['velocity_field_generator']
    vel_gen_settings = aero_solver_settings['velocity_field_input']

    return vel_gen_name, vel_gen_settings
