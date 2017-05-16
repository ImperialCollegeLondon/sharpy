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


def wind2body_rot(alpha, beta):
    cb = np.cos(beta)
    sb = np.sin(beta)
    beta_rot = algebra.rotation3d_z(beta)
    alpha_rot = algebra.rotation3d_y(alpha)
    return np.dot(beta_rot, alpha_rot)
