"""Aero utilities functions"""
import numpy as np
import sharpy.utils.algebra as algebra
from sharpy.utils import algebra as algebra


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


def magnitude_and_direction_of_relative_velocity(displacement, displacement_vel, for_vel, cga, uext,
                                                 add_rotation=False, rot_vel_g=np.zeros((3)), centre_rot_g=np.zeros((3))):
    r"""
    Calculates the magnitude and direction of the relative velocity ``u_rel`` at a local section of the wing.

    .. math::

       u_{rel, i}^G = \bar{U}_{\infty, i}^G - C^{GA}(\chi)(\dot{\eta}_i^A + v^A + \tilde{\omega}^A\eta_i^A

    where :math:`\bar{U}_{\infty, i}^G` is the average external velocity across all aerodynamic nodes at the
    relevant cross section.

    Args:
        displacement (np.array): Unit vector in the direction of the free stream velocity expressed in A frame.
        displacement_vel (np.array): Unit vector in the direction of the local chord expressed in A frame.
        for_vel (np.array): ``A`` frame of reference (FoR) velocity. Expressed in A FoR
        cga (np.array): Rotation vector from FoR ``G`` to FoR ``A``
        uext (np.array): Background flow velocity on solid grid nodes
        add_rotation (bool): Adds rotation velocity. Probalby needed in steady computations
        rot_vel_g (np.array): Rotation velocity. Only used if add_rotation = True
        centre_rot_g (np.array): Centre of rotation. Only used if add_rotation = True

    Returns:
        tuple: ``u_rel``, ``dir_u_rel`` expressed in the inertial, ``G`` frame.
    """
    urel = (displacement_vel +
            for_vel[0:3] +
            algebra.cross3(for_vel[3:6], displacement))
    urel = -np.dot(cga, urel)
    urel += np.average(uext, axis=1)

    if add_rotation:
        urel -= algebra.cross3(rot_vel_g,
                               np.dot(cga, displacement) - centre_rot_g)
    dir_urel = algebra.unit_vector(urel)

    return urel, dir_urel


def local_stability_axes(dir_urel, dir_chord):
    """
    Rotates the body axes onto stability axes. This rotation is equivalent to the projection of a vector in S onto B.

    The stability axes are defined as:

        * ``x_s``: parallel to the free stream

        * ``z_s``: perpendicular to the free stream and part of the plane formed by the local chord and the vertical
          body axis ``z_b``.

        * ``y_s``: completes the set

    Args:
        dir_urel (np.array): Unit vector in the direction of the free stream velocity expressed in B frame.
        dir_chord (np.array): Unit vector in the direction of the local chord expressed in B frame.

    Returns:
        np.array: Rotation matrix from B to S, equivalent to the projection matrix :math:`C^{BS}` that projects a
        vector from S onto B.
    """
    xs = dir_urel

    zb = np.array([0, 0, 1.])
    zs = algebra.cross3(algebra.cross3(dir_chord, zb), dir_urel)

    ys = -algebra.cross3(xs, zs)

    return algebra.triad2rotation(xs, ys, zs)


def span_chord(i_node_surf, zeta):
    """
    Retrieve the local span and local chord

    Args:
        i_node_surf (int): Node index in aerodynamic surface
        zeta (np.array): Aerodynamic surface coordinates ``(3 x n_chord x m_span)``

    Returns:
        tuple: ``dir_span``, ``span``, ``dir_chord``, ``chord``
    """
    N = zeta.shape[2] - 1 # spanwise vertices in surface (-1 for index)

    # Deal with the extremes
    if i_node_surf == 0:
        node_p = 1
        node_m = 0
    elif i_node_surf == N:
        node_p = N
        node_m = N - 1
    else:
        node_p = i_node_surf + 1
        node_m = i_node_surf - 1

    # Define the span and the span direction
    dir_span = 0.5 * (zeta[:, 0, node_p] - zeta[:, 0, node_m])

    span = np.linalg.norm(dir_span)
    dir_span = algebra.unit_vector(dir_span)

    # Define the chord and the chord direction
    dir_chord = zeta[:, -1, i_node_surf] - zeta[:, 0, i_node_surf]
    chord = np.linalg.norm(dir_chord)
    dir_chord = algebra.unit_vector(dir_chord)

    return dir_span, span, dir_chord, chord


def find_aerodynamic_solver_settings(settings):
    """
    Retrieves the settings of the first aerodynamic solver used in the solution ``flow``. 
    
    For coupled solvers, the aerodynamic solver is found in the aero solver settings. 
    The StaticTrim solver can either contain a coupled or aero solver in its solver 
    settings (making it into a possible 3-level Matryoshka).

    Args:
        settings (dict): SHARPy settings (usually found in ``data.settings`` )

    Returns:
        tuple: Aerodynamic solver settings
    """
    flow = settings['SHARPy']['flow']
    for solver_name in ['StaticUvlm', 'StaticCoupled', 'StaticTrim', 'DynamicCoupled', 'StepUvlm']:
        if solver_name in flow:
            aero_solver_settings = settings[solver_name]
            if solver_name == 'StaticTrim':
                aero_solver_settings = aero_solver_settings['solver_settings']['aero_solver_settings']
            elif 'aero_solver' in settings[solver_name].keys():
                aero_solver_settings = aero_solver_settings['aero_solver_settings']
                
            return aero_solver_settings

    raise KeyError("ERROR: Aerodynamic solver not found.")

def find_velocity_generator(settings):
    """
    Retrieves the name and settings of the fluid velocity generator in the first aerodynamic solver used in the
    solution ``flow``.

    Args:
        settings (dict): SHARPy settings (usually found in ``data.settings`` )

    Returns:
        tuple: velocity generator name and velocity generator settings
    """
    aero_solver_settings = find_aerodynamic_solver_settings(settings)

    vel_gen_name = aero_solver_settings['velocity_field_generator']
    vel_gen_settings = aero_solver_settings['velocity_field_input']

    return vel_gen_name, vel_gen_settings
