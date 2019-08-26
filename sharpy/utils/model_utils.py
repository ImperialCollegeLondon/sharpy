"""
Modelling Utilities
"""
import numpy as np
import sharpy.utils.algebra as algebra


def mass_matrix_generator(m, xcg, inertia):
    """
    This function takes the mass, position of the center of
    gravity wrt the elastic axis and the inertia matrix J (3x3) and
    returns the complete 6x6 mass matrix.
    """
    mass = np.zeros((6, 6))
    m_chi_cg = algebra.skew(m*xcg)
    mass[np.diag_indices(3)] = m
    mass[3:, 3:] = inertia
    mass[0:3, 3:6] = -m_chi_cg
    mass[3:6, 0:3] = m_chi_cg

    return mass
