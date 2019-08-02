"""Airfoil Geometry Utils

"""

import numpy as np


def generate_naca_camber(M=0, P=0):
    """
    Defines the x and y coordinates of a 4-digit NACA profile's camber line (i.e no thickness).

    The NACA 4-series airfoils follow the nomenclature: NACA MPTT where:
        * M indicates the maximum camber :math:`M = 100m`
        * P indicates the position of the maximum camber :math:`P=10p`
        * TT indicates the thickness to chord ratio :math:`TT=(t/c)*100`

    Args:
        M (float): maximum camber times 100 (i.e. the first of the 4 digits)
        P (float): position of the maximum camber times 10 (i.e. the second of the 4 digits)

    Returns:
        (x_vec,y_vec): ``x`` and ``y`` coordinates of the chosen airfoil

    Example:
        The NACA2400 airfoil would have 2% camber with the maximum at 40% of the chord and 0 thickness. To plot the
        camber line one would use this function as:

            ``x_vec, y_vec = generate_naca_camber(M = 2, P = 4)``

    """
    m = M * 1e-2
    p = P * 1e-1

    def naca(x, m, p):
        if x < 1e-6:
            return 0.0
        elif x < p:
            return m / (p * p) * (2 * p * x - x * x)
        elif x > p and x < 1 + 1e-6:
            return m / ((1 - p) * (1 - p)) * (1 - 2 * p + 2 * p * x - x * x)

    x_vec = np.linspace(0, 1, 1000)
    y_vec = np.array([naca(x, m, p) for x in x_vec])

    return x_vec, y_vec


def interpolate_naca_camber(eta, M00, P00, M01, P01):
    """
    Interpolate aerofoil camber at non-dimensional coordinate eta in (0,1), 
    where (M00,P00) and (M01,P01) define the camber properties at eta=0 and 
    eta=1 respectively.

    Notes:
        For two surfaces, eta can be in (-1,1). In this case, the root is eta=0
        and the tips are at eta=+-1.
    """

    # define domain
    eta = np.abs(eta)
    assert np.max(eta) < 1. + 1e-16, 'eta exceeding +/- 1!'

    # define reference
    x00, y00 = generate_naca_camber(M00, P00)
    x01, y01 = generate_naca_camber(M01, P01)

    # interpolate
    x_vec = x00 * (1. - eta) + x01 * eta
    y_vec = y00 * (1. - eta) + y01 * eta

    return x_vec, y_vec

if __name__ == '__main__':
    a = 1