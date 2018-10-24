'''
Utilities to define wing geometry
S. Maraniello, Jul 2018
'''

import numpy as np


def generate_naca_camber(M=0,P=0):
    '''
    Defines the x and y coordinates of a 4-digit NACA profile.
    '''
    m = M*1e-2
    p = P*1e-1

    def naca(x, m, p):
        if x < 1e-6:
            return 0.0
        elif x < p:
            return m/(p*p)*(2*p*x - x*x)
        elif x > p and x < 1+1e-6:
            return m/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

    x_vec = np.linspace(0, 1, 1000)
    y_vec = np.array([naca(x, m, p) for x in x_vec])
    return x_vec, y_vec


def interpolate_naca_camber(eta,M00,P00,M01,P01):
    '''
    Interpolate aerofoil camber at non-dimensional coordinate eta in (0,1), 
    where (M00,P00) and (M01,P01) define the camber properties at eta=0 and 
    eta=1 respectively.

    Ps: for two surfaces, eta can be in (-1,1). In this case, the root is eta=0
    and the tips are at eta=+-1.
    '''

    # define domain
    eta=np.abs(eta)
    assert np.max(eta)<1.+1e-16, 'eta exceeding +/- 1!'     

    # define reference
    x00,y00=generate_naca_camber(M00,P00)
    x01,y01=generate_naca_camber(M01,P01)

    # interpolate
    x_vec = x00*(1.-eta)+x01*eta
    y_vec = y00*(1.-eta)+y01*eta

    return x_vec, y_vec
