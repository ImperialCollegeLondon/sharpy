import numpy as np
import ctypes as ct
import scipy as sc


def check_symmetric(mat):
    return np.allclose(mat.transpose(), mat, atol=1e-3)


if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    print(check_symmetric(a))

    a = np.array([[1, 2], [2, 1]])
    print(check_symmetric(a))
