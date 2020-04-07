import unittest
import numpy as np
import sharpy.rom.utils.librom_interp as librominterp


class TestInterpolationSchemes(unittest.TestCase):

    def test_lagrange(self):

        p = np.poly1d([1, -1, 1, -1, 1, -1, 0])  # source polynomial

        # source points
        x_source = np.linspace(-0.75, 0.75, 7)
        y_source = p(x_source)

        x0 = 0.5
        interpolation_degree = 5
        weights = librominterp.lagrange_interpolation(x_source, x0, interpolation_degree)
        y_interp = sum(weights * p(x_source))

        np.testing.assert_almost_equal(y_interp, p(x0), 6)


if __name__ == '__main__':
    unittest.main()
