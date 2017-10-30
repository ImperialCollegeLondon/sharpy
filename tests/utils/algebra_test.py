import sharpy.utils.algebra as algebra
import numpy as np
import unittest


class TestAlgebra(unittest.TestCase):
    """
    Tests the algebra module
    """

    def test_unit_vector(self):
        """
        Tests the routine for normalising vectors
        :return:
        """
        vector_in = 1
        result = algebra.unit_vector(vector_in)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, 5)

        vector_in = 0
        result = algebra.unit_vector(vector_in)
        self.assertAlmostEqual(np.linalg.norm(result), 0.0, 5)

        vector_in = np.array([1, 0, 0])
        result = algebra.unit_vector(vector_in)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, 5)

        vector_in = np.array([2, -1, 1])
        result = algebra.unit_vector(vector_in)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, 5)

        vector_in = np.array([1e-8, 0, 0])
        result = algebra.unit_vector(vector_in)
        self.assertAlmostEqual(np.linalg.norm(result), 1e-8, 5)

        vector_in = 'aa'
        with self.assertRaises(ValueError):
            algebra.unit_vector(vector_in)

    # def test_rotation_matrix_around_axis(self):
    #     axis = np.array([1, 0, 0])
    #     angle = 90
    #     self.assert











