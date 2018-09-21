import sharpy.utils.algebra as algebra
import numpy as np
import unittest
from IPython import embed


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


    def test_rotation_vectors_conversions(self):
        '''
        Checks routine to convert rotation vectors.

        Note: test only includes CRV <-> quaternions conversions
        '''

        print(60*'-')
        print('Testing rotations vectors conversion functions')
        print('quat2crv\n' + 'crv2quat')
        print('Note: Euler and triad not included')

        N=1000
        for nn in range(N):
            # def random rotation in [-pi,pi]
            a=np.pi*( 2.*np.random.rand()-1 )
            nv=2.*np.random.rand(3)-1
            nv=nv/np.linalg.norm(nv)
            # reference
            fv0=a*nv
            quat0=np.zeros((4,))
            quat0[0]=np.cos(.5*a)
            quat0[1:]=np.sin(.5*a)*nv
            # check against reference
            assert np.linalg.norm(fv0-algebra.quat2crv(quat0))<1e-12,\
                                                             'Error in quat2crv'
            assert np.linalg.norm(quat0-algebra.crv2quat(fv0))<1e-12,\
                                                             'Error in crv2quat'
        print(50*'-'+' all good!\n')



    def test_rotation_matrices(self):
        '''
        Checks routines and consistency of functions to generate rotation 
        matrices.

        Note: test only includes CRV <-> quaternions conversions
        '''

        print(60*'-')
        print('Testing functions to build rotation matrices')
        print('quat2rot\n' + 'crv2rot')
        print('Note: Euler and triad not included')


        ### Verify that function build rotation matrix (not projection matrix)
        # set an easy rotation (x axis)
        a=np.pi/6.
        nv=np.array([1,0,0])
        sa,ca=np.sin(a),np.cos(a)
        Cab_exp=np.array([[1,  0,   0], 
                          [0, ca, -sa],
                          [0, sa,  ca],])

        # rot from crv
        fv=a*nv
        Cab_num=algebra.crv2rot(fv)
        assert np.linalg.norm(Cab_num-Cab_exp)<1e-15,\
                                        'crv2rot not producing the right result'

        # rot from quat
        quat=algebra.crv2quat(fv)
        Cab_num=algebra.quat2rot(quat)
        assert np.linalg.norm(Cab_num-Cab_exp)<1e-15,\
                                       'quat2rot not producing the right result'

        # N=1000
        # for nn in range(N):
        #     # def random rotation in [-pi,pi]
        #     a=np.pi*( 2.*np.random.rand()-1 )
        #     nv=2.*np.random.rand(3)-1
        #     nv=nv/np.linalg.norm(nv)
        #     # reference
        #     fv0=a*nv
        #     quat0=np.zeros((4,))
        #     quat0[0]=np.cos(.5*a)
        #     quat0[1:]=np.sin(.5*a)*nv
        #     # check against reference
        #     assert np.linalg.norm(fv0-algebra.quat2crv(quat0))<1e-12,\
        #                                                      'Error in quat2crv'
        #     assert np.linalg.norm(quat0-algebra.crv2quat(fv0))<1e-12,\
        #                                                      'Error in crv2quat'
        print(50*'-'+' all good!\n')



    def test_rotation_matrices_derivatives(self):
        '''
        Checks derivatives of rotation matrix derivatives with respect to
        quaternions and Cartesian rotation vectors

        Note: test only includes CRV <-> quaternions conversions
        '''

        print(60*'-')
        print('Testing functions to build rotation matrices derivatives')
        print('der_Cquat_by_v\n' + 'der_CquatT_by_v')

        ### linearisation point
        fi0=np.pi/6
        nv0=np.array([1,3,1])
        nv0=nv0/np.linalg.norm(nv0)
        fv0=fi0*nv0
        qv0=algebra.crv2quat(fv0)

        # direction of perturbation
        fi1=np.pi/3
        nv1=np.array([-2,4,1])
        nv1=nv1/np.linalg.norm(nv1)
        fv1=fi1*nv1
        qv1=algebra.crv2quat(fv1)

        # linearsation point
        Cga0=algebra.quat2rot(qv0)
        Cag0=Cga0.T

        # derivatives
        xv=np.ones((3,)) # dummy vector
        derCga=algebra.der_Cquat_by_v(qv0,xv)
        derCag=algebra.der_CquatT_by_v(qv0,xv)

        print('step\t\terror der_Cquat_by_v\terror der_CquatT_by_v')

        A=np.array([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])
        er_ag=10.
        er_ga=10.
        for a in A:

            # perturbed
            qv=a*qv1 + (1.-a)*qv0
            dqv=qv-qv0
            Cga=algebra.quat2rot(qv)
            Cag=Cga.T   

            dCag_num=np.dot(Cag-Cag0,xv)
            dCga_num=np.dot(Cga-Cga0,xv)
            dCag_an=np.dot(derCag,dqv)
            dCga_an=np.dot(derCga,dqv)

            er_ag_new=np.max(np.abs(dCag_num-dCag_an))
            er_ga_new=np.max(np.abs(dCga_num-dCga_an))

            print('%.3e\t%.3e\t\t%.3e'%(a,er_ag_new,er_ga_new) )
            assert er_ga_new<er_ga, 'der_Cquat_by_v error not converging to 0'
            assert er_ag_new<er_ag, 'der_CquatT_by_v error not converging to 0'
            er_ag=er_ag_new
            er_ga=er_ga_new

        assert er_ga<A[-2], 'der_Cquat_by_v error too large'
        assert er_ag<A[-2], 'der_CquatT_by_v error too large'






        #embed()








    # def test_rotation_matrix_around_axis(self):
    #     axis = np.array([1, 0, 0])
    #     angle = 90
    #     self.assert



if __name__=='__main__':
    # unittest.main()

    T=TestAlgebra()
    # T.setUp()

    T.test_rotation_vectors_conversions()
    T.test_rotation_matrices()
    T.test_rotation_matrices_derivatives()










