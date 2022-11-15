import sharpy.utils.algebra as algebra
import numpy as np
import unittest
import random


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

        vector_in = "aa"
        with self.assertRaises(ValueError):
            algebra.unit_vector(vector_in)

    def test_rotation_vectors_conversions(self):
        """
        Checks routine to convert rotation vectors.

        Note: test only includes CRV <-> quaternions conversions
        """

        N = 1000
        for nn in range(N):
            # def random rotation in [-pi,pi]
            a = np.pi * (2.0 * np.random.rand() - 1)
            nv = 2.0 * np.random.rand(3) - 1
            nv = nv / np.linalg.norm(nv)
            # reference
            fv0 = a * nv
            quat0 = np.zeros((4,))
            quat0[0] = np.cos(0.5 * a)
            quat0[1:] = np.sin(0.5 * a) * nv
            # check against reference
            assert (
                np.linalg.norm(fv0 - algebra.quat2crv(quat0)) < 1e-12
            ), "Error in quat2crv"
            assert (
                np.linalg.norm(quat0 - algebra.crv2quat(fv0)) < 1e-12
            ), "Error in crv2quat"

    def test_rotation_matrices(self):
        """
        Checks routines and consistency of functions to generate rotation
        matrices.

        Note: test only includes triad <-> CRV <-> quaternions conversions
        """

        ### Verify that function build rotation matrix (not projection matrix)
        # set an easy rotation (x axis)
        a = np.pi / 6.0
        nv = np.array([1, 0, 0])
        sa, ca = np.sin(a), np.cos(a)
        Cab_exp = np.array(
            [
                [1, 0, 0],
                [0, ca, -sa],
                [0, sa, ca],
            ]
        )
        ### rot from triad
        Cab_num = algebra.triad2rotation(Cab_exp[:, 0], Cab_exp[:, 1], Cab_exp[:, 2])
        assert (
            np.linalg.norm(Cab_num - Cab_exp) < 1e-15
        ), "crv2rotation not producing the right result"
        ### rot from crv
        fv = a * nv
        Cab_num = algebra.crv2rotation(fv)
        assert (
            np.linalg.norm(Cab_num - Cab_exp) < 1e-15
        ), "crv2rotation not producing the right result"
        ### rot from quat
        quat = algebra.crv2quat(fv)
        Cab_num = algebra.quat2rotation(quat)
        assert (
            np.linalg.norm(Cab_num - Cab_exp) < 1e-15
        ), "quat2rotation not producing the right result"

        ### inverse relations
        # check crv2rotation and rotation2crv are biunivolcal in [-pi,pi]
        # check quat2rotation and rotation2quat are biunivocal in [-pi,pi]
        N = 100
        for nn in range(N):
            # def random rotation in [-pi,pi]
            a = np.pi * (2.0 * np.random.rand() - 1)
            nv = 2.0 * np.random.rand(3) - 1
            nv = nv / np.linalg.norm(nv)

            # inverse crv
            fv0 = a * nv
            Cab = algebra.crv2rotation(fv0)
            fv = algebra.rotation2crv(Cab)
            assert (
                np.linalg.norm(fv - fv0) < 1e-12
            ), "rotation2crv not producing the right result"

            # triad2crv
            xa, ya, za = Cab[:, 0], Cab[:, 1], Cab[:, 2]
            assert (
                np.linalg.norm(algebra.triad2crv(xa, ya, za) - fv0) < 1e-12
            ), "triad2crv not producing the right result"

            # inverse quat
            quat0 = np.zeros((4,))
            quat0[0] = np.cos(0.5 * a)
            quat0[1:] = np.sin(0.5 * a) * nv
            quat = algebra.rotation2quat(algebra.quat2rotation(quat0))
            assert (
                np.linalg.norm(quat - quat0) < 1e-12
            ), "rotation2quat not producing the right result"

        ### combined rotation
        # assume 3 FoR, G, A and B where:
        #   - G is the initial FoR
        #   - A derives from a 90 deg rotation about zG
        #   - B derives from a 90 deg rotation about yA
        crv_G_to_A = 0.5 * np.pi * np.array([0, 0, 1])
        crv_A_to_B = 0.5 * np.pi * np.array([0, 1, 0])
        Cga = algebra.crv2rotation(crv_G_to_A)
        Cab = algebra.crv2rotation(crv_A_to_B)

        # rotation G to B (i.e. projection B onto G)
        Cgb = np.dot(Cga, Cab)
        Cgb_exp = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        assert (
            np.linalg.norm(Cgb - Cgb_exp) < 1e-15
        ), "combined rotation not as expected!"

    def test_rotation_matrices_derivatives(self):
        """
        Checks derivatives of rotation matrix derivatives with respect to
        quaternions and Cartesian rotation vectors

        Note: test only includes CRV <-> quaternions conversions
        """

        ### linearisation point
        # fi0=np.pi/6
        # nv0=np.array([1,3,1])
        fi0 = 2.0 * np.pi * random.random() - np.pi
        nv0 = np.array([random.random(), random.random(), random.random()])
        nv0 = nv0 / np.linalg.norm(nv0)
        fv0 = fi0 * nv0
        qv0 = algebra.crv2quat(fv0)
        ev0 = algebra.quat2euler(qv0)

        # direction of perturbation
        # fi1=np.pi/3
        # nv1=np.array([-2,4,1])
        fi1 = 2.0 * np.pi * random.random() - np.pi
        nv1 = np.array([random.random(), random.random(), random.random()])
        nv1 = nv1 / np.linalg.norm(nv1)
        fv1 = fi1 * nv1
        qv1 = algebra.crv2quat(fv1)
        ev1 = algebra.quat2euler(qv1)

        # linearsation point
        Cga0 = algebra.quat2rotation(qv0)
        Cag0 = Cga0.T
        Cab0 = algebra.crv2rotation(fv0)
        Cba0 = Cab0.T
        Cga0_euler = algebra.euler2rot(ev0)
        Cag0_euler = Cga0_euler.T

        # derivatives
        # xv=np.ones((3,)) # dummy vector
        xv = np.array(
            [random.random(), random.random(), random.random()]
        )  # dummy vector
        derCga = algebra.der_Cquat_by_v(qv0, xv)
        derCag = algebra.der_CquatT_by_v(qv0, xv)
        derCab = algebra.der_Ccrv_by_v(fv0, xv)
        derCba = algebra.der_CcrvT_by_v(fv0, xv)
        derCga_euler = algebra.der_Ceuler_by_v(ev0, xv)
        derCag_euler = algebra.der_Peuler_by_v(ev0, xv)

        A = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        er_ag = 10.0
        er_ga = 10.0
        er_ab = 10.0
        er_ba = 10.0
        er_ag_euler = 10.0
        er_ga_euler = 10.0

        for a in A:
            # perturbed
            qv = a * qv1 + (1.0 - a) * qv0
            fv = a * fv1 + (1.0 - a) * fv0
            ev = a * ev1 + (1.0 - a) * ev0
            dqv = qv - qv0
            dfv = fv - fv0
            dev = ev - ev0
            Cga = algebra.quat2rotation(qv)
            Cag = Cga.T
            Cab = algebra.crv2rotation(fv)
            Cba = Cab.T
            Cga_euler = algebra.euler2rot(ev)
            Cag_euler = Cga_euler.T

            dCag_num = np.dot(Cag - Cag0, xv)
            dCga_num = np.dot(Cga - Cga0, xv)
            dCag_an = np.dot(derCag, dqv)
            dCga_an = np.dot(derCga, dqv)
            er_ag_new = np.max(np.abs(dCag_num - dCag_an))
            er_ga_new = np.max(np.abs(dCga_num - dCga_an))

            dCab_num = np.dot(Cab - Cab0, xv)
            dCba_num = np.dot(Cba - Cba0, xv)
            dCab_an = np.dot(derCab, dfv)
            dCba_an = np.dot(derCba, dfv)
            er_ab_new = np.max(np.abs(dCab_num - dCab_an))
            er_ba_new = np.max(np.abs(dCba_num - dCba_an))

            dCag_num_euler = np.dot(Cag_euler - Cag0_euler, xv)
            dCga_num_euler = np.dot(Cga_euler - Cga0_euler, xv)
            dCag_an_euler = np.dot(derCag_euler, dev)
            dCga_an_euler = np.dot(derCga_euler, dev)
            er_ag_euler_new = np.max(np.abs(dCag_num_euler - dCag_an_euler))
            er_ga_euler_new = np.max(np.abs(dCga_num_euler - dCga_an_euler))

            assert er_ga_new < er_ga, "der_Cquat_by_v error not converging to 0"
            assert er_ag_new < er_ag, "der_CquatT_by_v error not converging to 0"
            assert er_ab_new < er_ab, "der_Ccrv_by_v error not converging to 0"
            assert er_ba_new < er_ba, "der_CcrvT_by_v error not converging to 0"
            assert (
                er_ga_euler_new < er_ga_euler
            ), "der_Ceuler_by_v error not converging to 0"
            assert (
                er_ag_euler_new < er_ag_euler
            ), "der_Peuler_by_v error not converging to 0"

            er_ag = er_ag_new
            er_ga = er_ga_new
            er_ab = er_ab_new
            er_ba = er_ba_new
            er_ag_euler = er_ag_euler_new
            er_ga_euler = er_ga_euler_new

        assert er_ga < A[-2], "der_Cquat_by_v error too large"
        assert er_ag < A[-2], "der_CquatT_by_v error too large"
        assert er_ab < A[-2], "der_Ccrv_by_v error too large"
        assert er_ba < A[-2], "der_CcrvT_by_v error too large"
        assert er_ag_euler < A[-2], "der_Peuler_by_v error too large"
        assert er_ga_euler < A[-2], "der_Ceuler_by_v error too large"

    def test_crv_tangential_operator(self):
        """Checks Cartesian rotation vector tangential operator"""

        # linearisation point
        fi0 = -np.pi / 6
        nv0 = np.array([1, 3, 1])
        nv0 = np.array([1, 0, 0])
        nv0 = nv0 / np.linalg.norm(nv0)
        fv0 = fi0 * nv0
        Cab = algebra.crv2rotation(fv0)  # fv0 is rotation from A to B

        # dummy
        fi1 = np.pi / 3
        nv1 = np.array([2, 4, 1])
        nv1 = nv1 / np.linalg.norm(nv1)
        fv1 = fi1 * nv1

        er_tan = 10.0
        A = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        for a in A:
            # perturbed
            fv = a * fv1 + (1.0 - a) * fv0
            dfv = fv - fv0

            ### Compute relevant quantities
            dCab = algebra.crv2rotation(fv0 + dfv) - Cab
            T = algebra.crv2tan(fv0)
            Tdfv = np.dot(T, dfv)
            Tdfv_skew = algebra.skew(Tdfv)
            dCab_an = np.dot(Cab, Tdfv_skew)

            er_tan_new = np.max(np.abs(dCab - dCab_an)) / np.max(np.abs(dCab_an))
            assert er_tan_new < er_tan, "crv2tan error not converging to 0"
            er_tan = er_tan_new

        assert er_tan < A[-2], "crv2tan error too large"

    def test_crv_tangetial_operator_derivative(self):
        """Checks Cartesian rotation vector tangential operator"""

        # linearisation point
        fi0 = np.pi / 6
        nv0 = np.array([1, 3, 1])
        nv0 = nv0 / np.linalg.norm(nv0)
        fv0 = fi0 * nv0
        T0 = algebra.crv2tan(fv0)

        # dummy vector
        xv = np.ones((3,))
        T0xv = np.dot(T0, xv)
        # derT_an=dTxv(fv0,xv)
        derT_an = algebra.der_Tan_by_xv(fv0, xv)
        # derT_an=algebra.der_Tan_by_xv_an(fv0,xv)
        # dummy
        fi1 = np.pi / 3
        nv1 = np.array([4, 1, -2])
        nv1 = nv1 / np.linalg.norm(nv1)
        fv1 = fi1 * nv1

        er = 10.0
        A = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        for a in A:
            # perturbed
            fv = a * fv1 + (1.0 - a) * fv0
            dfv = fv - fv0
            Tpert = algebra.crv2tan(fv)
            Tpertxv = np.dot(Tpert, xv)
            dT_num = Tpertxv - T0xv
            dT_an = np.dot(derT_an, dfv)

            er_new = np.max(np.abs(dT_num - dT_an)) / np.max(np.abs(dT_an))
            assert er_new < er, "der_Tan_by_xv error not converging to 0"
            er = er_new

        assert er < A[-2], "der_Tan_by_xv error too large"

    def test_crv_tangetial_operator_transpose_derivative(self):
        """Checks Cartesian rotation vector tangential operator transpose"""

        # dummy vector
        xv = np.random.rand(3)

        # linearisation point
        fi0 = 2.0 * np.pi * np.random.rand(1)
        nv0 = np.random.rand(3)
        nv0 = nv0 / np.linalg.norm(nv0)
        fv0 = fi0 * nv0
        T0_T = np.transpose(algebra.crv2tan(fv0))
        T0_Txv = np.dot(T0_T, xv)

        # Analytical solution
        derT_T_an = algebra.der_TanT_by_xv(fv0, xv)

        # End point
        fi1 = 2.0 * np.pi * np.random.rand()
        nv1 = np.random.rand(3)
        nv1 = nv1 / np.linalg.norm(nv1)
        fv1 = fi1 * nv1

        er = 10.0
        A = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        for a in A:
            # perturbed
            fv = a * fv1 + (1.0 - a) * fv0
            dfv = fv - fv0
            Tpert_T = np.transpose(algebra.crv2tan(fv))
            Tpert_Txv = np.dot(Tpert_T, xv)
            dT_T_num = Tpert_Txv - T0_Txv
            dT_T_an = np.dot(derT_T_an, dfv)

            # Error
            er_new = np.max(np.abs(dT_T_num - dT_T_an)) / np.max(np.abs(dT_T_an))
            assert er_new < er, "der_TanT_by_xv error not converging to 0"
            er = er_new

        assert er < A[-2], "der_TanT_by_xv error too large"

    def test_quat_wrt_rot(self):
        """
        We define:
        - G: initial frame
        - A: frame obtained upon rotation, Cga, defined by the quaternion q0
        - B: frame obtained upon further rotation, Cab, of A defined by
        the "infinitesimal" Cartesian rotation vector dcrv
        The test verifies that:
        1. the total rotation matrix Cgb(q0+dq) is equal to
            Cgb = Cga(q0) Cab(dcrv)
        where
            dq = algebra.der_quat_wrt_crv(q0)
        2. the difference between analytically computed delta quaternion, dq,
        and the numerical delta
            dq_num = algebra.crv2quat(algebra.rotation2crv(Cgb_ref))-q0
        is comparable to the step used to compute the delta dcrv
        3. The equality:
            d(Cga(q0)*v)*dq = Cga(q0) * d(Cab*dv)*dcrv
        where d(Cga(q0)*v) and d(Cab*dv) are the derivatives computed through
            algebra.der_Cquat_by_v and algebra.der_Ccrv_by_v
        for a random vector v.

        Warning:
        - The relation dcrv->dquat is not uniquely defined. However, the
        resulting rotation matrix is, namely:
            Cga(q0+dq)=Cga(q0)*Cab(dcrv)
        """

        ### case 1: simple rotation about the same axis

        # linearisation point
        a0 = 30.0 * np.pi / 180
        n0 = np.array([0, 0, 1])
        n0 = n0 / np.linalg.norm(n0)
        q0 = algebra.crv2quat(a0 * n0)
        Cga = algebra.quat2rotation(q0)

        # direction of perturbation
        n2 = n0

        A = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        for a in A:
            drot = a * n2

            # build rotation manually
            atot = a0 + a
            Cgb_exp = algebra.crv2rotation(atot * n0)  # ok

            # build combined rotation
            Cab = algebra.crv2rotation(drot)
            Cgb_ref = np.dot(Cga, Cab)

            # verify expected vs combined rotation matrices
            assert (
                np.linalg.norm(Cgb_exp - Cgb_ref) / a < 1e-8
            ), "Verify test case - these matrices need to be identical"

            # verify analytical rotation matrix
            dq_an = np.dot(algebra.der_quat_wrt_crv(q0), drot)
            Cgb_an = algebra.quat2rotation(q0 + dq_an)
            erel_rot = np.linalg.norm(Cgb_an - Cgb_ref) / a
            assert erel_rot < 3e-3, (
                "Relative error of rotation matrix (%.2e) too large!" % erel_rot
            )

            # verify delta quaternion
            erel_dq = np.linalg.norm(Cgb_an - Cgb_ref)
            dq_num = algebra.crv2quat(algebra.rotation2crv(Cgb_ref)) - q0
            erel_dq = np.linalg.norm(dq_num - dq_an) / np.linalg.norm(dq_an) / a
            assert erel_dq < 0.3, (
                "Relative error delta quaternion (%.2e) too large!" % erel_dq
            )

            # verify algebraic relation
            v = np.ones((3,))
            D1 = algebra.der_Cquat_by_v(q0, v)
            D2 = algebra.der_Ccrv_by_v(np.zeros((3,)), v)
            res = np.dot(D1, dq_num) - np.dot(np.dot(Cga, D2), drot)
            erel_res = np.linalg.norm(res) / a
            assert erel_res < 5e-1 * a, (
                "Relative error of residual (%.2e) too large!" % erel_res
            )

        ### case 2: random rotation

        # linearisation point
        a0 = 30.0 * np.pi / 180
        n0 = np.array([-2, -1, 1])
        n0 = n0 / np.linalg.norm(n0)
        q0 = algebra.crv2quat(a0 * n0)
        Cga = algebra.quat2rotation(q0)

        # direction of perturbation
        n2 = np.array([0.5, 1.0, -2.0])
        n2 = n2 / np.linalg.norm(n2)

        A = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        for a in A:
            drot = a * n2

            # build combined rotation
            Cab = algebra.crv2rotation(drot)
            Cgb_ref = np.dot(Cga, Cab)

            # verify analytical rotation matrix
            dq_an = np.dot(algebra.der_quat_wrt_crv(q0), drot)
            Cgb_an = algebra.quat2rotation(q0 + dq_an)
            erel_rot = np.linalg.norm(Cgb_an - Cgb_ref) / a
            assert erel_rot < 3e-3, (
                "Relative error of rotation matrix (%.2e) too large!" % erel_rot
            )

            # verify delta quaternion
            erel_dq = np.linalg.norm(Cgb_an - Cgb_ref)
            dq_num = algebra.crv2quat(algebra.rotation2crv(Cgb_ref)) - q0
            erel_dq = np.linalg.norm(dq_num - dq_an) / np.linalg.norm(dq_an) / a
            assert erel_dq < 0.3, (
                "Relative error delta quaternion (%.2e) too large!" % erel_dq
            )

            # verify algebraic relation
            v = np.ones((3,))
            D1 = algebra.der_Cquat_by_v(q0, v)
            D2 = algebra.der_Ccrv_by_v(np.zeros((3,)), v)
            res = np.dot(D1, dq_num) - np.dot(np.dot(Cga, D2), drot)
            erel_res = np.linalg.norm(res) / a
            assert erel_res < 5e-1 * a, (
                "Relative error of residual (%.2e) too large!" % erel_res
            )

    def test_rotation_about_axis(self):
        """
        Tests the rotations about the cartesian axes
        """
        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        k = np.array([0, 0, 1])

        angle = 90 * np.pi / 180

        # Testing rotations about the x axis - rotation3d_x
        out = algebra.rotation3d_x(angle).dot(j)
        for ax in range(3):
            self.assertAlmostEqual(out[ax], k[ax])

        out = algebra.rotation3d_x(angle).dot(k)
        for ax in range(3):
            self.assertAlmostEqual(out[ax], -j[ax])

        # Testing rotations about the y axis - rotation3d_y
        out = algebra.rotation3d_y(angle).dot(i)
        for ax in range(3):
            self.assertAlmostEqual(out[ax], -k[ax])

        out = algebra.rotation3d_y(angle).dot(k)
        for ax in range(3):
            self.assertAlmostEqual(out[ax], i[ax])

        # Testing rotations about the z axis - rotation3d_z
        out = algebra.rotation3d_z(angle).dot(i)
        for ax in range(3):
            self.assertAlmostEqual(out[ax], j[ax])

        out = algebra.rotation3d_z(angle).dot(j)
        for ax in range(3):
            self.assertAlmostEqual(out[ax], -i[ax])

    def test_rotation_G_to_A(self):
        """
        Tests the rotation of a vector in G frame to a vector in A frame by a roll, pitch and yaw considering that
        SHARPy employs an SEU frame.

        In the inertial frame, the ``x_g`` axis is aligned with the longitudinal axis of the aircraft pointing backwards,
        ``z_g`` points upwards and ``y_g`` completes the set
        Returns:

        """

        aircraft_nose = np.array([-1, 0, 0])
        z_A = np.array([0, 0, 1])
        euler = np.array([90, 90, 90]) * np.pi / 180
        quat = algebra.euler2quat(euler)

        aircraft_nose_rotated = np.array([0, 0, 1])  # in G frame pointing upwards
        z_A_rotated_g = np.array([1, 0, 0])

        Cga = algebra.euler2rot(euler)
        Cga_quat = algebra.quat2rotation(quat)

        np.testing.assert_array_almost_equal(
            Cga.dot(aircraft_nose),
            aircraft_nose_rotated,
            err_msg="Rotation using Euler angles not performed properly",
        )
        np.testing.assert_array_almost_equal(
            Cga_quat.dot(aircraft_nose),
            aircraft_nose_rotated,
            err_msg="Rotation using quaternions not performed properly",
        )
        np.testing.assert_array_almost_equal(
            Cga.dot(z_A),
            z_A_rotated_g,
            err_msg="Rotation using Euler angles not performed properly",
        )
        np.testing.assert_array_almost_equal(
            Cga_quat.dot(z_A),
            z_A_rotated_g,
            err_msg="Rotation using quaternions not performed properly",
        )

        # Check projections
        Pag = Cga.T
        Pag_quat = Cga_quat.T

        np.testing.assert_array_almost_equal(
            Pag.dot(z_A_rotated_g),
            z_A,
            err_msg="Error in projection from A to G using Euler angles",
        )
        np.testing.assert_array_almost_equal(
            Pag.dot(aircraft_nose_rotated),
            aircraft_nose,
            err_msg="Error in projection from A to G using Euler angles",
        )
        np.testing.assert_array_almost_equal(
            Pag_quat.dot(z_A_rotated_g),
            z_A,
            err_msg="Error in projection from A to G using quaternions",
        )
        np.testing.assert_array_almost_equal(
            Pag_quat.dot(aircraft_nose_rotated),
            aircraft_nose,
            err_msg="Error in projection from A to G using quaternions",
        )


# if __name__=='__main__':
# unittest.main()
# # T=TestAlgebra()
# # # T.setUp()
# # T.test_rotation_vectors_conversions()
# # T.test_rotation_matrices()
# # T.test_rotation_matrices_derivatives()
# # T.test_crv_tangetial_operator()
# # T.test_crv_tangetial_operator_derivative()
# # T.test_crv_tangetial_operator_transpose_derivative()
