from typing import Callable, Any, Optional, Type, cast
from abc import ABC
import numpy as np
import scipy as sp

import jax
import jax.scipy.spatial.transform
import jax.numpy as jnp
from jax.numpy import ndarray as jarr

# global constants
jax.config.update("jax_enable_x64", True)
DICT_OF_LC = dict()
USE_JIT = True

# type definition for b subfunctions (Constraint, i_lm, b_mat, q, q_dot, u, u_dot)
b_type: Type = Optional[Callable[[Any, slice, jarr, jarr, jarr, jarr, jarr], jarr]]

# type definitions for  b functions, ordering is (q, q_dot, u, u_dot, lmh, lmn)
func_type: Type = Callable[[jarr, jarr, jarr, jarr, jarr, jarr], jarr]


# redefine jax.scipy rotation class to use [real, i, j, k] ordering by default
class Rot(jax.scipy.spatial.transform.Rotation):
    @classmethod
    def from_quat(cls, quat: jarr):
        return super().from_quat(jnp.array((*quat[1:4], quat[0])))

    def as_quat(self, canonical=True) -> jarr:
        return super().as_quat(canonical=canonical)


# decorator populating DICT_OF_LC as {lc_id: Constraint}, not type hinted as preceeds Constraint declaration
def constraint(constraint_):
    global DICT_OF_LC
    if constraint_.lc_id is not None:
        DICT_OF_LC[constraint_.lc_id] = constraint_
    else:
        raise AttributeError('Class defined as lagrange constraint has no lc_id attribute')
    return constraint_


# JAX vector skew
def skew(vec: jarr) -> jarr:
    if vec.shape != (3,) or jnp.iscomplexobj(vec):
        raise ValueError("Incompatible input vector dimention or data type")
    return jnp.array(((0., -vec[2], vec[1]),
                      (vec[2], 0., -vec[0]),
                      (-vec[1], vec[0], 0.)))


# JAX cartesian rotation vector to rotation matrix as version from SciPy's Rotation class doesn't differentiate well
def crv2rot(crv: jarr) -> jarr:
    ang = jnp.linalg.norm(crv)
    crv_skew = skew(crv)
    return jax.lax.cond(ang > 1e-15,
                        lambda: jnp.eye(3) + jnp.sin(ang) / ang * crv_skew
                                + (1 - jnp.cos(ang)) / ang ** 2 * crv_skew @ crv_skew,
                        lambda: jnp.eye(3) + crv_skew + 0.5 * crv_skew @ crv_skew)


# JAX cartesian rotation vector tangent operator
def crv2tan(psi: jarr) -> jarr:
    ang = jnp.linalg.norm(psi)
    psi_skew = skew(psi)
    return jax.lax.cond(ang > 1e-8,
                        lambda: jnp.eye(3) + (jnp.cos(ang) - 1.) / ang ** 2 * psi_skew + (ang - jnp.sin(ang))
                                / ang ** 3 * psi_skew @ psi_skew,
                        lambda: jnp.eye(3) - 0.5 * psi_skew + psi_skew @ psi_skew / 6.)


@constraint
class Constraint(ABC):
    # this might make it a bit faster, might be overkill
    __slots__ = ('settings', 's_fact', 'p_fact', 'use_p_fact', 'jac_func', 'num_h_funcs', 'num_n_funcs', 'i_first_lm',
                 'num_lmh', 'num_lmn', 'num_lm', 'num_lm_tot', 'num_bodys', 'is_free', 'num_elem_body', 'num_node_body',
                 'num_dof_body', 'num_dof_tot', 'num_eq_tot', 'i_sys', 'i_lm_tot', 'i_lmh_eq', 'i_lmn_eq', 'i_lm_eq',
                 'i_v0', 'i_omega0', 'i_quat0', 'i_v1', 'i_omega1', 'i_quat1', 'i_r0', 'i_psi0', 'q_i_global',
                 'num_active_q', 'bh', 'bn', 'gn', 'c', 'd', 'bht', 'bnt', 'bht_lmh', 'bnt_lmn', 'p_func_h', 'p_func_n',
                 'run')

    lc_id = '_base_constraint'
    required_params: tuple[str, ...] = tuple()  # required parameters for the given constraint
    bh_funcs: tuple[b_type, ...] = list()  # tuple of holonomic B matrix funcs
    bn_funcs: tuple[b_type | None, ...] = tuple()  # tuple of non holonomic B matrix funcs, or None for zero matrix
    gn_funcs: tuple[b_type | None, ...] = tuple()  # tuple of non holonomic g matrix funcs, or None for zero matrix
    num_lmh_eq: tuple[int, ...] = tuple()  # number of LMs for each holonomic constraint
    num_lmn_eq: tuple[int, ...] = tuple()  # number of LMs for each nonholonomic constraint
    postproc_funcs: tuple[Callable, ...] = tuple()  # postprocessing functions to be run

    def __init__(self, data, i_lm: int, constraint_settings: dict): # case data, index of first LM, input settings
        self.settings = constraint_settings  # input constraint parameters

        # scaling factor, with 1/dt^2 as default
        self.s_fact = constraint_settings.get('scaling_factor', data.data.settings['DynamicCoupled']['dt'] ** -2)
        self.p_fact = constraint_settings.get('penalty_factor', 0.)  # penalty factor for constraint
        self.use_p_fact = abs(self.p_fact) > 1e-6   # if true, penalty equations are included in run function

        # choose either forward or reverse mode autodifferentiation
        # I believe that only forward works as of current, and for this case the difference should be negligible
        match data.settings['jacobian_method']:
            case 'forward':
                self.jac_func = jax.jacfwd
            case 'reverse':
                self.jac_func = jax.jacrev

        self.num_h_funcs = len(self.bh_funcs)           # number of holonomic functions (this constraint)
        self.num_n_funcs = len(self.bn_funcs)           # number of non holonomic functions (this constraint)
        self.i_first_lm = i_lm                          # index of first LM (this constraint)
        self.num_lmh = sum(self.num_lmh_eq)             # total number of holonomic LMs (this constraint)
        self.num_lmn = sum(self.num_lmn_eq)             # total number of non holonomic LMs (this constraint)
        self.num_lm = self.num_lmh + self.num_lmn       # total number of LMs (this constraint)
        self.num_lm_tot = data.num_lm_tot               # total numer of LMs (all constraints)

        self.num_bodys = data.data.structure.ini_mb_dict['num_bodies']  # number of bodies
        self.is_free = [data.data.structure.ini_mb_dict[f'body_{i:02d}']['FoR_movement'] == 'free'
                        for i in range(self.num_bodys)]  # boolean for beam freedom condition

        self.num_elem_body = [list(data.data.structure.body_number).count(i)
                              for i in range(self.num_bodys)]  # number of elements per body
        self.num_node_body = [self.num_elem_body[i] * (data.data.structure.num_node_elem - 1) + 1
                              for i in range(self.num_bodys)]  # number of nodes per body
        self.num_dof_body = [(self.num_node_body[i] - 1) * 6 + self.is_free[i] * 10
                             for i in range(self.num_bodys)]  # number of DoFs per body

        self.num_dof_tot = data.sys_size                     # number of equations representing the unconstrained system
        self.num_eq_tot = self.num_dof_tot + self.num_lm_tot  # number of equations representing the constrainted system

        self.i_sys = slice(0, self.num_dof_tot)  # index of equations representing the unconstrained system

        self.i_lm_tot = jnp.arange(self.num_dof_tot, self.num_dof_tot + self.num_lm_tot)  # index of LMs in full system

        self.i_lmh_eq: list[slice] = []  # list of slices of holonomic LMs for this constraint
        self.i_lmn_eq: list[slice] = []  # list of slices of non holonomic LMs for this constraint

        i_eq = self.i_first_lm
        for i in range(len(self.num_lmh_eq)):
            self.i_lmh_eq.append(slice(i_eq, i_eq + self.num_lmh_eq[i]))
            i_eq += self.num_lmh_eq[i]
        for i in range(len(self.num_lmn_eq)):
            self.i_lmn_eq.append(slice(i_eq, i_eq + self.num_lmn_eq[i]))
            i_eq += self.num_lmn_eq[i]
        self.i_lm_eq = self.i_lmh_eq + self.i_lmn_eq  # list of slices of LMs for this constraint, holonomic first

        # the below indexes are slices of q and q_dot in the subset of active q terms
        self.i_v0: Optional[slice] = None        # index of body 0 linear velocity
        self.i_omega0: Optional[slice] = None    # index of body 0 rotational velocity
        self.i_quat0: Optional[slice] = None     # index of body 0 orientation quaternion
        self.i_v1: Optional[slice] = None        # index of body 1 linear velocity
        self.i_omega1: Optional[slice] = None    # index of body 1 rotational velocity
        self.i_quat1: Optional[slice] = None     # index of body 1 orientation quaternion
        self.i_r0: Optional[slice] = None        # index of body 0 node position
        self.i_psi0: Optional[slice] = None      # index of body 0 node orientation
        self.q_i_global: Optional[jnp.ndarray] = None     # index of q used in constraints (global)
        self.num_active_q: Optional[int] = None     # number of elements of q required in constraint
        self.create_index()                      # assign all required indexes

        self.bh: func_type = self.create_bh()                                          # overall holonomic b function
        self.bn: func_type = self.create_bn()                                          # overall nonholonomic b function
        self.gn: func_type = self.create_gn()                                          # overall nonholonomic g function

        # cannot type hint lambda expressions, so casting the type will do
        self.c: func_type = cast(func_type, lambda *args: self.bh(*args) @ args[0])                    # C function
        self.d: func_type = cast(func_type, lambda *args: self.bn(*args) @ args[1] + self.gn(*args))   # D function
        self.bht: func_type = cast(func_type, lambda *args: self.bh(*args).T)                          # B_h^T
        self.bnt: func_type = cast(func_type, lambda *args: self.bn(*args).T)                          # B_n^T
        self.bht_lmh: func_type = cast(func_type, lambda *args: self.bht(*args) @ args[4])             # B_h^T @ lm_h
        self.bnt_lmn: func_type = cast(func_type, lambda *args: self.bnt(*args) @ args[5])             # B_n^T @ lm_n
        self.p_func_h: func_type = cast(func_type, lambda *args: self.bht(*args) @ self.c(*args))
        self.p_func_n: func_type = cast(func_type, lambda *args: self.bnt(*args) @ self.d(*args))

        # create function which returns contribution to structural equations
        self.run: Callable[[jarr, jarr, jarr, jarr, jarr, jarr], tuple[jarr, jarr, jarr]] = self.create_run()

    def postprocess(self, mb_beam, mb_tstep) -> None:
        for func in self.postproc_funcs:
            func(self, mb_beam, mb_tstep)

    @classmethod
    def get_n_lm(cls) -> int:
        return sum(cls.num_lmh_eq) + sum(cls.num_lmn_eq)

    def create_index(self) -> None:
        # check all required parameters are in settings
        for param in self.required_params:
            if param not in self.settings.keys():
                raise KeyError(f"Parameter {param} is undefined in constraint settings")

        i_global = []   # index in full q
        i_count = 0     # start of current slice

        if 'body' in self.required_params:
            i_for0 = self.settings['body']
            i_start_for0 = sum(self.num_dof_body[:i_for0 + 1]) - 10
            self.i_v0 = np.arange(0, 3, dtype=int)
            self.i_omega0 = np.arange(3, 6, dtype=int)
            self.i_quat0 = np.arange(6, 10, dtype=int)
            i_global.append(np.arange(i_start_for0, i_start_for0 + 10))
            i_count += 10

        if 'node_in_body' in self.required_params:
            i_node = self.settings['node_in_body']
            self.i_r0 = np.arange(i_count, i_count + 3, dtype=int)
            self.i_psi0 = np.arange(i_count + 3, i_count + 6, dtype=int)
            i_global.append(jnp.arange((i_node - 1) * 6, (i_node - 1) * 6 + 6))
            i_count += 6

        if 'body_FoR' in self.required_params:
            i_for1 = self.settings['body_FoR']
            i_start_for1 = sum(self.num_dof_body[:i_for1 + 1]) - 10
            self.i_v1 = np.arange(i_count, i_count + 3, dtype=int)
            self.i_omega1 = np.arange(i_count + 3, i_count + 6, dtype=int)
            self.i_quat1 = np.arange(i_count + 6, i_count + 10, dtype=int)
            i_global.append(np.arange(i_start_for1, i_start_for1 + 10))
            i_count += 10

        self.q_i_global = jnp.hstack(i_global)
        self.num_active_q = self.q_i_global.shape[0]

    def create_bh(self) -> func_type:
        def bh(*args: jarr) -> jarr:
            bh_mat = jnp.zeros((self.num_lm_tot, self.num_active_q))
            for i_func, bh_func in enumerate(self.bh_funcs):
                bh_mat = bh_func(self, self.i_lmh_eq[i_func], bh_mat, *args[:4])
            return bh_mat

        return bh

    def create_bn(self) -> func_type:
        def bn(*args: jarr) -> jarr:
            bn_mat = jnp.zeros((self.num_lm_tot, self.num_active_q))
            for i_func, bn_func in enumerate(self.bn_funcs):
                if bn_func is not None:
                    bn_mat = bn_func(self, self.i_lmn_eq[i_func], bn_mat, *args[:4])
            return bn_mat

        return bn

    def create_gn(self) -> func_type:
        def gn(*args: jarr) -> jarr:
            gn_mat = jnp.zeros(self.num_lm_tot)
            for i_func, gn_func in enumerate(self.gn_funcs):
                if gn_func is not None:
                    gn_mat = gn_func(self, self.i_lmn_eq[i_func], gn_mat, *args[:4])
            return gn_mat

        return gn

    def create_run(self):
        def run(q: jarr, q_dot: jarr, u: jarr, u_dot: jarr, lmh: jarr, lmn: jarr) -> tuple[jarr, jarr, jarr]:
            args = (q, q_dot, u, u_dot, lmh, lmn)

            c = jnp.zeros((self.num_eq_tot, self.num_eq_tot))
            c = c.at[jnp.ix_(self.q_i_global, self.q_i_global)].add(self.jac_func(self.bht_lmh, 1)(*args))
            c = c.at[jnp.ix_(self.q_i_global, self.q_i_global)].add(self.jac_func(self.bnt_lmn, 1)(*args))
            c = c.at[jnp.ix_(self.q_i_global, self.i_lm_tot)].add(self.bnt(*args))
            c = c.at[jnp.ix_(self.i_lm_tot, self.q_i_global)].add(self.s_fact * (self.jac_func(self.c, 1)(*args)))
            c = c.at[jnp.ix_(self.i_lm_tot, self.q_i_global)].add(self.s_fact * (self.jac_func(self.d, 1)(*args)))

            k = jnp.zeros((self.num_eq_tot, self.num_eq_tot))
            k = k.at[jnp.ix_(self.q_i_global, self.q_i_global)].add(self.jac_func(self.bht_lmh, 0)(*args))
            k = k.at[jnp.ix_(self.q_i_global, self.q_i_global)].add(self.jac_func(self.bnt_lmn, 0)(*args))
            k = k.at[jnp.ix_(self.q_i_global, self.i_lm_tot)].add(self.bht(*args))
            k = k.at[jnp.ix_(self.i_lm_tot, self.q_i_global)].add(self.s_fact * (self.jac_func(self.c, 0)(*args)))
            k = k.at[jnp.ix_(self.i_lm_tot, self.q_i_global)].add(self.s_fact * (self.jac_func(self.d, 0)(*args)))

            rhs = jnp.zeros(self.num_eq_tot)
            rhs = rhs.at[self.q_i_global].add(self.bht_lmh(*args) + self.bnt_lmn(*args))
            rhs = rhs.at[self.i_lm_tot].add(self.s_fact * (self.c(*args) + self.d(*args)))

            if self.use_p_fact:
                c = c.at[jnp.ix_(self.q_i_global, self.q_i_global)].add(self.p_fact * (self.jac_func(self.p_func_h, 1)(*args)))
                c = c.at[jnp.ix_(self.q_i_global, self.q_i_global)].add(self.p_fact * (self.jac_func(self.p_func_n, 1)(*args)))
                k = k.at[jnp.ix_(self.q_i_global, self.q_i_global)].add(self.p_fact * (self.jac_func(self.p_func_h, 0)(*args)))
                k = k.at[jnp.ix_(self.q_i_global, self.q_i_global)].add(self.p_fact * (self.jac_func(self.p_func_n, 0)(*args)))
                rhs = rhs.at[self.q_i_global].add(self.p_fact * (self.p_func_h(*args) + self.p_func_n(*args)))
            return c, k, rhs

        return run


def combine_constraints(csts: list[Constraint]) -> Callable:
    def combined_run(q: jarr, q_dot: jarr, u: list[None | jarr, ...], u_dot: list[None | jarr, ...], lmh: jarr,
                     lmn: jarr) -> tuple[jarr, jarr, jarr]:
        q_i = [cst.q_i_global for cst in csts]

        n_cst = len(csts)
        out_mats = [csts[i_cst].run(q[q_i[i_cst]], q_dot[q_i[i_cst]], u[i_cst], u_dot[i_cst], lmh, lmn)
                    for i_cst in range(n_cst)]
        c = jnp.sum(jnp.array([out_mats[i][0] for i in range(n_cst)]), axis=0)
        k = jnp.sum(jnp.array([out_mats[i][1] for i in range(n_cst)]), axis=0)
        rhs = jnp.sum(jnp.array([out_mats[i][2] for i in range(n_cst)]), axis=0)
        return c, k, rhs

    return jax.jit(combined_run) if USE_JIT else combined_run


class BaseFunc(ABC):
    n_eq = 3


class CstFuncs(ABC):
    class EqualNodeFoR(BaseFunc):
        @staticmethod
        def b_lin_vel(constraint_: Constraint, i_lm: slice, b: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            b = b.at[i_lm, constraint_.i_r0].add(
                -Rot.from_quat(q_dot[constraint_.i_quat0]).as_matrix())
            b = b.at[i_lm, constraint_.i_v0].add(
                -Rot.from_quat(q_dot[constraint_.i_quat0]).as_matrix())
            b = b.at[i_lm, constraint_.i_v1].add(
                Rot.from_quat(q_dot[constraint_.i_quat1]).as_matrix())
            b = b.at[i_lm, constraint_.i_omega0].add(
                Rot.from_quat(q_dot[constraint_.i_quat0]).as_matrix() @ skew(q[constraint_.i_r0]))
            return b

        @staticmethod
        def b_ang_vel(constraint_: Constraint, i_lm: slice, b: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            b = b.at[i_lm, constraint_.i_psi0].add(crv2tan(q[constraint_.i_psi0]))
            b = b.at[i_lm, constraint_.i_omega1].add(-crv2rot(q[constraint_.i_psi0]).T
                                                     @ Rot.from_quat(q_dot[constraint_.i_quat0]).as_matrix().T
                                                     @ Rot.from_quat(q_dot[constraint_.i_quat1]).as_matrix())
            b = b.at[i_lm, constraint_.i_omega0].add(crv2rot(q[constraint_.i_psi0]).T)
            return b

    class ControlFoR(BaseFunc):
        @staticmethod
        def b_ang_vel(constraint_: Constraint, i_lm: slice, b: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            return b.at[i_lm, constraint_.i_omega1].add(jnp.eye(3))

        @staticmethod
        def g_ang_vel(constraint_: Constraint, i_lm: slice, g: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            return g.at[i_lm].add(-crv2tan(u) @ u_dot)

    class ControlNodeFoR(EqualNodeFoR):
        @staticmethod
        def g_ang_vel(constraint_: Constraint, i_lm: slice, g: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            return g.at[i_lm].add(crv2rot(u).T @ crv2tan(u) @ u_dot)

    class ZeroFoR(BaseFunc):
        @staticmethod
        def b_lin_vel(constraint_: Constraint, i_lm: slice, b: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            return b.at[i_lm, constraint_.i_v1].add(jnp.eye(3))

        @staticmethod
        def b_ang_vel(constraint_: Constraint, i_lm: slice, b: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            return b.at[i_lm, constraint_.i_omega1].add(jnp.eye(3))

    class HingeNodeFoR(BaseFunc):
        n_eq = 2

        @staticmethod
        def b_ang_vel(constraint_: Constraint, i_lm: slice, b: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            r_ax_a = jnp.array(constraint_.settings['rot_axisA2'])
            r_ax_a /= jnp.linalg.norm(r_ax_a)
            a_skew = skew(r_ax_a)

            r_ax_b = jnp.array(constraint_.settings['rot_axisB'])
            r_ax_b /= jnp.linalg.norm(r_ax_b)
            b_skew = skew(r_ax_b)

            rot_psi = crv2rot(q[constraint_.i_psi0])

            aux = (rot_psi.T @ Rot.from_quat(q[constraint_.i_quat0]).as_matrix().T
                   @ Rot.from_quat(q[constraint_.i_quat1]).as_matrix()) @ a_skew

            aux_norms = jnp.linalg.norm(aux, axis=1)
            dirs = jnp.array(((1, 2), (0, 2), (0, 1)))[jnp.argmin(aux_norms)]

            b = b.at[i_lm, constraint_.i_omega0].add((b_skew @ rot_psi.T)[dirs, :])
            b = b.at[i_lm, constraint_.i_psi0].add((b_skew @ crv2tan(q[constraint_.i_psi0]))[dirs, :])
            b = b.at[i_lm, constraint_.i_omega1].add(-aux[dirs, :])
            return b

    class HingeFoR(BaseFunc):
        n_eq = 2

        @classmethod
        def b_ang_vel(cls, constraint_: Constraint, i_lm: slice, b: jarr, q: jarr, q_dot: jarr, u: jarr, u_dot: jarr) \
                -> jarr:
            axis = jnp.array(constraint_.settings['rot_axis_AFoR'])
            axis = axis / jnp.linalg.norm(axis)
            axis_skew = skew(axis)
            dirs = jnp.array(((1, 2), (0, 2), (0, 1)))[jnp.argmax(axis)]  # directions for axis matrix
            i_omega1_indep = jnp.arange(constraint_.i_omega1[0], constraint_.i_omega1[2] + 1)[dirs]

            b = jax.lax.cond(jnp.abs(jnp.linalg.norm(axis) - 1.) < 1e-6,
                             lambda: b.at[i_lm, i_omega1_indep].add(jnp.eye(2)),
                             lambda: b.at[i_lm, constraint_.i_omega1].add(axis_skew[dirs, :]))
            return b


def move_for_to_node(constraint_: Constraint, mb_beam, mb_tstep) -> None:
    i_body_node = constraint_.settings['body']  # beam number which contains node
    i_node = constraint_.settings['node_in_body']  # node number in beam
    i_body_for = constraint_.settings['body_FoR']
    rel_pos_b = constraint_.settings.get('rel_posB', np.zeros(3))

    i_elem, i_node_in_elem = mb_beam[i_body_node].node_master_elem[i_node]
    c_ga = mb_tstep[i_body_node].cga()
    c_ab = sp.spatial.transform.Rotation.from_rotvec(mb_tstep[i_body_node].psi[i_elem, i_node_in_elem, :]).as_matrix()

    mb_tstep[i_body_for].for_pos[:3] = (c_ga @ (mb_tstep[i_body_node].pos[i_node, :] + c_ab @ rel_pos_b)
                                        + mb_tstep[i_body_node].for_pos[:3])


@constraint
class FullyConstrainedNodeFoR(Constraint):
    lc_id = 'fully_constrained_node_FoR'
    required_params = ('node_in_body', 'body', 'body_FoR')
    bn_funcs = (CstFuncs.EqualNodeFoR.b_lin_vel, CstFuncs.EqualNodeFoR.b_ang_vel)
    gn_funcs = (None, None)
    num_lmn_eq = (CstFuncs.EqualNodeFoR.n_eq, CstFuncs.EqualNodeFoR.n_eq)
    postproc_funcs = (move_for_to_node,)


@constraint
class FullyConstrainedFoR(Constraint):
    lc_id = 'fully_constrained_FoR'
    required_params = ('body_FoR',)
    bn_funcs = (CstFuncs.ZeroFoR.b_lin_vel, CstFuncs.ZeroFoR.b_ang_vel)
    gn_funcs = (None, None)
    num_lmn_eq = (CstFuncs.ZeroFoR.n_eq, CstFuncs.ZeroFoR.n_eq)


@constraint
class SphericalFoR(Constraint):
    lc_id = 'spherical_FoR'
    required_params = ('body_FoR',)
    bn_funcs = (CstFuncs.ZeroFoR.b_lin_vel,)
    gn_funcs = (None,)
    num_lmn_eq = (CstFuncs.ZeroFoR.n_eq,)


@constraint
class Free(Constraint):
    lc_id = 'free'


@constraint
class ControlledRotNodeFoR(Constraint):
    lc_id = 'control_node_FoR_rot_vel'
    required_params = ('controller_id', 'node_in_body', 'body', 'body_FoR')
    bn_funcs = (CstFuncs.ControlNodeFoR.b_lin_vel, CstFuncs.ControlNodeFoR.b_ang_vel)
    gn_funcs = (None, CstFuncs.ControlNodeFoR.g_ang_vel)
    num_lmn_eq = (CstFuncs.ControlNodeFoR.n_eq, CstFuncs.ControlNodeFoR.n_eq)
    postproc_funcs = (move_for_to_node,)


@constraint
class ControlledRotFoR(Constraint):
    lc_id = 'control_rot_vel_FoR'
    required_params = ('controller_id', 'body_FoR')
    bn_funcs = (CstFuncs.ZeroFoR.b_lin_vel, CstFuncs.ControlFoR.b_ang_vel)
    gn_funcs = (None, CstFuncs.ControlFoR.g_ang_vel)
    num_lmn_eq = (CstFuncs.ZeroFoR.n_eq, CstFuncs.ControlFoR.n_eq)


@constraint
class HingeFoR(Constraint):
    lc_id = 'hinge_FoR'
    required_params = ('body_FoR', 'rot_axis_AFoR')
    bn_funcs = (CstFuncs.ZeroFoR.b_lin_vel, CstFuncs.HingeFoR.b_ang_vel)
    gn_funcs = (None, None)
    num_lmn_eq = (CstFuncs.ZeroFoR.n_eq, CstFuncs.HingeFoR.n_eq)


@constraint
class SphericalNodeFor(Constraint):
    lc_id = 'spherical_node_FoR'
    required_params = ('body', 'body_FoR', 'node_in_body')
    bn_funcs = (CstFuncs.EqualNodeFoR.b_lin_vel, )
    gn_funcs = (None, )
    num_lmn_eq = (CstFuncs.EqualNodeFoR.n_eq, )
    postproc_funcs = (move_for_to_node,)


@constraint
class HingeNodeFoR(Constraint):
    lc_id = 'hinge_node_FoR'
    required_params = ('body', 'body_FoR', 'node_in_body', 'rot_axisA2', 'rot_axisB')
    bn_funcs = (CstFuncs.EqualNodeFoR.b_lin_vel, CstFuncs.HingeNodeFoR.b_ang_vel)
    gn_funcs = (None, None)
    num_lmn_eq = (CstFuncs.EqualNodeFoR.n_eq, CstFuncs.HingeNodeFoR.n_eq)
    postproc_funcs = (move_for_to_node,)


if __name__ == '__main__':
    print("Available constraints:")
    for constraint in DICT_OF_LC.values():
        print(constraint.lc_id)
