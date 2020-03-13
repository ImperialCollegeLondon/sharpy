"""Force Mapping Utilities"""
import numpy as np
import sharpy.utils.algebra as algebra
import sharpy.aero.utils as aero_utils
from sharpy.utils.constants import deg2rad
import sharpy.aero.utils.airfoilpolars as ap
import sharpy.aero.utils.uvlmlib as uvlmlib
import ctypes as ct


def aero2struct_force_mapping(aero_forces,
                              struct2aero_mapping,
                              zeta,
                              pos_def,
                              psi_def,
                              master,
                              conn,
                              cag=np.eye(3),
                              aero_dict=None):
    r"""
    Maps the aerodynamic forces at the lattice to the structural nodes

    The aerodynamic forces from the UVLM are always in the inertial ``G`` frame of reference and have to be transformed
    to the body or local ``B`` frame of reference in which the structural forces are defined.

    Since the structural nodes and aerodynamic panels are coincident in a spanwise direction, the aerodynamic forces
    that correspond to a structural node are the summation of the ``M+1`` forces defined at the lattice at that
    spanwise location.

    .. math::
        \mathbf{f}_{struct}^B &= \sum\limits_{i=0}^{m+1}C^{BG}\mathbf{f}_{i,aero}^G \\
        \mathbf{m}_{struct}^B &= \sum\limits_{i=0}^{m+1}C^{BG}(\mathbf{m}_{i,aero}^G +
        \tilde{\boldsymbol{\zeta}}^G\mathbf{f}_{i, aero}^G)

    where :math:`\tilde{\boldsymbol{\zeta}}^G` is the skew-symmetric matrix of the vector between the lattice
    grid vertex and the structural node.

    It is possible to introduce efficiency and constant terms in the mapping of forces that are user-defined. For more
    info see :func:`~sharpy.aero.utils.mapping.efficiency_local_aero2struct_forces`.

    The efficiency and constant terms are introduced by means of the array ``airfoil_efficiency`` in the ``aero.h5``
    input file. If this variable has been defined, the function used to map the forces will be
    :func:`~sharpy.aero.utils.mapping.efficiency_local_aero2struct_forces`. Else, the standard formulation
    :func:`~sharpy.aero.utils.mapping.local_aero2struct_forces` will be used.

    Args:
        aero_forces (list): Aerodynamic forces from the UVLM in inertial frame of reference
        struct2aero_mapping (dict): Structural to aerodynamic node mapping
        zeta (list): Aerodynamic grid coordinates
        pos_def (np.ndarray): Vector of structural node displacements
        psi_def (np.ndarray): Vector of structural node rotations (CRVs)
        master: Unused
        conn (np.ndarray): Connectivities matrix
        cag (np.ndarray): Transformation matrix between inertial and body-attached reference ``A``
        aero_dict (dict): Dictionary containing the grid's information.

    Returns:
        np.ndarray: structural forces in an ``n_node x 6`` vector
    """

    n_node, _ = pos_def.shape
    n_elem, _, _ = psi_def.shape
    struct_forces = np.zeros((n_node, 6))

    nodes = []

    # load airfoil efficiency (if it exists); else set to one (to avoid multiple ifs in the loops)
    force_efficiency = None
    moment_efficiency = None
    struct2aero_force_function = local_aero2struct_forces

    if aero_dict is not None:
        try:
            airfoil_efficiency = aero_dict['airfoil_efficiency']
            # force efficiency dimensions [n_elem, n_node_elem, 2, [fx, fy, fz]] - all defined in B frame
            force_efficiency = np.zeros((n_elem, 3, 2, 3))
            force_efficiency[:, :, :, 1] = airfoil_efficiency[:, :, :, 0]
            force_efficiency[:, :, :, 2] = airfoil_efficiency[:, :, :, 1]

            # moment efficiency dimensions [n_elem, n_node_elem, 2, [mx, my, mz]] - all defined in B frame
            moment_efficiency = np.zeros_like(force_efficiency)
            moment_efficiency[:, :, :, 0] = airfoil_efficiency[:, :, :, 2]

            struct2aero_force_function = efficiency_local_aero2struct_forces

        except KeyError:
            pass

    for i_elem in range(n_elem):
        for i_local_node in range(3):

            i_global_node = conn[i_elem, i_local_node]
            if i_global_node in nodes:
                continue

            nodes.append(i_global_node)
            for mapping in struct2aero_mapping[i_global_node]:
                i_surf = mapping['i_surf']
                i_n = mapping['i_n']
                _, n_m, _ = aero_forces[i_surf].shape

                crv = psi_def[i_elem, i_local_node, :]
                cab = algebra.crv2rotation(crv)
                cbg = np.dot(cab.T, cag)

                for i_m in range(n_m):
                    chi_g = zeta[i_surf][:, i_m, i_n] - np.dot(cag.T, pos_def[i_global_node, :])
                    struct_forces[i_global_node, :] += struct2aero_force_function(aero_forces[i_surf][:, i_m, i_n],
                                                                                  chi_g,
                                                                                  cbg,
                                                                                  force_efficiency,
                                                                                  moment_efficiency,
                                                                                  i_elem,
                                                                                  i_local_node)

    return struct_forces


def local_aero2struct_forces(local_aero_forces, chi_g, cbg, force_efficiency=None, moment_efficiency=None, i_elem=None,
                             i_local_node=None):
    r"""
    Maps the local aerodynamic forces at a given vertex to its corresponding structural node.

    .. math::
        \mathbf{f}_{struct}^B &= C^{BG}\mathbf{f}_{i,aero}^G\\
        \mathbf{m}_{struct}^B &= C^{BG}(\mathbf{m}_{i,aero}^G +
        \tilde{\boldsymbol{\zeta}}^G\mathbf{f}_{i, aero}^G)

    Args:
        local_aero_forces (np.ndarray): aerodynamic forces and moments at a grid vertex
        chi_g (np.ndarray): vector between grid vertex and structural node in inertial frame
        cbg (np.ndarray): transformation matrix between inertial and body frames of reference
        force_efficiency (np.ndarray): Unused. See :func:`~sharpy.aero.utils.mapping.efficiency_local_aero2struct_forces`.
        moment_efficiency (np.ndarray): Unused. See :func:`~sharpy.aero.utils.mapping.efficiency_local_aero2struct_forces`.
        i_elem (int):

    Returns:
         np.ndarray: corresponding aerodynamic force at the structural node from the force and moment at a grid vertex

    """
    local_struct_forces = np.zeros(6)
    local_struct_forces[0:3] += np.dot(cbg, local_aero_forces[0:3])
    local_struct_forces[3:6] += np.dot(cbg, local_aero_forces[3:6])
    local_struct_forces[3:6] += np.dot(cbg, algebra.cross3(chi_g, local_aero_forces[0:3]))

    return local_struct_forces


def efficiency_local_aero2struct_forces(local_aero_forces, chi_g, cbg, force_efficiency, moment_efficiency, i_elem,
                                        i_local_node):
    r"""
    Maps the local aerodynamic forces at a given vertex to its corresponding structural node, introducing user-defined
    efficiency and constant value factors.

    .. math::
        \mathbf{f}_{struct}^B &= \varepsilon^f_0 C^{BG}\mathbf{f}_{i,aero}^G + \varepsilon^f_1\\
        \mathbf{m}_{struct}^B &= \varepsilon^m_0 (C^{BG}(\mathbf{m}_{i,aero}^G +
        \tilde{\boldsymbol{\zeta}}^G\mathbf{f}_{i, aero}^G)) + \varepsilon^m_1

    Args:
        local_aero_forces (np.ndarray): aerodynamic forces and moments at a grid vertex
        chi_g (np.ndarray): vector between grid vertex and structural node in inertial frame
        cbg (np.ndarray): transformation matrix between inertial and body frames of reference
        force_efficiency (np.ndarray): force efficiency matrix for all structural elements. Its size is ``n_elem x n_node_elem x 2 x 3``
        moment_efficiency (np.ndarray): moment efficiency matrix for all structural elements. Its size is ``n_elem x n_node_elem x 2 x 3``
        i_elem (int): element index
        i_local_node (int): local node index within element

    Returns:
         np.ndarray: corresponding aerodynamic force at the structural node from the force and moment at a grid vertex
    """
    local_struct_forces = np.zeros(6)
    local_struct_forces[0:3] += np.dot(cbg, local_aero_forces[0:3]) * force_efficiency[i_elem, i_local_node, 0] # element wise multiplication
    local_struct_forces[0:3] += force_efficiency[i_elem, i_local_node, 1]
    local_struct_forces[3:6] += np.dot(cbg, local_aero_forces[3:6]) * moment_efficiency[i_elem, i_local_node, 0]
    local_struct_forces[3:6] += np.dot(cbg, algebra.cross3(chi_g, local_aero_forces[0:3])) * moment_efficiency[i_elem, i_local_node, 0]
    local_struct_forces[3:6] += moment_efficiency[i_elem, i_local_node, 1]

    return local_struct_forces

def correct_forces_polars(aerogrid, beam, aero_kstep, structural_kstep, struct_forces):

    rho = 1.225
    aero_dict = aerogrid.aero_dict
    if aerogrid.polars is None:
        return struct_forces

    nnode = struct_forces.shape[0]
    for inode in range(nnode):
        if aero_dict['aero_node'][inode]:

            ielem, inode_in_elem = beam.node_master_elem[inode]
            iairfoil = aero_dict['airfoil_distribution'][ielem, inode_in_elem]
            isurf = aerogrid.struct2aero_mapping[inode][0]['i_surf']
            i_n = aerogrid.struct2aero_mapping[inode][0]['i_n']
            N = aerogrid.aero_dimensions[isurf, 1]
            polar = aerogrid.polars[iairfoil]
            cab = algebra.crv2rotation(structural_kstep.psi[ielem, inode_in_elem, :])
            cgb = np.dot(structural_kstep.cga(), cab)

            # Deal with the extremes
            if i_n == 0:
                node1 = 0
                node2 = 1
            elif i_n == N:
                node1 = nnode - 1
                node2 = nnode - 2
            else:
                node1 = inode + 1
                node2 = inode - 1

            # Define the span and the span direction
            dir_span = 0.5*np.dot(structural_kstep.cga(),
                              structural_kstep.pos[node1, :] - structural_kstep.pos[node2, :])
            span = np.linalg.norm(dir_span)
            dir_span = algebra.unit_vector(dir_span)

            # Define the chord and the chord direction
            dir_chord = aero_kstep.zeta[isurf][:, -1, i_n] - aero_kstep.zeta[isurf][:, 0, i_n]
            chord = np.linalg.norm(dir_chord)
            dir_chord = algebra.unit_vector(dir_chord)

            # Define the relative velocity and its direction
            urel = (structural_kstep.pos_dot[inode, :] +
                    structural_kstep.for_vel[0:3] +
                    np.cross(structural_kstep.for_vel[3:6],
                             structural_kstep.pos[inode, :]))
            urel = -np.dot(structural_kstep.cga(), urel)
            urel += np.average(aero_kstep.u_ext[isurf][:, :, i_n], axis=1)
            # uind = uvlmlib.uvlm_calculate_total_induced_velocity_at_points(aero_kstep,
            #                                                                np.array([structural_kstep.pos[inode, :] - np.array([0, 0, 1])]),
            #                                                                structural_kstep.for_pos,
            #                                                                ct.c_uint(8))[0]
            # print(inode, urel, uind)
            # urel -= uind
            dir_urel = algebra.unit_vector(urel)


            # Force in the G frame of reference
            force = np.dot(cgb,
                           struct_forces[inode, 0:3])
            dir_force = algebra.unit_vector(force)

            # Coefficient to change from aerodynamic coefficients to forces (and viceversa)
            coef = 0.5*rho*np.linalg.norm(urel)**2*chord*span

            # Divide the force in drag and lift
            drag_force = np.dot(force, dir_urel)*dir_urel
            lift_force = force - drag_force

            # Compute the associated lift
            cl = np.linalg.norm(lift_force)/coef

            # Compute the angle of attack assuming that UVLM giveas a 2pi polar
            aoa_deg_2pi = polar.get_aoa_deg_from_cl_2pi(cl)

            # Compute the coefficients assocaited to that angle of attack
            cl_new, cd, cm = polar.get_coefs(aoa_deg_2pi)
            # print(cl, cl_new)

            # Recompute the forces based on the coefficients
            lift_force = cl*algebra.unit_vector(lift_force)*coef
            drag_force += cd*dir_urel*coef
            force = lift_force + drag_force
            struct_forces[inode, 0:3] = np.dot(cgb.T,
                                               force)

    return struct_forces
