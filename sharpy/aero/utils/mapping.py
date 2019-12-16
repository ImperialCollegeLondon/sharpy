"""Force Mapping Utilities"""
import numpy as np
import sharpy.utils.algebra as algebra


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
        \mathbf{f}_{struct}^B &= \varepsilon^f_0 C^{BG}\mathbf{f}_{i,aero}^G + \varepsilon^f_1\\
        \mathbf{m}_{struct}^B &= \varepsilon^m_0 (C^{BG}(\mathbf{m}_{i,aero}^G +
        \tilde{\boldsymbol{\zeta}}^G\mathbf{f}_{i, aero}^G)) + \varepsilon^m_1

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
