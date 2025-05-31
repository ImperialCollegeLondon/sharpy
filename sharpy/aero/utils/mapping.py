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
                              data_dict=None,
                              skip_moments_generated_by_forces = False):
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

    Args:
        aero_forces (list): Aerodynamic forces from the UVLM in inertial frame of reference
        struct2aero_mapping (dict): Structural to aerodynamic node mapping
        zeta (list): Aerodynamic grid coordinates
        pos_def (np.ndarray): Vector of structural node displacements
        psi_def (np.ndarray): Vector of structural node rotations (CRVs)
        master: Unused
        conn (np.ndarray): Connectivities matrix
        cag (np.ndarray): Transformation matrix between inertial and body-attached reference ``A``
        data_dict (dict): Dictionary containing the grid's information.
        skip_moments_generated_by_forces (bool): Flag to skip local moment calculation.

    Returns:
        np.ndarray: structural forces in an ``n_node x 6`` vector
    """

    n_node, _ = pos_def.shape
    n_elem, _, _ = psi_def.shape
    struct_forces = np.zeros((n_node, 6))

    nodes = []

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
                    struct_forces[i_global_node, 0:3] += np.dot(cbg, aero_forces[i_surf][0:3, i_m, i_n])
                    struct_forces[i_global_node, 3:6] += np.dot(cbg, aero_forces[i_surf][3:6, i_m, i_n])
                    """
                        The calculation of the local moment is skipped for fuselage bodies, since this 
                        leads to noticeably asymmetric aeroforces. This transitional solution makes 
                        sense since we have only considered the stiff fuselage so far and the pitching 
                        moment coming from the fuselage is mainly caused by the longitudinal force distribution.
                        TODO: Correct the calculation of the local moment for the fuselage model.
                    """
                    if not skip_moments_generated_by_forces:
                        chi_g = zeta[i_surf][:, i_m, i_n] - np.dot(cag.T, pos_def[i_global_node, :])
                        struct_forces[i_global_node, 3:6] += np.dot(cbg, algebra.cross3(chi_g, aero_forces[i_surf][0:3, i_m, i_n]))

    return struct_forces


def total_forces_moments(forces_nodes_a,
                         pos_def,
                         ref_pos=np.array([0., 0., 0.])):
    """
    Performs a summation of the forces and moments expressed at the structural nodes in the A frame of reference.

    Note:
        If you need to transform forces and moments at the nodes from B to A, use the
        :func:`~sharpy.structure.models.beam.Beam.nodal_b_for_2_a_for()` function.

    Args:
        forces_nodes_a (np.array): ``n_node x 6`` vector of forces and moments at the nodes in A
        pos_def (np.array): ``n_node x 3`` vector of nodal positions in A
        ref_pos (np.array (optional)): Location in A about which to compute moments. Defaults to ``[0, 0, 0]``

    Returns:
        np.array: Vector of length 6 containing the total forces and moments expressed in A at the desired location.
    """

    num_node = pos_def.shape[0]
    ra_vec = pos_def - ref_pos

    total_forces = np.zeros(3)
    total_moments = np.zeros(3)
    for i_global_node in range(num_node):
        total_forces += forces_nodes_a[i_global_node, :3]
        total_moments += forces_nodes_a[i_global_node, 3:] + algebra.cross3(ra_vec[i_global_node], forces_nodes_a[i_global_node, :3])

    return np.concatenate((total_forces, total_moments))
