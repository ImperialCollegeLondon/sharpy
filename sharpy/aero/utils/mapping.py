import numpy as np
import sharpy.utils.algebra as algebra


def aero2struct_force_mapping(aero_forces,
                              struct2aero_mapping,
                              zeta,
                              pos_def,
                              psi_def,
                              master):

    n_node, _ = pos_def.shape
    struct_forces = np.zeros((n_node, 6))

    n_surf = len(aero_forces)
    for i_surf in range(n_surf):
        _, m, n = aero_forces[i_surf].shape
        for i_n in range(n):
            i_global_node = struct2aero_mapping[i_surf][i_n]
            i_local_node = master[i_global_node, 1]
            i_elem = master[i_global_node, 0]

            Cba = algebra.crv2rot(psi_def[i_elem, i_local_node, :])

            for i_m in range(m):
                chi_a = pos_def[i_global_node, :] - zeta[i_surf][:, i_m, i_n]
                chi_b = np.dot(Cba, chi_a)
                skew_chi_b = algebra.skew(chi_b)

                struct_forces[i_global_node, 0:3] += np.dot(Cba, aero_forces[i_surf][0:3, i_m, i_n])
                struct_forces[i_global_node, 3:6] += np.dot(np.dot(skew_chi_b, Cba), aero_forces[i_surf][0:3, i_m, i_n])
                struct_forces[i_global_node, 3:6] += np.dot(Cba, aero_forces[i_surf][3:6, i_m, i_n])

    return struct_forces


