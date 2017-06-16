import numpy as np
import sharpy.utils.algebra as algebra


def inertial2aero_rotation(alpha, beta):
    alpha_rot = algebra.rotation_matrix_around_axis(np.array([0, 1, 0]), alpha)
    beta_rot = algebra.rotation_matrix_around_axis(np.array([0, 0, 1]), beta)

    return np.dot(alpha_rot, beta_rot).T


def aero2struct_force_mapping(aero_forces,
                              struct2aero_mapping,
                              zeta,
                              pos_def,
                              psi_def,
                              master,
                              master_elem,
                              inertial2aero=None):

    n_node, _ = pos_def.shape
    struct_forces = np.zeros((n_node, 6))

    if inertial2aero is None:
        Cag = np.eye(3)
    else:
        Cag = inertial2aero

    for i_global_node in range(n_node):
        for mapping in struct2aero_mapping[i_global_node]:
            i_surf = mapping['i_surf']
            i_n = mapping['i_n']
            _, n_m, _ = aero_forces[i_surf].shape

            i_elem, i_local_node = master[i_global_node, :]
            i_master_elem, master_elem_local_node = master_elem[i_elem, i_local_node, :]
            if i_master_elem == -1:
                i_master_elem = i_elem
                master_elem_local_node = i_local_node

            crv = psi_def[i_master_elem, master_elem_local_node, :]
            Cab = algebra.crv2rot(crv)
            Cbg = np.dot(Cab.T, Cag)

            for i_m in range(n_m):
                chi_g = zeta[i_surf][:, i_m, i_n] - np.dot(Cag.T, pos_def[i_global_node, :])

                struct_forces[i_global_node, 0:3] += np.dot(Cbg, aero_forces[i_surf][0:3, i_m, i_n])
                struct_forces[i_global_node, 3:6] += np.dot(Cbg, aero_forces[i_surf][3:6, i_m, i_n])
                struct_forces[i_global_node, 3:6] += np.dot(Cbg, np.cross(chi_g, aero_forces[i_surf][0:3, i_m, i_n]))

    # n_surf = len(aero_forces)
    # for i_surf in range(n_surf):
    #     _, m, n = aero_forces[i_surf].shape
    #     for i_n in range(n):
    #         i_global_node = struct2aero_mapping[i_surf][i_n]
    #         i_local_node = master[i_global_node, 1]
    #         i_elem = master[i_global_node, 0]
    #
    #         if inertial2aero is not None:
    #             Cag = inertial2aero
    #         else:
    #             Cag = np.eye(3)
    #
    #         Cbg = np.dot(algebra.crv2rot(psi_def[i_elem, i_local_node, :]).T,
    #                      Cag)
    #
    #         for i_m in range(m):
    #             chi_g = zeta[i_surf][:, i_m, i_n] - np.dot(Cag.T, pos_def[i_global_node, :])
    #             chi_b = np.dot(Cbg, chi_g)
    #             # skew_chi_b = algebra.skew(chi_b)
    #
    #             struct_forces[i_global_node, 0:3] += np.dot(Cbg, aero_forces[i_surf][0:3, i_m, i_n])
    #             struct_forces[i_global_node, 3:6] += np.dot(Cbg, np.cross(chi_g, aero_forces[i_surf][0:3, i_m, i_n]))
    #             struct_forces[i_global_node, 3:6] += np.dot(Cbg, aero_forces[i_surf][3:6, i_m, i_n])

    return struct_forces


if __name__ == '__main__':
    m_alpha = 10*np.pi/180
    m_beta = 0.0

    m_rot = inertial2aero_rotation(m_alpha, m_beta)
    m_dir = np.array([0, 0, 1])
    print(m_dir)
    print(np.dot(m_rot.T, m_dir))
