import numpy as np
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
from tvtk.api import tvtk, write_data


def frequency_damping(eigenvalue):
    omega_n = np.abs(eigenvalue)
    omega_d = np.abs(eigenvalue.imag)
    f_n = omega_n / 2 / np.pi
    f_d = omega_d / 2 / np.pi
    if f_d < 1e-8:
        damping_ratio = 1.
        period = np.inf
    else:
        damping_ratio = -eigenvalue.real / omega_n
        period = 1 / f_d

    return omega_n, omega_d, damping_ratio, f_n, f_d, period


class EigenvalueTable(cout.TablePrinter):
    def __init__(self, filename=None):
        super().__init__(7, 12, ['g', 'f', 'f', 'f', 'f', 'f', 'f'], filename)

        self.headers = ['mode', 'eval_real', 'eval_imag', 'freq_n (Hz)', 'freq_d (Hz)',
                        'damping', 'period (s)']

    def print_evals(self, eigenvalues):
        for i in range(len(eigenvalues)):
            omega_n, omega_d, damping_ratio, f_n, f_d, period = frequency_damping(eigenvalues[i])
            self.print_line([i, eigenvalues[i].real, eigenvalues[i].imag, f_n, f_d,
                             damping_ratio, period])


def cg(M, use_euler=False):
    if use_euler:
        Mrr = M[-9:, -9:]
    else:
        Mrr = M[-10:, -10:]
    return -np.array([Mrr[2, 4], Mrr[0, 5], Mrr[1, 3]]) / Mrr[0, 0]


def scale_mode(data, eigenvector, rot_max_deg=15, perc_max=0.15):
    """
    Scales the eigenvector such that:
        1) the maximum change in component of the beam cartesian rotation vector
    is equal to rot_max_deg degrees.
        2) the maximum translational displacement does not exceed perc_max the
    maximum nodal position.

    Warning:
        If the eigenvector is in state-space form, only the first
        half of the eigenvector is scanned for determining the scaling.
    """

    ### initialise
    struct = data.structure
    tsstr = data.structure.timestep_info[data.ts]

    jj = 0  # structural dofs index
    RotMax = 0.0
    RaMax = 0.0
    dRaMax = 0.0

    for node_glob in range(struct.num_node):
        ### detect bc at node (and no. of dofs)
        bc_here = struct.boundary_conditions[node_glob]
        if bc_here == 1:  # clamp
            dofs_here = 0
            continue
        elif bc_here == -1 or bc_here == 0:
            dofs_here = 6
            jj_tra = [jj, jj + 1, jj + 2]
            jj_rot = [jj + 3, jj + 4, jj + 5]
        jj += dofs_here

        # check for max rotation
        RotMaxHere = np.max(np.abs(eigenvector[jj_rot].real))
        if RotMaxHere > RotMax:
            RotMax = RotMaxHere

        # check for maximum position
        RaNorm = np.linalg.norm(tsstr.pos[node_glob, :])
        if RaNorm > RaMax:
            RaMax = RaNorm

        # check for maximum displacement
        dRaNorm = np.linalg.norm(eigenvector[jj_tra].real)
        if dRaNorm > dRaMax:
            dRaMax = dRaNorm

    RotMaxDeg = RotMax * 180 / np.pi

    if RotMaxDeg > 1e-4:
        fact = rot_max_deg / RotMaxDeg
        if dRaMax * fact > perc_max * RaMax:
            fact = perc_max * RaMax / dRaMax
    else:
        fact = perc_max * RaMax / dRaMax
    # correct factor to ensure max disp is perc
    return fact


def get_mode_zeta(data, eigvect):
    """
    Retrieves the UVLM grid nodal displacements associated to the eigenvector ``eigvect``
    """

    ### initialise
    aero = data.aero
    struct = data.structure
    tsaero = data.aero.timestep_info[data.ts]
    tsstr = data.structure.timestep_info[data.ts]

    try:
        num_dof = struct.num_dof.value
    except AttributeError:
        num_dof = struct.num_dof

    eigvect = eigvect[:num_dof]

    zeta_mode = []
    for ss in range(aero.n_surf):
        zeta_mode.append(tsaero.zeta[ss].copy())

    jj = 0  # structural dofs index
    Cga0 = algebra.quat2rotation(tsstr.quat)
    Cag0 = Cga0.T
    for node_glob in range(struct.num_node):

        ### detect bc at node (and no. of dofs)
        bc_here = struct.boundary_conditions[node_glob]
        if bc_here == 1:  # clamp
            dofs_here = 0
            continue
        elif bc_here == -1 or bc_here == 0:
            dofs_here = 6
            jj_tra = [jj, jj + 1, jj + 2]
            jj_rot = [jj + 3, jj + 4, jj + 5]
        jj += dofs_here

        # retrieve element and local index
        ee, node_loc = struct.node_master_elem[node_glob, :]

        # get original position and crv
        Ra0 = tsstr.pos[node_glob, :]
        psi0 = tsstr.psi[ee, node_loc, :]
        Rg0 = np.dot(Cga0, Ra0)
        Cab0 = algebra.crv2rotation(psi0)
        Cbg0 = np.dot(Cab0.T, Cag0)

        # update position and crv of mode
        Ra = tsstr.pos[node_glob, :] + eigvect[jj_tra]
        psi = tsstr.psi[ee, node_loc, :] + eigvect[jj_rot]
        Rg = np.dot(Cga0, Ra)
        Cab = algebra.crv2rotation(psi)
        Cbg = np.dot(Cab.T, Cag0)

        ### str -> aero mapping
        # some nodes may be linked to multiple surfaces...
        for str2aero_here in aero.struct2aero_mapping[node_glob]:

            # detect surface/span-wise coordinate (ss,nn)
            nn, ss = str2aero_here['i_n'], str2aero_here['i_surf']
            # print('%.2d,%.2d'%(nn,ss))

            # surface panelling
            M = aero.aero_dimensions[ss][0]
            N = aero.aero_dimensions[ss][1]

            for mm in range(M + 1):
                # get position of vertex in B FoR
                zetag0 = tsaero.zeta[ss][:, mm, nn]  # in G FoR, w.r.t. origin A-G
                Xb = np.dot(Cbg0, zetag0 - Rg0)  # in B FoR, w.r.t. origin B

                # update vertex position
                zeta_mode[ss][:, mm, nn] = Rg + np.dot(np.dot(Cga0, Cab), Xb)

    return zeta_mode


def write_zeta_vtk(zeta, zeta_ref, filename_root):
    '''
    Given a list of arrays representing the coordinates of a set of n_surf UVLM
    lattices and organised as:
        zeta[n_surf][3,M+1,N=1]
    this function writes a vtk for each of the n_surf surfaces.

    Input:
        - zeta: lattice coordinates to plot
        - zeta_ref: reference lattice used to compute the magnitude of displacements
        - filename_root: initial part of filename (full path) without file
        extension (.vtk)
    '''

    # from IPython import embed
    # embed()
    for i_surf in range(len(zeta)):

        filename = filename_root + "_%02u.vtu" % (i_surf,)
        _, M, N = zeta[i_surf].shape

        M -= 1
        N -= 1
        point_data_dim = (M + 1) * (N + 1)
        panel_data_dim = M * N

        coords = np.zeros((point_data_dim, 3))
        conn = []
        panel_id = np.zeros((panel_data_dim,), dtype=int)
        panel_surf_id = np.zeros((panel_data_dim,), dtype=int)
        point_struct_id = np.zeros((point_data_dim,), dtype=int)
        point_struct_mag = np.zeros((point_data_dim,), dtype=float)

        counter = -1
        # coordinates of corners
        for i_n in range(N + 1):
            for i_m in range(M + 1):
                counter += 1
                coords[counter, :] = zeta[i_surf][:, i_m, i_n]

        counter = -1
        node_counter = -1
        for i_n in range(N + 1):
            # global_counter = aero.aero2struct_mapping[i_surf][i_n]
            for i_m in range(M + 1):
                node_counter += 1
                # point data
                # point_struct_id[node_counter]=global_counter
                point_struct_mag[node_counter] = \
                    np.linalg.norm(zeta[i_surf][:, i_m, i_n] \
                                   - zeta_ref[i_surf][:, i_m, i_n])

                if i_n < N and i_m < M:
                    counter += 1
                else:
                    continue

                conn.append([node_counter + 0,
                             node_counter + 1,
                             node_counter + M + 2,
                             node_counter + M + 1])
                # cell data
                panel_id[counter] = counter
                panel_surf_id[counter] = i_surf

        ug = tvtk.UnstructuredGrid(points=coords)
        ug.set_cells(tvtk.Quad().cell_type, conn)
        ug.cell_data.scalars = panel_id
        ug.cell_data.scalars.name = 'panel_n_id'
        ug.cell_data.add_array(panel_surf_id)
        ug.cell_data.get_array(1).name = 'panel_surface_id'

        ug.point_data.scalars = np.arange(0, coords.shape[0])
        ug.point_data.scalars.name = 'n_id'
        # ug.point_data.add_array(point_struct_id)
        # ug.point_data.get_array(1).name = 'point_struct_id'
        ug.point_data.add_array(point_struct_mag)
        ug.point_data.get_array(1).name = 'point_displacement_magnitude'

        write_data(ug, filename)


def write_modes_vtk(data, eigenvectors, NumLambda, filename_root,
                    rot_max_deg=15., perc_max=0.15, ts=-1):
    """
    Writes a vtk file for each of the first ``NumLambda`` eigenvectors. When these
    are associated to the state-space form of the structural equations, only
    the displacement field is saved.
    """

    ### initialise
    aero = data.aero
    struct = data.structure
    tsaero = data.aero.timestep_info[ts]
    tsstr = data.structure.timestep_info[ts]

    num_dof = struct.num_dof.value
    eigenvectors = eigenvectors[:num_dof, :]

    # Check whether rigid body motion is selected
    # Skip rigid body modes
    if data.settings['Modal']['rigid_body_modes'].value:
        num_rigid_body = 10
    else:
        num_rigid_body = 0

    for mode in range(num_rigid_body, NumLambda - num_rigid_body):
        # scale eigenvector
        eigvec = eigenvectors[:num_dof, mode]
        fact = scale_mode(data, eigvec, rot_max_deg, perc_max)
        eigvec = eigvec * fact
        zeta_mode = get_mode_zeta(data, eigvec)
        write_zeta_vtk(zeta_mode, tsaero.zeta, filename_root + "_%06u" % (mode,))


def free_modes_principal_axes(phi, mass_matrix, use_euler=False):
    """
    Transforms the rigid body modes defined at with the A frame as reference to the centre of mass position and aligned
    with the principal axes of inertia.


    References:
        Marc Artola, 2020
    """
    if use_euler:
        num_rigid_modes = 9
    else:
        num_rigid_modes = 10

    r_cg = cg(mass_matrix, use_euler)  # centre of gravity
    mrr = mass_matrix[-num_rigid_modes:-num_rigid_modes + 6, -num_rigid_modes:-num_rigid_modes + 6]
    m = mrr[0, 0]  # mass

    # principal axes of inertia matrix and transformation matrix
    j_cm, t_rb = np.linalg.eig(mrr[-3:, -3:] + algebra.multiply_matrices(algebra.skew(r_cg), algebra.skew(r_cg)) * m)

    # rigid body mass matrix about CM and inertia in principal axes
    m_cm = np.eye(6) * m
    m_cm[-3:, -3:] = np.diag(j_cm)

    # rigid body modes about CG - mass normalised
    rb_cm = np.eye(6)
    rb_cm /= np.sqrt(np.diag(rb_cm.T.dot(m_cm.dot(rb_cm))))

    # transform to A frame reference position
    trb_diag = np.zeros((6, 6))  # matrix with (t_rb, t_rb) in the diagonal
    trb_diag[:3, :3] = t_rb
    trb_diag[-3:, -3:] = t_rb
    rb_a = np.block([[np.eye(3), algebra.skew(r_cg)], [np.zeros((3, 3)), np.eye(3)]]).dot(trb_diag.dot(rb_cm))

    phit = np.block([np.zeros((phi.shape[0], num_rigid_modes)), phi[:, num_rigid_modes:]])
    phit[-num_rigid_modes:-num_rigid_modes + 6, :6] = rb_a

    phit[-num_rigid_modes + 6:, 6:num_rigid_modes] = np.eye(num_rigid_modes - 6)  # euler or quaternion modes

    return phit


def mode_sign_convention(bocos, eigenvectors, rigid_body_motion=False, use_euler=False):
    """
    When comparing against different cases, it is important that the modes share a common sign convention.

    In this case, modes will be arranged such that the z-coordinate of the first free end is positive.

    If the z-coordinate is 0, then the y-coordinate is forced to be positive, then x, followed by the CRV in y, x and z.

    Returns:
        np.ndarray: Eigenvectors following the aforementioned sign convention.
    """

    if use_euler:
        num_rigid_modes = 9
    else:
        num_rigid_modes = 10

    if rigid_body_motion:
        eigenvectors = order_rigid_body_modes(eigenvectors, use_euler)

        # A frame reference
        z_coord = -num_rigid_modes + 2
        y_coord = -num_rigid_modes + 1
        x_coord = -num_rigid_modes + 0
        mz_coord = -num_rigid_modes + 5
        my_coord = -num_rigid_modes + 4
        mx_coord = -num_rigid_modes + 3
    else:
        first_free_end_node = np.where(bocos == -1)[0][0]

        z_coord = 6 * (first_free_end_node - 1) + 2
        y_coord = 6 * (first_free_end_node - 1) + 1
        x_coord = 6 * (first_free_end_node - 1) + 0
        my_coord = 6 * (first_free_end_node - 1) + 4
        mz_coord = 6 * (first_free_end_node - 1) + 5
        mx_coord = 6 * (first_free_end_node - 1) + 3

    for i in range(0, eigenvectors.shape[1]):
        if np.abs(eigenvectors[z_coord, i]) > 1e-8:
            eigenvectors[:, i] = np.sign(eigenvectors[z_coord, i]) * eigenvectors[:, i]

        elif np.abs(eigenvectors[y_coord, i]) > 1e-8:
            eigenvectors[:, i] = np.sign(eigenvectors[y_coord, i]) * eigenvectors[:, i]

        elif np.abs(eigenvectors[x_coord, i]) > 1e-8:
            eigenvectors[:, i] = np.sign(eigenvectors[x_coord, i]) * eigenvectors[:, i]

        elif np.abs(eigenvectors[my_coord, i]) > 1e-8:
            eigenvectors[:, i] = np.sign(eigenvectors[my_coord, i]) * eigenvectors[:, i]

        elif np.abs(eigenvectors[mx_coord, i]) > 1e-8:
            eigenvectors[:, i] = np.sign(eigenvectors[mx_coord, i]) * eigenvectors[:, i]

        elif np.abs(eigenvectors[mz_coord, i]) > 1e-8:
            eigenvectors[:, i] = np.sign(eigenvectors[mz_coord, i]) * eigenvectors[:, i]

        else:
            if rigid_body_motion:
                cout.cout_wrap('Mode component at the A frame is 0.', 3)
            else:
                # cout.cout_wrap('Mode component at the first free end (node {:g}) is 0.'.format(first_free_end_node), 3)

                # this will be the case for symmetric clamped structures, where modes will be present for the left and
                # right wings. Method should be called again when symmetric modes are removed.
                pass

    return eigenvectors


def order_rigid_body_modes(eigenvectors, use_euler):

    if use_euler:
        num_rigid_modes = 9
    else:
        num_rigid_modes = 10

    phi_rr = np.zeros((num_rigid_modes, num_rigid_modes))
    num_node = eigenvectors.shape[0]

    for i in range(num_rigid_modes):
        index_max_node = np.where(eigenvectors[:, i] == np.max(eigenvectors[:, i]))[0][0]
        print(index_max_node)
        index_mode = num_rigid_modes - (num_node - index_max_node)
        print(index_mode)
        phi_rr[:, index_mode] = eigenvectors[-num_rigid_modes:, i]

    eigenvectors[-num_rigid_modes:, :num_rigid_modes] = phi_rr

    return eigenvectors


def scale_mass_normalised_modes(eigenvectors, mass_matrix):
    r"""
    Scales eigenvector matrix such that the modes are mass normalised:

    .. math:: \phi^\top\boldsymbol{M}\phi = \boldsymbol{I}

    and

    .. math:: \phi^\top\boldsymbol{K}\phi = \mathrm{diag}(\omega^2)

    Args:
        eigenvectors (np.array): Eigenvector matrix.
        mass_matrix (np.array): Mass matrix.

    Returns:
        np.array: Mass-normalised eigenvectors.
    """
    # mass normalise (diagonalises M and K)
    dfact = np.diag(np.dot(eigenvectors.T, np.dot(mass_matrix, eigenvectors)))
    eigenvectors = (1./np.sqrt(dfact))*eigenvectors

    return eigenvectors


def assert_orthogonal_eigenvectors(u, v, decimal, raise_error=False):
    """
    Checks orthogonality between eigenvectors

    Args:
        u (np.ndarray): Eigenvector 1.
        v (np.ndarray): Eigenvector 2.
        decimal (int): Number of decimal points to compare
        raise_error (bool): Raise an error or print a warning

    Raises:
        AssertionError: if ``raise_error == True`` it raises an error.

    """
    try:
        np.testing.assert_almost_equal(u.dot(v), 0, decimal=decimal,
                                       err_msg='Eigenvectors not orthogonal')  # random eigenvector to test orthonality
    except AssertionError as e:
        if raise_error:
            raise e
        else:
            cout.cout_wrap('Eigenvectors not orthogonal', 3)


def assert_modes_mass_normalised(phi, m, tolerance, raise_error=False):
    """
    Asserts the eigenvectors result in an identity modal mass matrix.

    Args:
        phi (np.ndarray): Eigenvector matrix
        m (np.ndarray): Mass matrix
        tolerance (float): Absolute tolerance.
        raise_error (bool): Raise ``AssertionError`` if modes not mass normalised.

    Returns:
        AssertionError: if ``raise_error == True`` it raises an error.

    """
    modal_mass = phi.T.dot(m.dot(phi))

    try:
        np.testing.assert_allclose(modal_mass - np.eye(modal_mass.shape[0]), np.zeros_like(modal_mass),
                                   atol=tolerance, err_msg='Eigenvectors are not mass normalised')
    except AssertionError as e:
        if raise_error:
            raise e
        else:
            cout.cout_wrap('Eigenvectors are not mass normalised', 3)

