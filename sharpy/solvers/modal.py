import ctypes as ct
import numpy as np
import scipy as sc
import os
import itertools
import warnings
import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout
import sharpy.structure.utils.modalutils as modalutils


@solver
class Modal(BaseSolver):
    """
    ``Modal`` solver class, inherited from ``BaseSolver``

    Extracts the ``M``, ``K`` and ``C`` matrices from the ``Fortran`` library for the beam. Depending on the choice of
    modal projection, these may or may not be transformed to a state-space form to compute the eigenvalues and mode shapes
    of the structure.
    """
    solver_id = 'Modal'
    solver_classification = 'Linear'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write status to screen'

    # solution options
    settings_types['rigid_body_modes'] = 'bool'
    settings_default['rigid_body_modes'] = False
    settings_description['rigid_body_modes'] = 'Write modes with rigid body mode shapes'

    settings_types['use_undamped_modes'] = 'bool'  # basis for modal projection
    settings_default['use_undamped_modes'] = True
    settings_description['use_undamped_modes'] = 'Project the modes onto undamped mode shapes'

    settings_types['NumLambda'] = 'int'  # no. of different modes to retain
    settings_default['NumLambda'] = 20  # doubles if use_undamped_modes is False
    settings_description['NumLambda'] = 'Number of modes to retain'

    # output options
    settings_types['write_modes_vtk'] = 'bool'  # write displacements mode shapes in vtk file
    settings_default['write_modes_vtk'] = True
    settings_description['write_modes_vtk'] = 'Write Paraview files with mode shapes'

    settings_types['print_matrices'] = 'bool'  # print M,C,K matrices to dat file
    settings_default['print_matrices'] = False
    settings_description['print_matrices']  = 'Write M, C and K matrices to file'

    settings_types['save_data'] = 'bool'  # write modes shapes/freq./damp. to dat file
    settings_default['save_data'] = True
    settings_description['save_data'] = 'Write mode shapes, frequencies and damping to file'

    settings_types['continuous_eigenvalues'] = 'bool'
    settings_default['continuous_eigenvalues'] = False
    settings_description['continuous_eigenvalues'] = 'Use continuous time eigenvalues'

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0
    settings_description['dt'] = 'Time step to compute discrete time eigenvalues'

    settings_types['delta_curved'] = 'float'
    settings_default['delta_curved'] = 1e-2
    settings_description['delta_curved'] = 'Threshold for linear expressions in rotation formulas'

    settings_types['plot_eigenvalues'] = 'bool'
    settings_default['plot_eigenvalues'] = False
    settings_description['plot_eigenvalues'] = 'Plot to screen root locus diagram'

    settings_types['max_rotation_deg'] = 'float'
    settings_default['max_rotation_deg'] = 15.
    settings_description['max_rotation_deg'] = 'Scale mode shape to have specified maximum rotation'

    settings_types['max_displacement'] = 'float'
    settings_default['max_displacement'] = 0.15
    settings_description['max_displacement'] = 'Scale mode shape to have specified maximum displacement'

    settings_types['use_custom_timestep'] = 'int'
    settings_default['use_custom_timestep'] = -1
    settings_description['use_custom_timestep'] = 'If > -1, it will use that time step geometry for calculating the modes'

    settings_types['rigid_modes_ppal_axes'] = 'bool'
    settings_default['rigid_modes_ppal_axes'] = False
    settings_description['rigid_modes_ppal_axes'] = 'Modify the ridid body modes such that they are defined wrt ' \
                                                    'to the CG and aligned with the principal axes of inertia'

    settings_types['rigid_modes_cg'] = 'bool'
    settings_default['rigid_modes_cg'] = False
    settings_description['rigid_modes_cg'] = 'Not implemente yet'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

        self.folder = None
        self.eigenvalue_table = None
        self.filename_freq = None
        self.filename_damp = None
        self.filename_shapes = None
        self.rigid_body_motion = None

    def initialise(self, data, custom_settings=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                           self.settings_types,
                           self.settings_default)

        self.rigid_body_motion = self.settings['rigid_body_modes']

        self.data.ts = len(self.data.structure.timestep_info) - 1
        if self.settings['use_custom_timestep'] > -1:
            self.data.ts = self.settings['use_custom_timestep']

        # load info from dyn dictionary
        self.data.structure.add_unsteady_information(
                                            self.data.structure.dyn_dict,
                                            self.data.ts)

        # create folder for containing files if necessary
        self.folder = data.output_folder + '/beam_modal_analysis/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.filename_freq = (self.folder +
                              'tstep' + ("%06d" % self.data.ts) +
                              '_ModalFrequencies.dat')
        self.filename_damp = (self.folder +
                              'tstep' + ("%06d" % self.data.ts) +
                              '_ModalDamping.dat')
        self.filename_shapes = (self.folder +
                                'tstep' + ("%06d" % self.data.ts) +
                                '_ModalShape')

        if self.settings['print_info']:
            cout.cout_wrap('Structural eigenvalues')
            eigenvalue_filename = self.folder + '/eigenvaluetable.txt'
            self.eigenvalue_table = modalutils.EigenvalueTable(filename=eigenvalue_filename)
            self.eigenvalue_table.print_header(self.eigenvalue_table.headers)

    def run(self, **kwargs):
        r"""
        Extracts the eigenvalues and eigenvectors of the clamped structure.

        If ``use_undamped_modes == True`` then the free vibration modes of the clamped structure are found solving:

            .. math:: \mathbf{M\,\ddot{\eta}} + \mathbf{K\,\eta} = 0

        that flows down to solving the non-trivial solutions to:

            .. math:: (-\omega_n^2\,\mathbf{M} + \mathbf{K})\mathbf{\Phi} = 0

        On the other hand, if the damped modes are chosen because the system has damping, the free vibration
        modes are found solving the equation of motion of the form:

            .. math:: \mathbf{M\,\ddot{\eta}} + \mathbf{C\,\dot{\eta}} + \mathbf{K\,\eta} = 0

        which can be written in state space form, with the state vector :math:`\mathbf{x} = [\eta^T,\,\dot{\eta}^T]^T`
        as

            .. math:: \mathbf{\dot{x}} = \begin{bmatrix} 0 & \mathbf{I} \\ -\mathbf{M^{-1}K} & -\mathbf{M^{-1}C}
                \end{bmatrix} \mathbf{x}

        and therefore the mode shapes and frequencies correspond to the solution of the eigenvalue problem

            .. math:: \mathbf{A\,\Phi} = \mathbf{\Lambda\,\Phi}.

        From the eigenvalues, the following system characteristics are provided:

            * Natural Frequency: :math:`\omega_n = |\lambda|`

            * Damped natural frequency: :math:`\omega_d = \text{Im}(\lambda) = \omega_n \sqrt{1-\zeta^2}`

            * Damping ratio: :math:`\zeta = -\frac{\text{Re}(\lambda)}{\omega_n}`

        In addition to the above, the modal output dictionary includes the following:

            * ``M``: Tangent mass matrix

            * ``C``: Tangent damping matrix

            * ``K``: Tangent stiffness matrix

            * ``Ccut``: Modal damping matrix :math:`\mathbf{C}_m = \mathbf{\Phi}^T\mathbf{C}\mathbf{\Phi}`

            * ``Kin_damp``: Forces gain matrix (when damped): :math:`K_{in} = \mathbf{\Phi}_L^T \mathbf{M}^{-1}`

            * ``eigenvectors``: Right eigenvectors

            * ``eigenvectors_left``: Left eigenvectors given when the system is damped

        Returns:
            PreSharpy: updated data object with modal analysis as part of the last structural time step.

        """

        # Number of degrees of freedom
        num_str_dof = self.data.structure.num_dof.value
        if self.rigid_body_motion:
            num_rigid_dof = 10
        else:
            num_rigid_dof = 0

        num_dof = num_str_dof + num_rigid_dof

        # if NumLambda

        # Initialize matrices
        FullMglobal = np.zeros((num_dof, num_dof),
                               dtype=ct.c_double, order='F')
        FullKglobal = np.zeros((num_dof, num_dof),
                               dtype=ct.c_double, order='F')
        FullCglobal = np.zeros((num_dof, num_dof),
                               dtype=ct.c_double, order='F')

        if self.rigid_body_motion:
            # Settings for the assembly of the matrices
            # try:
            #     full_matrix_settings = self.data.settings['StaticCoupled']['structural_solver_settings']
            #     full_matrix_settings['dt'] = ct.c_double(0.01)  # Dummy: required but not used
            #     full_matrix_settings['newmark_damp'] = ct.c_double(1e-2)  # Dummy: required but not used
            # except KeyError:
                # full_matrix_settings = self.data.settings['DynamicCoupled']['structural_solver_settings']
            import sharpy.solvers._basestructural as basestructuralsolver
            full_matrix_settings = basestructuralsolver._BaseStructural().settings_default
            settings_utils.to_custom_types(full_matrix_settings, basestructuralsolver._BaseStructural().settings_types, full_matrix_settings)


            # Obtain the tangent mass, damping and stiffness matrices
            FullMglobal, FullCglobal, FullKglobal, FullQ = xbeamlib.xbeam3_asbly_dynamic(self.data.structure,
                                          self.data.structure.timestep_info[self.data.ts],
                                          full_matrix_settings)

            cg = modalutils.cg(FullMglobal)
        else:
            xbeamlib.cbeam3_solv_modal(self.data.structure,
                                       self.settings, self.data.ts,
                                       FullMglobal, FullCglobal, FullKglobal)
            cg = None

        # Print matrices
        if self.settings['print_matrices']:
            np.savetxt(self.folder + "Mglobal.dat", FullMglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')
            np.savetxt(self.folder + "Cglobal.dat", FullCglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')
            np.savetxt(self.folder + "Kglobal.dat", FullKglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')

        # Check if the damping matrix is zero (issue working)
        if self.settings['use_undamped_modes']:
            zero_FullCglobal = True
            for i,j in itertools.product(range(num_dof),range(num_dof)):
                if np.absolute(FullCglobal[i, j]) > np.finfo(float).eps:
                    zero_FullCglobal = False
                    warnings.warn('Projecting a system with damping on undamped modal shapes')
                    break
        # Check if the damping matrix is skew-symmetric
        # skewsymmetric_FullCglobal = True
        # for i in range(num_dof):
        #     for j in range(i:num_dof):
        #         if((i==j) and (np.absolute(FullCglobal[i, j]) > np.finfo(float).eps)):
        #             skewsymmetric_FullCglobal = False
        #         elif(np.absolute(FullCglobal[i, j] + FullCglobal[j, i]) > np.finfo(float).eps):
        #             skewsymmetric_FullCglobal = False

        NumLambda = min(num_dof, self.settings['NumLambda'])

        if self.settings['use_undamped_modes']:

            # Solve for eigenvalues (with unit eigenvectors)
            eigenvalues,eigenvectors=np.linalg.eig(
                                       np.linalg.solve(FullMglobal,FullKglobal))
            eigenvectors_left=None
            # Define vibration frequencies and damping
            freq_natural = np.sqrt(eigenvalues)
            order = np.argsort(freq_natural)[:NumLambda]
            freq_natural = freq_natural[order]
            #freq_damped = freq_natural
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:,order]
            damping = np.zeros((NumLambda,))

        else:
            # State-space model
            Minv_neg = -np.linalg.inv(FullMglobal)
            A = np.zeros((2*num_dof, 2*num_dof), dtype=ct.c_double, order='F')
            A[:num_dof, num_dof:] = np.eye(num_dof)
            A[num_dof:, :num_dof] = np.dot(Minv_neg, FullKglobal)
            A[num_dof:, num_dof:] = np.dot(Minv_neg, FullCglobal)

            # Solve the eigenvalues problem
            eigenvalues, eigenvectors_left, eigenvectors = \
                sc.linalg.eig(A,left=True,right=True)
            freq_natural = np.abs(eigenvalues)
            damping = np.zeros_like(freq_natural)
            iiflex = freq_natural > 1e-16*np.mean(freq_natural)  # Pick only structural modes
            damping[iiflex] = -eigenvalues[iiflex].real/freq_natural[iiflex]
            freq_damped = freq_natural * np.sqrt(1-damping**2)

            # Order & downselect complex conj:
            # this algorithm assumes that complex conj eigenvalues appear consecutively
            # in eigenvalues. For symmetrical systems, this relies  on the fact that:
            # - complex conj eigenvalues have the same absolute value (to machine
            # precision)
            # - couples of eigenvalues with multiplicity higher than 1, show larger
            # numerical difference
            order = np.argsort(freq_damped)[:2*NumLambda]
            freq_damped = freq_damped[order]
            freq_natural = freq_natural[order]
            eigenvalues = eigenvalues[order]

            include = np.ones((2*NumLambda,), dtype=np.bool)
            ii = 0
            tol_rel = np.finfo(float).eps * freq_damped[ii]
            while ii < 2*NumLambda:
                # check complex
                if np.abs(eigenvalues[ii].imag) > 0.:
                    if np.abs(eigenvalues[ii+1].real-eigenvalues[ii].real) > tol_rel or\
                       np.abs(eigenvalues[ii+1].imag+eigenvalues[ii].imag) > tol_rel:
                        raise NameError('Complex conjugate expected but not found!')
                    ii += 1
                    try:
                        include[ii] = False
                    except IndexError:
                        pass
                ii += 1
            freq_damped = freq_damped[include]
            eigenvalues = eigenvalues[include]
            if self.settings['continuous_eigenvalues']:
                if self.settings['dt'] == 0.:
                    raise ValueError('Cannot compute the continuous eigenvalues without a dt value')
                eigenvalues = np.log(eigenvalues)/self.settings['dt']

            order = order[include]
            damping = damping[order]
            eigenvectors = eigenvectors[:, order]
            eigenvectors_left = eigenvectors_left[:, order].conj()

        # Modify rigid body modes for them to be defined wrt the CG
        eigenvectors = modalutils.mode_sign_convention(self.data.structure.boundary_conditions, eigenvectors,
                                                       self.rigid_body_motion)
        if not eigenvectors_left:
            if self.settings['rigid_modes_ppal_axes']:
                eigenvectors, t_pa, r_pa = modalutils.free_modes_principal_axes(eigenvectors, FullMglobal,
                                                                          return_transform=True)
            else:
                t_pa = None  # Transformation matrix from the A frame to the P frame (principal axes of inertia)
                r_pa = None
        # Scaling
        eigenvectors, eigenvectors_left = self.scale_modes_unit_mass_matrix(eigenvectors, FullMglobal, eigenvectors_left)

        # Other terms required for state-space realisation
        # non-zero damping matrix
        # Modal damping matrix
        if self.settings['use_undamped_modes'] and not(zero_FullCglobal):
            Ccut = np.dot(eigenvectors.T, np.dot(FullCglobal, eigenvectors))
        else:
            Ccut = None

        # forces gain matrix (nodal -> modal)
        if not self.settings['use_undamped_modes']:
            Kin_damp = np.dot(eigenvectors_left[num_dof:, :].T, -Minv_neg)
        else:
            Kin_damp = None

        # Plot eigenvalues using matplotlib if specified in settings
        if self.settings['plot_eigenvalues']:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                plt.scatter(eigenvalues.real, eigenvalues.imag)
                plt.show()
                plt.savefig(self.folder + 'eigenvalues.png', transparent=True, bbox_inches='tight')
            except ModuleNotFoundError:
                warnings.warn('Unable to import matplotlib, skipping plot')

        # Write dat files
        if self.settings['save_data']:
            if type(eigenvalues) == complex:
                np.savetxt(self.folder + "eigenvalues.dat", eigenvalues.view(float).reshape(-1, 2), fmt='%.12f',
                           delimiter='\t', newline='\n')
            else:
                np.savetxt(self.folder + "eigenvalues.dat", eigenvalues.view(float), fmt='%.12f',
                           delimiter='\t', newline='\n')
            np.savetxt(self.folder + "eigenvectors.dat", eigenvectors[:num_dof].real,
                       fmt='%.12f', delimiter='\t', newline='\n')

            if not self.settings['use_undamped_modes']:
                np.savetxt(self.folder + 'frequencies.dat', freq_damped[:NumLambda],
                           fmt='%e', delimiter='\t', newline='\n')
            else:
                np.savetxt(self.folder + 'frequencies.dat', freq_natural[:NumLambda],
                           fmt='%e', delimiter='\t', newline='\n')

            np.savetxt(self.filename_damp, damping[:NumLambda],
                       fmt='%e', delimiter='\t', newline='\n')

        # Write vtk
        if self.settings['write_modes_vtk']:
            try:
                self.data.aero
            except AttributeError:
                warnings.warn('No aerodynamic model found - unable to project the mode onto aerodynamic grid')
            else:
                modalutils.write_modes_vtk(
                    self.data,
                    eigenvectors[:num_dof],
                    NumLambda,
                    self.filename_shapes,
                    self.settings['max_rotation_deg'],
                    self.settings['max_displacement'],
                    ts=self.settings['use_custom_timestep'])

        outdict = dict()

        if self.settings['use_undamped_modes']:
            outdict['modes'] = 'undamped'
            outdict['freq_natural'] = freq_natural
            if not zero_FullCglobal:
                outdict['warning'] =\
                    'system with damping: mode shapes and natural frequencies do not account for damping!'
        else:
            outdict['modes'] = 'damped'
            outdict['freq_damped'] = freq_damped
            outdict['freq_natural'] = freq_natural

        outdict['damping'] = damping
        outdict['eigenvalues'] = eigenvalues
        outdict['eigenvectors'] = eigenvectors

        if Ccut is not None:
            outdict['Ccut'] = Ccut
        if Kin_damp is not None:
            outdict['Kin_damp'] = Kin_damp
        if not self.settings['use_undamped_modes']:
            outdict['eigenvectors_left'] = eigenvectors_left

        if cg is not None:
            outdict['cg'] = cg

        outdict['M'] = FullMglobal
        outdict['C'] = FullCglobal
        outdict['K'] = FullKglobal

        if t_pa is not None:
            outdict['t_pa'] = t_pa
            outdict['r_pa'] = r_pa
        self.data.structure.timestep_info[self.data.ts].modal = outdict

        if self.settings['print_info']:
            if self.settings['use_undamped_modes']:
                self.eigenvalue_table.print_evals(np.sqrt(eigenvalues[:NumLambda])*1j)
            else:
                self.eigenvalue_table.print_evals(eigenvalues[:NumLambda])
            self.eigenvalue_table.close_file()

        return self.data

    def scale_modes_unit_mass_matrix(self, eigenvectors, FullMglobal, eigenvectors_left=None):
        if self.settings['use_undamped_modes']:
            # mass normalise (diagonalises M and K)
            eigenvectors = modalutils.scale_mass_normalised_modes(eigenvectors, FullMglobal)
        else:
            # unit normalise (diagonalises A)
            if not self.rigid_body_motion:
                for ii in range(eigenvectors.shape[1]):  # Issue - dot product = 0 when you have arbitrary damping
                    fact = 1./np.sqrt(np.dot(eigenvectors_left[:, ii], eigenvectors[:, ii]))
                    eigenvectors_left[:, ii] = fact*eigenvectors_left[:, ii]
                    eigenvectors[:, ii] = fact*eigenvectors[:, ii]

        return eigenvectors, eigenvectors_left

    def free_free_modes(self, phi, M):
        r"""

        Warning:
            This function is deprecated. See :func:`~sharpy.structure.utils.modalutils.free_modes_principal_axes`
            for a transformation to the CG and with respect to the principal axes of inertia.

        Returns the rigid body modes defined with respect to the centre of gravity

        The transformation from the modes defined at the FoR A origin, :math:`\boldsymbol{\Phi}`, to the modes defined
        using the centre of gravity as a reference is


        .. math:: \boldsymbol{\Phi}_{rr,CG}|_{TRA} = \boldsymbol{\Phi}_{RR}|_{TRA} + \tilde{\mathbf{r}}_{CG}
            \boldsymbol{\Phi}_{RR}|_{ROT}

        .. math:: \boldsymbol{\Phi}_{rr,CG}|_{ROT} = \boldsymbol{\Phi}_{RR}|_{ROT}

        Returns:
            (np.array): Transformed eigenvectors
        """

        # NG - 26/7/19 This is the transformation being performed by K_vec
        # Leaving this here for now in case it becomes necessary
        # .. math:: \boldsymbol{\Phi}_{ss,CG}|_{TRA} = \boldsymbol{\Phi}_{SS}|_{TRA} +\boldsymbol{\Phi}_{RS}|_{TRA}  -
        # \tilde{\mathbf{r}}_{A}\boldsymbol{\Phi}_{RS}|_{ROT}
        #
        # .. math:: \boldsymbol{\Phi}_{ss,CG}|_{ROT} = \boldsymbol{\Phi}_{SS}|_{ROT}
        # + (\mathbf{T}(\boldsymbol{\Psi})^\top)^{-1}\boldsymbol{\Phi}_{RS}|_{ROT}
        warnings.warn('This function is deprecated. See sharpy.structure.utils.modalutils.free_modes_principal_axes',
                      category=DeprecationWarning)
        if not self.rigid_body_motion:
            warnings.warn('No rigid body modes to transform because the structure is clamped')
            return phi
        else:
            pos = self.data.structure.timestep_info[self.data.ts].pos
            r_cg = modalutils.cg(M)

            jj = 0
            K_vec = np.zeros((phi.shape[0], phi.shape[0]))

            jj_for_vel = range(self.data.structure.num_dof.value, self.data.structure.num_dof.value + 3)
            jj_for_rot = range(self.data.structure.num_dof.value + 3, self.data.structure.num_dof.value + 6)

            for node_glob in range(self.data.structure.num_node):
                ### detect bc at node (and no. of dofs)
                bc_here = self.data.structure.boundary_conditions[node_glob]

                if bc_here == 1:  # clamp (only rigid-body)
                    dofs_here = 0
                    jj_tra, jj_rot = [], []
                    continue

                elif bc_here == -1 or bc_here == 0:  # (rigid+flex body)
                    dofs_here = 6
                    jj_tra = 6 * self.data.structure.vdof[node_glob] + np.array([0, 1, 2], dtype=int)
                    jj_rot = 6 * self.data.structure.vdof[node_glob] + np.array([3, 4, 5], dtype=int)
                # jj_tra=[jj  ,jj+1,jj+2]
                # jj_rot=[jj+3,jj+4,jj+5]
                else:
                    raise NameError('Invalid boundary condition (%d) at node %d!' \
                                    % (bc_here, node_glob))

                jj += dofs_here

                ee, node_loc = self.data.structure.node_master_elem[node_glob, :]
                psi = self.data.structure.timestep_info[self.data.ts].psi[ee, node_loc, :]

                Ra = pos[node_glob, :]  # in A FoR with respect to G

                K_vec[np.ix_(jj_tra, jj_tra)] += np.eye(3)
                K_vec[np.ix_(jj_tra, jj_for_vel)] += np.eye(3)
                K_vec[np.ix_(jj_tra, jj_for_rot)] -= algebra.skew(Ra)

                K_vec[np.ix_(jj_rot, jj_rot)] += np.eye(3)
                K_vec[np.ix_(jj_rot, jj_for_rot)] += np.linalg.inv(algebra.crv2tan(psi).T)

            # Rigid-Rigid modes transform
            Krr = np.eye(10)
            Krr[np.ix_([0, 1, 2], [3, 4, 5])] += algebra.skew(r_cg)

            # Assemble transformed modes
            phirr = Krr.dot(phi[-10:, :10])
            phiss = K_vec.dot(phi[:, 10:])

            # Get rigid body modes to be positive in translation and rotation
            for i in range(10):
                ind = np.argmax(np.abs(phirr[:, i]))
                phirr[:, i] = np.sign(phirr[ind, i]) * phirr[:, i]

            # NG - 26/7/19 - Transformation of the rigid part of the elastic modes ended up not being necessary but leaving
            # here in case it becomes useful in the future
            phit = np.block([np.zeros((phi.shape[0], 10)), phi[:, 10:]])
            phit[-10:, :10] = phirr

            return phit
