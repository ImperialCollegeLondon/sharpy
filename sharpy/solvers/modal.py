"""
@modified   Alfonso del Carre
"""

import ctypes as ct
import numpy as np
import scipy as sc
import os
import itertools
import warnings
from tvtk.api import tvtk, write_data
import scipy.linalg

import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra



@solver
class Modal(BaseSolver):
    """
    ``Modal`` solver class, inherited from ``BaseSolver``

    Extracts the ``M``, ``K`` and ``C`` matrices from the ``Fortran`` library for the beam. Depending on the choice of
    modal projection, these may or may not be transformed to a state-space form to compute the eigenvalues and mode shapes
    of the structure.
    """
    solver_id = 'Modal'
    solver_classification = 'modal'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write status to screen'

    settings_types['folder'] = 'str'
    settings_default['folder'] = './output'
    settings_description['folder'] = 'Output folder'

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

    settings_types['keep_linear_matrices'] = 'bool'  # attach linear M,C,K matrices to output dictionary
    settings_default['keep_linear_matrices'] = True
    settings_description['keep_linear_matrices'] = 'Save M, C and K matrices to output dictionary'

    # output options
    settings_types['write_modes_vtk'] = 'bool'  # write displacements mode shapes in vtk file
    settings_default['write_modes_vtk'] = True
    settings_description['write_modes_vtk'] = 'Write Paraview files with mode shapes'

    settings_types['print_matrices'] = 'bool'  # print M,C,K matrices to dat file
    settings_default['print_matrices'] = False
    settings_description['print_matrices']  = 'Write M, C and K matrices to file'

    settings_types['write_dat'] = 'bool'  # write modes shapes/freq./damp. to dat file
    settings_default['write_dat'] = True
    settings_description['write_dat'] = 'Write mode shapes, frequencies and damping to file'

    settings_types['continuous_eigenvalues'] = 'bool'
    settings_default['continuous_eigenvalues'] = False
    settings_description['continuous_eigenvalues'] = 'Use continuous time eigenvalues'

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0
    settings_description['dt'] = 'Time step to compute discrete time eigenvalues'

    settings_types['plot_eigenvalues'] = 'bool'
    settings_default['plot_eigenvalues'] = False
    settings_description['plot_eigenvalues'] = 'Plot to screen root locus diagram'

    settings_types['max_rotation_deg'] = 'float'
    settings_default['max_rotation_deg'] = 15.
    settings_description['max_rotation_deg'] = 'Scale mode shape to have specified maximum rotation'

    settings_types['max_displacement'] = 'float'
    settings_default['max_displacement'] = 0.15
    settings_description['max_displacement'] = 'Scale mode shape to have specified maximum displacement'

    settings_types['rigid_modes_cg'] = 'bool'
    settings_default['rigid_modes_cg'] = False
    settings_description['rigid_modes_cg'] = 'Modify the ridid body modes such that they are defined wrt to the CG'

    settings_table = settings.SettingsTable()
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

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default)

        self.rigid_body_motion = self.settings['rigid_body_modes'].value

        # load info from dyn dictionary
        self.data.structure.add_unsteady_information(
                                            self.data.structure.dyn_dict,
                                            self.data.ts)

        # create folder for containing files if necessary
        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])
        self.folder = (self.settings['folder'] + '/' +
                       self.data.settings['SHARPy']['case'] +
                       '/beam_modal_analysis/')
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
            self.eigenvalue_table = cout.TablePrinter(7, 12, ['d', 'f', 'f', 'f', 'f', 'f', 'f'])
            self.eigenvalue_table.print_header(['mode', 'eval_real', 'eval_imag', 'freq_n (Hz)', 'freq_d (Hz)',
                                                'damping', 'period (s)'])


    def run(self):
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
        self.data.ts = len(self.data.structure.timestep_info) - 1

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
            settings.to_custom_types(full_matrix_settings, basestructuralsolver._BaseStructural().settings_types, full_matrix_settings)


            # Obtain the tangent mass, damping and stiffness matrices
            FullMglobal, FullCglobal, FullKglobal, FullQ = xbeamlib.xbeam3_asbly_dynamic(self.data.structure,
                                          self.data.structure.timestep_info[self.data.ts],
                                          full_matrix_settings)
        else:
            xbeamlib.cbeam3_solv_modal(self.data.structure,
                                       self.settings, self.data.ts,
                                       FullMglobal, FullCglobal, FullKglobal)

        # Print matrices
        if self.settings['print_matrices'].value:
            np.savetxt(self.folder + "Mglobal.dat", FullMglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')
            np.savetxt(self.folder + "Cglobal.dat", FullCglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')
            np.savetxt(self.folder + "Kglobal.dat", FullKglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')

        # Check if the damping matrix is zero (issue working)
        if self.settings['use_undamped_modes'].value:
            zero_FullCglobal = True
            for i,j in itertools.product(range(num_dof),range(num_dof)):
                if(np.absolute(FullCglobal[i, j]) > np.finfo(float).eps):
                    zero_FullCglobal = False
                    warnings.warn(
                        'Projecting a system with damping on undamped modal shapes')
                    break
        # Check if the damping matrix is skew-symmetric
        # skewsymmetric_FullCglobal = True
        # for i in range(num_dof):
        #     for j in range(i:num_dof):
        #         if((i==j) and (np.absolute(FullCglobal[i, j]) > np.finfo(float).eps)):
        #             skewsymmetric_FullCglobal = False
        #         elif(np.absolute(FullCglobal[i, j] + FullCglobal[j, i]) > np.finfo(float).eps):
        #             skewsymmetric_FullCglobal = False

        NumLambda = min(num_dof, self.settings['NumLambda'].value)

        if self.settings['use_undamped_modes'].value:

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
                if self.settings['dt'].value == 0.:
                    raise ValueError('Cannot compute the continuous eigenvalues without a dt value')
                eigenvalues = np.log(eigenvalues)/self.settings['dt'].value

            order = order[include]
            damping = damping[order]
            eigenvectors = eigenvectors[:, order]
            eigenvectors_left = eigenvectors_left[:, order].conj()

        # Modify rigid body modes for them to be defined wrt the CG
        if self.settings['rigid_modes_cg']:
            if not eigenvectors_left:
                eigenvectors = self.free_free_modes(eigenvectors, FullMglobal)

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
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.scatter(eigenvalues.real, eigenvalues.imag)
            plt.show()
            plt.savefig(self.folder + 'eigenvalues.png', transparent=True, bbox_inches='tight')


        # Write dat files
        if self.settings['write_dat'].value:
            if type(eigenvalues) == complex:
                np.savetxt(self.folder + "eigenvalues.dat", eigenvalues.view(float).reshape(-1, 2), fmt='%.12f',
                           delimiter='\t', newline='\n')
            else:
                np.savetxt(self.folder + "eigenvalues.dat", eigenvalues.view(float), fmt='%.12f',
                           delimiter='\t', newline='\n')
            np.savetxt(self.folder + "eigenvectors.dat", eigenvectors[:num_dof].real,
                       fmt='%.12f', delimiter='\t', newline='\n')
            try:
                np.savetxt(self.folder + 'frequencies.dat', freq_damped[:NumLambda],
                           fmt='%e', delimiter='\t', newline='\n')
            except NameError:
                np.savetxt(self.folder + 'frequencies.dat', freq_natural[:NumLambda],
                           fmt='%e', delimiter='\t', newline='\n')
            np.savetxt(self.filename_damp, damping[:NumLambda],
                       fmt='%e', delimiter='\t', newline='\n')

        # Write vtk
        if self.settings['write_modes_vtk'].value:
            try:
                self.data.aero
                aero_model = True
            except AttributeError:
                warnings.warn('No aerodynamic model found - unable to project the mode onto aerodynamic grid')
                aero_model = False

            if aero_model:
                write_modes_vtk(
                    self.data,
                    eigenvectors[:num_dof],
                    NumLambda,
                    self.filename_shapes,
                    self.settings['max_rotation_deg'],
                    self.settings['max_displacement'])

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

        if self.settings['keep_linear_matrices'].value:
            outdict['M'] = FullMglobal
            outdict['C'] = FullCglobal
            outdict['K'] = FullKglobal
        self.data.structure.timestep_info[self.data.ts].modal = outdict

        if self.settings['print_info']:
            for i in range(NumLambda):
                eigenvalue = eigenvalues[i]
                if self.settings['use_undamped_modes']:
                    eigenvalue = np.sqrt(eigenvalue)*1j
                omega_n = np.abs(eigenvalue)
                omega_d = np.abs(eigenvalue.imag)
                damping_ratio = -eigenvalue.real / omega_n
                f_n = omega_n / 2 / np.pi
                f_d = omega_d / 2 / np.pi
                period = 1 / f_d
                self.eigenvalue_table.print_line([i, eigenvalue.real, eigenvalue.imag, f_n, f_d,
                                                  damping_ratio, period])

        return self.data

    def scale_modes_unit_mass_matrix(self, eigenvectors, FullMglobal, eigenvectors_left=None):
        if self.settings['use_undamped_modes']:
            # mass normalise (diagonalises M and K)
            dfact = np.diag(np.dot(eigenvectors.T, np.dot(FullMglobal, eigenvectors)))
            eigenvectors = (1./np.sqrt(dfact))*eigenvectors
        else:
            # unit normalise (diagonalises A)
            if not self.rigid_body_motion:
                for ii in range(NumLambda):  # Issue - dot product = 0 when you have arbitrary damping
                    fact = 1./np.sqrt(np.dot(eigenvectors_left[:, ii], eigenvectors[:, ii]))
                    eigenvectors_left[:, ii] = fact*eigenvectors_left[:, ii]
                    eigenvectors[:, ii] = fact*eigenvectors[:, ii]

        return eigenvectors, eigenvectors_left

    def free_free_modes(self, phi, M):
        r"""
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

        pos = self.data.structure.timestep_info[self.data.ts].pos
        r_cg = cg(M)

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

        # NG - 26/7/19 - Transformation of the rigid part of the elastic modes ended up not being necessary but leaving
        # here in case it becomes useful in the future
        phit = np.block([np.zeros((phi.shape[0], 10)), phi[:, 10:]])
        phit[-10:, :10] = phirr

        return phit


def cg(M):
    Mrr = M[-10:, -10:]
    return -np.array([Mrr[2, 4], Mrr[0, 5], Mrr[1, 3]]) / Mrr[0, 0]


def scale_mode(data,eigenvector,rot_max_deg=15,perc_max=0.15):
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
    aero=data.aero
    struct=data.structure
    tsaero=data.aero.timestep_info[data.ts]
    tsstr=data.structure.timestep_info[data.ts]

    try:
        num_dof=struct.num_dof.value
    except AttributeError:
        num_dof = struct.num_dof

    eigvect=eigvect[:num_dof]

    zeta_mode=[]
    for ss in range(aero.n_surf):
        zeta_mode.append(tsaero.zeta[ss].copy())


    jj=0 # structural dofs index
    Cga0=algebra.quat2rotation(tsstr.quat)
    Cag0=Cga0.T
    for node_glob in range(struct.num_node):

        ### detect bc at node (and no. of dofs)
        bc_here=struct.boundary_conditions[node_glob]
        if bc_here==1: # clamp
            dofs_here=0
            continue
        elif bc_here==-1 or bc_here==0: 
            dofs_here=6
            jj_tra=[jj  ,jj+1,jj+2]
            jj_rot=[jj+3,jj+4,jj+5]
        jj+=dofs_here

        # retrieve element and local index
        ee,node_loc=struct.node_master_elem[node_glob,:]

        # get original position and crv 
        Ra0=tsstr.pos[node_glob,:]
        psi0=tsstr.psi[ee,node_loc,:]
        Rg0=np.dot(Cga0,Ra0) 
        Cab0=algebra.crv2rotation(psi0)
        Cbg0=np.dot(Cab0.T,Cag0)

        # update position and crv of mode
        Ra=tsstr.pos[node_glob,:]+eigvect[jj_tra]
        psi=tsstr.psi[ee,node_loc,:]+eigvect[jj_rot]
        Rg=np.dot(Cga0,Ra)
        Cab=algebra.crv2rotation(psi)
        Cbg=np.dot(Cab.T,Cag0)
    

        ### str -> aero mapping
        # some nodes may be linked to multiple surfaces...
        for str2aero_here in aero.struct2aero_mapping[node_glob]:

            # detect surface/span-wise coordinate (ss,nn)
            nn,ss=str2aero_here['i_n'],str2aero_here['i_surf']
            #print('%.2d,%.2d'%(nn,ss))

            # surface panelling
            M=aero.aero_dimensions[ss][0]
            N=aero.aero_dimensions[ss][1]

            for mm in range(M+1):

                # get position of vertex in B FoR
                zetag0=tsaero.zeta[ss][:,mm,nn]   # in G FoR, w.r.t. origin A-G
                Xb=np.dot(Cbg0,zetag0-Rg0)        # in B FoR, w.r.t. origin B

                # update vertex position
                zeta_mode[ss][:,mm,nn]=Rg+np.dot( np.dot(Cga0,Cab),Xb)

    return zeta_mode



def write_zeta_vtk(zeta,zeta_ref,filename_root):
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

        filename=filename_root+"_%02u.vtu" %(i_surf,)
        _,M,N=zeta[i_surf].shape

        M-=1
        N-=1
        point_data_dim = (M+1)*(N+1)
        panel_data_dim = M*N

        coords = np.zeros((point_data_dim, 3))
        conn = []
        panel_id = np.zeros((panel_data_dim,), dtype=int)
        panel_surf_id = np.zeros((panel_data_dim,), dtype=int)
        point_struct_id = np.zeros((point_data_dim,), dtype=int)
        point_struct_mag = np.zeros((point_data_dim,), dtype=float)

        counter = -1
        # coordinates of corners
        for i_n in range(N+1):
            for i_m in range(M+1):
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
                point_struct_mag[node_counter]=\
                    np.linalg.norm(zeta[i_surf][:, i_m, i_n]\
                                                   -zeta_ref[i_surf][:,i_m,i_n])

                if i_n < N and i_m < M:
                    counter += 1
                else:
                    continue

                conn.append([node_counter + 0,
                             node_counter + 1,
                             node_counter + M+2,
                             node_counter + M+1])
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
                                                rot_max_deg=15.,perc_max=0.15):
    """
    Writes a vtk file for each of the first ``NumLambda`` eigenvectors. When these
    are associated to the state-space form of the structural equations, only
    the displacement field is saved.
    """

    ### initialise
    aero = data.aero
    struct = data.structure
    tsaero = data.aero.timestep_info[data.ts]
    tsstr = data.structure.timestep_info[data.ts]

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

        # for i_surf in range(tsaero.n_surf):

        #     # filename=filename_root+"_%06u_%02u.vtu" %(mode,i_surf)
        #     filename = filename_root + "_%02u_%06u.vtu" % (i_surf, mode)

        #     dims = tsaero.dimensions[i_surf, :]
        #     point_data_dim = (dims[0]+1)*(dims[1]+1)  # + (dims_star[0]+1)*(dims_star[1]+1)
        #     panel_data_dim = (dims[0])*(dims[1])  # + (dims_star[0])*(dims_star[1])

        #     coords = np.zeros((point_data_dim, 3))
        #     conn = []
        #     panel_id = np.zeros((panel_data_dim,), dtype=int)
        #     panel_surf_id = np.zeros((panel_data_dim,), dtype=int)
        #     point_struct_id = np.zeros((point_data_dim,), dtype=int)
        #     point_struct_mag = np.zeros((point_data_dim,), dtype=float)

        #     counter = -1
        #     # coordinates of corners
        #     for i_n in range(dims[1]+1):
        #         for i_m in range(dims[0]+1):
        #             counter += 1
        #             coords[counter, :] = zeta_mode[i_surf][:, i_m, i_n]

        #     counter = -1
        #     node_counter = -1
        #     for i_n in range(dims[1] + 1):
        #         global_counter = aero.aero2struct_mapping[i_surf][i_n]
        #         for i_m in range(dims[0] + 1):
        #             node_counter += 1
        #             # point data
        #             point_struct_id[node_counter]=global_counter
        #             point_struct_mag[node_counter]=\
        #                 np.linalg.norm(zeta_mode[i_surf][:, i_m, i_n]\
        #                                     -tsaero.zeta[i_surf][:,i_m,i_n])

        #             if i_n < dims[1] and i_m < dims[0]:
        #                 counter += 1
        #             else:
        #                 continue

        #             conn.append([node_counter + 0,
        #                          node_counter + 1,
        #                          node_counter + dims[0]+2,
        #                          node_counter + dims[0]+1])
        #             # cell data
        #             panel_id[counter] = counter
        #             panel_surf_id[counter] = i_surf

        #     ug = tvtk.UnstructuredGrid(points=coords)
        #     ug.set_cells(tvtk.Quad().cell_type, conn)
        #     ug.cell_data.scalars = panel_id
        #     ug.cell_data.scalars.name = 'panel_n_id'
        #     ug.cell_data.add_array(panel_surf_id)
        #     ug.cell_data.get_array(1).name = 'panel_surface_id'

        #     ug.point_data.scalars = np.arange(0, coords.shape[0])
        #     ug.point_data.scalars.name = 'n_id'
        #     ug.point_data.add_array(point_struct_id)
        #     ug.point_data.get_array(1).name = 'point_struct_id'

        #     ug.point_data.add_array(point_struct_mag)
        #     ug.point_data.get_array(2).name = 'point_displacement_magnitude'

        #     write_data(ug, filename)





