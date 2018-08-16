"""
@modified   Alfonso del Carre
"""

import ctypes as ct
import numpy as np
import scipy as sc
import os
import itertools
from tvtk.api import tvtk, write_data

import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings

import scipy.linalg


@solver
class Modal(BaseSolver):
    solver_id = 'Modal'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['folder'] = 'str'
        self.settings_default['folder'] = './output'

        self.settings_types['NumLambda'] = 'int' # no. of modes to save in vtk
        self.settings_default['NumLambda'] = 20   

        self.settings_types['write_modes_vtk'] = 'bool' # write mode shapes in vtk file
        self.settings_default['write_modes_vtk'] = True        

        self.settings_types['print_matrices'] = 'bool' # print M,C,K matrices to dat file
        self.settings_default['print_matrices'] = False

        self.settings_types['write_dat'] = 'bool' # write modes shapes/freq./damp. to dat file
        self.settings_default['write_dat'] = True

        self.settings_types['keep_linear_matrices'] = 'bool' # attach linear M,C,K matrices to output dictionary
        self.settings_default['keep_linear_matrices'] = True



    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings,
                                 self.settings_types,
                                 self.settings_default)

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


    def run(self):

        # Initialize matrices
        num_dof = self.data.structure.num_dof.value
        FullMglobal = np.zeros((num_dof, num_dof),
                               dtype=ct.c_double, order='F')
        FullKglobal = np.zeros((num_dof, num_dof),
                               dtype=ct.c_double, order='F')
        FullCglobal = np.zeros((num_dof, num_dof),
                               dtype=ct.c_double, order='F')

        # Obtain the matrices from the fortran library
        xbeamlib.cbeam3_solv_modal(self.data.structure,
                                   self.settings, self.data.ts,
                                   FullMglobal, FullCglobal, FullKglobal)

        # Print matrices
        if(self.settings['print_matrices'].value):
            np.savetxt(self.folder + "Mglobal.dat", FullMglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')
            np.savetxt(self.folder + "Cglobal.dat", FullCglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')
            np.savetxt(self.folder + "Kglobal.dat", FullKglobal, fmt='%.12f',
                       delimiter='\t', newline='\n')

        # Check if the damping matrix is zero
        zero_FullCglobal = True
        for i,j in itertools.product(range(num_dof),range(num_dof)):
            if(np.absolute(FullCglobal[i, j]) > np.finfo(float).eps):
                zero_FullCglobal = False
                break

        # Check if the damping matrix is skew-symmetric
        # skewsymmetric_FullCglobal = True
        # for i in range(num_dof):
        #     for j in range(i:num_dof):
        #         if((i==j) and (np.absolute(FullCglobal[i, j]) > np.finfo(float).eps)):
        #             skewsymmetric_FullCglobal = False
        #         elif(np.absolute(FullCglobal[i, j] + FullCglobal[j, i]) > np.finfo(float).eps):
        #             skewsymmetric_FullCglobal = False

        NumLambda = min(self.data.structure.num_dof.value,
                                               self.settings['NumLambda'].value)

        if(zero_FullCglobal):
            # DAMPING MATRIX EQUAL TO ZERO
            print("Damping matrix equal to zero")

            # Solve the eigenvalues problem
            eigenvalues,eigenvectors=np.linalg.eig(
                                       np.linalg.solve(FullMglobal,FullKglobal))
            # Define vibration frequencies and damping
            vib_freq = np.sqrt(eigenvalues)
            damping = np.zeros((num_dof,),)

        else:
            # CASES WITH DAMPING
            print("Damping matrix different from zero")
            NumLambda = 2*NumLambda

            # Initialize eigenvalues and eigenvectors problem
            eigenvectors = np.zeros((num_dof, 2*num_dof),
                                                   dtype=ct.c_double, order='F')
            A = np.zeros((2*num_dof, 2*num_dof), dtype=ct.c_double, order='F')

            # State-space model
            Minv_neg=-np.linalg.inv(FullMglobal)
            A[range(num_dof),range(num_dof,2*num_dof)]=1.
            A[num_dof:,:num_dof] = np.dot(Minv_neg,FullKglobal)
            A[num_dof:,num_dof:] = np.dot(Minv_neg,FullCglobal)

            # Solve the eigenvalues problem
            # (with unit eigenvectors)
            #eigenvalues, aux_eigenvectors = np.linalg.eig(A)
            eigenvalues, aux_eigenvectors_left, aux_eigenvectors = sc.linalg.eig(A,left=True,right=True)
            vib_freq=np.abs(eigenvalues)
            damping=np.zeros_like(vib_freq)
            iiflex=vib_freq>1e-10
            damping[iiflex]=eigenvalues[iiflex].real/vib_freq[iiflex]

            # Remove eigenvectors second part and re-normalise first part
            for aux_i in range(2*num_dof):
                sign = np.ones((num_dof,),)
                for j in range(num_dof):
                    if((np.real(aux_eigenvectors[j,aux_i])>np.imag(aux_eigenvectors[j,aux_i])) and (np.real(aux_eigenvectors[j,aux_i]) < 0)):
                        sign[j] = -1
                    elif((np.real(aux_eigenvectors[j,aux_i]) < np.imag(aux_eigenvectors[j,aux_i])) and (np.imag(aux_eigenvectors[j,aux_i]) < 0)):
                        sign[j] = -1
                eigenvectors[:, aux_i] = sign*np.absolute(aux_eigenvectors[:num_dof,aux_i])/np.linalg.norm(aux_eigenvectors[:num_dof,aux_i])

        # Order
        order = np.argsort(vib_freq)
        damping = damping[order]
        vib_freq = vib_freq[order]
        eigenvectors = eigenvectors[:,order]

        # Write dat files
        if(self.settings['write_dat'].value):
            np.savetxt(self.folder + "eigenvalues.dat", eigenvalues, fmt='%.12f',
                       delimiter='\t', newline='\n')
            if (not zero_FullCglobal):
                np.savetxt(self.folder + "eigenvectors.dat", aux_eigenvectors[:num_dof,:],
                       fmt='%.12f', delimiter='\t', newline='\n')
            np.savetxt(self.filename_freq, vib_freq[0:NumLambda],
                       fmt='%e', delimiter='\t', newline='\n')
            np.savetxt(self.filename_damp, damping[0:NumLambda],
                       fmt='%e', delimiter='\t', newline='\n')

        # Write vtk
        if self.settings['write_modes_vtk'].value:
            self.write_modes_vtk(np.real(eigenvectors),NumLambda)


        # attach output to timestep_info
        outdict=dict()
        outdict['eigenvalues']=eigenvalues
        outdict['eigenvectors']=eigenvectors[:,:NumLambda] 
        outdict['damping']=damping
        outdict['vib_freq']=vib_freq
        if self.settings['keep_linear_matrices'].value:
            outdict['M']=FullMglobal
            #outdict['Minv']=-Minv_neg
            outdict['C']=FullCglobal
            outdict['K']=FullKglobal
        self.data.structure.timestep_info[self.data.ts].modal=outdict

        return self.data


    def write_modes_vtk(self, eigenvectors, NumLambda):

        # Define and initialize some variables
        num_nodes = self.data.structure.num_node
        num_elem = self.data.structure.num_elem

        node_id = np.zeros((num_nodes,),)
        elem_id = np.zeros((num_elem,),)
        conn = np.zeros((num_elem, 3),)
        coordinates = np.zeros((self.data.structure.num_node, 3),)
        #angle = np.zeros((self.data.structure.num_node, 3),)

        # Write the modal shapes for each mode
        # They are writen as time steps to make visualization easier

        for iMode in range(NumLambda):
            inode_free = 0
            #Define the coordinates
            for inode in range(self.data.structure.num_node):
                if(self.data.structure.boundary_conditions[inode] == 1):
                    # Clamped point
                    coordinates[inode,:] = self.data.structure.timestep_info[self.data.ts].pos[inode,:]
                    #angle[inode,:] = self.data.structure.timestep_info[self.data.ts].psi[inode,:]
                else:
                    coordinates[inode,:] = (self.data.structure.timestep_info[self.data.ts].pos[inode,:] +
                                            np.real(eigenvectors[inode_free*6+0:inode_free*6+3,iMode]))
                    #angle[inode,:] = (self.data.structure.timestep_info[self.data.ts].psi[ielem,inodeinelem,:] +
                    #                        np.real(eigenvectors[inode_free*6+4:inode_free*6+6,iMode]))
                    inode_free = inode_free + 1

            # Define node and element number and the connectivities
            for i_node in range(num_nodes):
                node_id[i_node] = i_node

            for i_elem in range(num_elem):
                conn[i_elem, :] = self.data.structure.elements[i_elem].reordered_global_connectivities
                elem_id[i_elem] = i_elem

            # Define structure to be writen
            ug = tvtk.UnstructuredGrid(points=coordinates)
            ug.set_cells(tvtk.Line().cell_type, conn)
            ug.cell_data.scalars = elem_id
            ug.cell_data.scalars.name = 'elem_id'
            ug.point_data.scalars = node_id
            ug.point_data.scalars.name = 'node_id'

            # Write data
            write_data(ug, ("%s%06u.vtu" % (self.filename_shapes, iMode)))

            # Write text version of the eigenvectors
            # if(self.settings['print_matrices'].value):
            #     np.savetxt(("%s%06u.dat" % (self.filename_shapes, iMode)), coordinates, fmt='%.12f',
            #                delimiter='\t', newline='\n')



def sort_eigvals(eigv,eigabsv,tol=1e-6):
    ''' sort by magnitude (frequency) and imaginary part if complex conj '''

    order=np.argsort(np.abs(eigv))
    eigv=eigv[order]

    for ii in range(len(eigv)-1):
        # check if ii and ii+1 are the same eigenvalue
        if np.abs(eigv[ii].imag+eigv[ii+1].imag)/eigabsv[ii]<tol:
            if np.abs(eigv[ii].real-eigv[ii+1].real)/eigabsv[ii]<tol:

                # swap if required
                if eigv[ii].imag>eigv[ii+1].imag:
                    temp=eigv[ii]
                    eigv[ii]=eigv[ii+1]
                    eigv[ii+1]=temp

                    temp=order[ii]
                    order[ii]=order[ii+1]
                    order[ii+1]=temp

    return order,eigv