import ctypes as ct
import numpy as np
import copy

from sharpy.structure.basestructure import BaseStructure
import sharpy.structure.models.beamstructures as beamstructures
import sharpy.utils.algebra as algebra
from sharpy.utils.datastructures import StructTimeStepInfo
import sharpy.utils.multibody as mb


class Beam(BaseStructure):
    def __init__(self):
        self.settings = None
        # basic info
        self.num_node_elem = -1
        self.num_node = -1
        self.num_elem = -1

        self.timestep_info = []
        self.ini_info = None
        self.dynamic_input = []

        self.connectivities = None

        self.elem_stiffness = None
        self.stiffness_db = None
        self.inv_stiffness_db = None
        self.n_stiff = 0

        self.elem_mass = None
        self.mass_db = None
        self.n_mass = 0

        self.frame_of_reference_delta = None
        self.structural_twist = None
        self.boundary_conditions = None
        self.beam_number = None

        self.lumped_mass = None
        self.lumped_mass_nodes = None
        self.lumped_mass_inertia = None
        self.lumped_mass_position = None
        self.lumped_mass_mat = None
        self.lumped_mass_mat_nodes = None
        self.n_lumped_mass = 0

        self.steady_app_forces = None

        self.elements = []

        self.master = None
        self.node_master_elem = None

        self.vdof = None
        self.fdof = None
        self.num_dof = 0

        self.fortran = dict()

        # Multibody variabes
        self.ini_mb_dict = dict()
        self.body_number = None
        self.num_bodies = None
        self.FoR_movement = None

        self.global_nodes_num = None
        self.global_elems_num = None


    def generate(self, in_data, settings):
        self.settings = settings
        # read and store data
        # type of node
        self.num_node_elem = in_data['num_node_elem']
        # node info
        self.num_node = in_data['num_node']
        self.num_elem = in_data['num_elem']
        # Body number
        try:
            self.body_number = in_data['body_number'].copy()
            self.num_bodies = np.max(self.body_number) + 1
        except KeyError:
            self.body_number = np.zeros((self.num_elem, ), dtype=int)
            self.num_bodies = 1

        # boundary conditions
        self.boundary_conditions = in_data['boundary_conditions'].copy()
        self.generate_dof_arrays()

        # ini info
        self.ini_info = StructTimeStepInfo(self.num_node, self.num_elem, self.num_node_elem,
                                           num_dof=self.num_dof, num_bodies=self.num_bodies)

        # mutibody: FoR information
        try:
            for ibody in range(self.num_bodies):
                self.ini_info.mb_FoR_pos[ibody,:] = self.ini_mb_dict["body_%02d" % ibody]["FoR_position"].copy()
                self.ini_info.mb_FoR_vel[ibody,:] = self.ini_mb_dict["body_%02d" % ibody]["FoR_velocity"].copy()
                self.ini_info.mb_FoR_acc[ibody,:] = self.ini_mb_dict["body_%02d" % ibody]["FoR_acceleration"].copy()
                self.ini_info.mb_quat[ibody,:] = self.ini_mb_dict["body_%02d" % ibody]["quat"].copy()
                self.ini_info.mb_dict = copy.deepcopy(self.ini_mb_dict)
        except KeyError:
            self.ini_info.mb_FoR_pos[0,:] = self.ini_info.for_pos
            self.ini_info.mb_FoR_vel[0,:] = self.ini_info.for_vel
            self.ini_info.mb_FoR_acc[0,:] = self.ini_info.for_acc
            self.ini_info.mb_quat[0,:] = self.ini_info.quat

        # attention, it has to be copied, not only referenced
        self.ini_info.pos = in_data['coordinates'].astype(dtype=ct.c_double, order='F')

        # connectivity information
        self.connectivities = in_data['connectivities'].astype(dtype=ct.c_int, order='F')

        self.global_elems_num = np.arange(self.num_elem)
        self.global_nodes_num = np.arange(self.num_node)

        # stiffness data
        self.elem_stiffness = in_data['elem_stiffness'].copy()
        self.stiffness_db = in_data['stiffness_db'].copy()
        (self.n_stiff, _, _) = self.stiffness_db.shape
        self.inv_stiffness_db = np.zeros_like(self.stiffness_db, dtype=ct.c_double, order='F')
        for i in range(self.n_stiff):
            self.inv_stiffness_db[i, :, :] = np.linalg.inv(self.stiffness_db[i, :, :])

        # mass data
        self.elem_mass = in_data['elem_mass'].copy()
        self.mass_db = in_data['mass_db'].copy()
        (self.n_mass, _, _) = self.mass_db.shape

        # frame of reference delta
        self.frame_of_reference_delta = in_data['frame_of_reference_delta'].copy()
        # structural twist
        self.structural_twist = in_data['structural_twist'].copy()
        # boundary conditions
        # self.boundary_conditions = in_data['boundary_conditions'].copy()
        # beam number for every elem
        try:
            self.beam_number = in_data['beam_number'].copy()
        except KeyError:
            self.beam_number = np.zeros((self.num_elem, ), dtype=int)

        # applied forces
        try:
            in_data['app_forces'][self.num_node - 1, 5]
        except IndexError:
            in_data['app_forces'] = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')

        self.steady_app_forces = in_data['app_forces'].astype(dtype=ct.c_double, order='F')

        # generate the Element array
        for ielem in range(self.num_elem):
            self.elements.append(
                beamstructures.Element(
                    ielem,
                    self.num_node_elem,
                    self.connectivities[ielem, :],
                    self.ini_info.pos[self.connectivities[ielem, :], :],
                    self.frame_of_reference_delta[ielem, :, :],
                    self.structural_twist[ielem, :],
                    self.beam_number[ielem],
                    self.elem_stiffness[ielem],
                    self.elem_mass[ielem]))
        # now we need to add the attributes like mass and stiffness index
        for ielem in range(self.num_elem):
            dictionary = dict()
            dictionary['stiffness_index'] = self.elem_stiffness[ielem]
            dictionary['mass_index'] = self.elem_mass[ielem]
            self.elements[ielem].add_attributes(dictionary)

        # psi calculation
        self.generate_psi()

        # master-slave structure
        self.generate_master_structure()

        # the timestep_info[0] is the steady state or initial state for unsteady solutions
        self.ini_info.steady_applied_forces = self.steady_app_forces.astype(dtype=ct.c_double, order='F')
        # rigid body rotations
        self.ini_info.quat = self.settings['orientation'].astype(dtype=ct.c_double, order='F')
        self.ini_info.for_pos[0:3] = self.settings['for_pos'].astype(dtype=ct.c_double, order='F')

        self.timestep_info.append(self.ini_info.copy())
        self.timestep_info[-1].steady_applied_forces = self.steady_app_forces.astype(dtype=ct.c_double, order='F')

        # lumped masses
        self.n_lumped_mass = 0
        try:
            self.lumped_mass = in_data['lumped_mass'].copy()
        except KeyError:
            self.lumped_mass = None
        else:
            self.lumped_mass_nodes = in_data['lumped_mass_nodes'].copy()
            self.lumped_mass_inertia = in_data['lumped_mass_inertia'].copy()
            self.lumped_mass_position = in_data['lumped_mass_position'].copy()
            self.n_lumped_mass += self.lumped_mass_position.shape[0]

        # lumped masses given as matrices
        try:
            self.lumped_mass_mat = in_data['lumped_mass_mat'].copy()
        except KeyError:
            self.lumped_mass_mat = None
        else:
            self.lumped_mass_mat_nodes = in_data['lumped_mass_mat_nodes'].copy()
            self.n_lumped_mass += self.lumped_mass_mat.shape[0]

        # lumped masses to element mass
        if self.lumped_mass is not None or self.lumped_mass_mat is not None:
            self.lump_masses()

        # self.generate_dof_arrays()
        self.generate_fortran()

    def generate_psi(self):
        # it will just generate the CRV for all the nodes of the element
        self.ini_info.psi = np.zeros((self.num_elem, 3, 3), dtype=ct.c_double, order='F')
        for elem in self.elements:
            self.ini_info.psi[elem.ielem, :, :] = elem.psi_ini

    def add_unsteady_information(self, dyn_dict, num_steps):
        # data storage for time dependant input
        for it in range(num_steps):
            self.dynamic_input.append(dict())

        try:
            for it in range(num_steps):
                self.dynamic_input[it]['dynamic_forces'] = dyn_dict['dynamic_forces'][it, :, :]
        except KeyError:
            for it in range(num_steps):
                self.dynamic_input[it]['dynamic_forces'] = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')

        try:
            for it in range(num_steps):
                self.dynamic_input[it]['for_pos'] = dyn_dict['for_pos'][it, :]
        except KeyError:
            for it in range(num_steps):
                self.dynamic_input[it]['for_pos'] = np.zeros((6, ), dtype=ct.c_double, order='F')

        try:
            for it in range(num_steps):
                self.dynamic_input[it]['for_vel'] = dyn_dict['for_vel'][it, :]
        except KeyError:
            for it in range(num_steps):
                self.dynamic_input[it]['for_vel'] = np.zeros((6, ), dtype=ct.c_double, order='F')

        try:
            for it in range(num_steps):
                self.dynamic_input[it]['for_acc'] = dyn_dict['for_acc'][it, :]
        except KeyError:
            for it in range(num_steps):
                self.dynamic_input[it]['for_acc'] = np.zeros((6, ), dtype=ct.c_double, order='F')

        # try:
        #     for it in range(num_steps):
        #         self.dynamic_input[it]['trayectories'] = dyn_dict['trayectories'][it, :, :]
        # except KeyError:
        #     for it in range(num_steps):
        #         self.dynamic_input[it]['trayectories'] = None

# TODO ADC: necessary? I don't think so
        # try:
            # for it in range(num_steps):
                # self.dynamic_input[it]['enforce_trajectory'] = dyn_dict['enforce_trayectory'][it, :, :]
        # except KeyError:
            # for it in range(num_steps):
                # self.dynamic_input[it]['enforce_trajectory'] = np.zeros((self.num_node, 3), dtype=bool)

    def generate_dof_arrays(self):
        self.vdof = np.zeros((self.num_node,), dtype=ct.c_int, order='F') - 1
        self.fdof = np.zeros((self.num_node,), dtype=ct.c_int, order='F') - 1

        vcounter = -1
        fcounter = -1
        for inode in range(self.num_node):
            if self.boundary_conditions[inode] == 0:
                vcounter += 1
                fcounter += 1
                self.vdof[inode] = vcounter
                self.fdof[inode] = fcounter
            elif self.boundary_conditions[inode] == -1:
                vcounter += 1
                self.vdof[inode] = vcounter
            elif self.boundary_conditions[inode] == 1:
                fcounter += 1
                self.fdof[inode] = fcounter

        self.num_dof = ct.c_int((vcounter + 1)*6)

    def generate_mass_matrix(self, mass, position, inertia):

        inertia_tensor = np.zeros((6, 6))
        pos_skew = algebra.skew(position)
        inertia_tensor[0:3, 0:3] = mass*np.eye(3)
        inertia_tensor[0:3, 3:6] = -mass*pos_skew
        inertia_tensor[3:6, 0:3] = mass*pos_skew
        inertia_tensor[3:6, 3:6] = inertia + mass*(np.dot(pos_skew.T, pos_skew))

        return inertia_tensor

    def lump_masses(self):
        if self.lumped_mass is not None:
            for i_lumped in range(self.lumped_mass.shape[0]):
                r = self.lumped_mass_position[i_lumped, :]
                m = self.lumped_mass[i_lumped]
                j = self.lumped_mass_inertia[i_lumped, :, :]

                inertia_tensor = self.generate_mass_matrix(m, r, j)
                i_lumped_node = self.lumped_mass_nodes[i_lumped]
                self.add_lumped_mass_to_element(i_lumped_node,
                                                   inertia_tensor)
        if self.lumped_mass_mat is not None:
            for i_lumped in range(self.lumped_mass_mat.shape[0]):
                inertia_tensor = self.lumped_mass_mat[i_lumped, :, :]
                i_lumped_node = self.lumped_mass_mat_nodes[i_lumped]
                self.add_lumped_mass_to_element(i_lumped_node,
                                                   inertia_tensor)

        return

    def add_lumped_mass_to_element(self, i_lumped_node, inertia_tensor, replace=False):

            i_lumped_master_elem, i_lumped_master_node_local = self.node_master_elem[i_lumped_node]

            if self.elements[i_lumped_master_elem].rbmass is None:
                # allocate memory
                self.elements[i_lumped_master_elem].rbmass = np.zeros((
                    self.elements[i_lumped_master_elem].max_nodes_elem, 6, 6))

            if replace:
                self.elements[i_lumped_master_elem].rbmass[i_lumped_master_node_local, :, :] = (
                    inertia_tensor)
            else:
                self.elements[i_lumped_master_elem].rbmass[i_lumped_master_node_local, :, :] += (
                    inertia_tensor) # += necessary in case multiple masses defined per node

    # def generate_master_structure(self):
    #     self.master = np.zeros((self.num_elem, self.num_node_elem, 2), dtype=int) - 1
    #     for i_elem in range(self.num_elem):
    #         for i_node_local in range(self.elements[i_elem].n_nodes):
    #             if not i_elem and not i_node_local:
    #                 continue
    #             j_elem = 0
    #             while self.master[i_elem, i_node_local, 0] == -1 and j_elem <= i_elem:
    #                 # for j_node_local in self.elements[j_elem].ordering:
    #                 for j_node_local in range(self.elements[j_elem].n_nodes):
    #                     if (self.connectivities[i_elem, i_node_local] ==
    #                             self.connectivities[j_elem, j_node_local]):
    #                         self.master[i_elem, i_node_local, :] = [j_elem, j_node_local]
    #                 j_elem += 1

    #     self.generate_node_master_elem()
    #     a = 1
    def generate_master_structure(self):
        self.master = np.zeros((self.num_elem, self.num_node_elem, 2), dtype=int) - 1
        for i_elem in range(self.num_elem):
            for i_node_local in range(self.elements[i_elem].n_nodes):
            # for i_node_local in self.elements[i_elem].ordering:
                if not i_elem and not i_node_local:
                    continue
                j_elem = 0
                while self.master[i_elem, i_node_local, 0] == -1 and j_elem <= i_elem:
                    # for j_node_local in self.elements[j_elem].ordering:
                    for j_node_local in range(self.elements[j_elem].n_nodes):
                    # for j_node_local in self.elements[j_elem].ordering:
                        if (self.connectivities[i_elem, i_node_local] ==
                                self.connectivities[j_elem, j_node_local]):
                            self.master[i_elem, i_node_local, :] = [j_elem, j_node_local]
                    j_elem += 1

        self.generate_node_master_elem()
        # a = 1

    def add_timestep(self, timestep_info):
        if len(timestep_info) == 0:
            # copy from ini_info
            timestep_info.append(self.ini_info.copy())
        else:
            timestep_info.append(self.timestep_info[-1].copy())

    def next_step(self):
        self.add_timestep(self.timestep_info)

    # def generate_node_master_elem(self):
    #     """
    #     Returns a matrix indicating the master element for a given node
    #     :return:
    #     """
    #     self.node_master_elem = np.zeros((self.num_node, 2), dtype=ct.c_int, order='F') - 1
    #     for i_elem in range(self.num_elem):
    #         for i_node_local in range(self.elements[i_elem].n_nodes):
    #             if self.master[i_elem, i_node_local, 0] == -1:
    #                 self.node_master_elem[self.connectivities[i_elem, i_node_local], 0] = i_elem
    #                 self.node_master_elem[self.connectivities[i_elem, i_node_local], 1] = i_node_local
    def generate_node_master_elem(self):
        """
        Returns a matrix indicating the master element for a given node
        :return:
        """
        self.node_master_elem = np.zeros((self.num_node, 2), dtype=ct.c_int, order='F') - 1
        for i_elem in range(self.num_elem):
            for i_node_local in range(self.elements[i_elem].n_nodes):
                if self.master[i_elem, i_node_local, 0] == -1:
                    if self.node_master_elem[self.connectivities[i_elem, i_node_local], 0] < 0:
                        self.node_master_elem[self.connectivities[i_elem, i_node_local], 0] = i_elem
                        self.node_master_elem[self.connectivities[i_elem, i_node_local], 1] = i_node_local
                else:
                    master_elem = self.master[i_elem, i_node_local, 0]
                    master_node = self.master[i_elem, i_node_local, 1]
                    if self.node_master_elem[self.connectivities[i_elem, i_node_local], 0] < 0:
                        self.node_master_elem[self.connectivities[i_elem, i_node_local], 0] = master_elem
                        self.node_master_elem[self.connectivities[i_elem, i_node_local], 1] = master_node


    def generate_fortran(self):
        # steady, no time-dependant information
        self.fortran['num_nodes'] = np.zeros((self.num_elem,), dtype=ct.c_int, order='F')
        for elem in self.elements:
            self.fortran['num_nodes'][elem.ielem] = elem.n_nodes

        self.fortran['num_mem'] = np.zeros_like(self.fortran['num_nodes'], dtype=ct.c_int)
        for elem in self.elements:
            self.fortran['num_mem'][elem.ielem] = elem.num_mem

        self.fortran['connectivities'] = self.connectivities.astype(ct.c_int, order='F') + 1
        self.fortran['master'] = self.master.astype(dtype=ct.c_int, order='F') + 1
        self.fortran['node_master_elem'] = self.node_master_elem.astype(dtype=ct.c_int, order='F') + 1

        self.fortran['length'] = np.zeros_like(self.fortran['num_nodes'], dtype=ct.c_double, order='F')
        for elem in self.elements:
            self.fortran['length'][elem.ielem] = elem.length

        self.fortran['mass'] = self.mass_db.astype(ct.c_double, order='F')
        self.fortran['stiffness'] = self.stiffness_db.astype(ct.c_double, order='F')
        self.fortran['inv_stiffness'] = self.inv_stiffness_db.astype(ct.c_double, order='F')
        self.fortran['mass_indices'] = self.elem_mass.astype(ct.c_int, order='F') + 1
        self.fortran['stiffness_indices'] = self.elem_stiffness.astype(ct.c_int, order='F') + 1

        self.fortran['frame_of_reference_delta'] = self.frame_of_reference_delta.astype(ct.c_double, order='F')

        self.fortran['vdof'] = self.vdof.astype(ct.c_int, order='F') + 1
        self.fortran['fdof'] = self.fdof.astype(ct.c_int, order='F') + 1

        # self.fortran['steady_applied_forces'] = self.steady_app_forces.astype(dtype=ct.c_double, order='F')

        # undeformed structure matrices
        self.fortran['pos_ini'] = self.ini_info.pos.astype(dtype=ct.c_double, order='F')
        self.fortran['psi_ini'] = self.ini_info.psi.astype(dtype=ct.c_double, order='F')

        max_nodes_elem = self.elements[0].max_nodes_elem
        rbmass_temp = np.zeros((self.num_elem, max_nodes_elem, 6, 6))
        for elem in self.elements:
            for inode in range(elem.n_nodes):
                if elem.rbmass is not None:
                    rbmass_temp[elem.ielem, inode, :, :] += elem.rbmass[inode, :, :]
        self.fortran['rbmass'] = rbmass_temp.astype(dtype=ct.c_double, order='F')

        if self.settings['unsteady']:
            pass
            # TODO
            # if self.dynamic_forces_amplitude is not None:
            #     self.dynamic_forces_amplitude_fortran = self.dynamic_forces_amplitude.astype(dtype=ct.c_double, order='F')
            #     self.dynamic_forces_time_fortran = self.dynamic_forces_time.astype(dtype=ct.c_double, order='F')
            # else:
            #     self.dynamic_forces_amplitude_fortran = np.zeros((self.num_node, 6), dtype=ct.c_double, order='F')
            #     self.dynamic_forces_time_fortran = np.zeros((self.n_tsteps, 1), dtype=ct.c_double, order='F')
            #
            # if self.forced_vel is not None:
            #     self.forced_vel_fortran = self.forced_vel.astype(dtype=ct.c_double, order='F')
            # else:
            #     self.forced_vel_fortran = np.zeros((self.n_tsteps, 6), dtype=ct.c_double, order='F')
            #
            # if self.forced_acc is not None:
            #     self.forced_acc_fortran = self.forced_acc.astype(dtype=ct.c_double, order='F')
            # else:
            #     self.forced_acc_fortran = np.zeros((self.n_tsteps, 6), dtype=ct.c_double, order='F')

    # def update_orientation(self, quat=None, ts=-1):
    #     if quat is None:
    #         quat = algebra.euler2quat(self.timestep_info[ts].for_pos[3:6])
    #     self.timestep_info[ts].update_orientation(quat)  # Cga going in here

    def integrate_position(self, ts, dt):
        try:
            ts.for_pos[0:3] += (
                dt*np.dot(ts.cga(),
                          ts.for_vel[0:3]))
        except AttributeError:
            self.timestep_info[ts].for_pos[0:3] += (
                dt*np.dot(self.timestep_info[ts].cga(),
                          self.timestep_info[ts].for_vel[0:3]))

    def nodal_premultiply_inv_T_transpose(self, nodal, tstep, filter=np.array([True]*6)):
        # nodal_t = np.zeros_like(nodal, dtype=ct.c_double, order='F')
        nodal_t = nodal.copy(order='F')
        for i_node in range(self.num_node):
            # get master elem and i_local_node
            i_master_elem, i_local_node = self.node_master_elem[i_node, :]
            crv = tstep.psi[i_master_elem, i_local_node, :]
            inv_tanT = algebra.crv2invtant(crv)
            temp = np.zeros((6,))
            temp[0:3] = np.dot(inv_tanT, nodal[i_node, 0:3])
            temp[3:6] = np.dot(inv_tanT, nodal[i_node, 3:6])
            for i in range(6):
                if filter[i]:
                    nodal_t[i_node, i] = temp[i]

        return nodal_t

    def get_body(self, ibody):
        """
        get_body

        Extract the body number ``ibody`` from a multibody system

        This function returns a :class:`~sharpy.structure.models.beam.Beam` class (``ibody_beam``)
        that only includes the body number ``ibody`` of the original system

        Args:
            self(:class:`~sharpy.structure.models.beam.Beam`): structural information of the multibody system
            ibody(int): body number to be extracted

        Returns:
        	ibody_beam(:class:`~sharpy.structure.models.beam.Beam`): structural information of the isolated body
        """

        ibody_beam = Beam()

        # Define the nodes and elements belonging to the body
        ibody_beam.global_elems_num, ibody_beam.global_nodes_num = mb.get_elems_nodes_list(self, ibody)

        # Renaming for clarity
        ibody_elements = ibody_beam.global_elems_num
        ibody_nodes = ibody_beam.global_nodes_num

        # Assign all the properties to the new StructTimeStepInfo
        ibody_beam.settings = self.settings.copy()

        ibody_beam.num_node_elem = self.num_node_elem.astype(dtype=ct.c_int, order='F', copy=True)
        ibody_beam.num_node = len(ibody_beam.global_nodes_num)
        ibody_beam.num_elem = len(ibody_beam.global_elems_num)

        ibody_beam.connectivities = self.connectivities[ibody_elements,:]
        # Renumber the connectivities
        int_list_nodes = np.arange(0, ibody_beam.num_node, 1)
        for ielem in range(ibody_beam.num_elem):
            for inode_in_elem in range(ibody_beam.num_node_elem):
                ibody_beam.connectivities[ielem, inode_in_elem] = int_list_nodes[ibody_nodes == ibody_beam.connectivities[ielem, inode_in_elem]]

        # TODO: I could copy only the needed stiffness and masses to save storage
        ibody_beam.elem_stiffness = self.elem_stiffness[ibody_elements].astype(dtype=ct.c_int, order='F', copy=True)
        ibody_beam.stiffness_db = self.stiffness_db.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_beam.inv_stiffness_db = self.inv_stiffness_db.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_beam.n_stiff = self.n_stiff

        ibody_beam.elem_mass = self.elem_mass[ibody_elements].astype(dtype=ct.c_int, order='F', copy=True)
        ibody_beam.mass_db = self.mass_db.astype(dtype=ct.c_double, order='F', copy=True)
        ibody_beam.n_mass = self.n_mass

        ibody_beam.frame_of_reference_delta = self.frame_of_reference_delta[ibody_elements,:,:].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_beam.structural_twist = self.structural_twist[ibody_elements, :].astype(dtype=ct.c_double, order='F', copy=True)
        ibody_beam.boundary_conditions = self.boundary_conditions[ibody_nodes].astype(dtype=ct.c_int, order='F', copy=True)
        ibody_beam.beam_number = self.beam_number[ibody_elements].astype(dtype=ct.c_int, order='F', copy=True)

        if not self.lumped_mass_nodes is None:
            is_first = True
            ibody_beam.n_lumped_mass = 0
            for inode in range(len(self.lumped_mass_nodes)):
                if self.lumped_mass_nodes[inode] in ibody_nodes:
                    if is_first:
                        is_first = False
                        ibody_beam.lumped_mass_nodes = int_list_nodes[ibody_nodes == self.lumped_mass_nodes[inode]]
                        ibody_beam.lumped_mass = np.array([self.lumped_mass[inode]])
                        ibody_beam.lumped_mass_inertia = np.array([self.lumped_mass_inertia[inode]])
                        ibody_beam.lumped_mass_position = np.array([self.lumped_mass_position[inode]])
                        ibody_beam.n_lumped_mass += 1
                    else:
                        ibody_beam.lumped_mass_nodes = np.concatenate((ibody_beam.lumped_mass_nodes ,int_list_nodes[ibody_nodes == self.lumped_mass_nodes[inode]]), axis=0)
                        ibody_beam.lumped_mass = np.concatenate((ibody_beam.lumped_mass ,np.array([self.lumped_mass[inode]])), axis=0)
                        ibody_beam.lumped_mass_inertia = np.concatenate((ibody_beam.lumped_mass_inertia ,np.array([self.lumped_mass_inertia[inode]])), axis=0)
                        ibody_beam.lumped_mass_position = np.concatenate((ibody_beam.lumped_mass_position ,np.array([self.lumped_mass_position[inode]])), axis=0)
                        ibody_beam.n_lumped_mass += 1

        if not self.lumped_mass_mat is None:
            is_first = True
            for inode in range(len(self.lumped_mass_mat_nodes)):
                if self.lumped_mass_mat_nodes[inode] in ibody_nodes:
                    if is_first:
                        is_first = False
                        ibody_beam.lumped_mass_mat_nodes = int_list_nodes[ibody_nodes == self.lumped_mass_mat_nodes[inode]]
                        ibody_beam.lumped_mass_mat = np.array([self.lumped_mass_mat[inode, :, :]])
                    else:
                        ibody_beam.lumped_mass_mat_nodes = np.concatenate((ibody_beam.lumped_mass_mat_nodes , int_list_nodes[ibody_nodes == self.lumped_mass_mat_nodes[inode]]), axis=0)
                        ibody_beam.lumped_mass_mat = np.concatenate((ibody_beam.lumped_mass_mat ,np.array([self.lumped_mass_mat[inode, :, :]])), axis=0)

        ibody_beam.steady_app_forces = self.steady_app_forces[ibody_nodes,:].astype(dtype=ct.c_double, order='F', copy=True)

        ibody_beam.num_bodies = 1

        ibody_beam.body_number = self.body_number[ibody_elements].astype(dtype=ct.c_int, order='F', copy=True)

        ibody_beam.generate_dof_arrays()

        ibody_beam.ini_info = self.ini_info.get_body(self, ibody_beam.num_dof, ibody)
        ibody_beam.timestep_info = self.timestep_info[-1].get_body(self, ibody_beam.num_dof, ibody)

        # generate the Element array
        for ielem in range(ibody_beam.num_elem):
            ibody_beam.elements.append(
                beamstructures.Element(
                    ielem,
                    ibody_beam.num_node_elem,
                    ibody_beam.connectivities[ielem, :],
                    ibody_beam.ini_info.pos[ibody_beam.connectivities[ielem, :], :],
                    ibody_beam.frame_of_reference_delta[ielem, :, :],
                    ibody_beam.structural_twist[ielem, :],
                    ibody_beam.beam_number[ielem],
                    ibody_beam.elem_stiffness[ielem],
                    ibody_beam.elem_mass[ielem]))
        # now we need to add the attributes like mass and stiffness index
        for ielem in range(ibody_beam.num_elem):
            dictionary = dict()
            dictionary['stiffness_index'] = ibody_beam.elem_stiffness[ielem]
            dictionary['mass_index'] = ibody_beam.elem_mass[ielem]
            ibody_beam.elements[ielem].add_attributes(dictionary)

        ibody_beam.generate_master_structure()

        if ibody_beam.lumped_mass is not None or ibody_beam.lumped_mass_mat is not None:
            ibody_beam.lump_masses()

        ibody_beam.generate_fortran()

        return ibody_beam
