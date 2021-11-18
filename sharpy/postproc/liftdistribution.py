import os

import numpy as np

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.aero.utils.mapping as mapping
import sharpy.utils.algebra as algebra
import sharpy.aero.utils.utils as aeroutils


@solver
class LiftDistribution(BaseSolver):
    """LiftDistribution
    
    Calculates and exports the lift distribution on lifting surfaces
    
    """
    solver_id = 'LiftDistribution'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['text_file_name'] = 'str'
    settings_default['text_file_name'] = 'lift_distribution'
    settings_description['text_file_name'] = 'Text file name'

    settings_default['coefficients'] = True
    settings_types['coefficients'] = 'bool'
    settings_description['coefficients'] = 'Calculate aerodynamic lift coefficients'

    settings_types['q_ref'] = 'float'
    settings_default['q_ref'] = 1
    settings_description['q_ref'] = 'Reference dynamic pressure'
    settings_types['rho'] = 'float'
    settings_default['rho'] = 1.225
    settings_description['rho'] = 'Reference density  [kg/m³]'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None
        self.folder = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.caller = caller
        self.folder = data.output_folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run(self, online=False):
        self.lift_distribution(self.data.structure.timestep_info[self.data.ts],
                               self.data.aero.timestep_info[self.data.ts])
        return self.data

    def lift_distribution(self, struct_tstep, aero_tstep):
        # Force mapping
        forces = mapping.aero2struct_force_mapping(
            aero_tstep.forces + aero_tstep.dynamic_forces,
            self.data.aero.struct2aero_mapping,
            aero_tstep.zeta,
            struct_tstep.pos,
            struct_tstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.connectivities,
            struct_tstep.cag(),
            self.data.aero.data_dict)
        # Prepare output matrix and file 
        N_nodes = self.data.structure.num_node
        numb_col = 4
        header = "x,y,z,fz"
        # get aero forces
        lift_distribution = np.zeros((N_nodes, numb_col))
        # get rotation matrix
        cga = algebra.quat2rotation(struct_tstep.quat)
        if self.settings["coefficients"]:
            # TODO: add nondimensional spanwise column y/s
            header += ", y/s ,cl"
            numb_col += 2
            lift_distribution = np.concatenate((lift_distribution, np.zeros((N_nodes, 2))), axis=1)

        for inode in range(N_nodes):
        
            lift_distribution[inode, 2] = struct_tstep.pos[inode, 2]  # z
            lift_distribution[inode, 1] = struct_tstep.pos[inode, 1]  # y
            lift_distribution[inode, 0] = struct_tstep.pos[inode, 0]  # x
            if self.data.aero.data_dict['aero_node'][inode]:
                if len(self.data.aero.struct2aero_mapping[inode]) > 0:
                    # print("local node = ", inode, ",   ", self.data.aero.struct2aero_mapping[inode])
                    local_node = self.data.aero.struct2aero_mapping[inode][0]["i_n"]
                    ielem, inode_in_elem = self.data.structure.node_master_elem[inode]
                    i_surf = int(self.data.aero.surface_distribution[ielem])
                    # get c_gb                
                    cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem, :])
                    cgb = np.dot(cga, cab)
                    # Get c_bs
                    urel, dir_urel = self.magnitude_and_direction_of_relative_velocity(struct_tstep.pos[inode, :],
                                                                                            struct_tstep.pos_dot[inode, :],
                                                                                            struct_tstep.for_vel[:],
                                                                                            cga,
                                                                                            aero_tstep.u_ext[i_surf][:, :,
                                                                                            local_node])                
                    dir_span, span, dir_chord, chord = self.span_chord(local_node, aero_tstep.zeta[i_surf])
                    # Stability axes - projects forces in B onto S
                    c_bs = self.local_stability_axes(cgb.T.dot(dir_urel), cgb.T.dot(dir_chord))
                    lift_force = c_bs.T.dot(forces[inode, :3])[2]
                    # Store data in export matrix
                    lift_distribution[inode, 3] = lift_force
                    if self.settings["coefficients"]:
                        # Get non-dimensional spanwise coordinate y/s
                        lift_distribution[inode, 4] = lift_distribution[inode, 1]/max(abs(aero_tstep.zeta[i_surf][1,0,:]))
                        # Get lift coefficient
                        lift_distribution[inode, 5] = np.sign(lift_force) * np.linalg.norm(lift_force) \
                                                    / (0.5 * self.settings['rho'] \
                                                        * np.linalg.norm(urel) ** 2 * span * chord) 
                        # Check if shared nodes from different surfaces exist (e.g. two wings joining at symmetry plane)
                        # Leads to error since panel area just donates for half the panel size while lift forces is summed up
                        lift_distribution[inode, 5] /= len(self.data.aero.struct2aero_mapping[inode])
     
        # Export lift distribution data
        np.savetxt(os.path.join(self.folder,self.settings['text_file_name']+'_' + str(self.data.ts) + '.csv'), lift_distribution, fmt='%10e,'*(numb_col-1)+'%10e', delimiter = ", ", header= header)
    
    def calculate_strip_area(self, aero_tstep):
        # Function to get the area of a strip, which has half of the panel area
        # of each adjacent panel. For one strip, all chordwise panels (from leading
        # to trailing edge) connected to the beam node are accounted.
        strip_area = []
        for i_surf in range(self.data.aero.n_surf):
            N_panel = self.data.aero.dimensions[i_surf][1]
            array_panel_area = np.zeros((N_panel))
            # the area is calculated for all chordwise panels together
            for i_panel in range(N_panel):
                array_panel_area[i_panel] = algebra.panel_area(
                    aero_tstep.zeta[i_surf][:, -1, i_panel],
                    aero_tstep.zeta[i_surf][:, 0, i_panel], 
                    aero_tstep.zeta[i_surf][:, 0, i_panel+1], 
                    aero_tstep.zeta[i_surf][:, -1, i_panel+1])
            # assume each strip shares half of each adjacent panel
            strip_area.append(np.zeros((N_panel+1)))
            strip_area[i_surf][:-1] = abs(np.roll(array_panel_area[:],1)+array_panel_area[:]).reshape(-1)
            strip_area[i_surf][0] = abs(array_panel_area[0])            
            strip_area[i_surf][-1] = abs(array_panel_area[-1])         
            strip_area[i_surf][:] /= 2
        
        return strip_area

    def magnitude_and_direction_of_relative_velocity(self, displacement, displacement_vel, for_vel, cga, uext):
        """
        Calculates the magnitude and direction of the relative velocity ``u_rel``

        Args:
            displacement (np.array): Unit vector in the direction of the free stream velocity expressed in B frame.
            displacement_vel (np.array): Unit vector in the direction of the local chord expressed in B frame.
            for_vel
            cga
            uext
        Returns:
            tuple: ``u_rel``, ``dir_u_rel``
        """
        urel = (displacement_vel+
                for_vel[0:3] +
                algebra.cross3(for_vel[3:6], displacement))
        urel = -np.dot(cga, urel)
        urel += np.average(uext, axis=1)

        dir_urel = algebra.unit_vector(urel)
        return urel, dir_urel

    def local_stability_axes(self, dir_urel, dir_chord):
        """
        Rotates the body axes onto stability axes. This rotation is equivalent to the projection of a vector in S onto B.

        The stability axes are defined as:

            * ``x_s``: parallel to the free stream

            * ``z_s``: perpendicular to the free stream and part of the plane formed by the local chord and the vertical
            body axis ``z_b``.

            * ``y_s``: completes the set

        Args:
            dir_urel (np.array): Unit vector in the direction of the free stream velocity expressed in B frame.
            dir_chord (np.array): Unit vector in the direction of the local chord expressed in B frame.

        Returns:
            np.array: Rotation matrix from B to S, equivalent to the projection matrix :math:`C^{BS}` that projects a
            vector from S onto B.
        """
        xs = dir_urel

        zb = np.array([0, 0, 1.])
        zs = algebra.cross3(algebra.cross3(dir_chord, zb), dir_urel)

        ys = -algebra.cross3(xs, zs)

        return algebra.triad2rotation(xs, ys, zs)

    def span_chord(self, i_node_surf, zeta):
        """
        Retrieve the local span and local chord

        Args:
            i_node_surf (int): Node index in aerodynamic surface
            zeta (np.array): Aerodynamic surface coordinates ``(3 x n_chord x m_span)``

        Returns:
            tuple: ``dir_span``, ``span``, ``dir_chord``, ``chord``
        """
        N = zeta.shape[2] - 1 # spanwise vertices in surface (-1 for index)

        # Deal with the extremes
        if i_node_surf == 0:
            node_p = 1
            node_m = 0
        elif i_node_surf == N:
            node_p = N
            node_m = N - 1
        else:
            node_p = i_node_surf + 1
            node_m = i_node_surf - 1

        # Define the span and the span direction
        dir_span = 0.5 * (zeta[:, 0, node_p] - zeta[:, 0, node_m])

        span = np.linalg.norm(dir_span)
        dir_span = algebra.unit_vector(dir_span)

        # Define the chord and the chord direction
        dir_chord = zeta[:, -1, i_node_surf] - zeta[:, 0, i_node_surf]
        chord = np.linalg.norm(dir_chord)
        dir_chord = algebra.unit_vector(dir_chord)

        return dir_span, span, dir_chord, chord