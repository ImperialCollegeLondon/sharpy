import os

import numpy as np

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
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
    settings_default['text_file_name'] = 'liftdistribution'
    settings_description['text_file_name'] = 'Text file name'

    settings_default['coefficients'] = True
    settings_types['coefficients'] = 'bool'
    settings_description['coefficients'] = 'Calculate aerodynamic lift coefficients'

    settings_types['rho'] = 'float'
    settings_default['rho'] = 1.225
    settings_description['rho'] = 'Reference freestream density [kg/m³]'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None
        self.folder = None
        self.caller = None

    def initialise(self, data, custom_settings=None, restart=False, caller=None):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.caller = caller
        self.folder = data.output_folder + '/liftdistribution/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run(self, **kwargs):
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
        numb_col = 6
        header = "x,y,z,fx,fy,fz"
        # get aero forces
        # get rotation matrix
        cga = algebra.quat2rotation(struct_tstep.quat)
        if self.settings["coefficients"]:
            # TODO: add nondimensional spanwise column y/s
            header += ", cfx, cfy, cfz"
            numb_col += 3
        lift_distribution = np.zeros((N_nodes, numb_col))

        for inode in range(N_nodes):
            if self.data.aero.data_dict['aero_node'][inode]:
                local_node = self.data.aero.struct2aero_mapping[inode][0]["i_n"]
                ielem, inode_in_elem = self.data.structure.node_master_elem[inode]
                i_surf = int(self.data.aero.surface_distribution[ielem])
                # get c_gb                
                cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem, :])
                cgb = np.dot(cga, cab)
                # Get c_bs
                urel, dir_urel = aeroutils.magnitude_and_direction_of_relative_velocity(struct_tstep.pos[inode, :],
                                                                                        struct_tstep.pos_dot[inode, :],
                                                                                        struct_tstep.for_vel[:],
                                                                                        cga,
                                                                                        aero_tstep.u_ext[i_surf][:, :,
                                                                                        local_node])
                dir_span, span, dir_chord, chord = aeroutils.span_chord(local_node, aero_tstep.zeta[i_surf])
                # Stability axes - projects forces in B onto S
                c_bs = aeroutils.local_stability_axes(cgb.T.dot(dir_urel), cgb.T.dot(dir_chord))
                aero_forces = c_bs.T.dot(forces[inode, :3])
                # Store data in export matrix
                lift_distribution[inode, 3:6] = aero_forces
                lift_distribution[inode, 2] = struct_tstep.pos[inode, 2]  # z
                lift_distribution[inode, 1] = struct_tstep.pos[inode, 1]  # y
                lift_distribution[inode, 0] = struct_tstep.pos[inode, 0]  # x
                if self.settings["coefficients"]:
                    # Get lift coefficient
                    for idim in range(3):
                        lift_distribution[inode, 6+idim] = np.sign(aero_forces[idim]) * np.linalg.norm(aero_forces[idim]) \
                                                    / (0.5 * self.settings['rho'] \
                                                        * np.linalg.norm(urel) ** 2 * span * chord)  
                        # Check if shared nodes from different surfaces exist (e.g. two wings joining at symmetry plane)
                        # Leads to error since panel area just donates for half the panel size while lift forces is summed up
                        lift_distribution[inode, 6+idim] /= len(self.data.aero.struct2aero_mapping[inode])

        # Export lift distribution data
        np.savetxt(os.path.join(self.folder,  self.settings['text_file_name'] + '_ts{}'.format(str(self.data.ts)) + '.txt'), lift_distribution,
                   fmt='%10e,' * (numb_col - 1) + '%10e', delimiter=", ", header=header)
