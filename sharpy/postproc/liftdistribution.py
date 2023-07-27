import os

import numpy as np

import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
from sharpy.utils.datastructures import init_matrix_structure, standalone_ctypes_pointer
import sharpy.aero.utils.mapping as mapping
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
    settings_default['text_file_name'] = 'lift_distribution.txt'
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

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.ts_max = len(self.data.structure.timestep_info)
        self.caller = caller
        self.folder = data.output_folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run(self, **kwargs):

        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        if not online:
            for self.ts in range(self.ts_max):
                self.lift_distribution()
            cout.cout_wrap('...Finished', 1)
        else:
            self.ts = len(self.data.structure.timestep_info) - 1
            self.lift_distribution()
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
            header += ", y/s, cl"
            numb_col += 2
            lift_distribution = np.concatenate((lift_distribution, np.zeros((N_nodes, 2))), axis=1)

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
                lift_force = c_bs.T.dot(forces[inode, :3])[2]
                # Store data in export matrix
                lift_distribution[inode, 3] = lift_force
                lift_distribution[inode, 2] = struct_tstep.pos[inode, 2]  # z
                lift_distribution[inode, 1] = struct_tstep.pos[inode, 1]  # y
                lift_distribution[inode, 0] = struct_tstep.pos[inode, 0]  # x
                if self.settings["coefficients"]:
                    # Get non-dimensional spanwise coordinate y/s
                    lift_distribution[inode, 4] = lift_distribution[inode, 1]/span
                    # Get lift coefficient
                    lift_distribution[inode, 5] = np.sign(lift_force) * np.linalg.norm(lift_force) \
                                                  / (0.5 * self.settings['rho'] \
                                                     * np.linalg.norm(urel) ** 2 * span * chord)  # strip_area[i_surf][local_node])
                    # Check if shared nodes from different surfaces exist (e.g. two wings joining at symmetry plane)
                    # Leads to error since panel area just donates for half the panel size while lift forces is summed up
                    lift_distribution[inode, 5] /= len(self.data.aero.struct2aero_mapping[inode])

        # Export lift distribution data
        np.savetxt(os.path.join(self.folder, self.settings['text_file_name']), lift_distribution,
                   fmt='%10e,' * (numb_col - 1) + '%10e', delimiter=", ", header=header)
