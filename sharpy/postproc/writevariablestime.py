import os
import numpy as np
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils


@solver
class WriteVariablesTime(BaseSolver):
    r"""
    Write variables with time

    ``WriteVariablesTime`` is a class inherited from ``BaseSolver``

    It is a postprocessor that outputs the value of variables with time onto a text file.

    Attributes:
        settings_types (dict): Acceptable data types of the input data
        settings_default (dict): Default values for input data should the user not provide them
        See the list of arguments
        dir (str): directory to output the information

    """

    solver_id = 'WriteVariablesTime'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['delimiter'] = 'str'
    settings_default['delimiter'] = ' '
    settings_description['delimiter'] = 'Delimiter to be used in the output file'

    settings_types['FoR_variables'] = 'list(str)'
    settings_default['FoR_variables'] = ['']
    settings_description['FoR_variables'] = 'Variables of :class:`~sharpy.utils.datastructures.StructTimeStepInfo` associated to the frame of reference to be writen'

    settings_types['FoR_number'] = 'list(int)'
    settings_default['FoR_number'] = np.array([0], dtype=int)
    settings_description['FoR_number'] = 'Number of the A frame of reference to output (for multibody configurations)'

    settings_types['structure_variables'] = 'list(str)'
    settings_default['structure_variables'] = ['']
    settings_description['structure_variables'] = 'Variables of :class:`~sharpy.utils.datastructures.StructTimeStepInfo` associated to the frame of reference to be writen'

    settings_types['structure_nodes'] = 'list(int)'
    settings_default['structure_nodes'] = np.array([-1])
    settings_description['structure_nodes'] = 'Number of the nodes to be writen'

    settings_types['nonlifting_nodes_variables'] = 'list(str)'
    settings_default['nonlifting_nodes_variables'] = ['']
    settings_description['nonlifting_nodes_variables'] = 'Variables of :class:`~sharpy.utils.datastructures.NonliftingBodyTimeStepInfo` associated to panels to be writen'

    settings_types['nonlifting_nodes_im'] = 'list(int)'
    settings_default['nonlifting_nodes_im'] = np.array([0])
    settings_description['nonlifting_nodes_im'] = 'Chordwise index of the nonlifting panels to be output'

    settings_types['nonlifting_nodes_in'] = 'list(int)'
    settings_default['nonlifting_nodes_in'] = np.array([0])
    settings_description['nonlifting_nodes_in'] = 'Spanwise index of the nonlifting panels to be output'

    settings_types['nonlifting_nodes_isurf'] = 'list(int)'
    settings_default['nonlifting_nodes_isurf'] = np.array([0])
    settings_description['nonlifting_nodes_isurf'] = "Number of the panels' surface to be output"

    settings_types['aero_panels_variables'] = 'list(str)'
    settings_default['aero_panels_variables'] = ['']
    settings_description['aero_panels_variables'] = 'Variables of :class:`~sharpy.utils.datastructures.AeroTimeStepInfo` associated to panels to be writen'

    settings_types['aero_panels_isurf'] = 'list(int)'
    settings_default['aero_panels_isurf'] = np.array([0])
    settings_description['aero_panels_isurf'] = "Number of the panels' surface to be output"

    settings_types['aero_panels_im'] = 'list(int)'
    settings_default['aero_panels_im'] = np.array([0])
    settings_description['aero_panels_im'] = 'Chordwise index of the panels to be output'

    settings_types['aero_panels_in'] = 'list(int)'
    settings_default['aero_panels_in'] = np.array([0])
    settings_description['aero_panels_in'] = 'Spanwise index of the panels to be output'

    settings_types['aero_nodes_variables'] = 'list(str)'
    settings_default['aero_nodes_variables'] = ['']
    settings_description['aero_nodes_variables'] = 'Variables of :class:`~sharpy.utils.datastructures.AeroTimeStepInfo` associated to nodes to be writen'

    settings_types['aero_nodes_isurf'] = 'list(int)'
    settings_default['aero_nodes_isurf'] = np.array([0])
    settings_description['aero_nodes_isurf'] = "Number of the nodes' surface to be output"

    settings_types['aero_nodes_im'] = 'list(int)'
    settings_default['aero_nodes_im'] = np.array([0])
    settings_description['aero_nodes_im'] = 'Chordwise index of the nodes to be output'

    settings_types['aero_nodes_in'] = 'list(int)'
    settings_default['aero_nodes_in'] = np.array([0])
    settings_description['aero_nodes_in'] = 'Spanwise index of the nodes to be output'

    settings_types['cleanup_old_solution'] = 'bool'
    settings_default['cleanup_old_solution'] = False
    settings_description['cleanup_old_solution'] = 'Remove the existing files'

    settings_types['vel_field_variables'] = 'list(str)'
    settings_default['vel_field_variables'] = list()
    settings_description['vel_field_variables'] = 'Variables associated to the velocity field. Only ``uext`` implemented so far'

    settings_types['vel_field_points'] = 'list(float)'
    settings_default['vel_field_points'] = np.array([0., 0., 0.])
    settings_description['vel_field_points'] = 'List of coordinates of the control points as x1, y1, z1, x2, y2, z2 ...'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None
        self.folder = None

        self.n_velocity_field_points = None
        self.velocity_field_points = None
        self.caller = None
        self.velocity_generator = None

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.folder = data.output_folder + '/WriteVariablesTime/'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        # Check inputs
        if not ((len(self.settings['aero_panels_isurf']) == len(self.settings['aero_panels_im'])) and (len(self.settings['aero_panels_isurf']) == len(self.settings['aero_panels_in']))):
            raise RuntimeError("aero_panels should be defined as [i_surf,i_m,i_n]")
        if not ((len(self.settings['aero_nodes_isurf']) == len(self.settings['aero_nodes_im'])) and (len(self.settings['aero_nodes_isurf']) == len(self.settings['aero_nodes_in']))):
            raise RuntimeError("aero_nodes should be defined as [i_surf,i_m,i_n]")

        if len(self.settings['vel_field_variables']) > 0:
            if not (len(self.settings['vel_field_points']) % 3 == 0):
                raise RuntimeError('Number of entries in ``vel_field_points`` has to be a multiple of 3')
            else:
                self.n_vel_field_points = len(self.settings['vel_field_points']) // 3
                self.vel_field_points = [np.zeros((3, self.n_vel_field_points, 1))]
                for ipoint in range(self.n_vel_field_points):
                    self.vel_field_points[0][:, ipoint, 0] = self.settings['vel_field_points'][ipoint*3:(ipoint + 1)*3]

        # Initialise files with headers and clean them if required
        for ivariable in range(len(self.settings['FoR_variables'])):
            for ifor in range(len(self.settings['FoR_number'])):
                filename = self.folder + "FoR_" + '%02d' % self.settings['FoR_number'][ifor] + "_" + self.settings['FoR_variables'][ivariable] + ".dat"
                if self.settings['cleanup_old_solution']:
                    if os.path.isfile(filename):
                        os.remove(filename)

        # Structure variables at nodes
        for ivariable in range(len(self.settings['structure_variables'])):
            for inode in range(len(self.settings['structure_nodes'])):
                node = self.settings['structure_nodes'][inode]
                filename = self.folder + "struct_" + self.settings['structure_variables'][ivariable] + "_node" + str(node) + ".dat"
                if self.settings['cleanup_old_solution']:
                    if os.path.isfile(filename):
                        os.remove(filename)

        # Nonlifting variables at panels
        for ivariable in range(len(self.settings['nonlifting_nodes_variables'])):
            for ipanel in range(len(self.settings['nonlifting_nodes_isurf'])):
                i_surf = self.settings['nonlifting_nodes_isurf'][ipanel]
                i_m = self.settings['nonlifting_nodes_im'][ipanel]
                i_n = self.settings['nonlifting_nodes_in'][ipanel]
                filename = self.folder + "nonlifting_" + self.settings['nonlifting_nodes_variables'][ivariable] + "_panel" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                if self.settings['cleanup_old_solution']:
                    if os.path.isfile(filename):
                        os.remove(filename)

        # Aerodynamic variables at panels
        for ivariable in range(len(self.settings['aero_panels_variables'])):
            for ipanel in range(len(self.settings['aero_panels_isurf'])):
                i_surf = self.settings['aero_panels_isurf'][ipanel]
                i_m = self.settings['aero_panels_im'][ipanel]
                i_n = self.settings['aero_panels_in'][ipanel]
                filename = self.folder + "aero_" + self.settings['aero_panels_variables'][ivariable] + "_panel" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                if self.settings['cleanup_old_solution']:
                    if os.path.isfile(filename):
                        os.remove(filename)

        # Aerodynamic variables at nodes
        for ivariable in range(len(self.settings['aero_nodes_variables'])):
            for inode in range(len(self.settings['aero_nodes_isurf'])):
                i_surf = self.settings['aero_nodes_isurf'][inode]
                i_m = self.settings['aero_nodes_im'][inode]
                i_n = self.settings['aero_nodes_in'][inode]
                filename = self.folder + "aero_" + self.settings['aero_nodes_variables'][ivariable] + "_node" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"
                if self.settings['cleanup_old_solution']:
                    if os.path.isfile(filename):
                        os.remove(filename)

        # Velocity field variables at points
        for ivariable in range(len(self.settings['vel_field_variables'])):
            for ipoint in range(self.n_vel_field_points):
                filename = self.folder + "vel_field_" + self.settings['vel_field_variables'][ivariable] + "_point" + str(ipoint) + ".dat"
                if self.settings['cleanup_old_solution']:
                    if os.path.isfile(filename):
                        os.remove(filename)
                if not os.path.isfile(filename):
                    fid = open(filename, 'w')
                    fid.write(("#t[s]%suext_x[m/s]%suext_y[m/s]%suext_z[m/s]\n" % ((self.settings['delimiter'],)*3)))
                    fid.close()

        # Initialise velocity generator
        self.caller = caller
        if ((not self.caller is None) and (not len(self.settings['vel_field_variables']) == 0)):
            if self.caller.solver_classification.lower() == 'aero':
                # For aerodynamic solvers
                self.velocity_generator = self.caller.velocity_generator
            elif self.caller.solver_classification.lower() == 'coupled':
                # For coupled solvers
                self.velocity_generator = self.caller.aero_solver.velocity_generator

    def run(self, **kwargs):
    
        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        if online:
            self.data = self.write(-1)
        else:
            for it in range(len(self.data.structure.timestep_info)):
                if self.data.structure.timestep_info[it] is not None:
                    self.data = self.write(it)

        return self.data

    def write(self, it):

        # FoR variables
        if 'FoR_number' in self.settings:
            pass
        else:
            self.settings['FoR_number'] = np.array([0], dtype=int)

        tstep = self.data.structure.timestep_info[it]

        for ivariable in range(len(self.settings['FoR_variables'])):
            if self.settings['FoR_variables'][ivariable] == '':
                continue
            for ifor in range(len(self.settings['FoR_number'])):
                filename = self.folder + "FoR_" + '%02d' % self.settings['FoR_number'][ifor] + "_" + self.settings['FoR_variables'][ivariable] + ".dat"

                with open(filename, 'a') as fid:
                    var = np.atleast_2d(getattr(tstep, self.settings['FoR_variables'][ivariable]))
                    rows, cols = var.shape
                    if ((cols == 1) and (rows == 1)):
                        self.write_value_to_file(fid, self.data.ts, var, self.settings['delimiter'])
                    elif ((cols > 1) and (rows == 1)):
                        self.write_nparray_to_file(fid, self.data.ts, var, self.settings['delimiter'])
                    elif ((cols == 1) and (rows >= 1)):
                        self.write_value_to_file(fid, self.data.ts, var[ifor], self.settings['delimiter'])
                    else:
                        self.write_nparray_to_file(fid, self.data.ts, var[ifor,:], self.settings['delimiter'])

        # Structure variables at nodes
        for ivariable in range(len(self.settings['structure_variables'])):
            if self.settings['structure_variables'][ivariable] == '':
                continue
            var = getattr(tstep, self.settings['structure_variables'][ivariable])
            num_indices = len(var.shape)
            if num_indices == 1:
                # Beam global variables (i.e. not node dependant)
                filename = self.folder + "struct_" + self.settings['structure_variables'][ivariable] + ".dat"
                with open(filename, 'a') as fid:
                    self.write_nparray_to_file(fid, self.data.ts, var, self.settings['delimiter'])

            else:  # These variables have nodal values (i.e the number of indices is either 2 or 3)
                for inode in range(len(self.settings['structure_nodes'])):
                    node = self.settings['structure_nodes'][inode]
                    filename = self.folder + "struct_" + self.settings['structure_variables'][ivariable] + "_node" + str(node) + ".dat"
                    with open(filename, 'a') as fid:
                        if num_indices == 2:
                            self.write_nparray_to_file(fid, self.data.ts, var[node,:], self.settings['delimiter'])
                        elif num_indices == 3:
                            ielem, inode_in_elem = self.data.structure.node_master_elem[node]
                            self.write_nparray_to_file(fid, self.data.ts, var[ielem,inode_in_elem,:], self.settings['delimiter'])


        # Aerodynamic variables at nonlifting panels
        for ivariable in range(len(self.settings['nonlifting_nodes_variables'])):
            if self.settings['nonlifting_nodes_variables'][ivariable] == '':
                continue
            
            for ipanel in range(len(self.settings['nonlifting_nodes_isurf'])):
                i_surf = self.settings['nonlifting_nodes_isurf'][ipanel]
                i_m = self.settings['nonlifting_nodes_im'][ipanel]
                i_n = self.settings['nonlifting_nodes_in'][ipanel]
                filename = self.folder + "nonlifting_" + self.settings['nonlifting_nodes_variables'][ivariable] + "_panel" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"

                with open(filename, 'a') as fid:
                    var = getattr(self.data.nonlifting_body.timestep_info[it], self.settings['nonlifting_nodes_variables'][ivariable])
                    self.write_value_to_file(fid, self.data.ts, var[i_surf][i_m,i_n], self.settings['delimiter'])

        # Aerodynamic variables at panels
        for ivariable in range(len(self.settings['aero_panels_variables'])):
            if self.settings['aero_panels_variables'][ivariable] == '':
                continue
            for ipanel in range(len(self.settings['aero_panels_isurf'])):
                i_surf = self.settings['aero_panels_isurf'][ipanel]
                i_m = self.settings['aero_panels_im'][ipanel]
                i_n = self.settings['aero_panels_in'][ipanel]

                filename = self.folder + "aero_" + self.settings['aero_panels_variables'][ivariable] + "_panel" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"

                with open(filename, 'a') as fid:
                    var = getattr(self.data.aero.timestep_info[it], self.settings['aero_panels_variables'][ivariable])
                    self.write_value_to_file(fid, self.data.ts, var[i_surf][i_m,i_n], self.settings['delimiter'])


        # Aerodynamic variables at nodes
        for ivariable in range(len(self.settings['aero_nodes_variables'])):
            if self.settings['aero_nodes_variables'][ivariable] == '':
                continue
            for inode in range(len(self.settings['aero_nodes_isurf'])):
                i_surf = self.settings['aero_nodes_isurf'][inode]
                i_m = self.settings['aero_nodes_im'][inode]
                i_n = self.settings['aero_nodes_in'][inode]

                filename = self.folder + "aero_" + self.settings['aero_nodes_variables'][ivariable] + "_node" + "_isurf" + str(i_surf) + "_im"+ str(i_m) + "_in"+ str(i_n) + ".dat"

                with open(filename, 'a') as fid:
                    var = getattr(self.data.aero.timestep_info[it], self.settings['aero_nodes_variables'][ivariable])
                    self.write_nparray_to_file(fid, self.data.ts, var[i_surf][:,i_m,i_n], self.settings['delimiter'])

        # Velocity field variables at points
        for ivariable in range(len(self.settings['vel_field_variables'])):
            if self.settings['vel_field_variables'][ivariable] == 'uext':
                uext = [np.zeros((3, self.n_vel_field_points, 1))]
                self.velocity_generator.generate({'zeta': self.vel_field_points,
                                    'for_pos': tstep.for_pos[0:3],
                                    't': self.data.ts*self.caller.settings['dt'],
                                    'is_wake': False,
                                    'override': True},
                                    uext)
                for ipoint in range(self.n_vel_field_points):
                    filename = self.folder + "vel_field_" + self.settings['vel_field_variables'][ivariable] + "_point" + str(ipoint) + ".dat"
                    with open(filename, 'a') as fid:
                        self.write_nparray_to_file(fid, self.data.ts, uext[0][:,ipoint,0], self.settings['delimiter'])

        return self.data

    def write_nparray_to_file(self, fid, ts, nparray, delimiter):

        fid.write("%d%s" % (ts,delimiter))
        for idim in range(nparray.shape[0]):
            try:
                for jdim in range(nparray.shape[1] - 1):
                    fid.write("%e%s" % (nparray[idim, jdim],delimiter))
                fid.write("%e" % (nparray[idim, -1]))
            except IndexError:
                fid.write("%e%s" % (nparray[idim],delimiter))

        fid.write("\n")

    def write_value_to_file(self, fid, ts, value, delimiter):

        fid.write("%d%s%e\n" % (ts,delimiter,value))
