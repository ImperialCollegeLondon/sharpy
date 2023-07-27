import numpy as np
import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
from sharpy.utils.datastructures import init_matrix_structure, standalone_ctypes_pointer
import sharpy.aero.utils.uvlmlib as uvlmlib


@solver
class StallCheck(BaseSolver):
    """
    Outputs the incidence angle of every panel of the surface as cell variables for visualisation in Paraview.

    Note:
        This postprocessor appends the information to the current SHARPy timestep being run, therefore,
        in order to visualise the result in Paraview, it must be run prior to `AerogridPlot`. Otherwise, the panel
        stall check will be performed but the actual angle not produced in the Paraview visualisation.

    It also checks that the angles do not exceed the specified limit, with a warning in the log if the angle of attack
    exceeds such limits (both positive and negative).

    The limits are set through the setting ``airfoil_stall_angles``, which takes a dictionary where the key is
    the ID to the airfoil (in string format) and the value is a 2-tuple containing the negative and positive limits
    in radians.
    """
    solver_id = 'StallCheck'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Print info to screen '

    settings_types['airfoil_stall_angles'] = 'dict'
    settings_default['airfoil_stall_angles'] = dict()
    settings_description['airfoil_stall_angles'] = 'Dictionary of stall angles for each airfoil as per the details ' \
                                                   'above'

    settings_types['output_degrees'] = 'bool'
    settings_default['output_degrees'] = False
    settings_description['output_degrees'] = 'Output incidence angles in degrees vs radians'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.settings = None
        self.data = None

        self.ts_max = None
        self.ts = None
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

    def run(self, **kwargs):
    
        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        if not online:
            for self.ts in range(self.ts_max):
                self.check_stall()
            cout.cout_wrap('...Finished', 1)
        else:
            self.ts = len(self.data.structure.timestep_info) - 1
            self.check_stall()
        return self.data

    def check_stall(self):
        # add entry to dictionary for postproc
        tstep = self.data.aero.timestep_info[self.ts]
        tstep.postproc_cell['incidence_angle'] = init_matrix_structure(dimensions=tstep.dimensions,
                                                                       with_dim_dimension=False)

        # create ctypes pointers
        tstep.postproc_cell['incidence_angle_ct_list'] = None
        tstep.postproc_cell['incidence_angle_ct_pointer'] = None
        tstep.postproc_cell['incidence_angle_ct_list'], tstep.postproc_cell['incidence_angle_ct_pointer'] = \
            standalone_ctypes_pointer(tstep.postproc_cell['incidence_angle'])

        # call calculate
        uvlmlib.uvlm_calculate_incidence_angle(self.data.aero.timestep_info[self.ts],
                                               self.data.structure.timestep_info[self.ts])

        # calculate ratio of stalled panels and print
        stalled_panels = False
        stalled_surfs = np.zeros((tstep.n_surf, ), dtype=int)
        added_panels = []
        for i_surf in range(tstep.n_surf):
            added_panels.append([])

        for i_elem in range(self.data.structure.num_elem):
            for i_local_node in range(self.data.structure.num_node_elem):
                airfoil_id = self.data.aero.data_dict['airfoil_distribution'][i_elem, i_local_node]
                if self.settings['airfoil_stall_angles']:
                    i_global_node = self.data.structure.connectivities[i_elem, i_local_node]
                    for i_dict in self.data.aero.struct2aero_mapping[i_global_node]:
                        i_surf = i_dict['i_surf']
                        i_n = i_dict['i_n']

                        if i_n in added_panels[i_surf]:
                            continue

                        if i_n == tstep.dimensions[i_surf][1]:
                            continue

                        limits = self.settings['airfoil_stall_angles'][str(airfoil_id)]
                        if tstep.postproc_cell['incidence_angle'][i_surf][0, i_n] < float(limits[0]):
                            stalled_panels = True
                            stalled_surfs[i_surf] += tstep.postproc_cell['incidence_angle'][i_surf].shape[1]
                        elif tstep.postproc_cell['incidence_angle'][i_surf][0, i_n] > float(limits[1]):
                            stalled_panels = True
                            stalled_surfs[i_surf] += tstep.postproc_cell['incidence_angle'][i_surf].shape[1]

        if stalled_panels:
            if self.settings['print_info']:
                cout.cout_wrap('Some panel has an incidence angle out of the linear region', 1)
                cout.cout_wrap('The number of stalled panels per surface id are:', 1)
                for i_surf in range(tstep.n_surf):
                    cout.cout_wrap('\ti_surf = ' + str(i_surf) + ': ' + str(stalled_surfs[i_surf]) + ' panels.', 1)
                # cout.cout_wrap('In total, the ratio of stalled panels is: ', str(stalled_surfs.sum()/))

        if self.settings['output_degrees']:
            for i_surf in range(tstep.n_surf):
                tstep.postproc_cell['incidence_angle'][i_surf] *= 180/np.pi
