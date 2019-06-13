"""
Control surface deflector for linear systems
"""


class ControlSurfaceDeflector(object):
    """
    Subsystem that deflects control surfaces for use with linear state space systems
    """

    def __init__(self, data, uvlm):
        self.data = data
        self.uvlm = uvlm
        self.ss = None

        self.n_node = 0
        self.n_elem = 0
        self.n_surf = 0
        self.n_control_surfaces = 0
        self.n_aeronode = 0

        self.aero_dict = dict()

    def assemble(self):
        """
        Assemble control surface deflector system

        Returns:

        """
        tsaero = self.data.aero.timestep_info[-1]
        self.n_node = self.data.aero.n_node
        self.n_aeronode = self.uvlm.Kzeta
        self.n_elem = self.data.aero.n_elem
        self.n_surf = self.uvlm.MS.n_surf

        self.aero_dict = self.data.aero.aero_dict

        # Find nodes with control surfaces
        # Control surface information
        try:
            self.aero_dict['control_surface']
            with_control_surfaces = True
        except KeyError:
            with_control_surfaces = False

        global_node_in_surface = []
        for i_surf in range(self.n_surf):
            global_node_in_surface.append([])

        control_surface_info = None
        for i_elem in range(self.n_elem):
            i_surf = self.aero_dict['surface_distribution'][i_elem]

            for i_local_node in range(len(self.beam.elements[i_elem].global_connectivities)):
                i_global_node = self.beam.elements[i_elem].global_connectivities[i_local_node]
                # i_global_node = self.beam.elements[i_elem].global_connectivities[
                #     self.beam.elements[i_elem].ordering[i_local_node]]
                if not self.aero_dict['aero_node'][i_global_node]:
                    continue
                if i_global_node in global_node_in_surface[i_surf]:
                    continue
                else:
                    global_node_in_surface[i_surf].append(i_global_node)

            if with_control_surfaces:
                it = len(self.data.beam.timestep_info)-1 # From aerogrid
                # 1) check that this node and elem have a control surface
                if self.aero_dict['control_surface'][i_elem, i_local_node] >= 0:
                    i_control_surface = self.aero_dict['control_surface'][i_elem, i_local_node]
                    # 2) type of control surface + write info
                    control_surface_info = dict()
                    if self.aero_dict['control_surface_type'][i_control_surface] == 0:
                        control_surface_info['type'] = 'static'
                        control_surface_info['deflection'] = self.aero_dict['control_surface_deflection'][i_control_surface]
                        control_surface_info['chord'] = self.aero_dict['control_surface_chord'][i_control_surface]
                        try:
                            control_surface_info['hinge_coords'] = self.aero_dict['control_surface_hinge_coords'][i_control_surface]
                        except KeyError:
                            control_surface_info['hinge_coords'] = None
                    elif self.aero_dict['control_surface_type'][i_control_surface] == 1:
                        control_surface_info['type'] = 'dynamic'
                        control_surface_info['chord'] = self.aero_dict['control_surface_chord'][i_control_surface]
                        try:
                            control_surface_info['hinge_coords'] = self.aero_dict['control_surface_hinge_coords'][i_control_surface]
                        except KeyError:
                            control_surface_info['hinge_coords'] = None

                        params = {'it': it}
                        control_surface_info['deflection'], control_surface_info['deflection_dot'] = \
                            self.cs_generators[i_control_surface](params)

                    elif self.aero_dict['control_surface_type'][i_control_surface] == 2:
                        raise NotImplementedError('control-type control surfaces are not yet implemented')
                    else:
                        raise NotImplementedError(str(self.aero_dict['control_surface_type'][i_control_surface]) +
                                                  ' control surfaces are not yet implemented')


        # Chord line by chord line
        # for i_surf in range(self.n_surf):
        #     M, N = self.data.aero.aero_dimensions[i_surf]
        #     for i_m in range(M+1):


