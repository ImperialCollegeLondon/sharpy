import numpy as np
import sharpy.utils.cout_utils as cout
import sharpy.rom.interpolation.interpolationspaces as interpolationspaces
import sharpy.linear.src.libss as libss
import sharpy.linear.src.lingebm as lingebm


def pmor_loader(rom_library, target_system, interpolation_space, projection_method,
                use_ct=True):
    cout.cout_wrap('Generating PMOR for %s system' % target_system)

    if target_system == 'structural' and interpolation_space == 'tangentspd':
        raise NotImplementedError("Tangent SPD Interpolation on the individual second order matrices is a WIP")

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  WIP
        # if interpolation_space == 'tangentspd':
        #     interpolate_spd = True
        # else:
        #     interpolate_spd = False
        #
        # pmor_list = rom_library.get_reduced_order_bases(target_system)
        # pmor = SetStructuralPMORs(pmor_list, method_proj=projection_method,
        #                           reference_case=rom_library.reference_case)
    else:
        ss_list, vv_list, wwt_list = rom_library.get_reduced_order_bases(target_system)

        if interpolation_space == 'direct':
            cout.cout_wrap('\tInterpolating Directly', 1)
            pmor = interpolationspaces.InterpROM()
        elif interpolation_space == 'tangent':
            cout.cout_wrap('\tInterpolating in the Tangent space', 1)
            pmor = interpolationspaces.TangentInterpolation()
        elif interpolation_space == 'real':
            cout.cout_wrap('\tInterpolating Real Matrices', 1)
            pmor = interpolationspaces.InterpolationRealMatrices()
        elif interpolation_space == 'tangentspd':
            cout.cout_wrap('\tInterpolating in the Tangent space', 1)
            pmor = interpolationspaces.TangentSPDInterpolation()

        pmor.initialise(ss_list, vv_list, wwt_list,
                        method_proj=projection_method,
                        reference_case=rom_library.reference_case,
                        use_ct=use_ct)

        # >>>>>>>>>>>>>>> Interpolate modal matrix to extract results in nodal coords
        # pmor.basis = interpolationspaces.BasisInterpolation(v_list=[U[:U.shape[0]//2, :U.shape[1]//2] for U in vv_list],
        #                                                     vt_list=[U[:U.shape[0]//2, :U.shape[1]//2] for U in wwt_list],
        #                                                     reference_case=rom_library.reference_case)

    # Transform onto gen coordinates
    pmor.project()

    # pmor.basis.create_tangent_space()
    return pmor


class CoupledPMOR:

    systems = ['aerodynamic', 'structural']

    def __init__(self, rom_library, interpolation_settings, use_ct=True):

        self.pmor = dict()  #: dictionary of interpolationspaces.InterpROM

        for sys in self.systems:
            self.pmor[sys] = pmor_loader(rom_library,
                                         target_system=sys,
                                         interpolation_space=interpolation_settings[sys]['interpolation_space'],
                                         projection_method=interpolation_settings[sys]['projection_method'],
                                         use_ct=use_ct)

        self.source_settings = rom_library.data_library[0].settings  # settings from the source case

    def __call__(self, weights):

        interpolated_systems = dict()
        for sys in self.systems:
            interpolated_systems[sys] = self.pmor[sys](weights)

        # Aeroelastic coupling
        # Future improvement: add support to normalised systems and scale with dynamic pressure
        t_as = np.eye(interpolated_systems['aerodynamic'].inputs, interpolated_systems['structural'].outputs)
        t_sa = np.eye(interpolated_systems['structural'].inputs, interpolated_systems['aerodynamic'].outputs)

        aeroelastic_ss = libss.couple(ss01=interpolated_systems['aerodynamic'],
                                      ss02=interpolated_systems['structural'],
                                      K12=t_as,
                                      K21=t_sa)

        # interpolated_modes = self.pmor['structural'].basis.get_interpolated_basis(weights)
        #
        # if self.source_settings['LinearAssembler']['inout_coordinates'] == 'nodes':
        #     # check if nodal
        #     from sharpy.linear.assembler.linearaeroelastic import LinearAeroelastic
        #     input_gain, output_gain = LinearAeroelastic.to_nodal(interpolated_modes,
        #                                                          interpolated_systems['aerodynamic'],
        #                                                          interpolated_systems['structural'])
        #
        #     # import pdb; pdb.set_trace()
        #     aeroelastic_ss.addGain(input_gain, where='in')
        #     aeroelastic_ss.addGain(output_gain, where='out')
        #
        # if len(self.source_settings['LinearAssembler']['retain_inputs']) != 0:
        #     aeroelastic_ss.remove_inout_channels(self.source_settings['LinearAssembler']['retain_inputs'], where='in')
        # if len(self.source_settings['LinearAssembler']['retain_outputs']) != 0:
        #     aeroelastic_ss.remove_inout_channels(self.source_settings['LinearAssembler']['retain_outputs'], where='out')
        #
        # aeroelastic_ss.summary()
        # import pdb; pdb.set_trace()

        return aeroelastic_ss

# >>>>>>>>>>>>> WIP
#
# class SetStructuralPMORs:
#
#     def __init__(self, pmor_list, method_proj, reference_case):
#         self.pmor_list = pmor_list
#
#         self.method_proj = method_proj
#         self.reference_case = reference_case
#
#         self.gamma_list = list()
#         self.reference_quintuplet = None
#
#     def project(self):
#         phi_list = [pmor.phi for pmor in self.pmor_list]
#         phi_t_list = [pmor.phi.T for pmor in self.pmor_list]
#
#         q_list = None
#         qinv_list = None
#
#         print('projecting...')
#
#         if self.method_proj == 'amsallem':
#             import sharpy.rom.interpolation.projectionmethods as projectionmethods
#             q_list, qinv_list = projectionmethods.amsallem(phi_list, phi_t_list, ref_case=self.reference_case)
#         else:
#             NotImplementedError('Interpolation method %s not implemented' % self.method_proj)
#
#         for i, pmor in enumerate(self.pmor_list):
#             pmor.rom_project()
#
#         self.reference_quintuplet = (
#             self.pmor_list[self.reference_case].m,
#             self.pmor_list[self.reference_case].c,
#             self.pmor_list[self.reference_case].k,
#             self.pmor_list[self.reference_case].bu,
#             self.pmor_list[self.reference_case].cy,
#         )
#
#         for i in self.reference_quintuplet:
#             print(i.shape)
#
#         for i, pmor in enumerate(self.pmor_list):
#             pmor.project(q_list[i], qinv_list[i], self.reference_quintuplet)
#
#             self.gamma_list.append(pmor.gamma)
#
#     def __call__(self, weights):
#         """Interpolation"""
#         assert self.gamma_list is not None, "System has not yet been projected onto congruent set of coordinates"
#
#         print('Interpolating')
#
#         m = np.zeros_like(self.gamma_list[0][0])
#         c = np.zeros_like(self.gamma_list[0][1])
#         k = np.zeros_like(self.gamma_list[0][2])
#
#         bu = np.zeros_like(self.gamma_list[0][3])
#         cy = np.zeros_like(self.gamma_list[0][4])
#
#         # from_tangent = interpolationspaces.TangentSPDInterpolation.from_tangent_manifold
#         from_tangent = StructuralPMOR.from_tangent
#         interp_gamma = [m, c, k, bu, cy]
#         interp_method = ['spd', None, 'spd', None, None]
#         for ith, gamma in enumerate(self.gamma_list):
#             for qth, matrix in enumerate(interp_gamma):
#                 matrix += weights[ith] * gamma[qth]
#
#         interpolated_matrices = [from_tangent(matrix, self.reference_quintuplet[ith], interp_method[ith])
#                                  for ith, matrix in enumerate(interp_gamma)]
#
#         for i in interpolated_matrices:
#             print(i.shape)
#
#         return lingebm.FlexDynamic.build_first_order_ct(
#             m=interpolated_matrices[0],
#             k=interpolated_matrices[2],
#             c_damp=interpolated_matrices[1],
#             bu=interpolated_matrices[3],
#             cy=interpolated_matrices[4]
#         )
#
#
# class StructuralPMOR:
#
#     system = 'structural'
#
#     def __init__(self, m, c, k, bu=None, cy=None, phi=None):
#         self.m = m  #: np.ndarray: Mass matrix
#         self.c = c  #: np.ndarray: Damping matrix
#         self.k = k  #: np.ndarray: Stiffness matrix
#
#         self.bu = bu  #: np.ndarray: Input matrix
#         self.cy = cy  #: np.ndarray: Output matrix
#
#         self.phi = phi  #: np.ndarray: Modal matrix
#
#         self.rom = False  #: bool: flag whether system has been projected onto modal coordinates
#         self.gen_project = False  #: bool: flag whether projected onto generalised coordinates
#
#         self.gamma = None  #: tuple: quintuple
#
#     def build_first_order_ct(self):
#         return lingebm.FlexDynamic.build_first_order_ct(self.m, self.k, self.c, self.phi)
#
#     def rom_project(self):
#
#         if not self.rom:
#             self.rom = True
#             self.m = self.phi.T.dot(self.m.dot(self.phi))
#             self.c = self.phi.T.dot(self.c.dot(self.phi))
#             self.k = self.phi.T.dot(self.k.dot(self.phi))
#
#             self.bu = self.phi.T.dot(self.bu)
#             self.cy = self.cy.dot(self.phi)
#
#             print('projected m: ' + str(self.m.shape))
#             print('projected c: ' + str(self.c.shape))
#             print('projected k: ' + str(self.k.shape))
#             print('projected bu: ' + str(self.bu.shape))
#             print('projected cy: ' + str(self.cy.shape))
#
#     def project(self, q, qinv, reference_quintlet):
#
#         if not self.gen_project:
#             self.gen_project = True
#
#             self.m = qinv.dot(self.m.dot(q))
#             self.c = qinv.dot(self.c.dot(q))
#             self.k = qinv.dot(self.k.dot(q))
#
#             self.bu = qinv.dot(self.bu)
#             self.cy = self.cy.dot(q)
#
#         self.gamma = (
#             self.to_tangent(self.m, reference_quintlet[0], method='spd'),
#             self.to_tangent(self.c, reference_quintlet[1]),
#             self.to_tangent(self.k, reference_quintlet[2], method='spd'),
#             self.to_tangent(self.bu, reference_quintlet[3]),
#             self.to_tangent(self.cy, reference_quintlet[4]),
#         )
#
#     @staticmethod
#     def to_tangent(matrix, ref_matrix, method=None):
#
#         if method is None or method == 'none':
#             return matrix
#
#         elif method == 'spd':
#             try:
#                 gamma = interpolationspaces.TangentSPDInterpolation.to_tangent_manifold(matrix, ref_matrix)
#             except np.linalg.LinAlgError:
#                 raise np.linalg.LinAlgError
#         elif method == 'tangent':
#             try:
#                 gamma = interpolationspaces.TangentInterpolation.to_tangent_manifold(matrix, ref_matrix)
#             except np.linalg.LinAlgError:
#                 raise np.linalg.LinAlgError
#         elif method == 'mxn':
#             gamma = interpolationspaces.InterpolationRealMatrices.to_tangent_manifold(matrix, ref_matrix)
#         else:
#             return matrix
#
#         return gamma
#
#     @staticmethod
#     def from_tangent(matrix, ref_matrix, method=None):
#
#         if method is None or method == 'none':
#             return matrix
#
#         elif method == 'spd':
#             try:
#                 out_matrix = interpolationspaces.TangentSPDInterpolation.from_tangent_manifold(matrix, ref_matrix)
#             except np.linalg.LinAlgError:
#                 raise np.linalg.LinAlgError
#         elif method == 'tangent':
#             try:
#                 out_matrix = interpolationspaces.TangentInterpolation.from_tangent_manifold(matrix, ref_matrix)
#             except np.linalg.LinAlgError:
#                 raise np.linalg.LinAlgError
#         elif method == 'mxn':
#             out_matrix = interpolationspaces.InterpolationRealMatrices.from_tangent_manifold(matrix, ref_matrix)
#         else:
#             return matrix
#
#         return out_matrix
#
# # class StructuralSystemMatrix(np.ndarray):
# #
# #     def __new__(cls, array, interpolation_method):
# #         obj = np.asarray(array).view(cls)
# #         return obj
