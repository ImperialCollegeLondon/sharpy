'''@package PyBeam.Utils.BeamLib
@brief      Loads the f90 subroutines with Python wrappers.
@author     Rob Simpson
@contact    r.simpson11@imperial.ac.uk
@version    0.0
@date       10/12/2012
@pre        Compile the beam code into wrapped dynamic library.
@warning    None
'''
import numpy as np
import ctypes as ct
import platform

BeamPath = 'lib/libxbeam'
if platform.system() == 'Darwin':
    ext = '.dylib'
elif platform.system() == 'Linux':
    ext = '.so'
else:
    raise NotImplementedError('The platform ' + platform.system() + 'is not supported')


class Xbopts(ct.Structure):
    """Structure skeleton for options input in xbeam

    """
    _fields_ = [("FollowerForce", ct.c_bool),
                ("FollowerForceRig", ct.c_bool),
                ("PrintInfo", ct.c_bool),
                ("OutInBframe", ct.c_bool),
                ("OutInaframe", ct.c_bool),
                ("ElemProj", ct.c_int),
                ("MaxIterations", ct.c_int),
                ("NumLoadSteps", ct.c_int),
                ("NumGauss", ct.c_int),
                ("Solution", ct.c_int),
                ("DeltaCurved", ct.c_double),
                ("MinDelta", ct.c_double),
                ("NewmarkDamp", ct.c_double)
                ]

    def __init__(self):
        ct.Structure.__init__(self)
        self.FollowerForce = ct.c_bool(True)
        self.FollowerForceRig = ct.c_bool(True)
        self.PrintInfo = ct.c_bool(True)
        self.OutInBframe = ct.c_bool(True)
        self.OutInaframe = ct.c_bool(False)
        self.ElemProj = ct.c_int(0)
        self.MaxIterations = ct.c_int(99)
        self.NumLoadSteps = ct.c_int(5)
        self.NumGauss = ct.c_int(1)
        self.Solution = ct.c_int(111)
        self.DeltaCurved = ct.c_double(1.0e-5)
        self.MinDelta = ct.c_double(1.0e-8)
        self.NewmarkDamp = ct.c_double(1.0e-4)


BeamPath += ext
BeamLib = ct.cdll.LoadLibrary(BeamPath)
f_cbeam3_solv_nlnstatic = BeamLib.cbeam3_solv_nlnstatic_python
f_cbeam3_solv_nlnstatic.restype = None
f_cbeam3_solv_nlnstatic.argtype = [ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_double),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_double),
                                   ct.POINTER(ct.c_double),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_double),
                                   ct.POINTER(Xbopts)]


def cbeam3_solv_nlnstatic(beam, settings):
    """@brief Python wrapper for f_cbeam3_solv_nlnstatic

    @details Numpy arrays are mutable so the changes (solution) made here are
     reflected in the data of the calling script after execution.
     Modified by Alfonso del Carre"""
    n_elem = ct.c_int(beam.num_elem)
    n_nodes = ct.c_int(beam.num_node)
    n_mass = ct.c_int(beam.n_mass)
    n_stiff = ct.c_int(beam.n_stiff)

    xbopts = Xbopts()
    xbopts.FollowerForce = ct.c_bool(False)
    xbopts.FollowerForceRig = ct.c_bool(False)
    xbopts.Solution = ct.c_int(112)
    xbopts.NumGauss = ct.c_int(beam.num_node_elem - 1)

    f_cbeam3_solv_nlnstatic(ct.byref(n_elem),
                            ct.byref(n_nodes),
                            beam.num_nodes_matrix.ctypes.data_as(ct.POINTER(ct.c_int)),
                            beam.num_mem_matrix.ctypes.data_as(ct.POINTER(ct.c_int)),
                            beam.connectivities_fortran.ctypes.data_as(ct.POINTER(ct.c_int)),
                            beam.master_nodes.ctypes.data_as(ct.POINTER(ct.c_int)),
                            ct.byref(n_mass),
                            beam.mass_matrix.ctypes.data_as(ct.POINTER(ct.c_double)),
                            beam.mass_indices.ctypes.data_as(ct.POINTER(ct.c_int)),
                            ct.byref(n_stiff),
                            beam.stiffness_matrix.ctypes.data_as(ct.POINTER(ct.c_double)),
                            beam.inv_stiffness_db.ctypes.data_as(ct.POINTER(ct.c_double)),
                            beam.stiffness_indices.ctypes.data_as(ct.POINTER(ct.c_int)),
                            beam.rbmass_matrix.ctypes.data_as(ct.POINTER(ct.c_double)),
                            beam.node_master_elem.ctypes.data_as(ct.POINTER(ct.c_int)),
                            beam.vdof.ctypes.data_as(ct.POINTER(ct.c_int)),
                            beam.fdof.ctypes.data_as(ct.POINTER(ct.c_int)),
                            ct.byref(xbopts)
                            )

#
# f_cbeam3_solv_nlnstatic = BeamLib.wrap_cbeam3_solv_nlnstatic
# f_cbeam3_solv_nlnstatic.restype = None
# f_cbeam3_solv_nlnstatic.argtypes = [ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_bool),
#                                     ct.POINTER(ct.c_bool),
#                                     ct.POINTER(ct.c_bool),
#                                     ct.POINTER(ct.c_bool),
#                                     ct.POINTER(ct.c_bool),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_int),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double),
#                                     ct.POINTER(ct.c_double)]
#
# def cbeam3_solv_nlnstatic(beam, settings):
#     """@brief Python wrapper for f_cbeam3_solv_nlnstatic
#
#     @details Numpy arrays are mutable so the changes (solution) made here are
#      reflected in the data of the calling script after execution.
#      Modified by Alfonso del Carre"""
#     solution = ct.c_int(112)
#
#     f_cbeam3_solv_nlnstatic(ct.byref(beam.num_dof),
#                             ct.byref(ct.c_int(beam.num_elem)),
#                             beam.num_nodes_matrix.ctypes.data_as(ct.POINTER(ct.c_int)),
#                             beam.num_mem_matrix.ctypes.data_as(ct.POINTER(ct.c_int)),
#                             beam.connectivities_fortran.ctypes.data_as(ct.POINTER(ct.c_int)),
#                             beam.master_nodes.ctypes.data_as(ct.POINTER(ct.c_int)),
#                             beam.length_matrix.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.precurv.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.psi.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.local_vec.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.mass_matrix.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.stiffness_matrix.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.inv_stiffness_matrix.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.rbmass_matrix.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             ct.byref(ct.c_int(beam.num_node)),
#                             beam.node_master_elem_fortran.ctypes.data_as(ct.POINTER(ct.c_int)),
#                             beam.vdof.ctypes.data_as(ct.POINTER(ct.c_int)),
#                             beam.fdof.ctypes.data_as(ct.POINTER(ct.c_int)),
#                             beam.app_forces_fortran.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.node_coordinates_fortran.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.psi_fortran.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.node_coordinates_defor_fortran.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             beam.psi_defor_fortran.ctypes.data_as(ct.POINTER(ct.c_double)),
#                             ct.byref(settings['follower_force']),
#                             ct.byref(settings['follower_force_rig']),
#                             ct.byref(settings['print_info']),
#                             ct.byref(settings['out_b_frame']),
#                             ct.byref(settings['out_a_frame']),
#                             ct.byref(settings['elem_proj']),
#                             ct.byref(settings['max_iterations']),
#                             ct.byref(settings['num_load_steps']),
#                             ct.byref(settings['num_gauss']),
#                             ct.byref(solution),
#                             ct.byref(settings['delta_curved']),
#                             ct.byref(settings['min_delta']),
#                             ct.byref(settings['newmark_damp']))

#
# def Cbeam3_Solv_NonlinearDynamic(XBINPUT, XBOPTS, NumNodes_tot, XBELEM, PosIni,\
#             PsiIni, XBNODE, NumDof, PosDefor, PsiDefor, NumSteps, Time,\
#             ForceTime, ForcedVel, ForcedVelDot, PosDotDef, PsiDotDef,\
#             PosPsiTime, VelocTime, DynOut, OutGrids):
#     """@brief Python wrapper for f_cbeam3_solv_nlndyn
#
#     @details Numpy arrays are mutable so the changes (solution) made here are
#      reflected in the data of the calling script after execution."""
#
#     f_cbeam3_solv_nlndyn(ct.byref(ct.c_int(XBINPUT.iOut)),\
#                 ct.byref(NumDof),\
#                 ct.byref(NumSteps),\
#                 Time.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(ct.c_int(XBINPUT.NumElems)),\
#                 XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Mass.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Stiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.InvStiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.RBMass.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 ct.byref(NumNodes_tot),\
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBINPUT.ForceStatic.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBINPUT.ForceDyn.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForceTime.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForcedVel.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForcedVelDot.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosPsiTime.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 VelocTime.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 DynOut.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 OutGrids.ctypes.data_as(ct.POINTER(ct.c_bool)),\
#                 ct.byref(XBOPTS.FollowerForce),\
#                 ct.byref(XBOPTS.FollowerForceRig),\
#                 ct.byref(XBOPTS.PrintInfo),\
#                 ct.byref(XBOPTS.OutInBframe),\
#                 ct.byref(XBOPTS.OutInaframe),\
#                 ct.byref(XBOPTS.ElemProj),\
#                 ct.byref(XBOPTS.MaxIterations),\
#                 ct.byref(XBOPTS.NumLoadSteps),\
#                 ct.byref(XBOPTS.NumGauss),\
#                 ct.byref(XBOPTS.Solution),\
#                 ct.byref(XBOPTS.DeltaCurved),\
#                 ct.byref(XBOPTS.MinDelta),\
#                 ct.byref(XBOPTS.NewmarkDamp) )
#
#
# def Cbeam3_Asbly_Static(XBINPUT, NumNodes_tot, XBELEM, XBNODE,\
#                         PosIni, PsiIni, PosDefor, PsiDefor,\
#                         ForceStatic, NumDof,\
#                         ks, KglobalFull, fs, FglobalFull, Qglobal,
#                         XBOPTS):
#     """@brief Python wrapper for f_cbeam3_asbly_static.
#
#     @details Numpy arrays are mutable so the changes made here are
#     reflected in the data of the calling script after execution."""
#
#     f_cbeam3_asbly_static( \
#                 ct.byref(ct.c_int(XBINPUT.NumElems)),\
#                 ct.byref(NumNodes_tot),\
#                 XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Mass.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Stiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.InvStiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.RBMass.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForceStatic.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(ct.c_int(Settings.DimMat)), \
#                 ct.byref(NumDof), \
#                 ct.byref(ks), \
#                 KglobalFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 ct.byref(fs), \
#                 FglobalFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 Qglobal.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(XBOPTS.FollowerForce),\
#                 ct.byref(XBOPTS.FollowerForceRig),\
#                 ct.byref(XBOPTS.PrintInfo),\
#                 ct.byref(XBOPTS.OutInBframe),\
#                 ct.byref(XBOPTS.OutInaframe),\
#                 ct.byref(XBOPTS.ElemProj),\
#                 ct.byref(XBOPTS.MaxIterations),\
#                 ct.byref(XBOPTS.NumLoadSteps),\
#                 ct.byref(XBOPTS.NumGauss),\
#                 ct.byref(XBOPTS.Solution),\
#                 ct.byref(XBOPTS.DeltaCurved),\
#                 ct.byref(XBOPTS.MinDelta),\
#                 ct.byref(XBOPTS.NewmarkDamp) )
#
#
# def Cbeam_Solv_Disp2State(NumNodes_tot, NumDof, XBINPUT, XBNODE,\
#                           PosDefor, PsiDefor, PosDotDef, PsiDotDef,
#                           x, dxdt):
#     """@brief Python wrapper for f_cbeam3_solv_disp2state.
#
#     @details Numpy arrays are mutable so the changes made here are
#     reflected in the data of the calling script after execution."""
#
#     f_cbeam3_solv_disp2state( \
#                 ct.byref(NumNodes_tot), \
#                 ct.byref(NumDof), \
#                 ct.byref(ct.c_int(XBINPUT.NumElems)), \
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDotDef.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PsiDotDef.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 x.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 dxdt.ctypes.data_as(ct.POINTER(ct.c_double)) )
#
#
# def Cbeam3_Solv_Update_Static(XBINPUT, NumNodes_tot, XBELEM, XBNODE, NumDof,\
#                               DeltaS, PosIni, PsiIni, PosDefor, PsiDefor):
#     """@brief Wrapper for f_cbeam3_solv_update_static.
#
#     @details Numpy arrays are mutable so the changes made here are
#     reflected in the data of the calling script after execution."""
#
#     f_cbeam3_solv_update_static( \
#                 ct.byref(ct.c_int(XBINPUT.NumElems)), \
#                 ct.byref(NumNodes_tot), \
#                 XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Mass.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Stiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.InvStiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.RBMass.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 ct.byref(NumDof), \
#                 DeltaS.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PosIni.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)) )
#
#
#
# def Cbeam3_Solv_NonlinearDynamicAccel(XBINPUT, XBOPTS, NumNodes_tot, XBELEM, PosIni,\
#             PsiIni, XBNODE, NumDof, PosDefor, PsiDefor, NumSteps, Time,\
#             ForceTime, ForcedVel, ForcedVelDot, PosDotDef, PsiDotDef,\
#             PosDotDotDef, PsiDotDotDef,\
#             PosPsiTime, VelocTime, DynOut, OutGrids):
#     """@brief Python wrapper for f_cbeam3_solv_nlndyn
#
#     @details Numpy arrays are mutable so the changes (solution) made here are
#      reflected in the data of the calling script after execution."""
#
#     f_cbeam3_solv_nlndyn_accel(ct.byref(ct.c_int(XBINPUT.iOut)),\
#                 ct.byref(NumDof),\
#                 ct.byref(NumSteps),\
#                 Time.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(ct.c_int(XBINPUT.NumElems)),\
#                 XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Mass.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Stiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.InvStiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.RBMass.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 ct.byref(NumNodes_tot),\
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBINPUT.ForceStatic.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBINPUT.ForceDyn.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForceTime.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForcedVel.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForcedVelDot.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDotDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDotDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosPsiTime.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 VelocTime.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 DynOut.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 OutGrids.ctypes.data_as(ct.POINTER(ct.c_bool)),\
#                 ct.byref(XBOPTS.FollowerForce),\
#                 ct.byref(XBOPTS.FollowerForceRig),\
#                 ct.byref(XBOPTS.PrintInfo),\
#                 ct.byref(XBOPTS.OutInBframe),\
#                 ct.byref(XBOPTS.OutInaframe),\
#                 ct.byref(XBOPTS.ElemProj),\
#                 ct.byref(XBOPTS.MaxIterations),\
#                 ct.byref(XBOPTS.NumLoadSteps),\
#                 ct.byref(XBOPTS.NumGauss),\
#                 ct.byref(XBOPTS.Solution),\
#                 ct.byref(XBOPTS.DeltaCurved),\
#                 ct.byref(XBOPTS.MinDelta),\
#                 ct.byref(XBOPTS.NewmarkDamp) )
#
#
# def Cbeam3_Asbly_Dynamic(XBINPUT, NumNodes_tot, XBELEM, XBNODE,\
#                          PosIni, PsiIni, PosDefor, PsiDefor,\
#                          PosDotDef, PsiDotDef, PosDotDotDef, PsiDotDotDef,\
#                          Force, Vrel, VrelDot,\
#                          NumDof, DimMat,\
#                          ms, MglobalFull, Mvel,\
#                          cs, CglobalFull, Cvel,\
#                          ks, KglobalFull, fs, FglobalFull,\
#                          Qglobal, XBOPTS, Cao):
#     """@brief Wrapper for f_cbeam3_asbly_dynamic."""
#
#     f_cbeam3_asbly_dynamic( \
#             ct.byref(ct.c_int(XBINPUT.NumElems)), \
#             ct.byref(NumNodes_tot), \
#             XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.Mass.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.Stiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.InvStiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.RBMass.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)), \
#             PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PosDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PsiDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PosDotDotDef.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             PsiDotDotDef.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Force.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Vrel.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             VrelDot.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(NumDof), \
#             ct.byref(ct.c_int(DimMat)), \
#             ct.byref(ms), \
#             MglobalFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Mvel.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(cs), \
#             CglobalFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Cvel.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(ks), \
#             KglobalFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(fs), \
#             FglobalFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Qglobal.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(XBOPTS.FollowerForce),\
#             ct.byref(XBOPTS.FollowerForceRig),\
#             ct.byref(XBOPTS.PrintInfo),\
#             ct.byref(XBOPTS.OutInBframe),\
#             ct.byref(XBOPTS.OutInaframe),\
#             ct.byref(XBOPTS.ElemProj),\
#             ct.byref(XBOPTS.MaxIterations),\
#             ct.byref(XBOPTS.NumLoadSteps),\
#             ct.byref(XBOPTS.NumGauss),\
#             ct.byref(XBOPTS.Solution),\
#             ct.byref(XBOPTS.DeltaCurved),\
#             ct.byref(XBOPTS.MinDelta),\
#             ct.byref(XBOPTS.NewmarkDamp),\
#             Cao.ctypes.data_as(ct.POINTER(ct.c_double)) )
#
#
# def Cbeam3_Solv_State2Disp(XBINPUT, NumNodes_tot, XBELEM, XBNODE,
#                            PosIni, PsiIni, NumDof, X, dXdt,\
#                            PosDefor, PsiDefor, PosDotDefor, PsiDotDefor):
#     """@brief Wrapper for f_cbeam3_solv_state2disp."""
#
#     f_cbeam3_solv_state2disp( \
#                 ct.byref(ct.c_int(XBINPUT.NumElems)), \
#                 ct.byref(NumNodes_tot), \
#                 XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Mass.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Stiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.InvStiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.RBMass.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(NumDof), \
#                 X.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 dXdt.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PosDotDefor.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 PsiDotDefor.ctypes.data_as(ct.POINTER(ct.c_double)) )
#
# def Cbeam3_fstifz(R0,Ri,K,z):
#     """@brief Local element stiffness forces as a function of non-dim ordinate z
#     .
#
#     @param eta0 Undeformed element displacements and rotations (2x6 or 3x6
#                  array).
#     @param eta Current element displacements and rotations.
#     @param K Element stiffness matrix.
#     @param z Non-dim ordinate on reference line.
#     @return stiffForce Discrete elastic force at z.
#     @details For three-noded elements R0(i,:) corresponds to LHS element for
#     i=1, RHS for i=2, and center for i=3. Also, z = 0 is the centre of the
#     element with z in [-1,1].
#     """
#     return (lib_cbeam3.cbeam3_fstifz(R0,Ri,K,z))
#
# def Cbeam3_strainz(R0,Ri,z):
#     """@brief Local element strain as a function of non-dim ordinate z.
#
#     @param eta0 Undeformed element displacements and rotations (2x6 or 3x6
#                  array).
#     @param eta Current element displacements and rotations.
#     @param z Non-dim ordinate on reference line.
#     @return stiffForce Discrete elastic force at z.
#     @details For three-noded elements R0(i,:) corresponds to LHS element for
#     i=1, RHS for i=2, and center for i=3. Also, z = 0 is the centre of the
#     element with z in [-1,1].
#     """
#     return (lib_cbeam3.cbeam3_strainz(R0,Ri,z))
#
#
# def Cbeam3_Asbly_Fglobal(XBINPUT, NumNodes_tot, XBELEM, XBNODE,\
#                         PosIni, PsiIni, PosDefor, PsiDefor,\
#                         ForceStatic, NumDof,\
#                         ksf, KglobalFull_foll, fsf, FglobalFull_foll, fsd, FglobalFull_dead, Cao):
#     """@brief Python wrapper for f_cbeam3_asbly_fglobal.
#
#     @details Numpy arrays are mutable so the changes made here are
#     reflected in the data of the calling script after execution."""
#
#     f_cbeam3_asbly_fglobal( \
#                 ct.byref(ct.c_int(XBINPUT.NumElems)),\
#                 ct.byref(NumNodes_tot),\
#                 XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForceStatic.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(ct.c_int(Settings.DimMat)), \
#                 ct.byref(NumDof), \
#                 ct.byref(ksf), \
#                 KglobalFull_foll.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 ct.byref(fsf), \
#                 FglobalFull_foll.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(fsd),\
#                 FglobalFull_dead.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 Cao.ctypes.data_as(ct.POINTER(ct.c_double)) )
#
#
# def Xbeam_Asbly_Dynamic(XBINPUT, NumNodes_tot, XBELEM, XBNODE,\
#                         PosIni, PsiIni, PosDefor, PsiDefor,\
#                         PosDotDef, PsiDotDef, PosDotDotDef, PsiDotDotDef,\
#                         Vrel, VrelDot, Quat,\
#                         NumDof, DimMat,\
#                         mr, MrsFull, Mrr,\
#                         cr, CrsFull, Crr, Cqr, Cqq,\
#                         kr, KrsFull, fr, FrigidFull,\
#                         Qrigid, XBOPTS, Cao):
#     """@brief Wrapper for f_xbeam_asbly_dynamic."""
#
#     f_xbeam_asbly_dynamic( \
#             ct.byref(ct.c_int(XBINPUT.NumElems)), \
#             ct.byref(NumNodes_tot), \
#             XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.Mass.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.Stiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.InvStiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             XBELEM.RBMass.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#             XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)), \
#             PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PosDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PsiDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             PosDotDotDef.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             PsiDotDotDef.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Vrel.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             VrelDot.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Quat.ctypes.data_as(ct.POINTER(ct.c_double)),\
#             ct.byref(NumDof), \
#             ct.byref(ct.c_int(DimMat)), \
#             ct.byref(mr), \
#             MrsFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Mrr.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(cr), \
#             CrsFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Crr.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Cqr.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Cqq.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(kr), \
#             KrsFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(fr), \
#             FrigidFull.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Qrigid.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             Cao.ctypes.data_as(ct.POINTER(ct.c_double)), \
#             ct.byref(XBOPTS.FollowerForce),\
#             ct.byref(XBOPTS.FollowerForceRig),\
#             ct.byref(XBOPTS.PrintInfo),\
#             ct.byref(XBOPTS.OutInBframe),\
#             ct.byref(XBOPTS.OutInaframe),\
#             ct.byref(XBOPTS.ElemProj),\
#             ct.byref(XBOPTS.MaxIterations),\
#             ct.byref(XBOPTS.NumLoadSteps),\
#             ct.byref(XBOPTS.NumGauss),\
#             ct.byref(XBOPTS.Solution),\
#             ct.byref(XBOPTS.DeltaCurved),\
#             ct.byref(XBOPTS.MinDelta),\
#             ct.byref(XBOPTS.NewmarkDamp) )
#
#
# def Xbeam_Asbly_Frigid(XBINPUT, NumNodes_tot, XBELEM, XBNODE, \
#                         PosIni, PsiIni, PosDefor, PsiDefor, NumDof, \
#                         frf, FrigidFull_foll, frd, FrigidFull_dead, Cao):
#     """@brief Python wrapper for f_xbeam_asbly_frigid.
#
#     @details Numpy arrays are mutable so the changes made here are
#     reflected in the data of the calling script after execution."""
#
#     f_xbeam_asbly_frigid( \
#                 ct.byref(ct.c_int(XBINPUT.NumElems)),\
#                 ct.byref(NumNodes_tot),\
#                 XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(ct.c_int(Settings.DimMat)),\
#                 ct.byref(NumDof),\
#                 ct.byref(frf), \
#                 FrigidFull_foll.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(frd),\
#                 FrigidFull_dead.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 Cao.ctypes.data_as(ct.POINTER(ct.c_double)) )
#
#
# def Xbeam_Solv_FreeNonlinDynamic(XBINPUT, XBOPTS, NumNodes_tot, XBELEM, PosIni,\
#             PsiIni, XBNODE, NumDof, PosDefor, PsiDefor, Quat, NumSteps, Time,\
#             ForceTime, ForcedVel, ForcedVelDot, PosDotDef, PsiDotDef,\
#             DynOut, OutGrids):
#     """@brief Python wrapper for f_xbeam_solv_couplednlndyn
#
#     @details Numpy arrays are mutable so the changes (solution) made here are
#      reflected in the data of the calling script after execution."""
#
#     f_xbeam_solv_couplednlndyn(ct.byref(ct.c_int(XBINPUT.iOut)),\
#                 ct.byref(NumDof),\
#                 ct.byref(NumSteps),\
#                 Time.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ct.byref(ct.c_int(XBINPUT.NumElems)),\
#                 XBELEM.NumNodes.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.MemNo.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Conn.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBELEM.Length.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.PreCurv.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Psi.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Mass.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.Stiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.InvStiff.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBELEM.RBMass.ctypes.data_as(ct.POINTER(ct.c_double)), \
#                 ct.byref(NumNodes_tot),\
#                 XBNODE.Master.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Fdof.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBNODE.Sflag.ctypes.data_as(ct.POINTER(ct.c_int)),\
#                 XBINPUT.ForceStatic.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 XBINPUT.ForceDyn.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForceTime.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForcedVel.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 ForcedVelDot.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 Quat.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiIni.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDefor.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PosDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 PsiDotDef.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 DynOut.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                 OutGrids.ctypes.data_as(ct.POINTER(ct.c_bool)),\
#                 ct.byref(XBOPTS.FollowerForce),\
#                 ct.byref(XBOPTS.FollowerForceRig),\
#                 ct.byref(XBOPTS.PrintInfo),\
#                 ct.byref(XBOPTS.OutInBframe),\
#                 ct.byref(XBOPTS.OutInaframe),\
#                 ct.byref(XBOPTS.ElemProj),\
#                 ct.byref(XBOPTS.MaxIterations),\
#                 ct.byref(XBOPTS.NumLoadSteps),\
#                 ct.byref(XBOPTS.NumGauss),\
#                 ct.byref(XBOPTS.Solution),\
#                 ct.byref(XBOPTS.DeltaCurved),\
#                 ct.byref(XBOPTS.MinDelta),\
#                 ct.byref(XBOPTS.NewmarkDamp) )
#
#
#
# if __name__ == '__main__':
#     print(lib_fem.__doc__)
#     print(lib_cbeam3.__doc__)
#
#     # Test forces at z
#     NumNodesElem = 2
#     R0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
#     Ri = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                    [2.0, 0.0, 0.0, 0.5*np.pi, 0.5*np.pi, 0.0]])
#     K = 1e3*np.eye(6)
#     z = 1.0 #local coordinate
#
#     print(Cbeam3_fstifz(R0, Ri, K, z))
#     print(Cbeam3_strainz(R0, Ri, z))
#
#     R0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],])
#     Ri = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                    [1.0, 0.0, 4.0, 0.0, np.pi, 0.0],
#                    [0.5, 0.0, 1.0, 0.0, 0.25*np.pi, 0.0]])
#
#     print(Cbeam3_fstifz(R0, Ri, K, z))
#     print(Cbeam3_strainz(R0, Ri, z))
#
#     quat = np.array([1.0,0.0,0.0,0.0])
#     print(Cbeam3_quat2psi(quat))
