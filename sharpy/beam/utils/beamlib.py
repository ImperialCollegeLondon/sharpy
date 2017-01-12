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
        self.NumGauss = ct.c_int(2)
        self.Solution = ct.c_int(111)
        self.DeltaCurved = ct.c_double(1.0e-5)
        self.MinDelta = ct.c_double(1.0e-8)
        self.NewmarkDamp = ct.c_double(1.0e-4)


BeamPath += ext
BeamLib = ct.cdll.LoadLibrary(BeamPath)
f_cbeam3_solv_nlnstatic = BeamLib.cbeam3_solv_nlnstatic_python
f_cbeam3_solv_nlnstatic.restype = None
# f_cbeam3_solv_nlnstatic.argtype = [ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_double),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_double),
#                                    ct.POINTER(ct.c_double),
#                                    ct.POINTER(ct.c_int),
#                                    ct.POINTER(ct.c_double),
#                                    ct.POINTER(Xbopts),
#                                    ct.POINTER(ct.c_double),
#                                    ct.POINTER(ct.c_double),
#                                    ct.POINTER(ct.c_double),
#                                    ct.POINTER(ct.c_double),
#                                    ct.POINTER(ct.c_double)
#                                    ]

# ctypes pointer types
doubleP = ct.POINTER(ct.c_double)
intP = ct.POINTER(ct.c_int)
charP = ct.POINTER(ct.c_char_p)

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
    xbopts.PrintInfo = ct.c_bool(settings['print_info'])
    xbopts.Solution = ct.c_int(112)
    xbopts.OutInaframe = ct.c_bool(settings['out_a_frame'])
    xbopts.OutInBframe = ct.c_bool(settings['out_b_frame'])
    xbopts.ElemProj = settings['elem_proj']
    xbopts.MaxIterations = settings['max_iterations']
    xbopts.NumLoadSteps = settings['num_load_steps']
    xbopts.NumGauss = settings['num_gauss']
    xbopts.DeltaCurved = settings['delta_curved']
    xbopts.MinDelta = settings['min_delta']
    xbopts.NewmarkDamp = settings['newmark_damp']

    # applied forces as 0=G, 1=a, 2=b
    # here we only need to set the flags at True, all the forces are follower
    xbopts.FollowerForce = ct.c_bool(True)
    xbopts.FollowerForceRig = ct.c_bool(True)

    f_cbeam3_solv_nlnstatic(ct.byref(n_elem),
                            ct.byref(n_nodes),
                            beam.num_nodes_matrix.ctypes.data_as(intP),
                            beam.num_mem_matrix.ctypes.data_as(intP),
                            beam.connectivities_fortran.ctypes.data_as(intP),
                            beam.master_nodes.ctypes.data_as(intP),
                            ct.byref(n_mass),
                            beam.mass_matrix.ctypes.data_as(doubleP),
                            beam.mass_indices.ctypes.data_as(intP),
                            ct.byref(n_stiff),
                            beam.stiffness_matrix.ctypes.data_as(doubleP),
                            beam.inv_stiffness_db.ctypes.data_as(doubleP),
                            beam.stiffness_indices.ctypes.data_as(intP),
                            beam.frame_of_reference_delta.ctypes.data_as(doubleP),
                            beam.rbmass_fortran.ctypes.data_as(doubleP),
                            beam.node_master_elem_fortran.ctypes.data_as(intP),
                            beam.vdof.ctypes.data_as(intP),
                            beam.fdof.ctypes.data_as(intP),
                            ct.byref(xbopts),
                            beam.pos_ini.ctypes.data_as(doubleP),
                            beam.psi_ini.ctypes.data_as(doubleP),
                            beam.pos_def.ctypes.data_as(doubleP),
                            beam.psi_def.ctypes.data_as(doubleP),
                            ct.byref(beam.n_app_forces),
                            beam.app_forces_fortran.ctypes.data_as(doubleP),
                            beam.node_app_forces_fortran.ctypes.data_as(intP)
                            )
    angle = 0*np.pi/180
    import presharpy.utils.algebra as algebra
    rot = np.zeros((3, 3))
    rot[0, :] = [np.cos(angle), -np.sin(angle), 0.0]
    rot[1, :] = [np.sin(angle), np.cos(angle), 0.0]
    rot[2, :] = [0, 0, 1.0]

    psi = beam.psi_def[-1, 2, :]
    total = algebra.crv2rot(psi)

    def_rot = np.dot(rot.T, total)
    psi_proj = algebra.rot2crv(def_rot)
    print(psi_proj)

    print(beam.pos_def[-1, :])
    # print(np.dot(rot.T, beam.psi_def[-1, 2, :]))
    print(beam.psi_def[-1, 2, :])

