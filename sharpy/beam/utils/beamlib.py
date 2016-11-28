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
                                   ct.POINTER(Xbopts),
                                   ct.POINTER(ct.c_double)
                                   ]


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
    xbopts.FollowerForce = ct.c_bool(settings['follower_force'])
    xbopts.FollowerForceRig = ct.c_bool(settings['follower_force_rig'])
    xbopts.Solution = ct.c_int(112)
    xbopts.NumGauss = settings['num_gauss']

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
                            beam.node_master_elem_fortran.ctypes.data_as(ct.POINTER(ct.c_int)),
                            beam.vdof.ctypes.data_as(ct.POINTER(ct.c_int)),
                            beam.fdof.ctypes.data_as(ct.POINTER(ct.c_int)),
                            ct.byref(xbopts),
                            beam.app_forces_fortran.ctypes.data_as(ct.POINTER(ct.c_double)),
                            beam.pos_ini.ctypes.data_as(ct.POINTER(ct.c_double)),
                            beam.psi_ini.ctypes.data_as(ct.POINTER(ct.c_double)),
                            beam.pos_def.ctypes.data_as(ct.POINTER(ct.c_double)),
                            beam.psi_def.ctypes.data_as(ct.POINTER(ct.c_double)),
                            )
