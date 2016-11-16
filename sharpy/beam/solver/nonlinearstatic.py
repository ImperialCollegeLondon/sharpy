"""@package PyBeam.Solver.NonlinearStatic
@brief      Nonlinear static solvers.
@author     Rob Simpson
@contact    r.simpson11@imperial.ac.uk
@version    0.0
@date       25/10/2012
@pre        None
@warning    None

@modified   Alfonso del Carre
"""

import sys
# import DerivedTypes
# import BeamIO
# import beam.utils.derivedtypes as derivedtypes
# import XbeamLib
# import BeamInit
import numpy as np
import ctypes as ct
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.beam.utils.beamlib as beamlib
from presharpy.utils.settings import str2bool


@solver
class NonLinearStatic(BaseSolver):
    solver_id = 'NonLinearStatic'
    solver_type = 'structural'

    def __init__(self):
        pass

    def initialise(self, data):
        self.data = data
        self.settings = data.settings[self.solver_id]
        self.convert_settings()
        data.beam.generate_aux_information()

    def run(self):
        beamlib.cbeam3_solv_nlnstatic(self.data.beam, self.settings)
        return self.data

    def convert_settings(self):
        self.settings['follower_force'] = ct.c_bool(str2bool(self.settings['follower_force']))
        self.settings['follower_force_rig'] = ct.c_bool(str2bool(self.settings['follower_force_rig']))
        self.settings['print_info'] = ct.c_bool(str2bool(self.settings['print_info']))
        self.settings['out_b_frame'] = ct.c_bool(str2bool(self.settings['out_b_frame']))
        self.settings['out_a_frame'] = ct.c_bool(str2bool(self.settings['out_a_frame']))
        self.settings['elem_proj'] = ct.c_int(int(self.settings['elem_proj']))
        self.settings['max_iterations'] = ct.c_int(int(self.settings['max_iterations']))
        self.settings['num_load_steps'] = ct.c_int(int(self.settings['num_load_steps']))
        self.settings['num_gauss'] = ct.c_int(int(self.settings['num_gauss']))
        self.settings['delta_curved'] = ct.c_double(float(self.settings['delta_curved']))
        self.settings['min_delta'] = ct.c_double(float(self.settings['min_delta']))
        self.settings['newmark_damp'] = ct.c_double(float(self.settings['newmark_damp']))


def Solve_F90(XBINPUT,XBOPTS):
    """@brief Nonlinear static structural solver using f90 solve routine."""
    
    "Check correct solution code"
    assert XBOPTS.Solution.value == 112, ('NonlinearStatic (F90) requested' +\
                                              ' with wrong solution code')
    
    "Initialise beam"
    XBINPUT, XBOPTS, NumNodes_tot, XBELEM, PosIni, PsiIni, XBNODE, NumDof \
                = BeamInit.Static(XBINPUT,XBOPTS)
    
    
    "Set initial conditions as undef config"
    PosDefor = PosIni.copy(order='F')
    PsiDefor = PsiIni.copy(order='F')
    
    
    if XBOPTS.PrintInfo.value==True:
        sys.stdout.write('Solve nonlinear static case (using .f90 routines) ... \n')
    
    BeamLib.Cbeam3_Solv_NonlinearStatic(XBINPUT, XBOPTS, NumNodes_tot, XBELEM,\
                                PosIni, PsiIni, XBNODE, NumDof,\
                                PosDefor, PsiDefor)
    
    if XBOPTS.PrintInfo.value==True:
        sys.stdout.write(' ... done\n')
    
    
    "Write deformed configuration to file"
    ofile = Settings.OutputDir + Settings.OutputFileRoot + '_SOL112_def.dat'
    if XBOPTS.PrintInfo.value==True:
        sys.stdout.write('Writing file %s ... ' %(ofile))
    fp = open(ofile,'w')
    fp.write('TITLE="Non-linear static solution: deformed geometry"\n')
    fp.write('VARIABLES="iElem" "iNode" "Px" "Py" "Pz" "Rx" "Ry" "Rz"\n')
    fp.close()
    if XBOPTS.PrintInfo.value==True:
        sys.stdout.write('done\n')
    WriteMode = 'a'
    
    BeamIO.OutputElems(XBINPUT.NumElems, NumNodes_tot.value, XBELEM, \
                       PosDefor, PsiDefor, ofile, WriteMode)
    
    "Print deformed configuration"
    if XBOPTS.PrintInfo.value==True:
        sys.stdout.write('--------------------------------------\n')
        sys.stdout.write('NONLINEAR STATIC SOLUTION\n')
        sys.stdout.write('%10s %10s %10s\n' %('X','Y','Z'))
        for inodi in range(NumNodes_tot.value):
            sys.stdout.write(' ')
            for inodj in range(3):
                sys.stdout.write('%12.5e' %(PosDefor[inodi,inodj]))
            sys.stdout.write('\n')
        sys.stdout.write('--------------------------------------\n')
        
    
    "Return solution as optional output argument"
    return PosDefor, PsiDefor


# def Solve_F90_steps(XBINPUT,XBOPTS):
#     """@brief Nonlinear static structural solver using f90 solve routine called
#     once per load-step."""
#
#     "Check correct solution code"
#     assert XBOPTS.Solution.value == 112, ('NonlinearStatic (F90) requested' +\
#                                               ' with wrong solution code')
#
#     "Initialise beam"
#     XBINPUT, XBOPTS, NumNodes_tot, XBELEM, PosIni, PsiIni, XBNODE, NumDof \
#                 = BeamInit.Static(XBINPUT,XBOPTS)
#
#
#     "Set initial conditions as undef config"
#     PosDefor = PosIni.copy(order='F')
#     PsiDefor = PsiIni.copy(order='F')
#
#
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write('Solve nonlinear static case (using .f90 routine at' +\
#                          ' each load-step) ... \n')
#
#
#
#     "Initialise load increments and set F90 load-step to 1"
#     LoadSteps = XBOPTS.NumLoadSteps.value
#     LoadIncrement = XBINPUT.ForceStatic/LoadSteps
#     XBOPTS.NumLoadSteps.value = 1
#
#
#     "Start loading loop"
#     for step in range(1,LoadSteps+1):
#         "Current load to be applied"
#         XBINPUT.ForceStatic = step*LoadIncrement
#
#         "Print load step"
#         if XBOPTS.PrintInfo.value == True:
#             print('     Python-based outer load step %d' %(step))
#
#         "Solve with one load step"
#         BeamLib.Cbeam3_Solv_NonlinearStatic(XBINPUT, XBOPTS, NumNodes_tot, XBELEM,\
#                                 PosIni, PsiIni, XBNODE, NumDof,\
#                                 PosDefor, PsiDefor)
#
#
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write(' ... done\n')
#
#
#     "Write deformed configuration to file"
#     ofile = Settings.OutputDir + Settings.OutputFileRoot + '_SOL112_def.dat'
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write('Writing file %s ... ' %(ofile))
#     fp = open(ofile,'w')
#     fp.write('TITLE="Non-linear static solution: deformed geometry"\n')
#     fp.write('VARIABLES="iElem" "iNode" "Px" "Py" "Pz" "Rx" "Ry" "Rz"\n')
#     fp.close()
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write('done\n')
#     WriteMode = 'a'
#
#     BeamIO.OutputElems(XBINPUT.NumElems, NumNodes_tot.value, XBELEM, \
#                        PosDefor, PsiDefor, ofile, WriteMode)
#
#
#     "Print deformed configuration"
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write('--------------------------------------\n')
#         sys.stdout.write('NONLINEAR STATIC SOLUTION\n')
#         sys.stdout.write('%10s %10s %10s\n' %('X','Y','Z'))
#         for inodi in range(NumNodes_tot.value):
#             sys.stdout.write(' ')
#             for inodj in range(3):
#                 sys.stdout.write('%12.5e' %(PosDefor[inodi,inodj]))
#             sys.stdout.write('\n')
#         sys.stdout.write('--------------------------------------\n')
#
#
#     "Return solution as optional output argument"
#     return PosDefor, PsiDefor
#
#
# def Solve_Py(XBINPUT,XBOPTS, moduleName = None):
#     """Nonlinear static solver using Python to solve residual
#     equation. Assembly of matrices is carried out with Fortran subroutines."""
#
#     "Check correct solution code"
#     assert XBOPTS.Solution.value == 112, ('NonlinearStatic requested' +\
#                                               ' with wrong solution code')
#
#     "Initialise beam"
#     XBINPUT, XBOPTS, NumNodes_tot, XBELEM, PosIni, PsiIni, XBNODE, NumDof \
#                 = BeamInit.Static(XBINPUT,XBOPTS, moduleName)
#
#
#     "Set initial conditions as undef config"
#     PosDefor = PosIni.copy(order='F')
#     PsiDefor = PsiIni.copy(order='F')
#
#
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write('Solve nonlinear static case in Python ... \n')
#
#     "Initialise rotation operators"
#     Unit = np.zeros((3,3), ct.c_double, 'F')
#     for i in range(3):
#         Unit[i,i] = 1.0
#
#     Cao = Unit.copy('F')
#
#     "Initialise structural eqn tensors"
#     KglobalFull = np.zeros((NumDof.value,NumDof.value),\
#                             ct.c_double, 'F'); ks = ct.c_int()
#     KglobalFull_foll = np.zeros((NumDof.value,NumDof.value),\
#                             ct.c_double, 'F');
#     FglobalFull = np.zeros((NumDof.value,NumDof.value),\
#                             ct.c_double, 'F'); fs = ct.c_int()
#
#     DeltaS  = np.zeros(NumDof.value, ct.c_double, 'F')
#     Qglobal = np.zeros(NumDof.value, ct.c_double, 'F')
#     x       = np.zeros(NumDof.value, ct.c_double, 'F')
#     dxdt    = np.zeros(NumDof.value, ct.c_double, 'F')
#
#
#     "Load Step tensors"
#     iForceStep     = np.zeros((NumNodes_tot.value,6), ct.c_double, 'F')
#     iForceStep_Dof = np.zeros(NumDof.value, ct.c_double, 'F')
#
#     "Start Load Loop"
#     for iLoadStep in range(XBOPTS.NumLoadSteps.value):
#
#         "Reset convergence parameters"
#         Iter = 0
#         ResLog10 = 1.0
#
#         "General load case"
#         iForceStep = XBINPUT.ForceStatic*float( (iLoadStep+1) /                \
#                                                 XBOPTS.NumLoadSteps.value)
#
#         if XBOPTS.PrintInfo.value == True:
#             sys.stdout.write('  iLoad: %-10d\n' %(iLoadStep+1))
#             sys.stdout.write('   SubIter DeltaF     DeltaX     ResLog10\n')
#
#
#         "Newton Iteration"
#         while( (ResLog10 > XBOPTS.MinDelta.value) \
#              & (Iter < XBOPTS.MaxIterations.value) ):
#
#             "Increment iteration counter"
#             Iter += 1
#             if XBOPTS.PrintInfo.value == True:
#                 sys.stdout.write('   %-7d ' %(Iter))
#
#
#             "Set structural eqn tensors to zero"
#             KglobalFull[:,:] = 0.0; ks = ct.c_int()
#             KglobalFull_foll[:,:] = 0.0;
#             FglobalFull[:,:] = 0.0; fs = ct.c_int()
#             Qglobal[:] = 0.0
#
#
#             "Assemble matrices for static problem"
#             BeamLib.Cbeam3_Asbly_Static(XBINPUT, NumNodes_tot, XBELEM, XBNODE,\
#                         PosIni, PsiIni, PosDefor, PsiDefor,\
#                         iForceStep, NumDof,\
#                         ks, KglobalFull, fs, FglobalFull, Qglobal,\
#                         XBOPTS)
#
#
#             "Get state vector from current deformation"
#             PosDot = np.zeros((NumNodes_tot.value,3), ct.c_double, 'F') #R
#             PsiDot = np.zeros((XBINPUT.NumElems,Settings.MaxElNod,3),\
#                                ct.c_double, 'F') #Psi
#
#             BeamLib.Cbeam_Solv_Disp2State(NumNodes_tot, NumDof, XBINPUT, XBNODE,\
#                           PosDefor, PsiDefor, PosDot, PsiDot,
#                           x, dxdt)
#
#
#             "Get forces on unconstrained nodes"
#             BeamLib.f_fem_m2v(ct.byref(NumNodes_tot),\
#                               ct.byref(ct.c_int(6)),\
#                               iForceStep.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                               ct.byref(NumDof),\
#                               iForceStep_Dof.ctypes.data_as(ct.POINTER(ct.c_double)),\
#                               XBNODE.Vdof.ctypes.data_as(ct.POINTER(ct.c_int)) )
#
#
#             "Calculate \Delta RHS"
#             Qglobal = Qglobal - np.dot(FglobalFull, iForceStep_Dof)
#
#
#             "Separate assembly of follower and dead loads"
#             Qforces, KglobalFull_foll = \
#                         XbeamLib.LoadAssembly(XBINPUT, XBELEM, XBNODE, XBOPTS, NumDof, \
#                                               PosIni, PsiIni, PosDefor, PsiDefor, \
#                                               XBINPUT.ForceStatic_foll,XBINPUT.ForceStatic_dead, \
#                                               Cao, float((iLoadStep+1)/XBOPTS.NumLoadSteps.value))[:2]
#
#             KglobalFull += KglobalFull_foll
#             Qglobal -= Qforces
#
#             "Calculate \Delta State Vector"
#             DeltaS = - np.dot(np.linalg.inv(KglobalFull), Qglobal)
#
#             if XBOPTS.PrintInfo.value == True:
#                 sys.stdout.write('%-10.4e %-10.4e ' \
#                                  % (max(abs(Qglobal)),max(abs(DeltaS))))
#
#
#             "Update Solution"
#             BeamLib.Cbeam3_Solv_Update_Static(XBINPUT, NumNodes_tot, XBELEM,\
#                                               XBNODE, NumDof, DeltaS,\
#                                               PosIni, PsiIni, PosDefor,PsiDefor)
#
#
#             "Residual at first iteration"
#             if(Iter == 1):
#                 Res0_Qglobal = max(max(abs(Qglobal)),1)
#                 Res0_DeltaX  = max(max(abs(DeltaS)),1)
#
#             "Update residual and compute log10"
#             Res_Qglobal = max(abs(Qglobal))
#             Res_DeltaX  = max(abs(DeltaS))
#
#             ResLog10 = max(Res_Qglobal/Res0_Qglobal, Res_DeltaX/Res0_DeltaX)
#
#
#             if XBOPTS.PrintInfo.value == True:
#                 sys.stdout.write('%8.4f\n' %(ResLog10))
#
#
#             "Stop the solution"
#             if(ResLog10 > 1.e10):
#                 sys.stderr.write(' STOP\n')
#                 sys.stderr.write(' The max residual is %e\n' %(ResLog10))
#                 exit(1)
#
#         "END Newton Loop"
#     "END Load Loop"
#
#
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write(' ... done\n')
#
#
#     "Write deformed configuration to file"
#     ofile = Settings.OutputDir + Settings.OutputFileRoot + '_SOL112_def.dat'
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write('Writing file %s ... ' %(ofile))
#     fp = open(ofile,'w')
#     fp.write('TITLE="Non-linear static solution: deformed geometry"\n')
#     fp.write('VARIABLES="iElem" "iNode" "Px" "Py" "Pz" "Rx" "Ry" "Rz"\n')
#     fp.close()
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write('done\n')
#     WriteMode = 'a'
#
#     BeamIO.OutputElems(XBINPUT.NumElems, NumNodes_tot.value, XBELEM, \
#                        PosDefor, PsiDefor, ofile, WriteMode)
#
#
#     "Print deformed configuration"
#     if XBOPTS.PrintInfo.value==True:
#         sys.stdout.write('--------------------------------------\n')
#         sys.stdout.write('NONLINEAR STATIC SOLUTION\n')
#         sys.stdout.write('%10s %10s %10s\n' %('X','Y','Z'))
#         for inodi in range(NumNodes_tot.value):
#             sys.stdout.write(' ')
#             for inodj in range(3):
#                 sys.stdout.write('%12.5e' %(PosDefor[inodi,inodj]))
#             sys.stdout.write('\n')
#         sys.stdout.write('--------------------------------------\n')
#
#
#     "Return solution as optional output argument"
#     return PosDefor, PsiDefor
#
#
# if __name__ == '__main__':
#     """Set up Xbopts for nonlinear static analysis defined in input_rob.f90
#     TPY0 test case"""
#     XBOPTS = DerivedTypes.Xbopts()
#     XBOPTS.Solution.value = 112
#     XBOPTS.NumLoadSteps.value = 10
#     XBOPTS.MinDelta.value = 1e-04
#     XBOPTS.PrintInfo.value = True
#     """Set up Xbinput for nonlinear static analysis defined in input_rob.f90
#     TPY0 test case"""
#     XBINPUT = DerivedTypes.Xbinput(2,8)
#     XBINPUT.BeamLength = 16.0
#     XBINPUT.BeamStiffness[0,0] = 1.0e+09
#     XBINPUT.BeamStiffness[1,1] = 1.0e+09
#     XBINPUT.BeamStiffness[2,2] = 1.0e+09
#     XBINPUT.BeamStiffness[3,3] = 1.0e+04
#     XBINPUT.BeamStiffness[4,4] = 2.0e+04
#     XBINPUT.BeamStiffness[5,5] = 4.0e+06
#     XBINPUT.BeamMass[0,0] = 0.75
#     XBINPUT.BeamMass[1,1] = 0.75
#     XBINPUT.BeamMass[2,2] = 0.75
#     XBINPUT.BeamMass[3,3] = 0.1
#     XBINPUT.BeamMass[4,4] = 0.001
#     XBINPUT.BeamMass[5,5] = 0.001
#     XBINPUT.ForceStatic[-1,2] = 800
#
#     #Solve_F90(XBINPUT,XBOPTS)
#     #Solve_F90_steps(XBINPUT,XBOPTS)
#     Solve_Py(XBINPUT,XBOPTS)
