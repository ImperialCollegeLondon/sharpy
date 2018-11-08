'''
Linear Time Invariant systems
author: S. Maraniello
date: 15 Sep 2017 (still basement...)

Library of methods to manipulate state-space models
'''

import copy
import numpy as np
import scipy.sparse as sparse
import scipy.signal as scsig
# # from IPython import embed


def couple(ss01,ss02,K12,K21):
	'''
	Couples 2 dlti systems ss01 and ss02 through the gains K12 and K21, where
	K12 transforms the output of ss02 into an input of ss01.
	'''

	assert np.abs(ss01.dt-ss02.dt)<1e-10*ss01.dt, 'Time-steps not matching!'
	assert K12.shape == (ss01.inputs,ss02.outputs),\
			 'Gain K12 shape not matching with systems number of inputs/outputs'
	assert K21.shape == (ss02.inputs,ss01.outputs),\
			 'Gain K21 shape not matching with systems number of inputs/outputs'


	A1,B1,C1,D1=ss01.A,ss01.B,ss01.C,ss01.D
	A2,B2,C2,D2=ss02.A,ss02.B,ss02.C,ss02.D

	# extract size
	Nx1,Nu1=B1.shape
	Ny1=C1.shape[0]
	Nx2,Nu2=B2.shape
	Ny2=C2.shape[0]

	#  terms to invert
	maxD1=np.max(np.abs(D1))
	maxD2=np.max(np.abs(D2))
	if maxD1<1e-32:
		pass 
	if maxD2<1e-32:
		pass 

	# compute self-influence gains
	K11=np.dot(K12,np.dot(D2,K21))
	K22=np.dot(K21,np.dot(D1,K12))

	# left hand side terms
	L1=np.eye(Nu1)-np.dot(K11,D1)
	L2=np.eye(Nu2)-np.dot(K22,D2)

	# invert left hand side terms
	L1inv=np.linalg.inv(L1)
	L2inv=np.linalg.inv(L2)

	# coupling terms
	cpl_12=np.dot(L1inv,K12)
	cpl_21=np.dot(L2inv,K21)	

	cpl_11=np.dot(cpl_12, np.dot(D2,K21) )
	cpl_22=np.dot(cpl_21, np.dot(D1,K12) )


	# Build coupled system
	A=np.block([
		[ A1+np.dot(np.dot(B1,cpl_11),C1),    np.dot(np.dot(B1,cpl_12),C2) ], 
		[    np.dot(np.dot(B2,cpl_21),C1), A2+np.dot(np.dot(B2,cpl_22),C2) ]])

	C=np.block([
		[ C1+np.dot(np.dot(D1,cpl_11),C1),    np.dot(np.dot(D1,cpl_12),C2) ], 
		[    np.dot(np.dot(D2,cpl_21),C1), C2+np.dot(np.dot(D2,cpl_22),C2) ]])

	B=np.block([
		[B1+np.dot(np.dot(B1,cpl_11),D1),    np.dot(np.dot(B1,cpl_12),D2) ],
		[   np.dot(np.dot(B2,cpl_21),D1), B2+np.dot(np.dot(B2,cpl_22),D2) ]])

	D=np.block([
		[D1+np.dot(np.dot(D1,cpl_11),D1),    np.dot(np.dot(D1,cpl_12),D2) ],
		[   np.dot(np.dot(D2,cpl_21),D1), D2+np.dot(np.dot(D2,cpl_22),D2) ]])


	if ss01.dt is None:
		sstot=scsig.lti(A,B,C,D)
	else:
		sstot=scsig.dlti(A,B,C,D,dt=ss01.dt)
	return sstot




def couple_wrong02(ss01,ss02,K12,K21):
	'''
	Couples 2 dlti systems ss01 and ss02 through the gains K12 and K21, where
	K12 transforms the output of ss02 into an input of ss01.
	'''

	assert ss01.dt==ss02.dt, 'Time-steps not matching!'
	assert K12.shape == (ss01.inputs,ss02.outputs),\
			 'Gain K12 shape not matching with systems number of inputs/outputs'
	assert K21.shape == (ss02.inputs,ss01.outputs),\
			 'Gain K21 shape not matching with systems number of inputs/outputs'


	A1,B1,C1,D1=ss01.A,ss01.B,ss01.C,ss01.D
	A2,B2,C2,D2=ss02.A,ss02.B,ss02.C,ss02.D

	# extract size
	Nx1,Nu1=B1.shape
	Ny1=C1.shape[0]
	Nx2,Nu2=B2.shape
	Ny2=C2.shape[0]

	#  terms to invert
	maxD1=np.max(np.abs(D1))
	maxD2=np.max(np.abs(D2))
	if maxD1<1e-32:
		pass 
	if maxD2<1e-32:
		pass 


	# terms solving for u21 (input of ss02 due to ss01)
	K11=np.dot(K12,np.dot(D2,K21))
	#L1=np.eye(Nu1)-np.dot(K11,D1)
	L1inv=np.linalg.inv( np.eye(Nu1)-np.dot(K11,D1) )

	# coupling terms for u21
	cpl_11=np.dot(L1inv,K11)
	cpl_12=np.dot(L1inv,K12)

	# terms solving for u12 (input of ss01 due to ss02)
	T=np.dot( np.dot( K21,D1 ),L1inv )

	# coupling terms for u21
	cpl_21=K21+np.dot(T,K11)
	cpl_22=np.dot(T,K12)


	# Build coupled system
	A=np.block([
		[ A1+np.dot(np.dot(B1,cpl_11),C1),    np.dot(np.dot(B1,cpl_12),C2) ], 
		[    np.dot(np.dot(B2,cpl_21),C1), A2+np.dot(np.dot(B2,cpl_22),C2) ]])

	C=np.block([
		[ C1+np.dot(np.dot(D1,cpl_11),C1),    np.dot(np.dot(D1,cpl_12),C2) ], 
		[    np.dot(np.dot(D2,cpl_21),C1), C2+np.dot(np.dot(D2,cpl_22),C2) ]])

	B=np.block([
		[B1+np.dot(np.dot(B1,cpl_11),D1),    np.dot(np.dot(B1,cpl_12),D2) ],
		[   np.dot(np.dot(B2,cpl_21),D1), B2+np.dot(np.dot(B2,cpl_22),D2) ]])

	D=np.block([
		[D1+np.dot(np.dot(D1,cpl_11),D1),    np.dot(np.dot(D1,cpl_12),D2) ],
		[   np.dot(np.dot(D2,cpl_21),D1), D2+np.dot(np.dot(D2,cpl_22),D2) ]])


	if ss01.dt is None:
		sstot=scsig.lti(A,B,C,D)
	else:
		sstot=scsig.dlti(A,B,C,D,dt=ss01.dt)
	return sstot



def couple_wrong(ss01,ss02,K12,K21):
	'''
	Couples 2 dlti systems ss01 and ss02 through the gains K12 and K21, where
	K12 transforms the output of ss02 into an input of ss01.
	'''

	assert ss01.dt==ss02.dt, 'Time-steps not matching!'
	assert K12.shape == (ss01.inputs,ss02.outputs),\
			 'Gain K12 shape not matching with systems number of inputs/outputs'
	assert K21.shape == (ss02.inputs,ss01.outputs),\
			 'Gain K21 shape not matching with systems number of inputs/outputs'


	A1,B1,C1,D1=ss01.A,ss01.B,ss01.C,ss01.D
	A2,B2,C2,D2=ss02.A,ss02.B,ss02.C,ss02.D

	# extract size
	Nx1,Nu1=B1.shape
	Ny1=C1.shape[0]
	Nx2,Nu2=B2.shape
	Ny2=C2.shape[0]

	#  terms to invert
	maxD1=np.max(np.abs(D1))
	maxD2=np.max(np.abs(D2))
	if maxD1<1e-32:
		pass 
	if maxD2<1e-32:
		pass 

	# compute self-coupling terms
	S1=np.dot(K12,np.dot(D2,K21))
	S2=np.dot(K21,np.dot(D1,K12))

	# left hand side terms
	L1=np.eye(Nu1)-np.dot(S1,D1)
	L2=np.eye(Nu2)-np.dot(S2,D2)

	# invert left hand side terms
	L1inv=np.linalg.inv(L1)
	L2inv=np.linalg.inv(L2)

	# recurrent terms
	L1invS1=np.dot(L1inv,S1)
	L2invS2=np.dot(L2inv,S2)

	L1invK12=np.dot(L1inv,K12)
	L2invK21=np.dot(L2inv,K21)

	# Build coupled system
	A=np.block([
		[ A1+np.dot(np.dot(B1,L1invS1), C1),    np.dot(np.dot(B1,L1invK12),C2) ], 
		[    np.dot(np.dot(B2,L2invK21),C1), A2+np.dot(np.dot(B2,L2invS2), C2) ]])

	C=np.block([
		[ C1+np.dot(np.dot(D1,L1invS1), C1),    np.dot(np.dot(D1,L1invK12),C2) ], 
		[    np.dot(np.dot(D2,L2invK21),C1), C2+np.dot(np.dot(D2,L2invS2), C2) ]])

	B=np.block([
		[B1+np.dot(np.dot(B1,L1invS1), D1),    np.dot(np.dot(B1,L1invK12),D2) ],
		[   np.dot(np.dot(B2,L2invK21),D1), B2+np.dot(np.dot(B2,L2invS2), D2) ]])

	D=np.block([
		[D1+np.dot(np.dot(D1,L1invS1), D1),    np.dot(np.dot(D1,L1invK12),D2) ],
		[   np.dot(np.dot(D2,L2invK21),D1), D2+np.dot(np.dot(D2,L2invS2), D2) ]])


	if ss01.dt is None:
		sstot=scsig.lti(A,B,C,D)
	else:
		sstot=scsig.dlti(A,B,C,D,dt=ss01.dt)
	return sstot



def freqresp(SS,wv,eng=None,method='standard',dlti=True,use_sparse=True):
	''' In-house frequency response function '''

	# matlab frequency response
	if method=='matlab':
		raise NameError('Please use freqresp function in matwrapper module!')

	# in-house implementation	
	elif method=='standard':

		if hasattr(SS,'dt') and dlti:
			Ts=SS.dt
			wTs=Ts*wv
			zv=np.cos(wTs)+1.j*np.sin(wTs)
		else:
			print('Assuming a continuous time system')
			zv=1.j*wv

		Nx=SS.A.shape[0]
		Ny=SS.D.shape[0]
		Nu=SS.B.shape[1]
		Nw=len(wv)
		Yfreq=np.empty((Ny,Nu,Nw,),dtype=np.complex_)

		if use_sparse:
			# csc format used for efficiency
			Asparse=sparse.csc_matrix(SS.A)
			Bsparse=sparse.csc_matrix(SS.B)
			Eye=sparse.eye(Nx,format='csc')
			for ii in range(Nw):
				sol_cplx=sparse.linalg.spsolve(zv[ii]*Eye-Asparse,Bsparse)
				Yfreq[:,:,ii]=np.dot(SS.C,sol_cplx.todense())+SS.D
		else:
			Eye=np.eye(Nx)
			for ii in range(Nw):
				Yfreq[:,:,ii]=np.dot(SS.C,
				              		 np.linalg.solve(zv[ii]*Eye-SS.A,SS.B))+SS.D

	return Yfreq



def series(SS01,SS02):
	'''
	Connects two state-space blocks in series. If these are instances of DLTI
	state-space systems, they need to have the same type and time-step.

	The connection is such that:
		u --> SS01 --> SS02 --> y 		==>		u --> SStot --> y
	'''

	if type(SS01) is not type(SS02):
		raise NameError('The two input systems need to have the same size!')
	if SS01.dt != SS02.dt:
		raise NameError('DLTI systems do not have the same time-step!')

	# if type(SS01) is control.statesp.StateSpace:
	# 	SStot=control.series(SS01,SS02)
	# else:

	# determine size of total system
	Nst01,Nst02=SS01.A.shape[0],SS02.A.shape[0]
	Nst=Nst01+Nst02
	Nin=SS01.inputs
	Nout=SS02.outputs

	# Build A matrix
	A=np.zeros((Nst,Nst))
	A[:Nst01,:Nst01]=SS01.A
	A[Nst01:,Nst01:]=SS02.A
	A[Nst01:,:Nst01]=np.dot(SS02.B,SS01.C)

	# Build the rest
	B=np.concatenate( ( SS01.B, np.dot(SS02.B,SS01.D) ), axis=0 )
	C=np.concatenate( ( np.dot(SS02.D,SS01.C), SS02.C ), axis=1 )		
	D=np.dot( SS02.D, SS01.D )

	SStot=scsig.dlti(A,B,C,D,dt=SS01.dt)

	return SStot



def parallel(SS01,SS02):
	'''
	Returns the sum (or paralle connection of two systems). Given two state-space
	models with the same output, but different input:
		u1 --> SS01 --> y
		u2 --> SS02 --> y

	'''

	if type(SS01) is not type(SS02):
		raise NameError('The two input systems need to have the same size!')
	if SS01.dt != SS02.dt:
		raise NameError('DLTI systems do not have the same time-step!')
	Nout=SS02.outputs
	if Nout != SS01.outputs: 
		raise NameError('DLTI systems need to have the same number of output!')


	# if type(SS01) is control.statesp.StateSpace:
	# 	SStot=control.parallel(SS01,SS02)
	# else:
	
	# determine size of total system
	Nst01,Nst02=SS01.A.shape[0],SS02.A.shape[0]
	Nst=Nst01+Nst02
	Nin01,Nin02=SS01.inputs,SS02.inputs
	Nin=Nin01+Nin02

	# Build A,B matrix
	A=np.zeros((Nst,Nst))
	A[:Nst01,:Nst01]=SS01.A
	A[Nst01:,Nst01:]=SS02.A
	B=np.zeros((Nst,Nin))
	B[:Nst01,:Nin01]=SS01.B
	B[Nst01:,Nin01:]=SS02.B

	# Build the rest
	C=np.block([ SS01.C,SS02.C ])		
	D=np.block([ SS01.D,SS02.D ])		

	SStot=scsig.dlti(A,B,C,D,dt=SS01.dt)

	return SStot




def SSconv(A,B0,B1,C,D,Bm1=None):
	'''
	Convert a DLTI system with prediction and delay of the form:
		x_{n+1} = A x_n + B0 u_n + B1 u_{n+1} + Bm1 u^{n-1}
		y_n = C x_n + D u_n
	into the state-space form
		h_{n+1} = Ah h_n + Bh u_n
		y_n = Ch h_n + Dh u_n

	If Bm1 is None, the original state is retrieved through
		x_n = h_n + B1 u_n
	and only the B and D matrices are modified.

	If Bm1 is not None, the SS is augmented with the new state
		g^{n} = u^{n-1}
	or, equivalently, with the equation
		g^{n+1}=u^n
	leading to the new form
		H^{n+1} = AA H^{n} + BB u^n
		y^n = CC H^{n} + DD u^n
	where H=(x,g)

	@ref: Franklin and Powell
	'''

	# Account for u^{n+1} terms (prediction)
	if type(A) is sparse.bsr.bsr_matrix: Bh=B0+A.dot(B1)
	else: Bh=B0+np.dot(A,B1)

	if type(C) is sparse.bsr.bsr_matrix: Dh=D+C.dot(B1)
	else: Dh=D+np.dot(C,B1)

	# Account for u^{n-1} terms (delay)
	if Bm1 is None:
		outs=(A,Bh,C,Dh)
	else:
		Nx,Nu,Ny=A.shape[0],B0.shape[1],C.shape[0]
		AA=np.block( [[A, Bm1],
			         [np.zeros((Nu,Nx)), np.zeros((Nu,Nu))]])
		BB=np.block( [[Bh],[np.eye(Nu)]] )
		CC=np.block( [C,np.zeros((Ny,Nu))] )
		DD=Dh
		outs=(AA,BB,CC,DD)

	return outs



def addGain(SShere,Kmat,where):
	'''
	Convert input u or output y of a SS DLTI system through gain matrix K. We
	have the following transformations:
	- where='in': the input dof of the state-space are changed
		u_new -> Kmat*u -> SS -> y  => u_new -> SSnew -> y
	- where='out': the output dof of the state-space are changed
	 	u -> SS -> y -> Kmat*u -> ynew => u -> SSnew -> ynew 
	- where='parallel': the input dofs are changed, but not the output 
		 -
		{u_1 -> SS -> y_1
	   { u_2 -> y_2= Kmat*u_2    =>    u_new=(u_1,u_2) -> SSnew -> y=y_1+y_2
		{y = y_1+y_2
		 -
	'''

	assert where in ['in', 'out', 'parallel-down', 'parallel-up'],\
							'Specify whether gains are added to input or output'

	if where=='in':
		A=SShere.A
		B=np.dot(SShere.B,Kmat)
		C=SShere.C
		D=np.dot(SShere.D,Kmat)

	if where=='out':
		A=SShere.A
		B=SShere.B
		C=np.dot(Kmat,SShere.C)
		D=np.dot(Kmat,SShere.D)

	if where=='parallel-down':
		A=SShere.A
		C=SShere.C
		B=np.block([SShere.B, np.zeros((SShere.B.shape[0],Kmat.shape[1]))])
		D=np.block([SShere.D, Kmat])	

	if where=='parallel-up':
		A=SShere.A
		C=SShere.C
		B=np.block([np.zeros((SShere.B.shape[0],Kmat.shape[1])),SShere.B])
		D=np.block([Kmat,SShere.D])	

	if SShere.dt==None:
		SSnew=scsig.lti(A,B,C,D)
	else:
		SSnew=scsig.dlti(A,B,C,D,dt=SShere.dt)

	return SSnew



def join(SS1,SS2):
	'''
	Join two state-spaces or gain matrices such that, given:
		u1 -> SS1 -> y1
		u2 -> SS2 -> y2
	we obtain:
		u -> SStot -> y
	with u=(u1,u2)^T and y=(y1,y2)^T.

	The output SStot is either a gain matrix or a state-space system according
	to the input SS1 and SS2
	'''

	type_dlti=scsig.ltisys.StateSpaceDiscrete


	if isinstance(SS1,np.ndarray) and isinstance(SS2,np.ndarray):

		Nin01,Nin02=SS1.shape[1],SS2.shape[1]
		Nout01,Nout02=SS1.shape[0],SS2.shape[0]
		SStot=np.block([[SS1, np.zeros((Nout01,Nin02))],
			            [np.zeros((Nout02,Nin01)),SS2 ]])


	elif isinstance(SS1,np.ndarray) and isinstance(SS2,type_dlti):

		Nin01,Nout01=SS1.shape[1],SS1.shape[0]
		Nin02,Nout02=SS2.inputs,SS2.outputs
		Nx02=SS2.A.shape[0]

		A=SS2.A
		B=np.block([np.zeros((Nx02,Nin01)),SS2.B])
		C=np.block([[np.zeros((Nout01,Nx02))],
					[SS2.C 				   ]])
		D=np.block([[SS1,np.zeros((Nout01,Nin02))],
					[np.zeros((Nout02,Nin01)),SS2.D]])

		SStot=scsig.StateSpace(A,B,C,D,dt=SS2.dt)		


	elif isinstance(SS1,type_dlti) and isinstance(SS2,np.ndarray):

		Nin01,Nout01=SS1.inputs,SS1.outputs
		Nin02,Nout02=SS2.shape[1],SS2.shape[0]
		Nx01=SS1.A.shape[0]

		A=SS1.A
		B=np.block([SS1.B,np.zeros((Nx01,Nin02))])
		C=np.block([[SS1.C 				   ],
					[np.zeros((Nout02,Nx01))]])
		D=np.block([[SS1.D,np.zeros((Nout01,Nin02))],
			        [np.zeros((Nout02,Nin01)),SS2]])

		SStot=scsig.StateSpace(A,B,C,D,dt=SS1.dt)	


	elif isinstance(SS1,type_dlti) and isinstance(SS2,type_dlti):

		assert SS1.dt==SS2.dt, 'State-space models must have the same time-step'

		Nin01,Nout01=SS1.inputs,SS1.outputs
		Nin02,Nout02=SS2.inputs,SS2.outputs
		Nx01,Nx02=SS1.A.shape[0],SS2.A.shape[0]

		A=np.block([[ SS1.A, np.zeros((Nx01,Nx02)) ],
					[ np.zeros((Nx02,Nx01)), SS2.A ]])
		B=np.block([[ SS1.B, np.zeros((Nx01,Nin02)) ],
					[ np.zeros((Nx02,Nin01)), SS2.B]])
		C=np.block([[ SS1.C, np.zeros((Nout01,Nx02))],
					[ np.zeros((Nout02,Nx01)), SS2.C]])
		D=np.block([[SS1.D, np.zeros((Nout01,Nin02))],
					[np.zeros((Nout02,Nin01)), SS2.D]])
		SStot=scsig.StateSpace(A,B,C,D,dt=SS1.dt)


	else:
		raise NameError('Input types not recognised in any implemented option!') 

	return SStot



def sum(SS1,SS2,negative=False):
	'''
	Given 2 systems or gain matrices (or a combination of the two) having the
	same amount of input/output, the function returns a gain or state space 
	model summing the two. Namely, given:
		u -> SS1 -> y1
		u -> SS2 -> y2
	we obtain:
		u -> SStot -> y1+y2 	if negative=False
	'''
	type_dlti=scsig.ltisys.StateSpaceDiscrete


	if isinstance(SS1,np.ndarray) and isinstance(SS2,np.ndarray):
		SStot=SS1+SS2

	elif isinstance(SS1,np.ndarray) and isinstance(SS2,type_dlti):
		Kmat=SS1
		A=SS2.A
		B=SS2.B
		C=SS2.C
		D=SS2.D+Kmat
		SStot=scsig.StateSpace(A,B,C,D,dt=SS2.dt)		

	elif isinstance(SS1,type_dlti) and isinstance(SS2,np.ndarray):
		Kmat=SS2
		A=SS1.A
		B=SS1.B
		C=SS1.C
		D=SS1.D+Kmat

		SStot=scsig.StateSpace(A,B,C,D,dt=SS2.dt)	

	elif isinstance(SS1,type_dlti) and isinstance(SS2,type_dlti):

		assert np.abs(1.-SS1.dt/SS2.dt)<1e-13,\
		                       'State-space models must have the same time-step'



		Nin01,Nout01=SS1.inputs,SS1.outputs
		Nin02,Nout02=SS2.inputs,SS2.outputs
		Nx01,Nx02=SS1.A.shape[0],SS2.A.shape[0]

		A=np.block([[ SS1.A, np.zeros((Nx01,Nx02)) ],
					[ np.zeros((Nx02,Nx01)), SS2.A ]])
		B=np.block([[ SS1.B,],
					[ SS2.B]])
		C=np.block([SS1.C, SS2.C])
		D=SS1.D+SS2.D

		SStot=scsig.StateSpace(A,B,C,D,dt=SS1.dt)


	else:
		raise NameError('Input types not recognised in any implemented option!') 

	return SStot


def scale_SS(SSin,input_scal=1.,output_scal=1.,state_scal=1.,byref=True):
	'''
	Given a state-space system, scales the equations such that the original
	input and output, u and y, are substituted by uad=u/uref and yad=y/yref.
	The entries input_scal/output_scal/state_scal can be:
		- floats: in this case all input/output are scaled by the same value
		- lists/arrays of length Nin/Nout: in this case each dof will be scaled
		by a different factor

	If the original system has form:
		xnew=A*x+B*u
		y=C*x+D*u
	the transformation is such that:
		xnew=A*x+(B*uref/xref)*uad
		yad=1/yref( C*xref*x+D*uref*uad )

	By default, the state-space model is manipulated by reference (byref=True)
	'''

	# check input:
	Nin,Nout=SSin.inputs,SSin.outputs
	Nstates=SSin.A.shape[0]

	if isinstance(input_scal,(list,np.ndarray)):
		assert len(input_scal)==Nin,\
			   'Length of input_scal not matching number of state-space inputs!'
	else:
		input_scal=Nin*[input_scal]

	if isinstance(output_scal,(list,np.ndarray)):
		assert len(output_scal)==Nout,\
			 'Length of output_scal not matching number of state-space outputs!'
	else:
		output_scal=Nout*[output_scal]

	if isinstance(state_scal,(list,np.ndarray)):
		assert len(state_scal)==Nstates,\
			   'Length of state_scal not matching number of state-space states!'
	else:
		state_scal=Nstates*[state_scal]


	if byref:
		SS=SSin
	else:
		print('deep-copying state-space model before scaling')
		SS=copy.deepcopy(SSin)


	# update input related matrices
	for ii in range(Nin):
		SS.B[:,ii]=SS.B[:,ii]*input_scal[ii]
		SS.D[:,ii]=SS.D[:,ii]*input_scal[ii]
	# SS.B*=input_scale
	# SS.D*=input_scale

	# update output related matrices
	for ii in range(Nout):
		SS.C[ii,:]=SS.C[ii,:]/output_scal[ii]
		SS.D[ii,:]=SS.D[ii,:]/output_scal[ii]

	# update state related matrices
	for ii in range(Nstates):
		SS.B[ii,:]=SS.B[ii,:]/state_scal[ii]
		SS.C[:,ii]=SS.C[:,ii]*state_scal[ii]
	# SS.B /= state_scal	

	return SS



def simulate(SShere,U,x0=None):
	'''
	Routine to simulate response to generic input.
	@warning: this routine is for testing and may lack of robustness. Use
		scipy.signal instead.
	'''

	A,B,C,D=SShere.A,SShere.B,SShere.C,SShere.D

	NT=U.shape[0]
	Nx=A.shape[0]
	Ny=C.shape[0]

	X=np.zeros((NT,Nx))
	Y=np.zeros((NT,Ny))

	if x0 is not None: X[0]=x0
	if len(U.shape)==1:
		U=U.reshape( (NT,1) )

	Y[0]=np.dot(C,X[0])+np.dot(D,U[0])

	for ii in range(1,NT):
		X[ii]=np.dot(A,X[ii-1])+np.dot(B,U[ii-1])
		Y[ii]=np.dot(C,X[ii])+np.dot(D,U[ii])

	return Y,X



def Hnorm_from_freq_resp(gv,method):
	'''
	Given a frequency response over a domain kv, this funcion computes the
	H norms through numerical integration.

	Note that if kv[-1]<np.pi/dt, the method assumed gv=0 for each frequency
	kv[-1]<k<np.pi/dt.
	'''

	if method is 'H2':
		Nk=len(gv)
		gvsq=gv*gv.conj()
		Gnorm=np.sqrt(np.trapz(gvsq/(Nk-1.)))

	elif method is 'Hinf':
		Gnorm=np.linalg.norm(gv,np.inf)
	
	if np.abs(Gnorm.imag/Gnorm.real)>1e-16:
		raise NameError('Norm is not a real number. Verify data/algorithm!')

	return Gnorm



def adjust_phase(y,deg=True):
	'''
	Modify the phase y of a frequency response to remove discontinuities.
	'''

	if deg is True: 
		shift=360.
	else: 
		shift=2.*np.pi

	dymax=0.0
	
	N=len(y)
	for ii in range(N-1):
		dy=y[ii+1]-y[ii]
		if np.abs(dy)>dymax: dymax=np.abs(dy)
		if dy>0.97*shift:
			print('Subtracting shift to frequency response phase diagram!')
			y[ii+1::]=y[ii+1::]-shift

		elif dy<-0.97*shift:
			y[ii+1::]=y[ii+1::]+shift
			print('Adding shift to frequency response phase diagram!')	

	return y




def SSderivative(ds):
	'''
	Given a time-step ds, and an single input time history u, this SS model 
	returns the output y=[u,du/ds], where du/dt is computed with second order 
	accuracy. 
	'''

	A=np.array([[0]])
	Bm1=np.array([0.5/ds])
	B0=np.array([[-2/ds]])
	B1=np.array([[1.5/ds]])
	C=np.array([[0],[1]])
	D=np.array([[1],[0]])

	# change state
	Aout,Bout,Cout,Dout=SSconv(A,B0,B1,C,D,Bm1)

	return Aout,Bout,Cout,Dout



def butter(order,Wn,N=1,btype='lowpass'):
	'''
	build MIMO butterworth filter of order ord and cut-off freq over Nyquist 
	freq ratio Wn.
	The filter will have N input and N output and N*ord states.

	Note: the state-space form of the digital filter does not depend on the 
	sampling time, but only on the Wn ratio. 
	As a result, this function only returns the A,B,C,D matrices of the filter
	state-space form.
	'''

	# build DLTI SISO
	num,den=scsig.butter(order,Wn,btype=btype,analog=False,output='ba')
	Af,Bf,Cf,Df=scsig.tf2ss(num,den)
	SSf=scsig.dlti(Af,Bf,Cf,Df,dt=1.0)

	SStot=SSf
	for ii in range(1,N):
		SStot=join(SStot,SSf)

	return SStot.A,SStot.B,SStot.C,SStot.D	




if __name__=='__main__':

	# check parallel connector
	Nout=2
	Nin01,Nin02=2,3
	Nst01,Nst02=4,2

	# build random systems
	fac=0.1
	A01,A02=fac*np.random.rand(Nst01,Nst01),fac*np.random.rand(Nst02,Nst02)
	B01,B02=np.random.rand(Nst01,Nin01),np.random.rand(Nst02,Nin02)
	C01,C02=np.random.rand(Nout,Nst01),np.random.rand(Nout,Nst02)
	D01,D02=np.random.rand(Nout,Nin01),np.random.rand(Nout,Nin02)

	dt=0.1
	SS01=scsig.StateSpace( A01,B01,C01,D01,dt=dt )
	SS02=scsig.StateSpace( A02,B02,C02,D02,dt=dt )

	# simulate
	NT=11
	U01,U02=np.random.rand(NT,Nin01),np.random.rand(NT,Nin02)

	# reference
	Y01,X01=simulate(SS01,U01)
	Y02,X02=simulate(SS02,U02)
	Yref=Y01+Y02

	# parallel
	SStot=parallel(SS01,SS02)
	Utot=np.block([U01,U02])
	Ytot,Xtot=simulate(SStot,Utot)

	# join method
	SStot=join(SS01,SS02)
	K=np.array([[1,2,3],[4,5,6]])
	SStot=join(K,SS02)
	SStot=join(SS02,K)
	K2=np.array([[10,20,30],[40,50,60]]).T
	Ktot=join(K,K2)

	# MIMO butterworth filter
	Af,Bf,C,Df=butter(4,.4,N=4)




	#embed()



