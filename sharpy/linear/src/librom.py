'''
Python tools for model reduction
S. Maraniello, 14 Feb 2018
'''

import warnings
import numpy as np 
import scipy as sc 
import scipy.linalg as scalg
import scipy.signal as scsig
# from IPython import embed

import libss # only for tune_rom


def balreal_direct_py(A,B,C,DLTI=True,Schur=False):
	'''
	Find balanced realisation of continuous (DLTI=False) and discrete (DLTI=True) 
	time of LTI systems using  scipy libraries. 

	Warning: this function may be less computationally efficient than the balreal
	Matlab implementation and does not offer the option to bound the realisation
	in frequency and time.

	Notes: Lyapunov equations are solved using Barlets-Stewart algorithm for
	Sylvester equation, which is based on A matrix Schur decomposition.
	'''


	### select solver for Lyapunov equation
	# Notation reminder:
	# scipy: A X A.T - X = -Q
	# contr: A W A.T - W = - B B.T
	# obser: A.T W A - W = - C.T C	
	if DLTI: 
		sollyap=scalg.solve_discrete_lyapunov
	else:
		sollyap=scalg.solve_lyapunov

	# solve Lyapunov
	if Schur:
		# decompose A
		Atri,U=scalg.schur(A)
		# solve Lyapunov
		BBtri=np.dot(U.T, np.dot(B, np.dot(B.T,U)))
		CCtri=np.dot(U.T, np.dot(C.T, np.dot(C,U)))
		Wctri=sollyap(Atri,BBtri)
		Wotri=sollyap(Atri.T,CCtri)		
		# reconstruct Wo,Wc
		Wc=np.dot(U,np.dot(Wctri,U.T))
		Wo=np.dot(U,np.dot(Wotri,U.T))
	else:
		Wc=sollyap(A,np.dot(B,B.T))
		Wo=sollyap(A.T,np.dot(C.T,C))


	# Choleski factorisation: W=Q Q.T
	Qc=scalg.cholesky(Wc).T
	Qo=scalg.cholesky(Wo).T

	# build M matrix and SVD
	M=np.dot(Qo.T,Qc)
	U,s,Vh=scalg.svd(M)
	S=np.diag(s)
	Sinv=np.diag(1./s)
	V=Vh.T

	# Build transformation matrices
	T=np.dot(Qc,np.dot(V,np.sqrt(Sinv)))
	Tinv=np.dot(np.sqrt(Sinv),np.dot(U.T,Qo.T))

	return S,T,Tinv


def balreal_iter(A,B,C,lowrank=True,tolSmith=1e-10,tolSVD=1e-6,kmin=None,
													tolAbs=False,PrintAll=False):
	'''
	Find balanced realisation of DLTI system. 

	Notes: Lyapunov equations are solved using iterative squared Smith 
	algorithm, in its low or full rank version. These implementations are
	as per the low_rank_smith and smith_iter functions respectively but, 
	for computational efficiency,, the iterations are rewritten here so as to 
	solve for the observability and controllability Gramians contemporary.
	'''

	### Solve Lyapunov equations
	# Notation reminder:
	# scipy: A X A.T - X = -Q
	# contr: A W A.T - W = - B B.T
	# obser: A.T W A - W = - C.T C	
	# low-rank smith: A.T X A - X = -Q Q.T

	if lowrank: # low-rank square-Smith iteration (with SVD)

		# matrices size
		N=A.shape[0]
		rC=B.shape[1]
		rO=C.shape[0]

		# initialise smith iteration
		DeltaNorm=1e6 					# error 
		DeltaNormNext=DeltaNorm**2		# error expected at next iter
		print('Iter\tMaxZ\t|\trank_c\trank_o\tA size')
		kk=0

		Qck=B
		Qok=C.T

		while DeltaNorm>tolSmith and DeltaNormNext>1e-3*tolSmith:

			###### controllability
			### compute Ak^2 * Qck
			# (future: use block Arnoldi)
			Qcright=np.dot(A,Qck)
			MaxZhere=np.max(np.abs(Qcright))

			### enlarge Z matrices
			Qck=np.concatenate((Qck,Qcright),axis=1)
			Qcright=None
			rC=Qck.shape[1]

			if kmin==None or kmin<rC:
				### "cheap" SVD truncation
				Uc,svc=scalg.svd(Qck,full_matrices=False,overwrite_a=True,
														  lapack_driver='gesdd')[:2]
				# import scipy.linalg.interpolative as sli
				# Ucnew,svcnew,temp=sli.svd(Qck,tolSVD)
				if tolAbs:
					rcmax=np.sum(svc>tolSVD)
				else:
					rcmax=np.sum(svc>tolSVD*svc[0])
				if kmin!=None:
					rC=max(rcmax,kmin)
				else:
					rC=rcmax
				Qck=Uc[:,:rC]*svc[:rC]
				# free memory
				Uc=None
				Qcright=None 


			###### observability
			### compute Ak^2 * Qok
			# (future: use block Arnoldi)
			Qoright=np.transpose( np.dot(Qok.T,A) )
			DeltaNorm=max(MaxZhere,np.max(np.abs(Qoright)))

			### enlarge Z matrices
			Qok=np.concatenate((Qok,Qoright),axis=1)
			Qoright=None
			rO=Qok.shape[1]

			if kmin==None or kmin<rO:
				### "cheap" SVD truncation
				Uo,svo=scalg.svd(Qok,full_matrices=False)[:2]

				if tolAbs:
					romax=np.sum(svo>tolSVD)
				else:
					romax=np.sum(svo>tolSVD*svo[0])
				if kmin!=None: 
					rO=max(romax,kmin)
				else:
					rO=romax
				Qok=Uo[:,:rO]*svo[:rO]
				Uo=None

			##### Prepare next time step
			print('%.3d\t%.2e\t%.5d\t%.5d\t%.5d' %(kk,DeltaNorm,rC,rO,N))
			DeltaNormNext=DeltaNorm**2


			if DeltaNorm>tolSmith and DeltaNormNext>1e-3*tolSmith:
				A=np.linalg.matrix_power(A,2)

			### update
			kk=kk+1

		A=None


	else: # full-rank squared smith iteration (with Cholevsky)

		raise NameError('Use balreal_iter_old instead!')

	# find min size (only if iter used)
	cc,co=Qck.shape[1],Qok.shape[1]
	print('cc=%.2d, co=%.2d'%(cc,co))
	
	# build M matrix and SVD
	M=np.dot(Qok.T,Qck)
	U,s,Vh=scalg.svd(M,full_matrices=False)
	sinv=s**(-0.5)
	T=np.dot(Qck,Vh.T*sinv)
	Tinv=np.dot((U*sinv).T,Qok.T)

	print('rank(Zc)=%.4d\trank(Zo)=%.4d'%(rcmax,romax) )

	return s,T,Tinv,rcmax,romax


def balreal_iter_old(A,B,C,lowrank=True,tolSmith=1e-10,tolSVD=1e-6,kmax=None,
												                  tolAbs=False):
	'''
	Find balanced realisation of DLTI system. 

	Notes: Lyapunov equations are solved using iterative squared Smith 
	algorithm, in its low or full rank version. These implementations are
	as per the low_rank_smith and smith_iter functions respectively but, 
	for computational efficiency,, the iterations are rewritten here so as to 
	solve for the observability and controllability Gramians contemporary.
	'''

	### Solve Lyapunov equations
	# Notation reminder:
	# scipy: A X A.T - X = -Q
	# contr: A W A.T - W = - B B.T
	# obser: A.T W A - W = - C.T C	
	# low-rank smith: A.T X A - X = -Q Q.T

	if lowrank: # low-rank square-Smith iteration (with SVD)

		# matrices size
		N=A.shape[0]
		rB=B.shape[1]
		rC=C.shape[0]

		# initialise smith iteration
		DeltaNorm=1e6
		print('Iter\tMaxZhere')
		kk=0
		Apow=A
		Qck=B
		Qok=C.T

		while DeltaNorm>tolSmith:
			### compute products Ak^2 * Zk
			### (use block Arnoldi)
			Qcright=np.dot(Apow,Qck)
			Qoright=np.dot(Apow.T,Qok)
			Apow=np.dot(Apow,Apow)

			### enlarge Z matrices
			Qck=np.concatenate((Qck,Qcright),axis=1)
			Qok=np.concatenate((Qok,Qoright),axis=1)

			### check convergence without reconstructing the added term
			MaxZhere=max(np.max(np.abs(Qoright)),np.max(np.abs(Qcright)))
			print('%.4d\t%.3e'%(kk,MaxZhere))
			DeltaNorm=MaxZhere

			# fixed columns chopping
			if kmax is None:
				# cheap SVD truncation
				if Qck.shape[1]>.4*N or Qok.shape[1]>.4*N:
					Uc,svc=scalg.svd(Qck,full_matrices=False)[:2]
					Uo,svo=scalg.svd(Qok,full_matrices=False)[:2]
					if tolAbs:
						rcmax=np.sum(svc>tolSVD)
						romax=np.sum(svo>tolSVD)
					else:
						rcmax=np.sum(svc>tolSVD*svc[0])
						romax=np.sum(svo>tolSVD*svo[0])
					pmax=max(rcmax,romax)
					Qck=Uc[:,:pmax]*svc[:pmax]
					Qok=Uo[:,:pmax]*svo[:pmax]
					# Qck_old=np.dot(Uc[:,:pmax],np.diag(svc[:pmax]))
					# Qok_old=np.dot(Uo[:,:pmax],np.diag(svo[:pmax]))
					# Qck=np.dot(Uc[:,:rcmax],np.diag(svc[:rcmax]))
					# Qok=np.dot(Uo[:,:romax],np.diag(svo[:romax]))
			else:
				if Qck.shape[1]>kmax:
					Uc,svc=scalg.svd(Qck,full_matrices=False)[:2]
					Qck=Uc[:,:kmax]*svc[:kmax]
				if Qok.shape[1]>kmax:
					Uo,svo=scalg.svd(Qok,full_matrices=False)[:2]
					Qok=Uo[:,:kmax]*svo[:kmax]

			### update
			kk=kk+1

		del Apow
		Qc,Qo=Qck,Qok

	else: # full-rank squared smith iteration (with Cholevsky)

		# first iteration
		Wc=np.dot(B,B.T)
		Wo=np.dot(C.T,C)
		Apow=A
		AXAobs=np.dot(np.dot(A.T,Wo),A)
		AXActrl=np.dot(np.dot(A,Wc),A.T)
		DeltaNorm=max(np.max(np.abs(AXAobs)),np.max(np.abs(AXActrl)))

		kk=1
		print('Iter\tRes')
		while DeltaNorm>tolSmith:
			kk=kk+1

			# update 
			Wo=Wo+AXAobs
			Wc=Wc+AXActrl

			# incremental
			Apow=np.dot(Apow,Apow)
			AXAobs=np.dot(np.dot(Apow.T,Wo),Apow)
			AXActrl=np.dot(np.dot(Apow,Wc),Apow.T)
			DeltaNorm=max(np.max(np.abs(AXAobs)),np.max(np.abs(AXActrl)))
			print('%.4d\t%.3e'%(kk,DeltaNorm))
		# final update (useless in very low tolerance)
		Wo=Wo+AXAobs
		Wc=Wc+AXActrl

		# Choleski factorisation: W=Q Q.T. If unsuccessful, directly solve 
		# eigenvalue problem
		Qc=scalg.cholesky(Wc).T
		Qo=scalg.cholesky(Wo).T	
		# # eigenvalues are normalised by one, hence Tinv and T matrices
		# # here are not scaled
		# ssq,Tinv,T=scalg.eig(np.dot(Wc,Wo),left=True,right=True)
		# Tinv=Tinv.T 
		# #Tinv02=Tinv02.T
		# S=np.diag(np.sqrt(ssq))
		# return S,T,Tinv

	# find min size (only if iter used)
	cc,co=Qc.shape[1],Qo.shape[1]
	cmin=min(cc,co)
	print('cc=%.2d, co=%.2d'%(cc,co))
	
	# build M matrix and SVD
	M=np.dot(Qo.T,Qc)

	# ### not optimised
	# U,s,Vh=scalg.svd(M,full_matrices=True)
	# U,Vh,s=U[:,:cmin],Vh[:cmin,:],s[:cmin]
	# S=np.diag(s)
	# Sinv=np.diag(1./s)
	# V=Vh.T
	# # Build transformation matrices
	# T=np.dot(Qc,np.dot(V,np.sqrt(Sinv)))
	# Tinv=np.dot(np.sqrt(Sinv),np.dot(U.T,Qo.T))

	### optimised
	U,s,Vh=scalg.svd(M,full_matrices=True) # as M is square, full_matrices has no effect
	sinv=s**(-0.5)
	T=np.dot(Qc,Vh.T*sinv)
	Tinv=np.dot((U*sinv).T,Qo.T)

	return s,T,Tinv





def smith_iter(S,T,tol=1e-8,Square=True):
	'''
	Solves the Stein equation
		S.T X S - X = -T
	by mean of Smith or squared-Smith algorithm. Note that a solution X exists 
	only if the eigenvalues of S are stricktly smaller than one, and the 
	algorithm will not converge otherwise. The algorithm can not exploit
	sparsity, hence, while convergence can be improved for very large matrices, 
	it can not be employed if matrices are too large to be stored in memory. 

	Ref. Penzt, "A cyclic low-rank Smith method for large sparse Lyapunov 
	equations", 2000.
	'''


	N=S.shape[0]

	if Square: 

		# first iteration
		X=T
		Spow=S
		STXS=np.dot(np.dot(S.T,X),S)
		DeltaNorm=np.max(np.abs(STXS))

		# # second iteration:
		# # can be removed using Spow=np.dot(Spow,Spow)
		# X=X+STXS
		# S=np.dot(S,S)
		# Spow=S
		# STXS=np.dot(np.dot(Spow.T,X),Spow)
		# DeltaNorm=np.max(np.abs(STXS))

		counter=1
		print('Iter\tRes')
		while DeltaNorm>tol:
			counter=counter+1

			# update 
			X=X+STXS

			# incremental
			#Spow=np.dot(Spow,S) # use this if uncomment second iter
			Spow=np.dot(Spow,Spow)
			STXS=np.dot(np.dot(Spow.T,X),Spow)
			DeltaNorm=np.max(np.abs(STXS))

			print('%.4d\t%.3e'%(counter,DeltaNorm))

	else:
		# first iteration
		X=T
		Spow=S
		STTS=np.dot(np.dot(Spow.T,T),Spow)
		DeltaNorm=np.max(np.abs(STTS))

		counter=1
		print('Iter\tRes')
		while DeltaNorm>tol:
			counter=counter+1

			# update 
			X=X+STTS

			# incremental
			Spow=np.dot(Spow,S)
			STTS=np.dot(np.dot(Spow.T,T),Spow)
			DeltaNorm=np.max(np.abs(STTS))

			print('%.4d\t%.3e'%(counter,DeltaNorm))


	print('Error %.2e achieved after %.4d iteration!'%(DeltaNorm,counter))

	return X



def res_discrete_lyap(A,Q,Z,Factorised=True):
	'''
	Provides residual of discrete Lyapunov equation:
		A.T X A - X = -Q Q.T
	If Factorised option is true, 
		X=Z*Z.T
	otherwise X=Z is chosen. 
	'''

	if Factorised: 
		X=np.dot(Z,Z.T)
	else:
		X=Z
	R=np.dot(A.T,np.dot(X,A)) - X + np.dot(Q,Q.T)
	resinf=np.max(np.abs(R))

	return resinf



def low_rank_smith(A,Q,tol=1e-10,Square=True,tolSVD=1e-12,tolAbs=False,
									   kmax=None,fullOut=True,Convergence='Zk'):
	'''
	Low-rank smith algorithm for Stein equation 
		A.T X A - X = -Q Q.T
	The algorithm can only be used if T is symmetric positive-definite, but this
	is not checked in this routine for computational performance. The solution X
	is provided in its factorised form:
		X=Z Z.T
	As in the most general case,  a solution X exists only if the eigenvalues of 
	S are stricktly smaller than one, and the algorithm will not converge 
	otherwise. The algorithm can not exploits parsity, hence, while convergence 
	can be improved for very large matrices, it can not be employed if matrices 
	are too large to be stored in memory.

	Parameters:
	- tol: tolerance for stopping convergence of Smith algorithm
	- Square: if true the squared-Smith algorithm is used
	- tolSVD: tolerance for reduce Z matrix based on singular values
	- kmax: if given, the Z matrix is forced to have size kmax
	- tolAbs: if True, the tolerance 
	- fullOut: not implemented
	- Convergence: 'Zk','res'. 
		- If 'Zk' the iteration is stopped when the inf norm of the incremental 
		matrix goes below tol. 
		- If 'res' the residual of the Lyapunov equation is computed. This
		strategy may fail to converge if kmax is too low or tolSVD too large!

	Ref. P. Benner, G.E. Khoury and M. Sadkane, "On the squared Smith method for
	large-scale Stein equations", 2014.
	'''

	N=A.shape[0]
	ncol=Q.shape[1]
	AT=A.T

	DeltaNorm=1e6
	print('Iter\tMaxZhere')

	kk=0
	SvList=[]
	ZcColList=[]


	if Square: # ------------------------------------------------- squared iter
		Zk=Q
		while DeltaNorm>tol:

			### compute product Ak^2 * Zk
			###  use block Arnoldi 

			## too expensive!!
			# Zright=Zk
			# for ii in range(2**kk):
			# 	Zright=np.dot(AT,Zright)
			Zright=np.dot(AT,Zk)
			AT=np.dot(AT,AT)

			### enlarge Z matrix
			Zk=np.concatenate((Zk,Zright),axis=1)

			### check convergence
			if Convergence=='Zk':
				### check convergence without reconstructing the added term
				MaxZhere=np.max(np.abs(Zright))
				print('%.4d\t%.3e'%(kk,MaxZhere))
				DeltaNorm=MaxZhere
			elif Convergence=='res':
				### check convergence through residual
				resinf=res_discrete_lyap(A,Q,Zk,Factorised=True)
				print('%.4d\t%.3e\t%.3e'%(kk,MaxZhere,resinf))
				DeltaNorm=resinf

			# cheap SVD truncation
			U,sv,Vh=scalg.svd(Zk,full_matrices=False)
			#embed()

			if kmax==None:
				if tolAbs:
					pmax=np.sum(sv>tolSVD)
				else:
					pmax=np.sum(sv>tolSVD*sv[0])
			else:
				pmax=kmax

			Ut=U[:,:pmax]
			svt=sv[:pmax]
			#Vht=Vh[:pmax,:]
			#Zkrec=np.dot(Ut,np.dot(np.diag(svt),Vht))
			Zk=np.dot(Ut,np.diag(svt))

			### update
			kk=kk+1


	else: # -------------------------------------------------------- smith iter
		raise NameError(
			   'Smith method without SVD will lead to extremely large matrices')

		Zk=[]
		Zk.append(Q)
		while DeltaNorm>tol:
			Zk.append(np.dot(AT,Zk[-1]))
			kk=kk+1
			# check convergence without reconstructing Z*Z.T
			MaxZhere=np.max(np.abs(Zk[-1]))
			print('%.4d\t%.3e'%(kk,MaxZhere))
			DeltaNorm=MaxZhere
		Zk=np.concatenate(tuple(Zk),axis=1)

	return Zk



def modred(SSb,N,method='residualisation'):
	'''
	Produces a reduced order model with N states from balanced or modal system 
	SSb.
	Both "truncation" and "residualisation" methods are employed.

	Note: 
	- this method is designed for small size systems, i.e. a deep copy of SSb is 
	produced by default.
	'''

	assert method in ['residualisation','realisation','truncation'],\
					"method must be equal to 'residualisation' or 'truncation'!"
	assert SSb.dt is not None, 'SSb is not a DLTI!'

	Nb=SSb.A.shape[0]
	if Nb==N:
		SSrom=scsig.dlti(SSb.A,SSb.B,SSb.C,SSb.D,dt=SSb.dt)
		return SSrom
			
	A11=SSb.A[:N,:N]
	B11=SSb.B[:N,:]
	C11=SSb.C[:,:N]
	D=SSb.D

	if method is 'truncation':
		SSrom=scsig.dlti(A11,B11,C11,D,dt=SSb.dt)
	else:
		Nb=SSb.A.shape[0]
		IA22inv=-SSb.A[N:,N:].copy()
		eevec=range(Nb-N)
		IA22inv[eevec,eevec]+=1.
		IA22inv=scalg.inv(IA22inv,overwrite_a=True)

		SSrom=scsig.dlti(
			A11+np.dot(SSb.A[:N,N:],np.dot(IA22inv,SSb.A[N:,:N])),
			B11+np.dot(SSb.A[:N,N:],np.dot(IA22inv,SSb.B[N:,:] )),
			C11+np.dot(SSb.C[:,N:] ,np.dot(IA22inv,SSb.A[N:,:N])),
			D  +np.dot(SSb.C[:,N:] ,np.dot(IA22inv,SSb.B[N:,:] )),
			dt=SSb.dt)

	return SSrom



def tune_rom(SSb,kv,tol,gv,method='realisation',convergence='all'):
	'''
	Starting from a balanced DLTI, this function determines the number of states
	N required in a ROM (obtained either through 'residualisation' or 
	'truncation' as specified in method - see also librom.modred) to match the 
	frequency response of SSb over the frequency array, kv, with absolute 
	accuracy tol. gv contains the balanced system Hankel singular value, and is 
	used to determine the upper bound for the ROM order N.

	Unless kv does not conver the full Nyquist frequency range, the ROM accuracy 
	is not guaranteed to increase monothonically with the number of states. To
	account for this, two criteria can be used to determine the ROM convergence:

		- convergence='all': in this case, the number of ROM states N is chosen
		such that any ROM of order greater than N produces an error smaller than 
		tol. To guarantee this the ROM frequency response is computed for all 
		N<=Nb, where Nb is the number of balanced states. This method is 
		numerically inefficient.

		- convergence='min': atempts to find the minimal number of states to 
		achieve the accuracy tol.

	Note:
	- the input state-space model, SSb, must be balanced.
	- the routine in not implemented for numerical efficiency and assumes that 
	SSb is small.
	'''

	# reference frequency response
	Nb=SSb.A.shape[0]
	Yb=libss.freqresp(SSb,kv,dlti=True)

	Nmax=min(np.sum(gv>tol)+1,Nb)



	if convergence=='all':
		# start from larger size and decrease untill the ROm accuracy is over tol
		Found=False
		N=Nmax
		while not Found:
			SSrom=modred(SSb,N,method)
			Yrom=libss.freqresp(SSrom,kv,dlti=True)
			er=np.max(np.abs(Yrom-Yb))
			print('N=%.3d, er:%.2e (tol=%.2e)' %(N,er,tol) )

			if N==Nmax and er>tol:
				warnings.warn(
					'librom.tune_rom: error %.2e above tolerance %.2e and HSV bound %.2e'\
																%(er,tol,gv[N-1]) )
				# raise NameError('Hankel singluar values do not '\
				# 				'provide a bound for error! '\
				# 				'The balanced system may not be accurate')
			if er<tol:
				N-=1 
			else:
				N+=1
				Found=True
				SSrom=modred(SSb,N,method)

	elif convergence=='min':
		Found=False
		N=1
		while not Found:
			SSrom=modred(SSb,N,method)
			Yrom=libss.freqresp(SSrom,kv,dlti=True)
			er=np.max(np.abs(Yrom-Yb))
			print('N=%.3d, er:%.2e (tol=%.2e)' %(N,er,tol) )
			if er<tol:
				Found=True

			else:
				N+=1

	else:
		raise NameError("'convergence' method not implemented")

	return SSrom



def eigen_dec(A,B,C,dlti=True,N=None,eigs=None,UR=None,URinv=None,
									   order_by='damp',tol=1e-10,complex=False):
	'''
	Eigen decomposition of state-space model (either discrete or continuous time) 
	defined by the A,B,C matrices. Eigen-states are organised in decreasing 
	damping order or increased frequency order such that the truncation
		A[:N,:N], B[:N,:], C[:,:N]
	will retain the least N damped (or lower frequency) modes.
	
	If the eigenvalues of A, eigs, are complex, the state-space is automatically
	convert into real by separating its real and imaginary part. This procedure
	retains the minimal number of states as only 2 equations are added for each
	pair of complex conj eigenvalues. Extra care is however required when 
	truncating the system, so as to ensure that the chosen value of N does not
	retain the real part, but not the imaginary part, of a complex pair. 

	For this reason, the function also returns an optional output, Nlist, such
	that, for each N in Nlist, the truncation
		A[:N,:N], B[:N,:], C[:,:N]
	does guarantee that both the real and imaginary part of a complex conj pair
	is included in the truncated model. Note that if order_by == None, the eigs 
	and UR must be given in input and must be such that complex pairs are stored
	consecutivelly. 


	Input:
	A,B,C: matrices of state-space model
	dlti: specifies whether discrete (True) or continuous-time. This information
		is only required to order the eigenvalues in decreasing dmaping order
	N: number of states to retain. If None, all states are retained
	eigs,Ur: eigenvalues and right eigenvector of A matrix as given by:
		eigs,Ur=scipy.linalg.eig(A,b=None,left=False,right=True)
	Urinv: inverse of Ur
	order_by={'damp','freq'}: order according to increasing damping or 
		decreasing frequency. If None, the same order as eigs/UR is followed. 
	tol: absolute tolerance used to identify complex conj pair of eigenvalues
	complex: if true, the system is left in complex form


	Output:
	(Aproj,Bproj,Cproj): state-space matrices projected over the first N (or N+1 
		if N removes the imaginary part equations of a complex conj pair of 
		eigenvalues) related to the least damped modes
	Nlist: list of acceptable truncation values
	'''

	if N==None: 
		N=A.shape[0]

	if order_by is None: 
		assert ((eigs is not None) and (UR is not None)),\
			'Specify criterion to order eigenvalues or provide both eigs and UR'

	### compute eigevalues/eigenvectors
	if eigs is None:
		eigs,UR=scalg.eig(A, b=None, left=False, right=True)
	if URinv is None:
		try:
			URinv=np.linalg.inv(UR)
		except LinAlgError:
			print('The A matrix can not be diagonalised as does not admit '\
											'linearly independent eigenvectors')


	### order eigenvalues/eigenvectors (or verify format)
	if order_by is None:
		# verify format
		nn=0
		while nn<N:
			if np.abs(eigs[nn].imag)>tol:
				if nn<N-1:
					assert np.abs(eigs[nn].imag+eigs[nn+1].imag)<tol,\
						'When order_by is None, eigs and UR much be organised such '\
						'that complex conj pairs are consecutives'
				else:
					assert np.abs(eigs[nn].imag+eigs[nn-1].imag)<tol,\
						'When order_by is None, eigs and UR much be organised such '\
						'that complex conj pairs are consecutives'				
				nn+=2
			else:
				nn+=1
	else:
		if order_by=='damp':
			if dlti:
				order=np.argsort(np.abs(eigs))[::-1]
			else:
				order=np.argsort(eigs.real)[::-1]
		elif order_by=='freq':
			if dlti:
				order=np.argsort(np.abs(np.angle(eigs)))
			else:
				order=np.argsort(np.abs(eigs.imag))	
		else:
			raise NameError("order_by must be equal to 'damp' or 'freq'")
		eigs=eigs[order]
		UR=UR[:,order]
		URinv=URinv[order,:]


	### compute list of available truncation size, Nlist
	Nlist=[]
	nn=0
	while nn<N:
		# check if eig are complex conj
		if nn<N-1 and np.abs(eigs[nn]-eigs[nn+1].conjugate())<tol:
			nn+=2
		else:
			nn+=1
		Nlist.append(nn)
	assert Nlist[-1]>=N,\
			 'Something failed when identifying the admissible truncation sizes'
	if Nlist[-1]>N:
		warnings.warn(
			'Resizing the eigendecomposition from %.3d to %.3d states'\
																 %(N,Nlist[-1]))
		N=Nlist[-1]


	### build complex form
	if complex:
		Aproj=np.diag(eigs[:N])
		Bproj=np.dot(URinv[:N,:],B)
		Cproj=np.dot(C,UR[:,:N])
		return Aproj,Bproj,Cproj,Nlist


	### build real values form
	Aproj=np.zeros((N,N))
	Bproj=np.zeros((N,B.shape[1]))
	Cproj=np.zeros((C.shape[0],N))
	nn=0
	while nn<N:
		# redundant check
		if (nn+1 in Nlist) and np.abs(eigs[nn].imag)<tol:
			Aproj[nn,nn]=eigs[nn].real
			Bproj[nn,:]=np.dot(URinv[nn,:].real,B)
			Cproj[:,nn]=np.dot(C,UR[:,nn].real)			
			nn+=1
		else:
			Aproj[nn  ,nn  ]= eigs[nn].real
			Aproj[nn  ,nn+1]=-eigs[nn].imag
			Aproj[nn+1,nn  ]= eigs[nn].imag
			Aproj[nn+1,nn+1]= eigs[nn].real
			#
			Bproj[nn  ,:]=np.dot(URinv[nn,:].real,B)
			Bproj[nn+1,:]=np.dot(URinv[nn,:].imag,B)
			#
			Cproj[:,nn  ]= 2.*np.dot(C,UR[:,nn].real)
			Cproj[:,nn+1]=-2.*np.dot(C,UR[:,nn].imag)
			nn+=2


	return Aproj,Bproj,Cproj,Nlist














if __name__=='__main__':
	gv=np.array([5,4,3,2,1])




