'''
@author: salvatore maraniello
@date: 15 Jan 2018
@brief: Fitting tools
'''

import numpy as np
import scipy.optimize as scopt


def fpoly(kv,B0,B1,B2,dyv,ddyv):
    return B0+B1*dyv+B2*ddyv


def fpade(kv,Cv,B0,B1a,B1b,B2,dyv,ddyv):
    # evaluate transfer function from unsteady function Cv
    return B0*Cv+(B1a*Cv+B1b)*dyv+B2*ddyv
  

def getC(kv,Yv,B0,B1a,B1b,B2,dyv,ddyv):
    # evaluate unsteady function C required to perfectly match Yv
    Den=B0+B1a*dyv
    #kkvec=np.abs(Den)<1e-8
    C=(Yv-B1b*dyv-B2*ddyv)/Den
    #C[kkvec]=1.0
    #if np.sum(kkvec)>0: embed()
    return C


def rfa(cnum,cden,kv,ds=None):
	'''
	Evaluates over the frequency range kv.the rational function approximation:
	[cnum[-1] + cnum[-2] z + ... + cnum[0] z**Nnum ]/...
								[cden[-1] + cden[-2] z + ... + cden[0] z**Nden]
	where the numerator and denominator polynomial orders, Nnum and Nden, are 
	the length of the cnum and cden arrays and:
		- z=exp(1.j*kv*ds), with ds sampling time if ds is given (discrete-time 
		system)
		- z=1.*kv, if ds is None (continuous time system)
	'''

	if ds==None: 
		# continuous-time LTI system
		zv=1.j*kv
	else: 
		# discrete-time LTI system
		zv=np.exp(1.j*kv*ds)

	return np.polyval(cnum,zv)/np.polyval(cden,zv)


def rfader(cnum,cden,kv,m=1,ds=None):
	'''
	Evaluates over the frequency range kv.the derivative of order m of the 
	rational function approximation:
	[cnum[-1] + cnum[-2] z + ... + cnum[0] z**Nnum ]/...
								[cden[-1] + cden[-2] z + ... + cden[0] z**Nden]
	where the numerator and denominator polynomial orders, Nnum and Nden, are 
	the length of the cnum and cden arrays and:
		- z=exp(1.j*kv*ds), with ds sampling time if ds is given (discrete-time 
		system)
		- z=1.*kv, if ds is None (continuous time system)
	'''

	if ds==None: 
		# continuous-time LTI system
		zv=1.j*kv
		dzv=1.j
		raise NameError('Never tested for continuous systems!')
	else: 
		# discrete-time LTI system
		zv=np.exp(1.j*kv*ds)
		dzv=1.j*ds*zv


	Nv=np.polyval(cnum,zv)
	Dv=np.polyval(cden,zv)
	dNv=np.polyval(np.polyder(cnum),zv)
	dDv=np.polyval(np.polyder(cden),zv)

	return dzv*(dNv*Dv-Nv*dDv)/Dv**2



def fitfrd(kv,yv,N,dt=None,mag=0,eng=None):
	'''
	Wrapper for fitfrd (mag=0) and fitfrdmag (mag=1) functions in continuous and 
	discrete time (if ds in input).
	Input:
	   kv,yv: frequency array and frequency response
	   N: order for rational function approximation
	   mag=1,0: Flag for determining method to use
	   dt (optional): sampling time for DLTI systems
	'''

	raise NameError('Please use fitfrd function in matwrapper module!')

	return None



def get_rfa_res(xv,kv,Yv,Nnum,Nden,ds=None):
	'''
	Returns magnitude of the residual Yfit-Yv of a RFA approximation at each
	point kv. The coefficients of the approximations are:
	- cnum=xv[:Nnum]
	- cdem=xv[Nnum:]
	where cnum and cden are as per the 'rfa' function. 
	'''

	assert Nnum+Nden==len(xv), 'Nnum+Nden must be equal to len(xv)!'
	cnum=xv[:Nnum]
	cden=xv[Nnum:]

	Yfit=rfa(cnum,cden,kv,ds)

	return np.abs(Yfit-Yv)	



def get_rfa_res_norm(xv,kv,Yv,Nnum,Nden,ds=None,method='mix'):
	'''
	Define residual scalar norm of Pade approximation of coefficients 
	cnum=xv[:Nnum] and cden[Nnum:] (see get_rfa_res and rfa function) and 
	time-step ds (if discrete time).
	'''

	ErvAbs=get_rfa_res(xv,kv,Yv,Nnum,Nden,ds)

	if method=='H2':
		res=np.sum( ErvAbs**2 )
	elif method=='Hinf':
		res=np.max(ErvAbs)
	elif method=='mix':
		res=np.sum(ErvAbs**2)+np.max(ErvAbs)

	return res



def rfa_fit_dev(kv,Yv,Nnum,Nden,ds=None,Ntrial=6,Cfbound=1e2,method='mix',
	                                                 OutFull=False,Print=False):
	'''
	Find best fitting RFA approximation from frequency response Yv over the 
	frequency range kv for both continuous (ds=None) and discrete (ds>0) LTI 
	systems.
	
	Other input:
	- Nnum,Ndem: number of coefficients for Pade approximation.
	- ds: sampling time for DLTI systems
	- Ntrial: number of repetition of global and least square optimisations
	- Cfbouds: maximum absolute values of coeffieicnts (only for evolutionary
	algorithm)
	- method: metric to compute error of RFA approximation

	Output:
	- cnopt: optimal coefficients (numerator)
	- cdopt: optimal coefficients (denominator)

	Important:
	- this function has the same objective as fitfrd in matwrapper module. While
	generally slower, the global optimisation approach allows to verify the
	results from fitfrd.
	'''

	Nx=Nnum+Nden
	cost_best=1e32
	XvOptDev,XvOptLsq=[],[]
	Cdev,Clsq=[],[]

	Tol=1e-14
	for tt in range(Ntrial):

		# Evolutionary algorithm
		res=scopt.differential_evolution(#popsize=100,
			strategy='best1bin',func=get_rfa_res_norm,
				args=(kv,Yv,Nnum,Nden,ds,method),bounds=Nx*((-Cfbound,Cfbound),))
		xvdev=res.x
		cost_dev=get_rfa_res_norm(xvdev,kv,Yv,Nnum,Nden,ds,'mix')
		XvOptDev.append(xvdev)
		Cdev.append(cost_dev)
		if cost_dev<cost_best:
			cost_best=cost_dev				
			xvopt=xvdev

		# Least squares fitting - unbounded
		#  method only local, but do not move to the end of global search: best 
		# results can be found even when starting from a "worse" solution
		xvlsq=scopt.leastsq(get_rfa_res,x0=xvdev, args=(kv,Yv,Nnum,Nden,ds))[0]
		cost_lsq=get_rfa_res_norm(xvlsq,kv,Yv,Nnum,Nden,ds,method)
		XvOptLsq.append(xvlsq)
		Clsq.append(cost_lsq)
		if cost_lsq<cost_best:
			cost_best=cost_lsq				
			xvopt=xvlsq

		if Print:
			print('Trial %.2d: cost dev: %.3e, cost lsq: %.3e'\
				                                      %(tt+1,cost_dev,cost_lsq))
		if cost_best<Tol:
			print('\tSearch terminated!')
			Ntrial=tt+1
			break

	# rescale coefficients
	cnopt=xvopt[:Nnum]
	cdopt=xvopt[Nnum:]
	if np.abs(cdopt[-1])>1e-2: cdscale=cdopt[-1]
	else: cdscale=1.0
	cnopt=cnopt/cdscale
	cdopt=cdopt/cdscale

	# determine outputs
	Outputs=(cnopt,cdopt)
	if OutFull:
		for tt in range(Ntrial):
			if np.abs(XvOptDev[tt][-1])>1e-2:
				XvOptDev[tt]=XvOptDev[tt]/XvOptDev[tt][-1]
			if np.abs(XvOptLsq[tt][-1])>1e-2:
				XvOptLsq[tt]=XvOptLsq[tt]/XvOptLsq[tt][-1]	
		Outputs=Outputs+( XvOptDev,XvOptLsq,Cdev,Clsq)

	return Outputs



def poly_fit(kv,Yv,dyv,ddyv,method='leastsq',Bup=None):

	'''
	Find best II order fitting polynomial from frequency response Yv over the 
	frequency range kv for both continuous (ds=None) and discrete (ds>0) LTI 
	systems.

	Input:
	- kv: frequency points
	- Yv: frequency response
	- dyv,ddyv: frequency responses of I and II order derivatives
	- method='leastsq','dev': algorithm for minimisation
	- Bup (only 'dev' method): bounds for bv coefficients as per 
	scipy.optimize.differential_evolution. This is a length 3 array.

	Important:
	- this function attributes equal weight to each data-point!
	'''



	if method=='leastsq':
		# pointwise residual
		def funRes(bv,kv,Yv,dyv,ddyv):
		    B0,B1,B2=bv
		    rv=fpoly(kv,B0,B1,B2,dyv,ddyv)-Yv
		    return np.concatenate((rv.real,rv.imag))
		# solve
		bvopt,cost=scopt.leastsq(funRes,x0=[0.,0.,0.],args=(kv,Yv,dyv,ddyv))


	elif method=='dev':
		# use genetic algorithm with objective a sum of H2 and Hinf norms of 
		# residual
		def funRes(bv,kv,Yv,dyv,ddyv):

			B0,B1,B2=bv
			rv=fpoly(kv,B0,B1,B2,dyv,ddyv)-Yv

			Nk=len(kv)
			rvsq=rv*rv.conj()
			#H2norm=np.sqrt(np.trapz(rvsq/(Nk-1.)))
			#return H2norm+np.linalg.norm(rv,np.inf)
			return np.sum(rvsq)

		# prepare bounds
		if Bup is None:
			Bounds=3*((-Bup,Bup),)
		else:
			assert len(Bup)==3, 'Bup must be a length 3 list/array'
			Bounds=( (-Bup[0],Bup[0]),(-Bup[1],Bup[1]),(-Bup[2],Bup[2]), )

		res=scopt.differential_evolution(
			func=funRes,args=(kv,Yv,dyv,ddyv),strategy='best1bin',bounds=Bounds)
		bvopt=res.x                 
		cost=funRes(bvopt,kv,Yv,dyv,ddyv)


	return bvopt,cost







if __name__=='__main__':

	import libss
	import scipy.signal as scsig
	import matplotlib.pyplot as plt


	# build state-space
	cfnum=np.array([4, 1.25, 1.5])
	cfden=np.array([2, .5, 1])
	A,B,C,D=scsig.tf2ss(cfnum,cfden)


	# -------------------------------------------------------------- Test cases
	# Comment/Uncomment as appropriate

	### Test01: 2nd order DLTI system
	ds=2./40.
	fs=1./ds 
	fn=fs/2.
	kn=2.*np.pi*fn
	kv=np.linspace(0,kn,301)
	SS=libss.ss(A,B,C,D,dt=ds)
	Cv=libss.freqresp(SS,kv)
	Cv=Cv[0,0,:]

	# ### Test02: 2nd order continuous-time LTI system
	# ds=None
	# kv=np.linspace(0,40,301)
	# kvout,Cv=scsig.freqresp((A,B,C,D),kv)

	# -------------------------------------------------------------------------


	# Find fitting
	Nnum,Nden=3,3
	cnopt,cdopt=rfa_fit_dev(kv,Cv,Nnum,Nden,ds=ds,Ntrial=6,Cfbound=1e2,
										 method='mix', OutFull=False,Print=True)

	print('Error coefficients (DLTI):')
	print('Numerator:   '+3*'%.2e  ' %tuple(np.abs(cnopt-cfnum)))
	print('Denominator: '+3*'%.2e  ' %tuple(np.abs(cnopt-cfnum)))

	# Visualise
	Cfit=rfa(cnopt,cdopt,kv,ds)

	fig=plt.figure('Transfer function',(10,4))
	ax1=fig.add_subplot(111)
	ax1.plot(kv,Cv.real,color='r',lw=2,ls='-',label=r'ref - real')
	ax1.plot(kv,Cv.imag,color='r',lw=2,ls='--',label=r'ref - imag')
	ax1.plot(kv,Cfit.real,color='k',lw=1,ls='-',label=r'RFA - real')
	ax1.plot(kv,Cfit.imag,color='k',lw=1,ls='--',label=r'RFA - imag')
	ax1.legend(ncol=1,frameon=True,columnspacing=.5,labelspacing=.4)
	ax1.grid(color='0.85', linestyle='-')
	plt.show()


