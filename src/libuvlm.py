'''
Methods for UVLM solution.

S. Maraniello, 1 Jun 2018
'''

import numpy as np
import libalg

cfact_biot=0.25/np.pi
VORTEX_RADIUS=1e-2 # numerical radious of vortex

# local mapping segment/vertices of a panel
svec=[0,1,2,3] # seg. number
avec=[0,1,2,3] # 1st vertex of seg.
bvec=[1,2,3,0] # 2nd vertex of seg.




def joukovski_qs_segment(zetaA,zetaB,v_mid,gamma=1.0,fact=0.5):
	'''
	Joukovski force over vetices A and B produced by the segment A->B.
	The factor fact allows to compute directly the contribution over the
	vertices A and B (e.g. 0.5) or include DENSITY.
	'''

	rab=zetaB-zetaA
	fs=libalg.cross3d(v_mid,rab)
	gfact=fact*gamma

	return gfact*fs


def biot_segment(zetaP,zetaA,zetaB,gamma=1.0):
	'''
	Induced velocity of segment A_>B of circulation gamma over point P.
	'''

	# differences
	ra=zetaP-zetaA
	rb=zetaP-zetaB
	rab=zetaB-zetaA
	ra_norm,rb_norm=libalg.norm3d(ra),libalg.norm3d(rb)
	vcross=libalg.cross3d(ra,rb)
	vcross_sq=np.dot(vcross,vcross)

	# numerical radious
	vortex_radious_here=VORTEX_RADIUS*libalg.norm3d(rab)
	if vcross_sq<vortex_radious_here**2:
		return np.zeros((3,))
	#assert vcross_sq>vortex_radious_here**2, 'P along AB line'

	# ra_unit=ra/ra_norm
	# rb_unit=rb/rb_norm
	q=((cfact_biot*gamma/vcross_sq)*\
		( np.dot(rab,ra)/ra_norm - np.dot(rab,rb)/rb_norm)) * vcross

	return q


def biot_panel(zetaC,ZetaPanel,gamma=1.0):
	'''
	Induced velocity over point ZetaC of a panel of vertices coordinates 
	ZetaPanel and circulaiton gamma, where:
		ZetaPanel.shape=(4,3)=[vertex local number, (x,y,z) component]
	'''

	q=np.zeros((3,))
	for ss,aa,bb in zip(svec,avec,bvec):
		q+=biot_segment(zetaC,ZetaPanel[aa,:],ZetaPanel[bb,:],gamma)

	return q


def panel_normal(ZetaPanel):
	'''
	return normal of panel with vertiex coordinates ZetaPanel, where:
		ZetaPanel.shape=(4,3)		
	'''
	
	# build cross-vectors
	r02=ZetaPanel[2,:]-ZetaPanel[0,:]
	r13=ZetaPanel[3,:]-ZetaPanel[1,:]

	nvec=libalg.cross3d(r02,r13)
	nvec=nvec/libalg.norm3d(nvec)

	return nvec


def panel_area(ZetaPanel):
	'''
	return area of panel with vertices coordinates ZetaPanel, where:
		ZetaPanel.shape=(4,3)
	using Bretschneider formula - for cyclic or non-cyclic quadrilaters.
	'''

	# build cross-vectors
	r02=ZetaPanel[2,:]-ZetaPanel[0,:]
	r13=ZetaPanel[3,:]-ZetaPanel[1,:]
	# build side vectors
	r01=ZetaPanel[1,:]-ZetaPanel[0,:]
	r12=ZetaPanel[2,:]-ZetaPanel[1,:]
	r23=ZetaPanel[3,:]-ZetaPanel[2,:]
	r30=ZetaPanel[0,:]-ZetaPanel[3,:]

	# compute distances
	d02=libalg.norm3d(r02)
	d13=libalg.norm3d(r13)
	d01=libalg.norm3d(r01)
	d12=libalg.norm3d(r12)
	d23=libalg.norm3d(r23)
	d30=libalg.norm3d(r30)

	A=0.25*np.sqrt(  (4.*d02**2*d13**2) - ((d12**2+d30**2)-(d01**2+d23**2))**2 )

	return A



if __name__=='__main__':

	zetaP=np.array([2,-1.5,1])
	zetaA=np.array([0,-1,5])
	zetaB=np.array([1,-2,2])
	q=biot_segment(zetaP,zetaA,zetaB,3.)
	#qcheck=biot_seg_check(zetaP,zetaA,zetaB,3.)

	##### verify biot-savart:
	rv_center=np.array([1,2,3])
	R=0.4
	Theta_rad=2.*np.pi 
	gamma=5.
	# expected
	Uabs_exp=gamma*Theta_rad/(4.*np.pi)/R
	Uexp=np.array([0,0,Uabs_exp])
	# numerical
	N=101
	thvec=np.linspace(0,Theta_rad,N)
	xv=rv_center[0]+np.cos(thvec)*R
	yv=rv_center[1]+np.sin(thvec)*R
	zv=rv_center[2]+0.0*xv 
	U=np.zeros((3,))
	for ii in range(N-1):
		ra=np.array([xv[ii],yv[ii],zv[ii]])
		rb=np.array([xv[ii+1],yv[ii+1],zv[ii+1]])
		U+=biot_segment( rv_center, ra, rb, gamma  )
	assert libalg.norm3d(U-Uexp)**2 < 1e-3*np.abs(Uabs_exp), 'Wrong velocity'


	pass

	import itertools
	M,N=4,6





