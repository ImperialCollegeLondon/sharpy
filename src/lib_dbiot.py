'''
Calculate derivative of Wnc(zeta)*Uc w.r.t local panel coordinates
'''

import numpy as np 
from IPython import embed
import libalg

### constants
cfact_biot=0.25/np.pi
VORTEX_RADIUS=1e-2 # numerical radious of vortex


### looping through panels
#svec =[ 0, 1, 2, 3] # seg. no.
avec =[ 0, 1, 2, 3] # 1st vertex no.
bvec =[ 1, 2, 3, 0] # 2nd vertex no.
LoopPanel=[(0,1),(1,2),(2,3),(3,0)]



def eval_seg(ZetaP,ZetaA,ZetaB,gamma_seg=1.0):
	'''
	Derivative of induced velocity Q w.r.t. collocation and segment coordinates 
	in format:
		[ (ZetaP,ZetaA,ZetaB), (x,y,z) of Zeta,  (x,y,z) of Q]

	Warning: function optimised for performance. Variables are scaled during the
	execution.
	'''

	Der=np.zeros((3,3,3))

	RA=ZetaP-ZetaA
	RB=ZetaP-ZetaB
	RAB=ZetaB-ZetaA
	Vcr=libalg.cross3d(RA,RB)
	vcr2=np.dot(Vcr,Vcr)

	# numerical radious
	vortex_radious_here=VORTEX_RADIUS*libalg.norm3d(RAB)
	if vcr2<vortex_radious_here**2:
		return Der


	# scaling
	ra1,rb1=libalg.norm3d(RA),libalg.norm3d(RB)
	ra2,rb2=ra1**2,rb1**2
	rainv=1./ra1
	rbinv=1./rb1
	ra_dir,rb_dir=RA*rainv,RB*rbinv
	ra3inv,rb3inv=rainv**3,rbinv**3
	Vcr=Vcr/vcr2

	diff_vec=ra_dir-rb_dir
	vdot_prod=np.dot(diff_vec,RAB)
	T2=vdot_prod/vcr2

	# Extract components
	ra_x,ra_y,ra_z=RA
	rb_x,rb_y,rb_z=RB
	rab_x,rab_y,rab_z=RAB
	vcr_x,vcr_y,vcr_z=Vcr
	ra2_x,ra2_y,ra2_z=RA**2
	rb2_x,rb2_y,rb2_z=RB**2
	ra_vcr_x, ra_vcr_y, ra_vcr_z=2.*libalg.cross3d(RA,Vcr)
	rb_vcr_x, rb_vcr_y, rb_vcr_z=2.*libalg.cross3d(RB,Vcr)
	vcr_sca_x,vcr_sca_y,vcr_sca_z=Vcr*ra3inv
	vcr_scb_x,vcr_scb_y,vcr_scb_z=Vcr*rb3inv


	# # ### derivatives indices:
	# # # the 1st is the component of the vaiable w.r.t derivative are taken.
	# # # the 2nd is the component of the output 
	dQ_dRA=np.array(
		 [[-vdot_prod*rb_vcr_x*vcr_x           + vcr_sca_x*(rab_x*(ra2 - ra2_x) - ra_x*ra_y*rab_y - ra_x*ra_z*rab_z),
		   -T2*rb_z - vdot_prod*rb_vcr_x*vcr_y + vcr_sca_y*(rab_x*(ra2 - ra2_x) - ra_x*ra_y*rab_y - ra_x*ra_z*rab_z),
		    T2*rb_y - vdot_prod*rb_vcr_x*vcr_z + vcr_sca_z*(rab_x*(ra2 - ra2_x) - ra_x*ra_y*rab_y - ra_x*ra_z*rab_z)],
		  [ T2*rb_z - vdot_prod*rb_vcr_y*vcr_x + vcr_sca_x*(rab_y*(ra2 - ra2_y) - ra_x*ra_y*rab_x - ra_y*ra_z*rab_z),
		   -vdot_prod*rb_vcr_y*vcr_y           + vcr_sca_y*(rab_y*(ra2 - ra2_y) - ra_x*ra_y*rab_x - ra_y*ra_z*rab_z),
		   -T2*rb_x - vdot_prod*rb_vcr_y*vcr_z + vcr_sca_z*(rab_y*(ra2 - ra2_y) - ra_x*ra_y*rab_x - ra_y*ra_z*rab_z)],
		  [-T2*rb_y - vdot_prod*rb_vcr_z*vcr_x + vcr_sca_x*(rab_z*(ra2 - ra2_z) - ra_x*ra_z*rab_x - ra_y*ra_z*rab_y),
		    T2*rb_x - vdot_prod*rb_vcr_z*vcr_y + vcr_sca_y*(rab_z*(ra2 - ra2_z) - ra_x*ra_z*rab_x - ra_y*ra_z*rab_y),
		   -vdot_prod*rb_vcr_z*vcr_z           + vcr_sca_z*(rab_z*(ra2 - ra2_z) - ra_x*ra_z*rab_x - ra_y*ra_z*rab_y)]])

	dQ_dRB=np.array( 
		 [[ vdot_prod*ra_vcr_x*vcr_x           + vcr_scb_x*(rab_x*(-rb2 + rb2_x) + rab_y*rb_x*rb_y + rab_z*rb_x*rb_z),
		    T2*ra_z + vdot_prod*ra_vcr_x*vcr_y + vcr_scb_y*(rab_x*(-rb2 + rb2_x) + rab_y*rb_x*rb_y + rab_z*rb_x*rb_z),
		   -T2*ra_y + vdot_prod*ra_vcr_x*vcr_z + vcr_scb_z*(rab_x*(-rb2 + rb2_x) + rab_y*rb_x*rb_y + rab_z*rb_x*rb_z)],
		  [-T2*ra_z + vdot_prod*ra_vcr_y*vcr_x + vcr_scb_x*(rab_x*rb_x*rb_y + rab_y*(-rb2 + rb2_y) + rab_z*rb_y*rb_z),
		    vdot_prod*ra_vcr_y*vcr_y           + vcr_scb_y*(rab_x*rb_x*rb_y + rab_y*(-rb2 + rb2_y) + rab_z*rb_y*rb_z),
		    T2*ra_x + vdot_prod*ra_vcr_y*vcr_z + vcr_scb_z*(rab_x*rb_x*rb_y + rab_y*(-rb2 + rb2_y) + rab_z*rb_y*rb_z)],
		  [ T2*ra_y + vdot_prod*ra_vcr_z*vcr_x + vcr_scb_x*(rab_x*rb_x*rb_z + rab_y*rb_y*rb_z + rab_z*(-rb2 + rb2_z)),
		   -T2*ra_x + vdot_prod*ra_vcr_z*vcr_y + vcr_scb_y*(rab_x*rb_x*rb_z + rab_y*rb_y*rb_z + rab_z*(-rb2 + rb2_z)),
		    vdot_prod*ra_vcr_z*vcr_z           + vcr_scb_z*(rab_x*rb_x*rb_z + rab_y*rb_y*rb_z + rab_z*(-rb2 + rb2_z))]])

	dQ_dRAB=np.array(
			  [[ vcr_x*diff_vec[0],
			     vcr_y*diff_vec[0],
			     vcr_z*diff_vec[0]],
					   [ vcr_x*diff_vec[1],
					     vcr_y*diff_vec[1],
					     vcr_z*diff_vec[1]],
							   [ vcr_x*diff_vec[2],
							     vcr_y*diff_vec[2],
							     vcr_z*diff_vec[2]]])

	Der[0,:,:]+=  dQ_dRA  + dQ_dRB # w.r.t. P
	Der[1,:,:]+= -dQ_dRAB - dQ_dRA # w.r.t. A	
	Der[2,:,:]+=  dQ_dRAB - dQ_dRB # w.r.t. B

	return (cfact_biot*gamma_seg)*Der



def eval_panel(zetaP,ZetaPanel,gamma_pan=1.0):
	'''
	Computes derivatives of induced velocity w.r.t. coordinates of target point,
	zetaP, and panel coordinates. Returns two elements:
		- DerP: derivative of induced velocity w.r.t. ZetaP, with:
			DerP.shape=(3,3) : DerC[ ZetaC_{x,y,z}, Uind_{x,y,z} ]
		- DerVertices: derivative of induced velocity w.r.t. panel vertices, with:
			DerVertices.shape=(4,3,3) : 
				DerVertices[ vertex number {0,1,2,3}, vertex_{x,y,z}, Uind_{x,y,z}]
	'''

	DerP=np.zeros((3,3))
	DerVertices=np.zeros((4,3,3))

	for aa,bb in LoopPanel: 
		DerSeg=eval_seg(zetaP,ZetaPanel[aa,:],ZetaPanel[bb,:],gamma_pan)
		# sum contribution
		DerP+=DerSeg[0,:,:]
		DerVertices[aa,:,:]+=DerSeg[1,:,:]
		DerVertices[bb,:,:]+=DerSeg[2,:,:]

	return DerP,DerVertices




if __name__=='__main__':
	pass	