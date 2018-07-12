'''
Calculate derivatives of induced velocity.

Methods:

- eval_seg_exp and eval_seg_exp_loop: profide ders in format 
	[Q_{x,y,z},ZetaPoint_{x,y,z}]
  and use fully-expanded analytical formula.
- eval_panel_exp: iterates through whole panel

- eval_seg_comp and eval_seg_comp_loop: profide ders in format 
	[Q_{x,y,z},ZetaPoint_{x,y,z}]
  and use compact analytical formula.


'''

import numpy as np 
from IPython import embed
import libalg

### constants
cfact_biot=0.25/np.pi
VORTEX_RADIUS=1e-2 # numerical radious of vortex


### looping through panels
#svec =[ 0, 1, 2, 3] # seg. no.
# avec =[ 0, 1, 2, 3] # 1st vertex no.
# bvec =[ 1, 2, 3, 0] # 2nd vertex no.
LoopPanel=[(0,1),(1,2),(2,3),(3,0)]



def eval_seg_exp(ZetaP,ZetaA,ZetaB,gamma_seg=1.0):
	'''
	Derivative of induced velocity Q w.r.t. collocation and segment coordinates 
	in format:
		[ (ZetaP,ZetaA,ZetaB), (x,y,z) of Zeta,  (x,y,z) of Q]

	Warning: function optimised for performance. Variables are scaled during the
	execution.
	'''

	DerP=np.zeros((3,3))
	DerA=np.zeros((3,3))
	DerB=np.zeros((3,3))
	eval_seg_exp_loop(DerP,DerA,DerB,ZetaP,ZetaA,ZetaB,gamma_seg)
	return DerP,DerA,DerB


def eval_seg_exp_loop(DerP,DerA,DerB,ZetaP,ZetaA,ZetaB,gamma_seg):
	'''
	Derivative of induced velocity Q w.r.t. collocation (DerC) and segment
	coordinates in format.
	
	To optimise performance, the function requires the derivative terms to be
	pre-allocated and passed as input.

	Each Der* term returns derivatives in the format

		[ (x,y,z) of Zeta,  (x,y,z) of Q]

	Warning: to optimise performance, variables are scaled during the execution.
	'''

	RA=ZetaP-ZetaA
	RB=ZetaP-ZetaB
	RAB=ZetaB-ZetaA
	Vcr=libalg.cross3d(RA,RB)
	vcr2=np.dot(Vcr,Vcr)

	# numerical radious
	vortex_radious_here=VORTEX_RADIUS*libalg.norm3d(RAB)
	if vcr2<vortex_radious_here**2:
		return


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

	DerP += (cfact_biot*gamma_seg)* ( dQ_dRA  + dQ_dRB).T # w.r.t. P
	DerA += (cfact_biot*gamma_seg)* (-dQ_dRAB - dQ_dRA).T # w.r.t. A	
	DerB += (cfact_biot*gamma_seg)* ( dQ_dRAB - dQ_dRB).T # w.r.t. B



def eval_panel_exp(zetaP,ZetaPanel,gamma_pan=1.0):
	'''
	Computes derivatives of induced velocity w.r.t. coordinates of target point,
	zetaP, and panel coordinates. Returns two elements:
		- DerP: derivative of induced velocity w.r.t. ZetaP, with:
			DerP.shape=(3,3) : DerC[ Uind_{x,y,z}, ZetaC_{x,y,z} ]
		- DerVertices: derivative of induced velocity wrt panel vertices, with:
			DerVertices.shape=(4,3,3) :
			DerVertices[ vertex number {0,1,2,3}, Uind_{x,y,z}, ZetaC_{x,y,z} ]
	'''

	DerP=np.zeros((3,3))
	DerVertices=np.zeros((4,3,3))

	for aa,bb in LoopPanel: 
		eval_seg_exp_loop(DerP,DerVertices[aa,:,:],DerVertices[bb,:,:],
								zetaP,ZetaPanel[aa,:],ZetaPanel[bb,:],gamma_pan)

	return DerP,DerVertices



# ------------------------------------------------------------------------------
#	Compact Formula
# ------------------------------------------------------------------------------

def Dvcross_by_skew3d(Dvcross,rv):
	'''
	Fast matrix multiplication of der(vcross)*skew(rv), where 
		vcross = (rv x sv)/|rv x sv|^2
	The function exploits the property that the output matrix is symmetric.
	'''
	P=np.empty((3,3))
	P[0,0]=Dvcross[0,1]*rv[2]-Dvcross[0,2]*rv[1]
	P[0,1]=Dvcross[0,2]*rv[0]-Dvcross[0,0]*rv[2]
	P[0,2]=Dvcross[0,0]*rv[1]-Dvcross[0,1]*rv[0]
	#
	P[1,0]=P[0,1]
	P[1,1]=Dvcross[1,2]*rv[0]-Dvcross[0,1]*rv[2]
	P[1,2]=Dvcross[0,1]*rv[1]-Dvcross[1,1]*rv[0]
	#
	P[2,0]=P[0,2]
	P[2,1]=P[1,2]
	P[2,2]=Dvcross[0,2]*rv[1]-Dvcross[1,2]*rv[0]
	return P

def der_runit(r,rinv,minus_rinv3):

	# alloc upper diag
	Der=np.empty((3,3))
	Der[0,0]=rinv+minus_rinv3*r[0]**2
	Der[0,1]=     minus_rinv3*r[0]*r[1]
	Der[0,2]=     minus_rinv3*r[0]*r[2]
	Der[1,1]=rinv+minus_rinv3*r[1]**2
	Der[1,2]=     minus_rinv3*r[1]*r[2]
	Der[2,2]=rinv+minus_rinv3*r[2]**2                                
	# alloc lower
	Der[1,0]=Der[0,1]
	Der[2,0]=Der[0,2]
	Der[2,1]=Der[1,2]
	return Der

def eval_seg_comp(ZetaP,ZetaA,ZetaB,gamma_seg=1.0):
	DerP=np.zeros((3,3))
	DerA=np.zeros((3,3))
	DerB=np.zeros((3,3))
	eval_seg_comp_loop(DerP,DerA,DerB,ZetaP,ZetaA,ZetaB,gamma_seg)
	return DerP,DerA,DerB


def eval_seg_comp_loop(DerP,DerA,DerB,ZetaP,ZetaA,ZetaB,gamma_seg):
	'''
	Derivative of induced velocity Q w.r.t. collocation and segment coordinates 
	in format:
		[ (x,y,z) of Q, (x,y,z) of Zeta ]
	Warning: function optimised for performance. Variables are scaled during the
	execution.
	'''

	Cfact=cfact_biot*gamma_seg

	RA=ZetaP-ZetaA
	RB=ZetaP-ZetaB
	RAB=ZetaB-ZetaA
	Vcr=libalg.cross3d(RA,RB)
	vcr2=np.dot(Vcr,Vcr)

	# numerical radious
	if vcr2<(VORTEX_RADIUS*libalg.norm3d(RAB))**2:
		return

	### cross-product term derivative - upper triangular part only
	ra1,rb1=libalg.norm3d(RA),libalg.norm3d(RB)
	rainv=1./ra1
	rbinv=1./rb1
	Tv=RA*rainv-RB*rbinv
	dotprod=np.dot(RAB,Tv)

	Dvcross=np.empty((3,3))
	vcr2inv=1./vcr2
	vcr4inv=vcr2inv*vcr2inv
	diag_fact=    Cfact*vcr2inv*dotprod
	off_fact =-2.*Cfact*vcr4inv*dotprod
	# Dvcross[0,0]=diag_fact+off_fact*Vcr[0]**2
	# Dvcross[0,1]=          off_fact*Vcr[0]*Vcr[1]
	# Dvcross[0,2]=          off_fact*Vcr[0]*Vcr[2]
	Dvcross[0,:]=[diag_fact+off_fact*Vcr[0]**2, 
       								off_fact*Vcr[0]*Vcr[1],
 														off_fact*Vcr[0]*Vcr[2]]
	Dvcross[1,1]=diag_fact+off_fact*Vcr[1]**2
	Dvcross[1,2]=          off_fact*Vcr[1]*Vcr[2]
	Dvcross[2,2]=diag_fact+off_fact*Vcr[2]**2



	### difference term derivative - no symmetry
	minus_rainv3=-rainv**3
	minus_rbinv3=-rbinv**3
	Vsc=Vcr*vcr2inv*Cfact

	Ddiff=np.empty((3,3))
	Ddiff[:,0]=RAB[0]*Vsc
	Ddiff[:,1]=RAB[1]*Vsc
	Ddiff[:,2]=RAB[2]*Vsc


	# ### RAB derivative
	# dQ_dRAB=np.empty((3,3))
	# dQ_dRAB[:,0]=Tv[0]*Vsc
	# dQ_dRAB[:,1]=Tv[1]*Vsc
	# dQ_dRAB[:,2]=Tv[2]*Vsc


	# ##### crucial part!
	# dQ_dRA=Dvcross_by_skew3d(Dvcross,-RB)\
	# 							+np.dot(Ddiff, der_runit(RA,rainv,minus_rainv3))
	# dQ_dRB=Dvcross_by_skew3d(Dvcross, RA)\
	# 							-np.dot(Ddiff, der_runit(RB,rbinv,minus_rbinv3))

	# DerP +=  dQ_dRA  + dQ_dRB # w.r.t. P
	# DerA -=  dQ_dRAB + dQ_dRA # w.r.t. A	
	# DerB +=  dQ_dRAB - dQ_dRB # w.r.t. B




	# collocation point only
	# ##### crucial part!
	# DerP +=Dvcross_by_skew3d(Dvcross,RA-RB)\
	# 		+np.dot(Ddiff, der_runit(RA,rainv,minus_rainv3))\
	# 							-np.dot(Ddiff, der_runit(RB,rbinv,minus_rbinv3))




def eval_panel_comp(zetaP,ZetaPanel,gamma_pan=1.0):
	'''
	Computes derivatives of induced velocity w.r.t. coordinates of target point,
	zetaP, and panel coordinates. Returns two elements:
		- DerP: derivative of induced velocity w.r.t. ZetaP, with:
			DerP.shape=(3,3) : DerC[ Uind_{x,y,z}, ZetaC_{x,y,z} ]
		- DerVertices: derivative of induced velocity wrt panel vertices, with:
			DerVertices.shape=(4,3,3) : 
			DerVertices[ vertex number {0,1,2,3},  Uind_{x,y,z}, ZetaC_{x,y,z} ]
	'''

	DerP=np.zeros((3,3))
	DerVertices=np.zeros((4,3,3))

	for aa,bb in LoopPanel: 
		eval_seg_comp_loop(DerP,DerVertices[aa,:,:],DerVertices[bb,:,:],
								zetaP,ZetaPanel[aa,:],ZetaPanel[bb,:],gamma_pan)

	return DerP,DerVertices






if __name__=='__main__':
	
	gamma=4.
	zetaP=np.array([3.,3.,-2.])	
	zetaA=np.array([2.,1., 2.])	
	zetaB=np.array([4.,7., 3.])

	DPexp,DAexp,DBexp=eval_seg_exp(zetaP,zetaA,zetaB,gamma_seg=gamma)
	DPcomp,DAcomp,DBcomp=eval_seg_comp(zetaP,zetaA,zetaB,gamma_seg=gamma)

	ermax=max( np.max(np.abs(DPexp-DPcomp)),
						np.max(np.abs(DAexp-DAcomp)),
									np.max(np.abs(DBexp-DBcomp)) )
	assert ermax<1e-16, 'Analytical models not matching' 


