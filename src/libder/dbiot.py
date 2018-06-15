'''
Calculate derivative of Wnc(zeta)*Uc w.r.t local panel coordinates
'''

import numpy as np 
from IPython import embed

##### constants
cfact_biot=0.25/np.pi
VORTEX_RADIUS=1e-2 # numerical radious of vortex

### derivatives indices for eval_seg:
# the 1st is the variable w.r.t. which derivatives are taken.
# the 2nd is the component of the vaiable w.r.t derivative are taken.
# the 3rd is the component of the output 
dRA=np.array([[[ 1, 0, 0], [0, 1, 0], [0, 0, 1]], 
			  [[-1, 0, 0], [0,-1, 0], [0, 0,-1]], 
			  [[ 0, 0, 0], [0, 0, 0], [0, 0, 0]]])
dRB=np.array([[[ 1, 0, 0], [0, 1, 0], [0, 0, 1]], 
			  [[ 0, 0, 0], [0, 0, 0], [0, 0, 0]], 
			  [[-1, 0, 0], [0,-1, 0], [0, 0,-1]]])
dRAB=np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
			  [[-1, 0, 0], [0,-1, 0], [0, 0,-1]], 
			  [[ 1, 0, 0], [0, 1, 0], [0, 0, 1]]])


def eval_seg(ZetaP,ZetaA,ZetaB,gamma_seg=1.0):
	'''
	Derivative of induced velocity Q w.r.t. collocation and segment coordinates 
	in format:
		[ (ZetaP,ZetaA,ZetaB), (x,y,z) of Zeta,  (x,y,z) of Q]
	'''

	Der=np.zeros((3,3,3))

	RA=ZetaP-ZetaA
	RB=ZetaP-ZetaB
	RAB=ZetaB-ZetaA
	ra_norm,rb_norm=np.linalg.norm(RA),np.linalg.norm(RB)
	ra3,rb3=ra_norm**3,rb_norm**3

	vcross=np.cross(RA,RB)
	vcross2=np.dot(vcross,vcross)
	vcross4=vcross2**2

	# numerical radious
	vortex_radious_here=VORTEX_RADIUS*np.linalg.norm(RAB)
	if vcross2<vortex_radious_here**2:
		return np.zeros((3,3,3))

	# Extract components
	ra_x,ra_y,ra_z=RA
	rb_x,rb_y,rb_z=RB
	rab_x,rab_y,rab_z=RAB

	# special terms
	vdot_prod=(rab_x*(-rb_x/rb_norm + ra_x/ra_norm) + 
							rab_y*(-rb_y/rb_norm + ra_y/ra_norm) + 
								rab_z*(-rb_z/rb_norm + ra_z/ra_norm))
	T2=vdot_prod/vcross2
	T4=vdot_prod/vcross4

	xdiff=-rb_x/rb_norm + ra_x/ra_norm
	ydiff=-rb_y/rb_norm + ra_y/ra_norm
	zdiff=-rb_z/rb_norm + ra_z/ra_norm
	Rcross=np.cross(RA,RB)
	rcross_x,rcross_y,rcross_z=Rcross
	ra_rcross_x, ra_rcross_y, ra_rcross_z=2.*np.cross(RA,Rcross)
	rb_rcross_x, rb_rcross_y, rb_rcross_z=2.*np.cross(RB,Rcross)


	# ### derivatives indices:
	# # the 1st is the variable w.r.t. which derivatives are taken.
	# # the 2nd is the component of the vaiable w.r.t derivative are taken.
	# # the 3rd is the component of the output 
	dQ=np.array(
		[
		 # derivs w.r.t. RA
		 [[-T4*rb_rcross_x*rcross_x + rcross_x*(rab_x*(1/ra_norm - ra_x**2/ra3) - ra_x*ra_y*rab_y/ra3 - ra_x*ra_z*rab_z/ra3)/vcross2,
		   -T2*rb_z - T4*rb_rcross_x*rcross_y + rcross_y*(rab_x*(1/ra_norm - ra_x**2/ra3) - ra_x*ra_y*rab_y/ra3 - ra_x*ra_z*rab_z/ra3)/vcross2,
		    T2*rb_y - T4*rb_rcross_x*rcross_z + rcross_z*(rab_x*(1/ra_norm - ra_x**2/ra3) - ra_x*ra_y*rab_y/ra3 - ra_x*ra_z*rab_z/ra3)/vcross2],
		  [ T2*rb_z - T4*rb_rcross_y*rcross_x + rcross_x*(rab_y*(1/ra_norm - ra_y**2/ra3) - ra_x*ra_y*rab_x/ra3 - ra_y*ra_z*rab_z/ra3)/vcross2,
		   -T4*rb_rcross_y*rcross_y + rcross_y*(rab_y*(1/ra_norm - ra_y**2/ra3) - ra_x*ra_y*rab_x/ra3 - ra_y*ra_z*rab_z/ra3)/vcross2,
		   -T2*rb_x - T4*rb_rcross_y*rcross_z + rcross_z*(rab_y*(1/ra_norm - ra_y**2/ra3) - ra_x*ra_y*rab_x/ra3 - ra_y*ra_z*rab_z/ra3)/vcross2],
		  [-T2*rb_y - T4*rb_rcross_z*rcross_x + rcross_x*(rab_z*(1/ra_norm - ra_z**2/ra3) - ra_x*ra_z*rab_x/ra3 - ra_y*ra_z*rab_y/ra3)/vcross2,
		    T2*rb_x - T4*rb_rcross_z*rcross_y + rcross_y*(rab_z*(1/ra_norm - ra_z**2/ra3) - ra_x*ra_z*rab_x/ra3 - ra_y*ra_z*rab_y/ra3)/vcross2,
		   -T4*rb_rcross_z*rcross_z + rcross_z*(rab_z*(1/ra_norm - ra_z**2/ra3) - ra_x*ra_z*rab_x/ra3 - ra_y*ra_z*rab_y/ra3)/vcross2]],
		 # derivs w.r.t. RB
		 [[ T4*ra_rcross_x*rcross_x + rcross_x*(rab_x*(-1/rb_norm + rb_x**2/rb3) + rab_y*rb_x*rb_y/rb3 + rab_z*rb_x*rb_z/rb3)/vcross2,
		    T2*ra_z + T4*ra_rcross_x*rcross_y + rcross_y*(rab_x*(-1/rb_norm + rb_x**2/rb3) + rab_y*rb_x*rb_y/rb3 + rab_z*rb_x*rb_z/rb3)/vcross2,
		   -T2*ra_y + T4*ra_rcross_x*rcross_z + rcross_z*(rab_x*(-1/rb_norm + rb_x**2/rb3) + rab_y*rb_x*rb_y/rb3 + rab_z*rb_x*rb_z/rb3)/vcross2],
		  [-T2*ra_z + T4*ra_rcross_y*rcross_x + rcross_x*(rab_x*rb_x*rb_y/rb3 + rab_y*(-1/rb_norm + rb_y**2/rb3) + rab_z*rb_y*rb_z/rb3)/vcross2,
		    T4*ra_rcross_y*rcross_y + rcross_y*(rab_x*rb_x*rb_y/rb3 + rab_y*(-1/rb_norm + rb_y**2/rb3) + rab_z*rb_y*rb_z/rb3)/vcross2,
		    T2*ra_x + T4*ra_rcross_y*rcross_z + rcross_z*(rab_x*rb_x*rb_y/rb3 + rab_y*(-1/rb_norm + rb_y**2/rb3) + rab_z*rb_y*rb_z/rb3)/vcross2],
		  [ T2*ra_y + T4*ra_rcross_z*rcross_x + rcross_x*(rab_x*rb_x*rb_z/rb3 + rab_y*rb_y*rb_z/rb3 + rab_z*(-1/rb_norm + rb_z**2/rb3))/vcross2,
		   -T2*ra_x + T4*ra_rcross_z*rcross_y + rcross_y*(rab_x*rb_x*rb_z/rb3 + rab_y*rb_y*rb_z/rb3 + rab_z*(-1/rb_norm + rb_z**2/rb3))/vcross2,
		    T4*ra_rcross_z*rcross_z + rcross_z*(rab_x*rb_x*rb_z/rb3 + rab_y*rb_y*rb_z/rb3 + rab_z*(-1/rb_norm + rb_z**2/rb3))/vcross2]],
		 # derivs. w.r.t. RAB
		 [[ rcross_x*xdiff/vcross2,
		    rcross_y*xdiff/vcross2,
		    rcross_z*xdiff/vcross2],
		  [ rcross_x*ydiff/vcross2,
		    rcross_y*ydiff/vcross2,
		    rcross_z*ydiff/vcross2],
		  [ rcross_x*zdiff/vcross2,
		    rcross_y*zdiff/vcross2,
		    rcross_z*zdiff/vcross2]]])

	for cc_q in range(3):
		for cc_zeta in range(3):
			for cc_r in range(3):
				for zz in range(3): # loop through (ZetaP,ZetaA,ZetaB)
					Der[zz,cc_zeta,cc_q]=Der[zz,cc_zeta,cc_q]+\
						dQ[0,cc_r,cc_q]*dRA[zz,cc_zeta,cc_r]+\
						dQ[1,cc_r,cc_q]*dRB[zz,cc_zeta,cc_r]+\
						dQ[2,cc_r,cc_q]*dRAB[zz,cc_zeta,cc_r]\

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

	for ll in range(0,3):
		zetaA=ZetaPanel[ll,:]
		zetaB=ZetaPanel[ll+1,:]
		DerSeg=eval_seg(zetaP,zetaA,zetaB)
		# sum contribution
		DerP+=DerSeg[0,:,:]
		DerVertices[ll,:,:]+=DerSeg[1,:,:]
		DerVertices[ll+1,:,:]+=DerSeg[2,:,:]

	# last segment
	zetaA=ZetaPanel[3,:]
	zetaB=ZetaPanel[0,:]
	DerSeg=eval_seg(zetaP,zetaA,zetaB)
	DerP+=DerSeg[0,:,:]
	DerVertices[3,:,:]+=DerSeg[1,:,:]
	DerVertices[0,:,:]+=DerSeg[2,:,:]	

	return gamma_pan*DerP,gamma_pan*DerVertices






if __name__=='__main__':

	pass	