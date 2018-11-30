'''
Analytical linearisation of biot-savart law for single semgnet.

Sign convention:

Scalar quantities are all lower case, e.g. zeta
Arrays begin with upper case, e.g. Zeta_i
2 D Matrices are all upper case, e.g. AW, ZETA=[Zeta_i]
3 D arrays (tensors) will be labelled with a 3 in the name, e.g. A3
'''

import numpy as np
import sympy as sm
import sympy.tensor.array as smarr
import linfunc



##### Define symbols

### vertices vectors
# coordinates
zetaA_x,zetaA_y,zetaA_z=sm.symbols('zetaA_x,zetaA_y,zetaA_z', real=True)
zetaB_x,zetaB_y,zetaB_z=sm.symbols('zetaB_x,zetaB_y,zetaB_z', real=True)
zetaP_x,zetaP_y,zetaP_z=sm.symbols('zetaP_x,zetaP_y,zetaP_z', real=True)

# vectors
ZetaA,ZetaB,ZetaP=sm.symbols('ZetaA ZetaB ZetaP', real=True)
ZetaA=smarr.MutableDenseNDimArray([zetaA_x,zetaA_y,zetaA_z])
ZetaB=smarr.MutableDenseNDimArray([zetaB_x,zetaB_y,zetaB_z])
ZetaP=smarr.MutableDenseNDimArray([zetaP_x,zetaP_y,zetaP_z])


### Difference vectors
RA=ZetaP-ZetaA
RB=ZetaP-ZetaB
RAB=ZetaB-ZetaA
dRA=sm.derive_by_array(RA,[ZetaP,ZetaA,ZetaB])
dRB=sm.derive_by_array(RB,[ZetaP,ZetaA,ZetaB])
dRAB=sm.derive_by_array(RAB,[ZetaP,ZetaA,ZetaB])



################################################################################


### redefine R02,R13
ra_x,ra_y,ra_z=sm.symbols('ra_x ra_y ra_z', real=True)
rb_x,rb_y,rb_z=sm.symbols('rb_x rb_y rb_z', real=True)
rab_x,rab_y,rab_z=sm.symbols('rab_x rab_y rab_z', real=True)
RA=smarr.MutableDenseNDimArray([ra_x,ra_y,ra_z])
RB=smarr.MutableDenseNDimArray([rb_x,rb_y,rb_z])
RAB=smarr.MutableDenseNDimArray([rab_x,rab_y,rab_z])


Vcross=linfunc.cross_product(RA,RB)
Vcross_sq=linfunc.scalar_product(Vcross,Vcross)
#Vcross_norm=linfunc.norm2(Vcross)
#Vcross_unit=Vcross/Vcross_norm
RA_unit=RA/linfunc.norm2(RA)
RB_unit=RB/linfunc.norm2(RB)
#Q=1/4/sm.pi*Vcross_unit*linfunc.scalar_product(RAB,(RA_unit-RB_unit))
Q=Vcross*linfunc.scalar_product(RAB,(RA_unit-RB_unit))/Vcross_sq


dQ=sm.derive_by_array(Q,[RA,RB,RAB])


##### shorted equation
ra_norm,rb_norm=sm.symbols('ra_norm rb_norm')
dQshort=dQ.copy()
dQshort=dQshort.subs(sm.sqrt(ra_x**2 + ra_y**2 + ra_z**2),ra_norm)
dQshort=dQshort.subs(sm.sqrt(rb_x**2 + rb_y**2 + rb_z**2),rb_norm)


vcross2,vcross4=sm.symbols('vcross2 vcross4')
dQshort=dQshort.subs((ra_x*rb_y - ra_y*rb_x)**2 + 
									(-ra_x*rb_z + ra_z*rb_x)**2 + 
										(ra_y*rb_z - ra_z*rb_y)**2,vcross2)
dQshort=dQshort.subs(vcross2**2,vcross4)

vdot_prod=sm.symbols('vdot_prod')
dQshort=dQshort.subs( (rab_x*(-rb_x/rb_norm + ra_x/ra_norm) + 
							rab_y*(-rb_y/rb_norm + ra_y/ra_norm) + 
								rab_z*(-rb_z/rb_norm + ra_z/ra_norm)),vdot_prod)


T2,T4=sm.symbols('T2 T4')
dQshort=dQshort.subs(vdot_prod/vcross2,T2)
dQshort=dQshort.subs(vdot_prod/vcross4,T4)


xdiff,ydiff,zdiff=sm.symbols('xdiff ydiff zdiff')
dQshort=dQshort.subs(-rb_x/rb_norm + ra_x/ra_norm,xdiff)
dQshort=dQshort.subs(-rb_y/rb_norm + ra_y/ra_norm,ydiff)
dQshort=dQshort.subs(-rb_z/rb_norm + ra_z/ra_norm,zdiff)

rcross_x,rcross_y,rcross_z=sm.symbols('rcross_x rcross_y rcross_z')
dQshort=dQshort.subs(ra_y*rb_z - ra_z*rb_y,rcross_x)
dQshort=dQshort.subs(-ra_x*rb_z + ra_z*rb_x,rcross_y)
dQshort=dQshort.subs(ra_x*rb_y - ra_y*rb_x,rcross_z)

ra3,rb3=sm.symbols('ra3 rb3')
dQshort=dQshort.subs(ra_norm**3,ra3)
dQshort=dQshort.subs(rb_norm**3,rb3)


ra_rcross_x, ra_rcross_y, ra_rcross_z = sm.symbols('ra_rcross_x ra_rcross_y ra_rcross_z')
rb_rcross_x, rb_rcross_y, rb_rcross_z = sm.symbols('rb_rcross_x rb_rcross_y rb_rcross_z')
dQshort=dQshort.subs(2*ra_y*rcross_z - 2*ra_z*rcross_y,ra_rcross_x)
dQshort=dQshort.subs(-2*ra_x*rcross_z + 2*ra_z*rcross_x,ra_rcross_y)
dQshort=dQshort.subs(2*ra_x*rcross_y - 2*ra_y*rcross_x,ra_rcross_z)
dQshort=dQshort.subs(2*rb_y*rcross_z - 2*rb_z*rcross_y,rb_rcross_x)
dQshort=dQshort.subs(-2*rb_x*rcross_z + 2*rb_z*rcross_x,rb_rcross_y)
dQshort=dQshort.subs(2*rb_x*rcross_y - 2*rb_y*rcross_x,rb_rcross_z)


