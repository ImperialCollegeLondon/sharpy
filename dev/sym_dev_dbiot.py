'''
Symbolic derivation for fast derivative of biot-savart law
S. Maraniello, 11 Jul 2018
'''


import sympy as sm
import sympy.tensor.array as smarr
import linfunc


# coordinates
# a00,a01,a02=sm.symbols('a00 a01 a02', real=True)
# a11,a12=sm.symbols('a11 a12', real=True)
# a22=sm.symbols('a22', real=True)

##### derivative w.r.t. Ra,Rb due to cross-product
# show derivative is symmetric
ra_x,ra_y,ra_z=sm.symbols('ra_x ra_y ra_z', real=True)
rb_x,rb_y,rb_z=sm.symbols('rb_x rb_y rb_z', real=True)
RA=smarr.MutableDenseNDimArray([ra_x,ra_y,ra_z])
RB=smarr.MutableDenseNDimArray([rb_x,rb_y,rb_z])
Vcross=linfunc.cross_product(RA,RB)

V2=linfunc.scalar_product(Vcross,Vcross)
V4=V2*V2


Dv=sm.zeros(3)
for ii in range(3):
	Dv[ii,ii]+=1/V2
	for jj in range(3):
		Dv[ii,jj]+=-2/V4*Vcross[ii]*Vcross[jj]
DvRAskew=linfunc.matrix_product(Dv,linfunc.skew(RA))
DvRBskew=linfunc.matrix_product(Dv,linfunc.skew(RB))

# verify symetry
print('Verify symmetry of Dv')
for ii in range(3):
	for jj in range(ii+1,3):
		print('ii,jj:%d,%d'%(ii,jj) )
		h=sm.simplify(Dv[ii,jj]-Dv[jj,ii])
		print(h)
# verify symetry
print('Verify symmetry of DvR.skew')
for ii in range(3):
	for jj in range(ii+1,3):
		print('ii,jj:%d,%d'%(ii,jj) )
		h=sm.simplify(DvRAskew[ii,jj]-DvRAskew[jj,ii])
		print(h)
		h=sm.simplify(DvRBskew[ii,jj]-DvRBskew[jj,ii])
		print(h)
