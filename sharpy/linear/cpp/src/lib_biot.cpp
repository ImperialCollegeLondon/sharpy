/*
Library for biot-savart law.
S. Maraniello, Jul 2018

The fundamental routine in this module are:
- biot_panel_map:
	This function implements the biot-savart law.
- der_biot_panel and der_biot_panel_map: 
	These functions are "identical", except for the input types. While eval_panel 
works with dense matrices, eval_panel_map works with Eigen Map objects, which are
required to build array based python input/output interface.
- dvinddzeta_map: computes derivatives w.r.t. grid coordinates of the velocity 
induced by a surface to a point.
*/

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <lib_biot.h>
#include <types.h>

using namespace Eigen;
using namespace std;

#define VORTEX_RADIUS_SQ 1e-4

# define Nvert 4
const double PI = 3.1415926535897932384626433832795028841971;
const double PIquart=0.25/PI;
const int svec[Nvert]={0, 1, 2, 3}; // seg. no.
const int avec[Nvert]={0, 1, 2, 3}; // seg. no.
const int bvec[Nvert]={1, 2, 3, 0}; // seg. no.
const int dm[Nvert]={0,1,1,0};
const int dn[Nvert]={0,0,1,1};




void biot_panel_map( map_RowVec3& velP,
					 const map_RowVec3 zetaP, 
					 const map_Mat4by3 ZetaPanel, 
					 const double gamma ){
	/* 
	This implementation works with mapping objects. 
	*/


	// declarations
	int ii,aa,bb;
	const double Cbiot=PIquart*gamma;
	double vcr2;	

	RowVector3d RAB, Vcr;
	Vector3d Vsc;
	Vector4d RABsq;

	Matrix4by3d R; 		// vectors P - vertex matrix
	Matrix4by3d Runit;  // unit vectors P - vertex matrix


	// ----------------------------------------------- Compute common variables
	// these are constants or variables depending only on vertices and P coords
	for(ii=0;ii<Nvert;ii++){
		R.row(ii)=zetaP-ZetaPanel.row(ii);
		Runit.row(ii)=R.row(ii)/R.row(ii).norm();
	}


	// -------------------------------------------------- Loop through segments
	for(ii=0;ii<Nvert;ii++){

		aa=avec[ii];
		bb=bvec[ii];

		RAB=ZetaPanel.row(bb)-ZetaPanel.row(aa);	// segment vector
		Vcr=R.row(aa).cross(R.row(bb));
		vcr2=Vcr.dot(Vcr);
		if (vcr2<VORTEX_RADIUS_SQ*RAB.dot(RAB)) continue;

		velP += ((Cbiot/vcr2) * RAB.dot(Runit.row(aa)-Runit.row(bb))) *Vcr;
	}
}



// -----------------------------------------------------------------------------


void der_biot_panel( Matrix3d& DerP, Matrix3d DerVertices[Nvert], 
	const RowVector3d zetaP, const Matrix4by3d ZetaPanel, const double gamma ){
	/* This implementation is no suitable for python interface */


	// declarations
	int ii,aa,bb;
	const double Cbiot=PIquart*gamma;
	double r1inv, vcr2, vcr2inv, vcr4inv, dotprod, diag_fact, off_fact;	

	RowVector3d RAB, Vcr, Tv;
	Vector3d Vsc;

	Matrix3d Dvcross, Ddiff, dQ_dRA, dQ_dRB, dQ_dRAB;

	Matrix4by3d R; 		// vectors P - vertex matrix
	Matrix4by3d Runit;  // unit vectors P - vertex matrix

	Matrix3d Array_Der_runit[Nvert]; // as a static arrays (we know size)


	// ----------------------------------------------- Compute common variables
	// these are constants or variables depending only on vertices and P coords
	for(ii=0;ii<Nvert;ii++){

		R.row(ii)=zetaP-ZetaPanel.row(ii);

		r1inv=1./R.row(ii).norm();
		Runit.row(ii)=R.row(ii)*r1inv;

		der_runit( Array_Der_runit[ii], R.row(ii), r1inv, -std::pow(r1inv,3) );
	}


	// -------------------------------------------------- Loop through segments
	for(ii=0;ii<Nvert;ii++){

		// vertices indices
		aa=avec[ii];
		bb=bvec[ii];

		// utility vars
		RAB=ZetaPanel.row(bb)-ZetaPanel.row(aa);	// segment vector
		Vcr=R.row(aa).cross(R.row(bb));
		vcr2=Vcr.dot(Vcr);
		if (vcr2<VORTEX_RADIUS_SQ*RAB.dot(RAB)){ 
			//cout << endl << "Skipping seg. " << ii << endl;
			continue;}
		Tv=Runit.row(aa)-Runit.row(bb);
		dotprod=RAB.dot(Tv);


		// ------------------------------------------ cross-product derivatives
		// lower triangular part only
		vcr2inv=1./vcr2;
		vcr4inv=vcr2inv*vcr2inv;
		diag_fact=    Cbiot*vcr2inv*dotprod;
		off_fact =-2.*Cbiot*vcr4inv*dotprod;

		Dvcross(0,0)=diag_fact+off_fact*Vcr[0]*Vcr[0];
		Dvcross(1,0)=off_fact*Vcr[0]*Vcr[1];
		Dvcross(1,1)=diag_fact+off_fact*Vcr[1]*Vcr[1]; 
		Dvcross(2,0)=off_fact*Vcr[0]*Vcr[2];
		Dvcross(2,1)=off_fact*Vcr[1]*Vcr[2];
		Dvcross(2,2)= diag_fact+off_fact*Vcr[2]*Vcr[2];


		// ------------------------------- difference and RAB terms derivatives
		Vsc=Vcr.transpose()*vcr2inv*Cbiot;
		Ddiff=Vsc*RAB;
		dQ_dRAB=Vsc*Tv;


		// ----------------------------------------------------- Final assembly
		dQ_dRA=Dvcross_by_skew3d(Dvcross,-R.row(bb))+Ddiff*Array_Der_runit[aa];
		dQ_dRB=Dvcross_by_skew3d(Dvcross, R.row(aa))-Ddiff*Array_Der_runit[bb];

		DerP += dQ_dRA + dQ_dRB;
		DerVertices[aa] -= dQ_dRAB + dQ_dRA;
		DerVertices[bb] += dQ_dRAB - dQ_dRB;
	}
}



void der_biot_panel_map( map_Mat3by3& DerP, 
					 Vec_map_Mat3by3& DerVertices,
					 const map_RowVec3 zetaP, 
					 const map_Mat4by3 ZetaPanel, 
					 const double gamma ){
	/* 
	This implementation works with mapping objects. 
	*/


	// declarations
	int ii,aa,bb;
	const double Cbiot=PIquart*gamma;
	double r1inv, vcr2, vcr2inv, vcr4inv, dotprod, diag_fact, off_fact;	

	RowVector3d RAB, Vcr, Tv;
	Vector3d Vsc;

	Matrix3d Dvcross, Ddiff, dQ_dRA, dQ_dRB, dQ_dRAB;

	Matrix4by3d R; 		// vectors P - vertex matrix
	Matrix4by3d Runit;  // unit vectors P - vertex matrix

	Matrix3d Array_Der_runit[Nvert]; // as a static arrays (we know size)


	// ----------------------------------------------- Compute common variables
	// these are constants or variables depending only on vertices and P coords
	for(ii=0;ii<Nvert;ii++){

		R.row(ii)=zetaP-ZetaPanel.row(ii);

		r1inv=1./R.row(ii).norm();
		Runit.row(ii)=R.row(ii)*r1inv;

		der_runit( Array_Der_runit[ii], R.row(ii), r1inv, -std::pow(r1inv,3) );
	}


	// -------------------------------------------------- Loop through segments
	for(ii=0;ii<Nvert;ii++){

		// vertices indices
		aa=avec[ii];
		bb=bvec[ii];

		// utility vars
		RAB=ZetaPanel.row(bb)-ZetaPanel.row(aa);	// segment vector
		Vcr=R.row(aa).cross(R.row(bb));
		vcr2=Vcr.dot(Vcr);
		if (vcr2<VORTEX_RADIUS_SQ*RAB.dot(RAB)){ 
			//cout << endl << "Skipping seg. " << ii << endl;
			continue;}
		Tv=Runit.row(aa)-Runit.row(bb);
		dotprod=RAB.dot(Tv);


		// ------------------------------------------ cross-product derivatives
		// lower triangular part only
		vcr2inv=1./vcr2;
		vcr4inv=vcr2inv*vcr2inv;
		diag_fact=    Cbiot*vcr2inv*dotprod;
		off_fact =-2.*Cbiot*vcr4inv*dotprod;

		Dvcross(0,0)=diag_fact+off_fact*Vcr[0]*Vcr[0];
		Dvcross(1,0)=off_fact*Vcr[0]*Vcr[1];
		Dvcross(1,1)=diag_fact+off_fact*Vcr[1]*Vcr[1]; 
		Dvcross(2,0)=off_fact*Vcr[0]*Vcr[2];
		Dvcross(2,1)=off_fact*Vcr[1]*Vcr[2];
		Dvcross(2,2)= diag_fact+off_fact*Vcr[2]*Vcr[2];


		// ------------------------------- difference and RAB terms derivatives
		Vsc=Vcr.transpose()*vcr2inv*Cbiot;
		Ddiff=Vsc*RAB;
		dQ_dRAB=Vsc*Tv;


		// ----------------------------------------------------- Final assembly
		dQ_dRA=Dvcross_by_skew3d(Dvcross,-R.row(bb))+Ddiff*Array_Der_runit[aa];
		dQ_dRB=Dvcross_by_skew3d(Dvcross, R.row(aa))-Ddiff*Array_Der_runit[bb];

		//cout << endl << "dQ_dRA = " << endl << dQ_dRA << endl;
		DerP += dQ_dRA + dQ_dRB;
		DerVertices[aa] -= dQ_dRAB + dQ_dRA;
		DerVertices[bb] += dQ_dRAB - dQ_dRB;
	}
/*	cout << "vcr2=" << vcr2 << endl;
	cout << "Tv=" << Tv << endl;
	cout << "dotprod=" << dotprod << endl;
	cout << "dQ_dRB=" << dQ_dRB << endl;
*/
}


// -----------------------------------------------------------------------------
// Sub-functions

void der_runit(Matrix3d& Der,const RowVector3d& rv, double rinv,double minus_rinv3){
	/*Warning: 
	1. RowVector3d needs to defined as constant if in main code RowVector
	is a row of a matrix. 
	2. The function will fail is Matrix3d is a sub-block of a matrix.
	 */

	// alloc upper diagonal part
	Der(0,0)=rinv+minus_rinv3*rv(0)*rv(0);
	Der(0,1)=     minus_rinv3*rv(0)*rv(1);
	Der(0,2)=     minus_rinv3*rv(0)*rv(2);
	Der(1,1)=rinv+minus_rinv3*rv(1)*rv(1);
	Der(1,2)=     minus_rinv3*rv(1)*rv(2);
	Der(2,2)=rinv+minus_rinv3*rv(2)*rv(2);                              
	// alloc lower diag
	Der(1,0)=Der(0,1);
	Der(2,0)=Der(0,2);
	Der(2,1)=Der(1,2);
	}



Matrix3d Dvcross_by_skew3d(const Matrix3d& Dvcross, const RowVector3d& rv){
	/*Warning: 
	1. RowVector3d needs to defined as constant if in main code RowVector
	is a row of a matrix. 
	 */

	Matrix3d P;

	P(0,0)=Dvcross(1,0)*rv(2)-Dvcross(2,0)*rv(1);
	P(0,1)=Dvcross(2,0)*rv(0)-Dvcross(0,0)*rv(2);
	P(0,2)=Dvcross(0,0)*rv(1)-Dvcross(1,0)*rv(0);
	//
	P(1,0)=P(0,1);
	P(1,1)=Dvcross(2,1)*rv(0)-Dvcross(1,0)*rv(2);
	P(1,2)=Dvcross(1,0)*rv(1)-Dvcross(1,1)*rv(0);
	//
	P(2,0)=P(0,2);
	P(2,1)=P(1,2);
	P(2,2)=Dvcross(2,0)*rv(1)-Dvcross(2,1)*rv(0);

	return P;
	}




// -----------------------------------------------------------------------------


void dvinddzeta(map_Mat3by3 DerC, 
				map_Mat DerV,
				const map_RowVec3 zetaC, 
				Vec_map_Mat ZetaIn,
				map_Mat GammaIn,
				int& M_in,
				int& N_in,
				int& Kzeta_in,
				bool& IsBound,
				int& M_in_bound, // M of bound surf associated
				int& Kzeta_in_bound
				)
{
	int cc, vv, mm, nn, jj, cc_in; //pp
	//int Kin=M_in*N_in;


	// below defined as maps to keep compatibility with der-biot_panel_map
	//Matrix4by3d ZetaPanel_in;
	//Matrix3d derv[Nvert];
	double p_ZetaPanel_in[12];
	double p_derv[36];
	map_Mat4by3 ZetaPanel_in(p_ZetaPanel_in);
	Vec_map_Mat3by3 derv;
	for(vv=0;vv<4;vv++) derv.push_back( map_Mat3by3(p_derv+9*vv) );


/*	cout << "Kzeta_in=" << endl << Kzeta_in << endl;
	cout << "DerV = " << endl << DerV << endl;
	cout << "GammaIn = " << endl << GammaIn << endl;
	for(cc=0;cc<3;cc++){
		cout << "ZetaIn[" << cc << "] = " << endl <<  ZetaIn[cc] << endl;
	}*/


	if (IsBound){// ------------------------------------------------ Bound case

		// Loop panels (mm,nn)
		for (mm=0; mm<M_in; mm++){
			for (nn=0; nn<N_in; nn++){
				//pp=mm*N_in+nn; // panel no.

				// get panel coords in 4x3 format
				for(cc=0; cc<3; cc++){
					for(vv=0; vv<Nvert; vv++){
						ZetaPanel_in(vv,cc)=ZetaIn[cc](mm+dm[vv],nn+dn[vv]);
					}
				}

				// init. local derivatives
				for(vv=0; vv<Nvert; vv++) derv[vv].setZero();
				
				// get local deriv
				der_biot_panel_map(DerC,derv,zetaC,ZetaPanel_in,GammaIn(mm,nn));
				//for(vv=0; vv<Nvert; vv++) cout << derv[vv] << endl;

				for(cc=0; cc<3; cc++){
					for(vv=0; vv<Nvert; vv++){
						for(cc_in=0; cc_in<3; cc_in++){
							jj= cc_in*Kzeta_in + (mm+dm[vv])*(N_in+1) + (nn+dn[vv]);
							DerV(cc,jj)+=derv[vv](cc,cc_in);
						}
					}
				}
			}
		}


	} else{ // ------------------------------------------------------ Wake case


		// scan TE first
		mm=0;
		for (nn=0; nn<N_in; nn++){
			//pp=mm*N_in+nn; // panel no.

			// get panel coords in 4x3 format
			for(cc=0; cc<3; cc++){
				for(vv=0; vv<Nvert; vv++){
					ZetaPanel_in(vv,cc)=ZetaIn[cc](mm+dm[vv],nn+dn[vv]);
				}
			}

			// init. local derivatives. only vertices 0 and 3 are on TE
			derv[0].setZero();
			derv[3].setZero();			

			// get local deriv
			der_biot_panel_map(DerC,derv,zetaC,ZetaPanel_in,GammaIn(mm,nn));

			for(cc=0; cc<3; cc++){
				for(cc_in=0; cc_in<3; cc_in++){
					// vv=0
					jj= cc_in*Kzeta_in_bound + M_in_bound*(N_in+1) + (nn);
					DerV(cc,jj)+=derv[0](cc,cc_in);
					// vv=3
					jj= cc_in*Kzeta_in_bound + M_in_bound*(N_in+1) + (nn+1);
					DerV(cc,jj)+=derv[3](cc,cc_in);
				}
				
			}
		}


		// Loop other panels (mm,nn) for colloc point
		for (mm=1; mm<M_in; mm++){
			for (nn=0; nn<N_in; nn++){

				// get panel coords in 4x3 format
				for(cc=0; cc<3; cc++){
					for(vv=0; vv<Nvert; vv++){
						ZetaPanel_in(vv,cc)=ZetaIn[cc](mm+dm[vv],nn+dn[vv]);
					}
				}
				// update DerC
				der_biot_panel_map(DerC,derv,zetaC,ZetaPanel_in,GammaIn(mm,nn));
			}// loop nn
		}// loop mm
	}// if-else
}




void aic3(	map_Mat AIC3,
			const map_RowVec3 zetaC, 
			Vec_map_Mat ZetaIn,
			int& M_in,
			int& N_in)
{

	int mm,nn,cc,vv;

	double p_ZetaPanel_in[12];
	map_Mat4by3 ZetaPanel_in(p_ZetaPanel_in);

	double p_vel[3];
	map_RowVec3 vel(p_vel);

	// Loop panels (mm,nn)
	for (mm=0; mm<M_in; mm++){
		for (nn=0; nn<N_in; nn++){
			//pp=mm*N_in+nn; // panel no.

			// get panel coords in 4x3 format
			for(cc=0; cc<3; cc++){
				for(vv=0; vv<Nvert; vv++){
					ZetaPanel_in(vv,cc)=ZetaIn[cc](mm+dm[vv],nn+dn[vv]);
				}
			}

			vel.setZero();
			biot_panel_map( vel, zetaC, ZetaPanel_in, 1.0);
			AIC3.col(mm*N_in+nn)=vel;

		}
	}
}



void ind_vel(map_RowVec3 velC,
			const map_RowVec3 zetaC, 
			Vec_map_Mat ZetaIn,
			map_Mat GammaIn,
			int& M_in,
			int& N_in)
{

	int mm,nn,cc,vv;

	double p_ZetaPanel_in[12];
	map_Mat4by3 ZetaPanel_in(p_ZetaPanel_in);

	// Loop panels (mm,nn)
	for (mm=0; mm<M_in; mm++){
		for (nn=0; nn<N_in; nn++){
			//pp=mm*N_in+nn; // panel no.

			// get panel coords in 4x3 format
			for(cc=0; cc<3; cc++){
				for(vv=0; vv<Nvert; vv++){
					ZetaPanel_in(vv,cc)=ZetaIn[cc](mm+dm[vv],nn+dn[vv]);
				}
			}

			biot_panel_map( velC, zetaC, ZetaPanel_in, GammaIn(mm,nn));
		}
	}
}


