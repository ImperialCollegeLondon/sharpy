/*
C++ interface.
S. Maraniello, Jul 2018

Routines to wrap C++ Eigen based function use 1d arrays as input/output.
*/


#include <Eigen/Dense>
#include <types.h>
#include <lib_biot.h>

using namespace Eigen;
using namespace std;

#define Nvert 4


extern "C" void call_der_biot_panel(double p_DerP[9], 
									double p_DerVertices[Nvert*9],
									double p_zetaP[3], 
									double p_ZetaPanel[12], 
									const double& gamma )
{ 
	/*
	To interface Eigen based routines with python, matrices need to be mapped
	into 1d arrays.
	*/

	int vv;

	map_Mat3by3 DerP(p_DerP);
	//Matrix3d DerVertices[Nvert];
	Vec_map_Mat3by3 DerVertices; // requires push_back to assigns
	const map_RowVec3 zetaP(p_zetaP);
	const map_Mat4by3 ZetaPanel(p_ZetaPanel);

	// initialise DerVertices - all done by reference
	for(vv=0;vv<4;vv++){
		DerVertices.push_back( map_Mat3by3(p_DerVertices+9*vv) );
		//DerVertices[vv]=0.0*DerVertices[vv];
	}

	/*	// Verify inputs (debugging)
	for(vv=0;vv<4;vv++){
		cout << endl << "DerVertices[" << vv << "] (in)=" << endl << DerVertices[vv] << endl;
	}
	cout << endl << "DerP (in)=" << endl << DerP << endl;
	cout << endl << "zetaP (in)=" << endl << zetaP << endl;
	cout << endl << "ZetaPanel (in)=" << endl << ZetaPanel << endl;
	cout << endl << "gamma (in)=" << endl << gamma << endl;*/
	der_biot_panel_map( DerP, DerVertices, zetaP, ZetaPanel, gamma );
	//cout << endl << "DerP (out)=" << endl << DerP << endl;
}



extern "C" void call_biot_panel(double p_vel[3], 
								double p_zetaP[3], 
								double p_ZetaPanel[12], 
								const double& gamma ){	
	/*
	To interface Eigen based routines with python, matrices need to be mapped
	into 1d arrays.
	*/

	map_RowVec3 velP(p_vel);
	const map_RowVec3 zetaP(p_zetaP);
	const map_Mat4by3 ZetaPanel(p_ZetaPanel);

	biot_panel_map(velP, zetaP, ZetaPanel, gamma);
}



extern "C" void call_dvinddzeta(double p_DerC[9], 
								double p_DerV[],
								double p_zetaC[3], 
								double p_ZetaIn[],
								double p_GammaIn[],
								int& M_in,
								int& N_in,
								bool& IsBound,
								int& M_in_bound // M of bound surf associated
								 )
{
	int cc;
	int Kzeta_in=(M_in+1)*(N_in+1);
	int Kzeta_in_bound=(M_in_bound+1)*(N_in+1);

	// interface
	map_Mat3by3 DerC(p_DerC);	
	const map_RowVec3 zetaC(p_zetaC);

	map_Mat DerV(p_DerV,3,3*Kzeta_in_bound);
	map_Mat GammaIn(p_GammaIn,M_in,N_in);

	Vec_map_Mat ZetaIn;
	for(cc=0;cc<3;cc++){
		ZetaIn.push_back( map_Mat(p_ZetaIn+cc*Kzeta_in, M_in+1, N_in+1) );
	}

	dvinddzeta( DerC,DerV, 
				zetaC,ZetaIn,GammaIn,
				M_in,N_in,Kzeta_in, 
				IsBound,M_in_bound,Kzeta_in_bound);

}



extern "C" void call_aic3(	double p_AIC3[],
							double p_zetaC[3], 
							double p_ZetaIn[],
							int& M_in,
							int& N_in)
{
	int cc;
	int K_in=M_in*N_in;

	map_Mat AIC3(p_AIC3,3,K_in);	
	const map_RowVec3 zetaC(p_zetaC);

	int Kzeta_in=(M_in+1)*(N_in+1);
	Vec_map_Mat ZetaIn;
	for(cc=0;cc<3;cc++){
		ZetaIn.push_back( map_Mat(p_ZetaIn+cc*Kzeta_in, M_in+1, N_in+1) );
	}

	aic3(AIC3, zetaC, ZetaIn, M_in, N_in);
}



extern "C" void call_ind_vel(
							double p_vel[3],
							double p_zetaC[3], 
							double p_ZetaIn[],
							double p_GammaIn[],
							int& M_in,
							int& N_in)
{
	int cc;

	map_RowVec3 velC(p_vel);
	const map_RowVec3 zetaC(p_zetaC);

	map_Mat GammaIn(p_GammaIn,M_in,N_in);

	int Kzeta_in=(M_in+1)*(N_in+1);
	Vec_map_Mat ZetaIn;
	for(cc=0;cc<3;cc++){
		ZetaIn.push_back( map_Mat(p_ZetaIn+cc*Kzeta_in, M_in+1, N_in+1) );
	}

	ind_vel(velC, zetaC, ZetaIn, GammaIn, M_in, N_in);
}




