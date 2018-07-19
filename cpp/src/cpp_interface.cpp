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




