/*
Test cpp functions.
S. Maraniello, 12 Jul 2018
*/

#include <iostream>
#include <Eigen/Dense>
#include <types.h>
//#include <lib_biot.h>
#include <cpp_interface.h>

using namespace std;
using namespace Eigen;

#define Nvert 4



void print_array(double A[], const int N){
	for (int ii = 0; ii < N; ii++){ cout << A[ii] << ", ";}
	cout << endl;
}



int main(){

	// ---------------------------------------------------------------- Factors
	double gamma=4.;
	int ii, vv;
	Vec_map_Mat3by3 ArrayDerVertices; // requires push_back for assignment



	// ----------------------------------------------- define input/output vars
	// to max speed, we define new types of known size, when this holds
	// see Eigen doc (Fixed vs Dynamic size)

	double p_vel[3]={0.,0.,0.};

	double p_ZetaPanel[12]={1.0,3.0,0.9, 
						    5.0,3.1,1.9,
				 			4.8,8.1,2.5,
				 			0.9,7.9,1.7};
	//double p_zetaP[3]={3.0,5.5,2.0};
	double p_zetaP[3]={2.07, 7.96, 1.94};

	double p_DerP[9] ={	0.,0.,0.,
						0.,0.,0.,
						0.,0.,0.};

	double p_DerVertices[Nvert*9];

	for (vv=0; vv<Nvert; vv++) {
		for (ii=0; ii<9; ii++) p_DerVertices[vv*9+ii]=0.0;
	}

	// Map into DerVertices
	for (vv=0; vv<Nvert; vv++) {
		ArrayDerVertices.push_back( map_Mat3by3(p_DerVertices+9*vv) );
		//cout << " vertex: " << vv << "= " << ArrayDerVertices[vv] << endl;		
	}

	call_der_biot_panel( p_DerP, p_DerVertices, p_zetaP, p_ZetaPanel, gamma );

	cout << endl << "p_DerVertices = ";
	print_array(p_DerVertices,9*Nvert);
	cout << endl << "p_DerP = ";
	print_array(p_DerP,9);


	call_biot_panel(p_vel, p_zetaP, p_ZetaPanel, gamma );
	cout << endl << "p_vel = ";
	print_array(p_vel,3);

	return 0;
}
