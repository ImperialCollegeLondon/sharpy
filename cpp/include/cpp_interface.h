/*
C++ interface.
S. Maraniello, Jul 2018

Routines to wrap C++ Eigen based function use 1d arrays as input/output.
*/


#include <Eigen/Dense>
#include <types.h>

using namespace Eigen;
using namespace std;

#define Nvert 4


extern "C" void call_der_biot_panel(double p_DerP[9], 
									double p_DerVertices[Nvert*9],
									double p_zetaP[3], 
									double p_ZetaPanel[12], 
									const double& gamma );


extern "C" void call_biot_panel(double p_vel[3], 
								double p_zetaP[3], 
								double p_ZetaPanel[12], 
								const double& gamma );







