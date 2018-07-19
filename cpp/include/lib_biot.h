/*
Library for derivatives of biot-savart law
*/

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <types.h>

using namespace Eigen;
using namespace std;

#define Nvert 4


void biot_panel_map( map_RowVec3& velP,
					 const map_RowVec3 zetaP, 
					 const map_Mat4by3 ZetaPanel, 
					 const double gamma );


void der_biot_panel(Matrix3d& DerP, 
				Matrix3d DerVertices[Nvert], 
	 			const RowVector3d zetaP,
	 			const Matrix4by3d ZetaPanel, 
	 			const double gamma );


void der_biot_panel_map( map_Mat3by3& DerP, 
					 Vec_map_Mat3by3& DerVertices,
					 const map_RowVec3 zetaP, 
					 const map_Mat4by3 ZetaPanel, 
					 const double gamma );


void der_runit( Matrix3d& Der,
				const RowVector3d& rv, 
				double rinv,
				double minus_rinv3);


Matrix3d Dvcross_by_skew3d(const Matrix3d& Dvcross, 
						   const RowVector3d& rv);












