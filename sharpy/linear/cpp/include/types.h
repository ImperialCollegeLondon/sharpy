/*
Define types for C++ interface.

To maximise speed, matrices size is specified whenever possible. 
(see Eigen doc, (Fixed vs Dynamic size)
*/

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;


typedef Matrix<double,4,3,RowMajor> Matrix4by3d;

// For mapping 1d arrays into Eigen Matrices (interface for 1D or 2D python arrays)
typedef Map< Matrix<double,Dynamic,Dynamic,RowMajor> > map_Mat; 
typedef Map< Matrix<double,4,3,RowMajor> > map_Mat4by3; 
typedef Map< Matrix<double,3,3,RowMajor> > map_Mat3by3; 
typedef Map< Matrix<double,1,3> > map_RowVec3; 

// For mapping 3D python arrays
// ps: using std::vector works for Map class. 
typedef std::vector<map_Mat3by3> Vec_map_Mat3by3;
typedef std::vector<map_Mat> Vec_map_Mat;









