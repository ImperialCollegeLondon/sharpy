######################################################################
##################  PYTHON PACKAGES  #################################
######################################################################
# Usual SHARPy
import h5py as h5
import numpy as np
import os
import sharpy.utils.algebra as algebra
import sharpy.utils.cout_utils as cout # to use color output
# Read excel files
import pandas as pd
from pandas import ExcelFile
# Generate errors during execution
import sys
# Use .isnan
import math

######################################################################
##################  MATHEMATICAL FUNCTIONS  ##########################
######################################################################
def rotate_vector(vector,direction,angle):
    # This function rotates a "vector" around a "direction" a certain "angle"
    # according to Rodrigues formula

    #Create numpy arrays from vectors
    vector=np.array(vector)
    direction=np.array(direction)

    # Assure that "direction" has unit norm
    direction/=np.linalg.norm(direction)

    rot_vector=vector*np.cos(angle)+np.cross(direction,vector)*np.sin(angle)+direction*np.dot(direction,vector)*(1.0-np.cos(angle))

    return rot_vector

def get_airfoil_camber(x,y,n_points_camber):
    # Returns the airfoil camber for a given set of coordinates (XFOIL format expected)

    x=np.array(x, dtype=float)
    y=np.array(y, dtype=float)
    n=len(x)
    imin_x=0

    # Look for the minimum x (it will be assumed as the LE position
    for i in range(0,n):
        if(x[i]<x[imin_x]):
            imin_x=i

    x_suction=np.zeros((imin_x+1, ))
    y_suction=np.zeros((imin_x+1, ))
    x_pressure=np.zeros((n-imin_x, ))
    y_pressure=np.zeros((n-imin_x, ))

    for i in range(0,imin_x+1):
        x_suction[i]=x[imin_x-i]
        y_suction[i]=y[imin_x-i]

    for i in range(imin_x,n):
        x_pressure[i-imin_x]=x[i]
        y_pressure[i-imin_x]=y[i]

    # Compute the camber coordinates
    camber_y=np.zeros((n_points_camber,))
    camber_x=np.linspace(0.0,1.0,n_points_camber)

    # camber_y=0.5*(np.interp(camber_x,x[imin_x::-1],y[imin_x::-1])+np.interp(camber_x,x[imin_x:],y[imin_x:]))
    camber_y=0.5*(np.interp(camber_x,x_suction,y_suction)+np.interp(camber_x,x_pressure,y_pressure))

    # The function should be called as: camber_x, camber_y = get_airfoil_camber(x,y)
    return camber_x, camber_y

######################################################################
##################  DEFINE CASE  #####################################
######################################################################
case_name = 'basic_wing'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

flow = ['BeamLoader',
        'AerogridLoader',
        'StaticCoupled',
        'SteadyHelicoidalWake',
        #'DynamicPrescribedCoupled',
        'AerogridPlot',
        'BeamPlot'
        ]

gravity = 'off'



######################################################################
##################  DEFINE CONSTANTS  ################################
######################################################################
n_node_elem=3
deg2rad=2*np.pi/360
n_points_camber=1000

######################################################################
##################  READ EXCEL FILES  ################################
######################################################################
WSP = 10.0
airDensity = 0.5
omega = 1.6

wake_revs = 3
sim_revs = wake_revs**2

dphi_deg = 1.0
dt = dphi_deg*deg2rad / omega

n_tstep = int(sim_revs*360/dphi_deg)
alpha = 0.0*deg2rad
beta = 0.0*deg2rad

# Number of panels on the blade (chordwise direction)
m = 1
# Number of panels on the wake (flow direction)
mstar = int(wake_revs*360/dphi_deg)


numberOfBlades = 1
tilt = 0.0*deg2rad
cone = 0.0*deg2rad
pitch = 0.0*deg2rad

angular_velocity=np.zeros((3,))
angular_velocity[0]=omega*np.cos(tilt)
angular_velocity[1]=omega*np.sin(tilt)
angular_velocity[2]=0.0

n_elem_structural = 6
length = 50.0

Radius = np.linspace(0.1*length,length,n_elem_structural)
BlFract = 0.0*np.ones((n_elem_structural, ))
AeroCent = 0.25*np.ones((n_elem_structural, ))
StrcTwst = 90.0*np.ones((n_elem_structural, ))*deg2rad
BMassDen = np.ones((n_elem_structural, ))
FlpStff = 1e10*np.ones((n_elem_structural, ))
EdgStff = 1e10*np.ones((n_elem_structural, ))
GJStff = 1e10*np.ones((n_elem_structural, ))
EAStff = 1e10*np.ones((n_elem_structural, ))
Alpha = np.ones((n_elem_structural, ))
FlpIner = np.ones((n_elem_structural, ))
EdgIner = np.ones((n_elem_structural, ))
PrecrvRef = np.zeros((n_elem_structural, ))
PreswpRef = np.zeros((n_elem_structural, ))
FlpcgOf = np.zeros((n_elem_structural, ))
EdgcgOf = np.zeros((n_elem_structural, ))
FlpEAOf = np.zeros((n_elem_structural, ))
EdgEAOf = np.zeros((n_elem_structural, ))

n_elem_aero = n_elem_structural
Rnodes = np.linspace(0.1*length,length,n_elem_aero)
AeroTwst = 0.0*np.ones((n_elem_aero, ))*deg2rad
Chord = np.ones((n_elem_aero, ))
pure_airfoils = ["" for x in range(n_elem_aero)]

pure_airfoils_camber=np.zeros((n_points_camber,2,n_elem_aero))
for ielem in range(0,n_elem_aero):
    pure_airfoils_camber[:,0,ielem] = np.linspace(0.0,1.0,n_points_camber)

######################################################################
######################  BLADE DISCRETIZATION  ########################
######################################################################

keep_excel_discretization = True

if(keep_excel_discretization):
    # Check that the number of nodes provided is consistent
    n_elem_main = n_elem_structural-2
    elem_r=np.zeros((n_elem_main,))
    elem_r=Radius[1:-1]

    n_node_main = int(n_elem_main*(n_node_elem-1) + 1)
    r=np.zeros((n_node_main, ))
    r[0]=Radius[0]
    r[-2]=Radius[-2]
    r[-1]=Radius[-1]

    for ielem in range(0,n_elem_main-1):
        r[ielem*(n_node_elem-1)+1]=elem_r[ielem]
        r[ielem*(n_node_elem-1)+2]=0.5*(elem_r[ielem]+elem_r[ielem+1])

else:
    # Choose whatever discretization
    sys.error("ERROR: Change excel discretization not implemented")

# Initialize variables
prebending=np.zeros((n_node_main, ))
presweept=np.zeros((n_node_main, ))
node_struct_twist=np.zeros((n_node_main, ))
EA=np.zeros((n_elem_main, ))
EIz=np.zeros((n_elem_main, ))
EIy=np.zeros((n_elem_main, ))
GJ=np.zeros((n_elem_main, ))

GAy=np.zeros((n_elem_main, ))
GAz=np.zeros((n_elem_main, ))

mass=np.zeros((n_elem_main, ))

inertia_xb=np.zeros((n_elem_main, ))
inertia_yb=np.zeros((n_elem_main, ))
inertia_zb=np.zeros((n_elem_main, ))

pos_cg_B=np.zeros((n_elem_main,3))

# Interpolate all the blade properties
prebending=np.interp(r,Radius,PrecrvRef)
presweept=np.interp(r,Radius,PreswpRef)
node_struct_twist=np.interp(r,Radius,StrcTwst)
EA=np.interp(elem_r,Radius,EAStff)
EIz=np.interp(elem_r,Radius,FlpStff)
EIy=np.interp(elem_r,Radius,EdgStff)
GJ=np.interp(elem_r,Radius,GJStff)

# cout.cout_wrap('WARNING: The poisson cofficient is supossed equal to 0.3', 1)
print('WARNING: The poisson cofficient is supossed equal to 0.3')
print('WARNING: Cross-section area is used as shear area')
poisson_coef=0.3
GAy=EA/2.0/(1+poisson_coef)
GAz=EA/2.0/(1+poisson_coef)

mass=np.interp(elem_r,Radius,BMassDen)

inertia_yb=np.interp(elem_r,Radius,EdgIner)
inertia_zb=np.interp(elem_r,Radius,FlpIner)
print('WARNING: Using perpendicular axis theorem to compute the inertia around xB')
inertia_xb=inertia_yb+inertia_zb

pos_cg_B[:,1]=np.interp(elem_r,Radius,EdgcgOf)
pos_cg_B[:,2]=np.interp(elem_r,Radius,FlpcgOf)

# Initialize variables
blade_x=np.zeros((n_node_main, ))
blade_y=np.zeros((n_node_main, ))
blade_z=np.zeros((n_node_main, ))
blade_y_BFoR=np.zeros((n_node_main,3))
mass_db=np.zeros((n_elem_main,6,6))
stiffness_matrix_db=np.zeros((n_elem_main,6,6))

######################################################################
######################  BLADE STRUCTURE  #############################
######################################################################

for inode in range(0,n_node_main):

    # COORDINATES
    # (x,y,z) are the coordinates (A frame) of the reference line at each position
    blade_x[inode]=-1.0*r[inode]*np.cos(90*deg2rad+cone-tilt)-prebending[inode]*np.sin(90*deg2rad+cone-tilt)
    blade_y[inode]=r[inode]*np.sin(90*deg2rad+cone-tilt)-prebending[inode]*np.cos(90*deg2rad+cone-tilt)
    blade_z[inode]=-1.0*presweept[inode]

    # FRAME_OF_REFERENCE_DELTA
    # y vector of the B frame of reference
    blade_y_BFoR[inode, :]=[-1.0,0.0,0.0]

for ielem in range(0,n_elem_main):
    # MASS DATABASE (1 element per node)
    # mass_db[inode, :, :] = mass[inode]
    mass_db[ielem, 0:3, 0:3] = np.diag(np.ones((3,))*mass[ielem])

    mass_db[ielem, 0:3, 3:6] = [[ 0.0                                , mass[ielem]*pos_cg_B[ielem,2]      ,  -1.0*mass[ielem]*pos_cg_B[ielem,1] ],
                                [ -1.0*mass[ielem]*pos_cg_B[ielem,2] , 0.0                                ,  mass[ielem]*pos_cg_B[ielem,0]      ],
                                [ mass[ielem]*pos_cg_B[ielem,1]      , -1.0*mass[ielem]*pos_cg_B[ielem,0] ,  0.0                                ]]

    mass_db[ielem, 3:6, 0:3] = -1.0*mass_db[ielem, 0:3, 3:6]

    mass_db[ielem, 3:6, 3:6] = np.diag([inertia_xb[ielem], inertia_yb[ielem], inertia_zb[ielem]])

    # STIFFNESS DATABASE (1 element per node)
    stiffness_matrix_db[ielem, :, :] = np.diag([EA[ielem], GAy[ielem], GAz[ielem], GJ[ielem], EIy[ielem], EIz[ielem]])


    # STRUCTURAL_TWIST
    # Already in node_struct_twist

######################################################################
######################  ROTOR STRUCTURE  #############################
######################################################################

if(r[0]==0):
    # Blades are joined at the centre
    n_node=(n_node_main-1)*numberOfBlades+1
    joined_blades=1
    sys.exit("ERROR: Joined blades at the origin not implemented yet.\n")
else:
    n_node=n_node_main*numberOfBlades
    joined_blades=0

n_elem=n_elem_main*numberOfBlades

# Initialize variables
x = np.zeros((n_node, ))
y= np.zeros((n_node, ))
z= np.zeros((n_node, ))
elem_stiffness=np.zeros((n_elem, ), dtype=int)
elem_mass=np.zeros((n_elem, ), dtype=int)
frame_of_reference_delta=np.zeros((n_elem,3,3))
structural_twist=np.zeros((n_node, ))
boundary_conditions=np.zeros((n_node, ), dtype=int)
beam_number=np.zeros((n_node, ), dtype=int)
app_forces=np.zeros((n_node,6))
connectivities=np.zeros((n_elem,3), dtype=int)

for iblade in range(0,numberOfBlades):
    blade_angle=iblade*(360.0/numberOfBlades)*deg2rad

    for inode_main in range(0,n_node_main):
        # Compute the global node
        inode = inode_main+iblade*n_node_main

        structural_twist[inode]=node_struct_twist[inode_main]

        # Boundary conditions
        if (inode_main==0):
            boundary_conditions[inode]=1
        elif(inode_main==n_node_main-1):
            boundary_conditions[inode]=-1
        else:
            boundary_conditions[inode]=0

        # Beam number
        beam_number[inode]=iblade

        # Coordinates
        [x[inode],y[inode],z[inode]]=rotate_vector([blade_x[inode_main],blade_y[inode_main],blade_z[inode_main]],[np.cos(tilt),np.sin(tilt),0.0],blade_angle)

        # Applied forces
        app_forces[inode,:]=np.zeros((6, ))
        if(inode_main==n_node_main-1):
            app_forces[inode,1]=0.0


    # Connectivities
    for ielem_main in range(n_elem_main):
        ielem=ielem_main+iblade*n_elem_main

        # Node characteristics
        elem_stiffness[ielem]=int(ielem_main)
        elem_mass[ielem]=int(ielem_main)


        connectivities[ielem, :] = (np.ones((3, ))*(ielem*(n_node_elem - 1)+iblade) + [0, 2, 1])
        # B frame of reference
        for inode in range(0,n_node_elem):
            frame_of_reference_delta[ielem,inode,:]=rotate_vector(blade_y_BFoR[connectivities[ielem_main,inode], :],[np.cos(tilt),np.sin(tilt),0.0],blade_angle)

######################################################################
######################  BLADE AERODYNAMICS  ##########################
######################################################################

# Look for the first element that is goint to be aerodynamic
first_aero_elem=0
while (elem_r[first_aero_elem]<=Rnodes[0]):
    first_aero_elem+=1

first_aero_node=first_aero_elem*(n_node_elem-1)
n_aero_node=(n_elem_main-first_aero_elem)*(n_node_elem-1)+1

blade_aero_node = np.zeros((n_node_main,), dtype=bool)
blade_aero_node[first_aero_node:]=np.ones((n_node_main-first_aero_node,),dtype=bool)

# Airfoil distribution
blade_airfoil_distribution=np.zeros((n_elem_main,n_node_elem), dtype=int)
for ielem_main in range(n_elem_main):
    blade_airfoil_distribution[ielem_main,:]=connectivities[ielem_main,:]

blade_airfoil=np.zeros((n_points_camber,2,n_node_main))

for inode_main in range(0,n_node_main):
    # camber_x, camber_y = get_airfoil_camber(x,y)

    iairfoil=0
    while(r[inode_main]<Rnodes[iairfoil]):
        iairfoil+=1
        if(iairfoil==len(Rnodes)):
            iairfoil-=1
            break

    beta=min((r[inode_main]-Rnodes[iairfoil-1])/(Rnodes[iairfoil]-Rnodes[iairfoil-1]),1.0)
    beta=max(0.0,beta)

    blade_airfoil[:,0,inode_main]=(1-beta)*pure_airfoils_camber[:,0,iairfoil-1]+beta*pure_airfoils_camber[:,0,iairfoil]
    blade_airfoil[:,1,inode_main]=(1-beta)*pure_airfoils_camber[:,1,iairfoil-1]+beta*pure_airfoils_camber[:,1,iairfoil]


# Node chord, elastic axis
node_chord=np.interp(r, Rnodes, Chord)
node_aero_twist=np.interp(r, Rnodes, AeroTwst)
node_elastic_axis=np.ones((n_node_main,))*0.25

######################################################################
######################  ROTOR AERODYNAMICS  ##########################
######################################################################

aero_node=np.ones((n_node,), dtype=bool)
airfoil_distribution=np.zeros((n_elem,n_node_elem), dtype=int)
surface_distribution=np.zeros((n_elem,), dtype=int)

chord=np.zeros((n_elem,n_node_elem))
elastic_axis=np.zeros((n_elem,n_node_elem))
aerodynamic_twist=np.zeros((n_elem,n_node_elem))

for iblade in range(0,numberOfBlades):

    for inode_main in range(0,n_node_main):
        # Compute the global node
        inode = inode_main+iblade*n_node_main

        # Node characteristics
        aero_node[inode]=blade_aero_node[inode_main]

    # Element characteristics
    for ielem_main in range(0,n_elem_main):
        for inode in range(0,n_node_elem):
            ielem=ielem_main+iblade*n_elem_main

            chord[ielem,inode]=node_chord[connectivities[ielem_main, inode]]
            elastic_axis[ielem,inode]=node_elastic_axis[connectivities[ielem_main, inode]]
            aerodynamic_twist[ielem,inode]=node_aero_twist[connectivities[ielem_main, inode]]

            surface_distribution[ielem]=iblade
            airfoil_distribution[ielem,:]=blade_airfoil_distribution[ ielem_main,: ]

surface_m = np.ones((numberOfBlades, ), dtype=int)*m
m_distribution = 'uniform'

######################################################################
######################  GENERATE FILES  ##############################
######################################################################

def clean_test_files():
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    dyn_file_name = route + '/' + case_name + '.dyn.h5'
    if os.path.isfile(dyn_file_name):
        os.remove(dyn_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    solver_file_name = route + '/' + case_name + '.solver.txt'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)


def generate_dyn_file():
    global dt
    global n_tstep
    global route
    global case_name
    global num_elem
    global num_node_elem
    global num_node
    global amplitude
    global period

    dynamic_forces_time = None
    with_dynamic_forces = False
    with_forced_vel = True
    # if with_dynamic_forces:
    #     f1 = 100
    #     dynamic_forces = np.zeros((num_node, 6))
    #     app_node = []
    #     dynamic_forces[app_node, 2] = f1
    #     force_time = np.zeros((n_tstep, ))
    #     limit = round(0.05/dt)
    #     force_time[50:61] = 1
    #
    #     dynamic_forces_time = np.zeros((n_tstep, num_node, 6))
    #     for it in range(n_tstep):
    #         dynamic_forces_time[it, :, :] = force_time[it]*dynamic_forces

    forced_for_vel = None
    if with_forced_vel:
        forced_for_vel = np.zeros((n_tstep, 6))
        forced_for_acc = np.zeros((n_tstep, 6))
        for it in range(n_tstep):
            # forced_for_vel[it, 3:6] = it/n_tstep*angular_velocity
            forced_for_vel[it, 3:6] = angular_velocity

    with h5.File(route + '/' + case_name + '.dyn.h5', 'a') as h5file:
        if with_dynamic_forces:
            h5file.create_dataset(
                'dynamic_forces', data=dynamic_forces_time)
        if with_forced_vel:
            h5file.create_dataset(
                'for_vel', data=forced_for_vel)
            h5file.create_dataset(
                'for_acc', data=forced_for_acc)
        h5file.create_dataset(
            'num_steps', data=n_tstep)

def generate_fem():

    # CHECKING
    if(elem_stiffness.shape[0]!=n_elem):
        sys.exit("ERROR: Element stiffness must be defined for each element")
    if(elem_mass.shape[0]!=n_elem):
        sys.exit("ERROR: Element mass must be defined for each element")
    if(frame_of_reference_delta.shape[0]!=n_elem):
        sys.exit("ERROR: The first dimension of FoR does not match the number of elements")
    if(frame_of_reference_delta.shape[1]!=n_node_elem):
        sys.exit("ERROR: The second dimension of FoR does not match the number of nodes element")
    if(frame_of_reference_delta.shape[2]!=3):
        sys.exit("ERROR: The third dimension of FoR must be 3")
    if(structural_twist.shape[0]!=n_node):
        sys.exit("ERROR: The structural twist must be defined for each node")
    if(boundary_conditions.shape[0]!=n_node):
        sys.exit("ERROR: The boundary conditions must be defined for each node")
    if(beam_number.shape[0]!=n_node):
        sys.exit("ERROR: The beam number must be defined for each node")
    if(app_forces.shape[0]!=n_node):
        sys.exit("ERROR: The first dimension of the applied forces matrix does not match the number of nodes")
    if(app_forces.shape[1]!=6):
        sys.exit("ERROR: The second dimension of the applied forces matrix must be 6")

    # Writting the file
    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
        conectivities = h5file.create_dataset('connectivities', data=connectivities)
        num_nodes_elem_handle = h5file.create_dataset(
            'num_node_elem', data=n_node_elem)
        num_nodes_handle = h5file.create_dataset(
            'num_node', data=n_node)
        num_elem_handle = h5file.create_dataset(
            'num_elem', data=n_elem)
        stiffness_db_handle = h5file.create_dataset(
            'stiffness_db', data=stiffness_matrix_db)
        stiffness_handle = h5file.create_dataset(
            'elem_stiffness', data=elem_stiffness)
        mass_db_handle = h5file.create_dataset(
            'mass_db', data=mass_db)
        mass_handle = h5file.create_dataset(
            'elem_mass', data=elem_mass)
        frame_of_reference_delta_handle = h5file.create_dataset(
            'frame_of_reference_delta', data=frame_of_reference_delta)
        structural_twist_handle = h5file.create_dataset(
            'structural_twist', data=structural_twist)
        bocos_handle = h5file.create_dataset(
            'boundary_conditions', data=boundary_conditions)
        beam_handle = h5file.create_dataset(
            'beam_number', data=beam_number)
        app_forces_handle = h5file.create_dataset(
            'app_forces', data=app_forces)
        # lumped_mass_nodes_handle = h5file.create_dataset(
        #     'lumped_mass_nodes', data=lumped_mass_nodes)
        # lumped_mass_handle = h5file.create_dataset(
        #     'lumped_mass', data=lumped_mass)
        # lumped_mass_inertia_handle = h5file.create_dataset(
        #     'lumped_mass_inertia', data=lumped_mass_inertia)
        # lumped_mass_position_handle = h5file.create_dataset(
        #     'lumped_mass_position', data=lumped_mass_position)

def generate_aero_file():

    # CHECKING
    if(aero_node.shape[0]!=n_node):
        sys.exit("ERROR: Aero node must be defined for each node")
    if(airfoil_distribution.shape[0]!=n_elem or airfoil_distribution.shape[1]!=n_node_elem):
            sys.exit("ERROR: Airfoil distribution must be defined for each element/local node")
    if(chord.shape[0]!=n_elem):
        sys.exit("ERROR: The first dimension of the chord matrix does not match the number of elements")
    if(chord.shape[1]!=n_node_elem):
        sys.exit("ERROR: The second dimension of the chord matrix does not match the number of nodes per element")
    if(elastic_axis.shape[0]!=n_elem):
        sys.exit("ERROR: The first dimension of the elastic axis matrix does not match the number of elements")
    if(elastic_axis.shape[1]!=n_node_elem):
        sys.exit("ERROR: The second dimension of the elastic axis matrix does not match the number of nodes per element")
    if(surface_distribution.shape[0]!=n_elem):
        sys.exit("ERROR: The surface distribution must be defined for each element")
    if(aerodynamic_twist.shape[0]!=n_elem):
        sys.exit("ERROR: The first dimension of the aerodynamic twist does not match the number of elements")
    if(aerodynamic_twist.shape[1]!=n_node_elem):
        sys.exit("ERROR: The second dimension of the aerodynamic twist does not match the number nodes per element")


    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add the airfoils
        for inode in range(0,n_node_main):
            airfoils_group.create_dataset("%d" % inode, data=blade_airfoil[:,:,inode])

        # chord
        chord_input = h5file.create_dataset('chord', data=chord)
        dim_attr = chord_input .attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=aerodynamic_twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
        surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

        #control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
        #control_surface_deflection_input = h5file.create_dataset('control_surface_deflection', data=control_surface_deflection)
        #control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
        #control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)

def generate_solver_file():
    file_name = route + '/' + case_name + '.solver.txt'
    settings = dict()
    aux_settings = dict()
    settings['SHARPy'] = {'case': case_name,
                          'route': route,
                          'flow': flow,
                          'write_screen': 'on',
                          'write_log': 'on',
                          'log_folder': route + '/output/',
                          'log_file': case_name + '.log'}

    # AUX DICTIONARIES
    aux_settings['velocity_field_input'] = {'u_inf': WSP,
                                        'u_inf_direction': [1., 0, 0]}

    # LOADERS

    settings['BeamLoader'] = {'unsteady': 'on',
                              'orientation': algebra.euler2quat(np.array([0.0,
                                                                          alpha,
                                                                          beta]))}

    settings['AerogridLoader'] = {'unsteady': 'on',
                                  'aligned_grid': 'on',
                                  'mstar': mstar,
                                  'freestream_dir': ['1', '0', '0']}

    # POSTPROCESS

    settings['AerogridPlot'] = {'folder': route + '/output/',
                                'include_rbm': 'on',
                                'include_forward_motion': 'off',
                                'include_applied_forces': 'on',
                                'minus_m_star': 0,
                                'u_inf': WSP,
                                'dt': dt}

    #settings['AeroForcesCalculator'] = {'folder': route + '/output/forces',
                                        #'write_text_file': 'on',
                                        #'text_file_name': case_name + '_aeroforces.csv',
                                        #'screen_output': 'on',
                                        #'unsteady': 'off'}

    settings['BeamPlot'] = {'folder': route + '/output/',
                            'include_rbm': 'on',
                            'include_applied_forces': 'on',
                            'include_forward_motion': 'on'}

    #settings['BeamCsvOutput'] = {'folder': route + '/output/',
                                 #'output_pos': 'on',
                                 #'output_psi': 'on',
                                 #'screen_output': 'off'}

    settings['BeamLoads'] = {}

    settings['WriteVariablesTime'] = {'delimiter': ' ',
                                      'structure_variables': ['AFoR_steady_forces', 'AFoR_unsteady_forces','AFoR_position', 'AFoR_velocity'],
                                      'structure_nodes': [4],
                                      'aero_panels_variables': ['gamma', 'norm_gamma', 'norm_gamma_star'],
                                      'aero_panels_isurf': [0],
                                      'aero_panels_im': [0],
                                      'aero_panels_in': [4],
                                      'aero_nodes_variables': ['GFoR_steady_force', 'GFoR_unsteady_force'],
                                      'aero_nodes_isurf': [0],
                                      'aero_nodes_im': [0],
                                      'aero_nodes_in': [4]}

    # STATIC COUPLED

    settings['NonLinearStatic'] = {'print_info': 'on',
                                   'max_iterations': 150,
                                   'num_load_steps': 1,
                                   'delta_curved': 1e-15,
                                   'min_delta': 1e-8,
                                   'gravity_on': gravity,
                                   'gravity': 9.81}

    settings['StaticUvlm'] = {'print_info': 'on',
                              'horseshoe': 'off',
                              'num_cores': 4,
                              'n_rollup': 0,
                              'rollup_dt': dt,
                              'rollup_aic_refresh': 1,
                              'rollup_tolerance': 1e-4,
                              'velocity_field_generator': 'SteadyVelocityField',
                              'velocity_field_input': aux_settings['velocity_field_input'],
                              'rho': 0.0}

    settings['StaticCoupled'] = {'print_info': 'on',
                                 'structural_solver': 'NonLinearStatic',
                                 'structural_solver_settings': settings['NonLinearStatic'],
                                 'aero_solver': 'StaticUvlm',
                                 'aero_solver_settings': settings['StaticUvlm'],
                                 'max_iter': 100,
                                 'n_load_steps': 4,
                                 'tolerance': 1e-8,
                                 'relaxation_factor': 0}

    # DYNAMIC PRESCRIBED COUPLED

    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'on',
                                               'max_iterations': 95000,
                                               'delta_curved': 1e-9,
                                               'min_delta': 1e-6,
                                               'newmark_damp': 1e-1,
                                               'gravity_on': gravity,
                                               'gravity': 9.81,
                                               'num_steps': n_tstep,
                                               'dt': dt}

    settings['StepUvlm'] = {'print_info': 'on',
                            'horseshoe': 'off',
                            'num_cores': 4,
                            'n_rollup': 0,
                            'convection_scheme': 2,
                            'rollup_dt': dt,
                            'rollup_aic_refresh': 1,
                            'rollup_tolerance': 1e-4,
                            'velocity_field_generator': 'SteadyVelocityField',
                            'velocity_field_input': aux_settings['velocity_field_input'],
                            'rho': airDensity,
                            'n_time_steps': n_tstep,
                            'dt': dt}

    settings['DynamicPrescribedCoupled'] = {'structural_solver': 'NonLinearDynamicPrescribedStep',
                                            'structural_solver_settings': settings['NonLinearDynamicPrescribedStep'],
                                            'aero_solver': 'StepUvlm',
                                            'aero_solver_settings': settings['StepUvlm'],
                                            'fsi_substeps': 20000,
                                            'fsi_tolerance': 1e-15,
                                            'fsi_vel_tolerance': 1e-12,
                                            'relaxation_factor': 0,
                                            'minimum_steps': 1,
                                            'relaxation_steps': 150,
                                            'final_relaxation_factor': 0.0,
                                            'n_time_steps': n_tstep,
                                            'dt': dt,
                                            'include_unsteady_force_contribution': 'off',
                                            'print_info': 'on',
                                            'postprocessors': ['WriteVariablesTime'],
                                            'postprocessors_settings': {'WriteVariablesTime': settings['WriteVariablesTime']}}


    # STEADY HELICOIDAL WAKE
    settings['SteadyHelicoidalWake'] = settings['DynamicPrescribedCoupled']


    import configobj
    config = configobj.ConfigObj()
    config.filename = file_name
    for k, v in settings.items():
        config[k] = v
    config.write()



clean_test_files()
generate_dyn_file()
generate_fem()
generate_aero_file()
generate_solver_file()

print("DONE")
