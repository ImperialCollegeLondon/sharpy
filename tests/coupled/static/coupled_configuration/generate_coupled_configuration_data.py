import h5py as h5
import numpy as np
import configparser
import os

case_name = 'coupled_configuration'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

# flight conditions
u_inf = 25
rho = 0.08891
alpha = 4
beta = 0
c_ref = 1
b_ref = 16

alpha_rad = alpha*np.pi/180

# main geometry data
main_span = 16
main_chord = 1.0
main_ea = 0.5
main_sigma = 1
main_airfoil_P = 0
main_airfoil_M = 0

fuselage_length = 10
fuselage_sigma = 1

tail_span = 2.0
tail_chord = 1
tail_ea = 0.25
tail_sigma = 0.7
tail_airfoil_P = 5
tail_airfoil_M = 5
tail_twist = 0*np.pi/180

fin_span = 1.5
fin_chord = 1
fin_ea = 0.25
fin_sigma = 1.5
fin_airfoil_P = 0
fin_airfoil_M = 0

n_surfaces = 2
force = 0
momenty = 0
momentx = 0

# discretisation data
num_elem_main = 20
num_elem_tail = 5
num_elem_fin = 4
num_elem_fuselage = 10


num_node_elem = 3
num_elem = num_elem_main + num_elem_main# + num_elem_fuselage + 2*num_elem_tail# + num_elem_fin
num_node_main = num_elem_main*(num_node_elem - 1) + 1
num_node_fuselage = num_elem_fuselage*(num_node_elem - 1) + 1
num_node_tail = num_elem_tail*(num_node_elem - 1) + 1
num_node_fin = num_elem_fin*(num_node_elem - 1) + 1

num_node = num_node_main + (num_node_main - 1)
# num_node += num_node_fuselage - 1
# num_node += 2*(num_node_tail - 1)
# num_node += num_node_fin - 1
nodes_distributed = num_node

m_main = 10
m_tail = 5
m_fin = 5


def clean_test_files():
    fem_file_name = route + '/' + case_name + '.fem.h5'
    if os.path.isfile(fem_file_name):
        os.remove(fem_file_name)

    aero_file_name = route + '/' + case_name + '.aero.h5'
    if os.path.isfile(aero_file_name):
        os.remove(aero_file_name)

    solver_file_name = route + '/' + case_name + '.solver.txt'
    if os.path.isfile(solver_file_name):
        os.remove(solver_file_name)

    flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
    if os.path.isfile(flightcon_file_name):
        os.remove(flightcon_file_name)


def generate_fem_file():
    # placeholders
    # coordinates
    global x, y, z
    x = np.zeros((num_node, ))
    y = np.zeros((num_node, ))
    z = np.zeros((num_node, ))
    # struct twist
    structural_twist = np.zeros_like(x)
    # beam number
    beam_number = np.zeros((num_elem, ), dtype=int)
    # frame of reference delta
    frame_of_reference_delta = np.zeros((num_elem, num_node_elem, 3))
    # connectivities
    conn = np.zeros((num_elem, num_node_elem), dtype=int)
    # stiffness
    num_stiffness = 4
    ea = 1e4
    ga = 1e4
    gj = 1e4
    eiy = 2e4
    eiz = 5e4
    sigma = 1
    base_stiffness = sigma*np.diag([ea, ga, ga, gj, eiy, eiz])
    stiffness = np.zeros((num_stiffness, 6, 6))
    stiffness[0, :, :] = main_sigma*base_stiffness
    stiffness[1, :, :] = fuselage_sigma*base_stiffness
    stiffness[2, :, :] = tail_sigma*base_stiffness
    stiffness[3, :, :] = fin_sigma*base_stiffness
    elem_stiffness = np.zeros((num_elem,), dtype=int)
    # mass
    num_mass = 4
    m_base = 0.75
    j_base = 0.1
    base_mass = np.diag([m_base, m_base, m_base, j_base, j_base, j_base])
    mass = np.zeros((num_mass, 6, 6))
    mass[0, :, :] = np.sqrt(main_sigma)*base_mass
    mass[1, :, :] = np.sqrt(fuselage_sigma)*base_mass
    mass[2, :, :] = np.sqrt(tail_sigma)*base_mass
    mass[3, :, :] = np.sqrt(fin_sigma)*base_mass
    elem_mass = np.zeros((num_elem,), dtype=int)
    # boundary conditions
    boundary_conditions = np.zeros((num_node, ), dtype=int)
    boundary_conditions[0] = 1
    # applied forces
    n_app_forces = nodes_distributed
    node_app_forces = np.zeros((n_app_forces,), dtype=int)
    app_forces = np.zeros((n_app_forces, 6))
    # orientation
    inertial2aero = np.zeros((3,3))
    inertial2aero[0, :] = [np.cos(alpha_rad), -np.sin(alpha_rad), 0.0]
    inertial2aero[1, :] = [-np.sin(alpha_rad), np.cos(alpha_rad), 0.0]
    inertial2aero[2, :] = [0.0, 0.0, 1.0]

    # app_forces = np.array([
#  [ 0.13492313660888947       , 5.7880903908409316  ,      65.655317319832761   ,     18.810001202584075   ,     2.4730050479267185E-003,  -5.7960192457458155E-002],
#  [ -4.3848548126468252       , 6.4266761958075396  ,      66.435488029473092   ,     19.039380630449916   ,    0.10937030506823936     ,   1.3518947061181612     ],
#  [ -4.9958347165176571       , 6.9741952192179042  ,      68.531961980614696   ,     19.879063391546580   ,    0.39420608778557120     ,   1.4538930450394916     ],
#  [ -5.3730078431558574       , 7.4928583130004771  ,      70.743743892404765   ,     20.727196528998284   ,    0.79791599119670797     ,   1.4536601085776593     ],
#  [ -5.4978605250849633       , 7.9508206068323179  ,      72.763980787179264   ,     21.456513915238762   ,     1.3534605580962022     ,   1.3302755640548014     ],
#  [ -5.3922334673526251       , 8.2786380052442947  ,      74.366113697389821   ,     21.967328200665357   ,     2.0855919466446515     ,   1.0796292942703101     ],
#  [ -5.0667158520067570       , 8.4852666031693396  ,      75.445338954028344   ,     22.174350862406410   ,     3.0351321961639512     ,  0.68707825639507036     ],
#  [ -4.5488003348592105       , 8.5165436970612607  ,      75.863654249968917   ,     21.961515735617247   ,     4.2110426458677708     ,  0.15175397378121908     ],
#  [ -3.8594384651605336       , 8.4195580536494692  ,      75.648399508822735   ,     21.205560635323877   ,     5.6127710778432842     , -0.53805401937516473     ],
#  [ -3.0244534062439143       , 8.2102314530117493  ,      74.873265654071446   ,     19.805116129274612   ,     7.1380418635826270     ,  -1.3739697793789845     ],
#  [ -2.1127612395646067       , 8.0619244929100642  ,      74.047485367878224   ,     17.762550681178855   ,     8.6252479740059229     ,  -2.3500823640106341     ],
#  [ -1.2045633503551016       , 8.0779705865954305  ,      72.971797183305640   ,     15.207700938878752   ,     9.7424550536704420     ,  -3.4066462302483234     ],
#  [-0.47113318735756687       , 8.3425334724923506  ,      71.159644032002433   ,     12.431490836514032   ,     10.122522828601154     ,  -4.4311295470349128     ],
#  [ -4.6591764645945222E-002  , 8.6150484524876134  ,      66.650187440555996   ,     9.8770608928274761   ,     9.4977843508960973     ,  -5.1697209869466025     ],
#  [  3.8858317798580677E-002  , 8.6408563874714091  ,      60.453990754713686   ,     7.9756985633778799   ,     8.2322684455291579     ,  -5.5266534600332484     ],
#  [ -5.2379793012643533E-002  , 8.2201128501649823  ,      53.031811654333538   ,     6.7719499628658655   ,     6.8732156405059799     ,  -5.4046177507111040     ],
#  [-0.15223206112761467       , 7.6050520179969379  ,      47.100571818520422   ,     6.0406136864578555   ,     5.8596942087939636     ,  -5.1238096265864943     ],
#  [-0.17807004425509840       , 6.9471660965166597  ,      42.093136758087468   ,     5.4981015776638591   ,     5.1255167881342496     ,  -4.6925372226462914     ],
#  [-0.12522344755245746       , 6.1446342674132968  ,      38.600342218618721   ,     4.9230466930197530   ,     4.5936895772917552     ,  -4.3697481260654563     ],
#  [ 0.14935628087318459       , 5.5952603153869704  ,      34.413211795312037   ,     3.9926591702559118   ,     3.8035827318568414     ,  -3.9255051596876211     ],
#  [  4.2663294546934853       , 13.124999919757265  ,      32.625413370302049   ,     1.2479290849561890   ,    0.54163671293285232     ,  -2.5964771547522885     ],
#  [ -3.1486945654875393       , 1.3511670026136726  ,     -5.8018520995138374   ,     2.7463659474131381   ,     3.0230439654051766     , -0.53928694956545908     ],
#  [-0.13781786952821065       , 1.8470308471158665  ,      28.648120623393666   ,     4.9282501070683278   ,     3.4586222423638802     , -0.66375908804539541     ],
#  [ 0.22576977783091073       , 4.3249457013907229  ,      42.889158830061092   ,     6.9772979487914011   ,     6.1939599357146617     ,  -1.7029118045395262     ],
#  [ 0.50253341616215452       , 6.4145181927263497  ,      49.495513386921118   ,     8.3146097433912338   ,     7.8855041086945823     ,  -2.5632593195767388     ],
#  [ 0.77980289997299546       , 7.7202127543868677  ,      53.586786543353924   ,     9.3554514897350778   ,     8.8655213323359945     ,  -3.0580855078853779     ],
#  [  1.0729953291232714       , 8.5372749624278494  ,      56.318627348273559   ,     10.462623639471516   ,     9.5210169285900808     ,  -3.2862712904244402     ],
#  [  1.4494519759196192       , 9.1093897980775083  ,      59.583645370654835   ,     11.869511622990759   ,     10.024809282772319     ,  -3.4180443167673245     ],
#  [  1.9165033455904907       , 9.5380797940770279  ,      62.648670896990723   ,     13.633130622795633   ,     10.226300168280289     ,  -3.4579714010228635     ],
#  [  2.5282272467774440       , 9.8710542887525783  ,      66.254957177722162   ,     15.683006042855204   ,     10.022591948698098     ,  -3.4607380408193080     ],
#  [  3.2209626824133073       , 10.071604153695819  ,      69.254169988538536   ,     17.736614382147973   ,     9.2704334105720303     ,  -3.3972624882800573     ],
#  [  3.9661634201170983       , 10.145044011405513  ,      71.926303176435354   ,     19.521693633931800   ,     8.1136759285473374     ,  -3.3049083645386270     ],
#  [  4.6512563523272377       , 10.074666389623811  ,      73.608924765431496   ,     20.831554384758526   ,     6.7290698510502223     ,  -3.1744757975766222     ],
#  [  5.2470757840273556       , 9.9033280864179662  ,      74.657296835703136   ,     21.664659622202556   ,     5.3691819545840556     ,  -3.0364067772140695     ],
#  [  5.7016474271333903       , 9.6299264265966080  ,      74.929886745670430   ,     22.071461725034759   ,     4.1378903692743521     ,  -2.8834957847873981     ],
#  [  6.0108798582312017       , 9.2871718949203501  ,      74.689769451300435   ,     22.152156860575733   ,     3.1061904071984801     ,  -2.7264418000619521     ],
#  [  6.1509558706756531       , 8.8564663983659582  ,      73.897327917857851   ,     21.960514358926915   ,     2.2569134129399866     ,  -2.5526105587370984     ],
#  [  6.1020227137071883       , 8.3633102260905829  ,      72.680850774528153   ,     21.553093001160171   ,     1.5819752062224324     ,  -2.3569602218165691     ],
#  [  5.8471428601942472       , 7.7930609510385471  ,      71.045975772411310   ,     20.956782347580692   ,     1.0465922641054706     ,  -2.1289812348415045     ],
#  [  5.3668358664255464       , 7.1927073861210200  ,      69.171272720605899   ,     20.233818402721344   ,    0.63375798667635175     ,  -1.8628084355845027     ],
#  [  4.6757073150879220       , 6.5937474082909020  ,      67.324964133570077   ,     19.486537977423701   ,    0.31899793968925244     ,  -1.5621457304124773     ]
# ])
    for i in range(n_app_forces):
        node_app_forces[i] = i
    #     app_forces[i, :] = [0, 0, force, momentx, momenty, 0]
    # lumped masses
    n_lumped_mass = 0
    lumped_mass_nodes = np.array([], dtype=int)
    lumped_mass = np.zeros((n_lumped_mass, ))
    lumped_mass_inertia = np.zeros((n_lumped_mass, 3, 3))
    lumped_mass_position = np.zeros((n_lumped_mass, 3))

    # right wing (beam 0) --------------------------------------------------------------
    working_elem = 0
    working_node = 0
    beam_number[working_elem:working_elem + num_elem_main] = 0
    y[working_node:working_node + num_node_main] = np.linspace(0.0, main_span, num_node_main)
    for ielem in range(num_elem_main):
        for inode in range(num_node_elem):
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(num_elem_main):
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1])
    elem_stiffness[working_elem:working_elem + num_elem_main] = 0
    elem_mass[working_elem:working_elem + num_elem_main] = 0
    boundary_conditions[0] = 1
    boundary_conditions[working_node + num_node_main - 1] = -1
    working_elem += num_elem_main
    working_node += num_node_main

    # left wing (beam 1) --------------------------------------------------------------
    beam_number[working_elem:working_elem + num_elem_main] = 1
    # tempy = np.linspace(0.0, -main_span, num_node_main)
    tempy = np.linspace(-main_span, 0.0, num_node_main)
    y[working_node:working_node + num_node_main - 1] = tempy[0:-1]
    for ielem in range(num_elem_main):
        for inode in range(num_node_elem):
            # frame_of_reference_delta[working_elem + ielem, inode, :] = [1, 0, 0]
            frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # connectivity
    for ielem in range(num_elem_main):
        # conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
        #                                  [0, 2, 1])
        conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
                                         [0, 2, 1]) + 1
    conn[working_elem + num_elem_main - 1, 1] = 0
    # conn[working_elem, 0] = 0
    elem_stiffness[working_elem:working_elem + num_elem_main] = 0
    elem_mass[working_elem:working_elem + num_elem_main] = 0
    # boundary_conditions[working_node + num_node_main - 1 - 1] = -1
    boundary_conditions[working_node] = -1
    # node_app_forces[1] = working_node + num_node_main - 1 - 1
    working_elem += num_elem_main
    working_node += num_node_main - 1

    # fuselage (beam 2) --------------------------------------------------------------
    # beam_number[working_elem:working_elem + num_elem_fuselage] = 2
    # tempx = np.linspace(0.0, fuselage_length, num_node_fuselage)
    # x[working_node:working_node + num_node_fuselage - 1] = tempx[1:]
    # for ielem in range(num_elem_fuselage):
    #     for inode in range(num_node_elem):
    #         frame_of_reference_delta[working_elem + ielem, inode, :] = [0, 1, 0]
    # # connectivity
    # for ielem in range(num_elem_fuselage):
    #     conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
    #                                      [0, 2, 1])
    # conn[working_elem, 0] = 0
    # elem_stiffness[working_elem:working_elem + num_elem_fuselage] = 1
    # elem_mass[working_elem:working_elem + num_elem_fuselage] = 1
    # node_app_forces[2] = working_node + num_node_fuselage - 1 - 1
    # app_forces[2, :] = [0, 0, force, 0, 0, 0]
    # # 60 nodes, 29 elems
    # working_elem += num_elem_fuselage
    # working_node += num_node_fuselage - 1

    # right tail (beam 3) --------------------------------------------------------------
    # beam_number[working_elem:working_elem + num_elem_tail] = 3
    # tempy = np.linspace(0.0, tail_span, num_node_tail)
    # y[working_node:working_node + num_node_tail - 1] = tempy[1:]
    # x[working_node:working_node + num_node_tail - 1] = x[working_node - 1]
    # for ielem in range(num_elem_tail):
    #     for inode in range(num_node_elem):
    #         frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # # connectivity
    # for ielem in range(num_elem_tail):
    #     conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
    #                                      [0, 2, 1])
    # conn[working_elem, 0] = conn[working_elem - 1 - 1, 1]
    # global end_of_fuselage_node
    # end_of_fuselage_node = conn[working_elem - 1 - 1, 1]
    # elem_stiffness[working_elem:working_elem + num_elem_tail] = 2
    # elem_mass[working_elem:working_elem + num_elem_fuselage] = 2
    # boundary_conditions[working_node + num_node_tail - 1 - 1] = -1
    # # 70 nodes, 34 elems
    # working_elem += num_elem_tail
    # working_node += num_node_tail - 1
    #
    # # left tail (beam 4) --------------------------------------------------------------
    # beam_number[working_elem:working_elem + num_elem_tail] = 4
    # tempy = np.linspace(0.0, -tail_span, num_node_tail)
    # y[working_node:working_node + num_node_tail - 1] = tempy[1:]
    # x[working_node:working_node + num_node_tail - 1] = x[working_node - 1]
    # for ielem in range(num_elem_tail):
    #     for inode in range(num_node_elem):
    #         frame_of_reference_delta[working_elem + ielem, inode, :] = [1, 0, 0]
    # # connectivity
    # for ielem in range(num_elem_tail):
    #     conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
    #                                      [0, 2, 1])
    # conn[working_elem, 0] = end_of_fuselage_node
    # elem_stiffness[working_elem:working_elem + num_elem_tail] = 2
    # elem_mass[working_elem:working_elem + num_elem_fuselage] = 2
    # boundary_conditions[working_node + num_node_tail - 1 - 1] = -1
    # # node_app_forces[2] = working_node + num_node_tail - 2
    # # app_forces[2, :] = [0, 0, 0*force, 0, 0, 0]
    # working_elem += num_elem_tail
    # working_node += num_node_tail - 1
    #
    # # fin (beam 5) --------------------------------------------------------------
    # beam_number[working_elem:working_elem + num_elem_fin] = 5
    # tempz = np.linspace(0.0, fin_span, num_node_fin)
    # x[working_node:working_node + num_node_fin - 1] = x[working_node - 1]
    # z[working_node:working_node + num_node_fin - 1] = tempz[1:]
    # for ielem in range(num_elem_fin):
    #     for inode in range(num_node_elem):
    #         frame_of_reference_delta[working_elem + ielem, inode, :] = [-1, 0, 0]
    # # connectivity
    # for ielem in range(num_elem_fin):
    #     conn[working_elem + ielem, :] = ((np.ones((3,))*(working_elem + ielem)*(num_node_elem - 1)) +
    #                                      [0, 2, 1])
    # conn[working_elem, 0] = end_of_fuselage_node
    # elem_stiffness[working_elem:working_elem + num_elem_fin] = 3
    # elem_mass[working_elem:working_elem + num_elem_fin] = 3
    # boundary_conditions[working_node + num_node_fin - 1 - 1] = -1
    # node_app_forces[3] = -1
    # app_forces[3, :] = [force, 0, 0, 0, 0, 0]
    # working_elem += num_elem_fin
    # working_node += num_node_fin - 1

    with h5.File(route + '/' + case_name + '.fem.h5', 'a') as h5file:
        coordinates = h5file.create_dataset('coordinates', data=np.column_stack((x, y, z)))
        conectivities = h5file.create_dataset('connectivities', data=conn)
        num_nodes_elem_handle = h5file.create_dataset(
            'num_node_elem', data=num_node_elem)
        num_nodes_handle = h5file.create_dataset(
            'num_node', data=num_node)
        num_elem_handle = h5file.create_dataset(
            'num_elem', data=num_elem)
        stiffness_db_handle = h5file.create_dataset(
            'stiffness_db', data=stiffness)
        stiffness_handle = h5file.create_dataset(
            'elem_stiffness', data=elem_stiffness)
        mass_db_handle = h5file.create_dataset(
            'mass_db', data=mass)
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
        node_app_forces_handle = h5file.create_dataset(
            'node_app_forces', data=node_app_forces)
        lumped_mass_nodes_handle = h5file.create_dataset(
            'lumped_mass_nodes', data=lumped_mass_nodes)
        lumped_mass_handle = h5file.create_dataset(
            'lumped_mass', data=lumped_mass)
        lumped_mass_inertia_handle = h5file.create_dataset(
            'lumped_mass_inertia', data=lumped_mass_inertia)
        lumped_mass_position_handle = h5file.create_dataset(
            'lumped_mass_position', data=lumped_mass_position)
        orientation_handle = h5file.create_dataset(
            'orientation', data=inertial2aero)


def generate_aero_file():
    global x, y, z
    airfoil_distribution = np.zeros((num_node,), dtype=int)
    surface_distribution = np.zeros((num_elem,), dtype=int) - 1
    surface_m = np.zeros((n_surfaces, ), dtype=int)
    m_distribution = 'uniform'
    aero_node = np.zeros((num_node,), dtype=bool)
    twist = np.zeros((num_node,))
    chord = np.zeros((num_node,))
    elastic_axis = np.zeros((num_node,))

    working_elem = 0
    working_node = 0
    # right wing (surface 0, beam 0)
    i_surf = 0
    airfoil_distribution[working_node:working_node + num_node_main] = 0
    surface_distribution[working_elem:working_elem + num_elem_main] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + num_node_main] = True
    chord[working_node:working_node + num_node_main] = main_chord
    elastic_axis[working_node:working_node + num_node_main] = main_ea
    working_elem += num_elem_main
    working_node += num_node_main

    # left wing (surface 1, beam 1)
    i_surf = 1
    airfoil_distribution[working_node:working_node + num_node_main - 1] = 0
    surface_distribution[working_elem:working_elem + num_elem_main] = i_surf
    surface_m[i_surf] = m_main
    aero_node[working_node:working_node + num_node_main - 1] = True
    chord[working_node:working_node + num_node_main - 1] = main_chord
    elastic_axis[working_node:working_node + num_node_main - 1] = main_ea
    working_elem += num_elem_main
    working_node += num_node_main - 1

    working_elem += num_elem_fuselage
    working_node += num_node_fuselage - 1 - 1

    # # right tail (surface 2, beam 3)
    # i_surf = 2
    # airfoil_distribution[working_node:working_node + num_node_tail] = 1
    # surface_distribution[working_elem:working_elem + num_elem_tail] = i_surf
    # surface_m[i_surf] = m_tail
    # aero_node[working_node:working_node + num_node_tail] = True
    # chord[working_node:working_node + num_node_tail] = tail_chord
    # elastic_axis[working_node:working_node + num_node_main] = tail_ea
    # twist[working_node + 1:working_node + num_node_tail] = tail_twist
    # working_elem += num_elem_tail
    # working_node += num_node_tail
    #
    # # left tail (surface 3, beam 4)
    # i_surf = 3
    # airfoil_distribution[working_node:working_node + num_node_tail] = 1
    # surface_distribution[working_elem:working_elem + num_elem_tail] = i_surf
    # surface_m[i_surf] = m_tail
    # aero_node[working_node:working_node + num_node_tail] = True
    # chord[working_node:working_node + num_node_tail] = tail_chord
    # elastic_axis[working_node:working_node + num_node_main] = tail_ea
    # twist[working_node + 1:working_node + num_node_tail] = -tail_twist
    # working_elem += num_elem_tail
    # working_node += num_node_tail

    # fin (surface 4, beam 5)
    # i_surf = 4
    # airfoil_distribution[working_node:working_node + num_node_fin] = 2
    # surface_distribution[working_elem:working_elem + num_elem_fin] = i_surf
    # surface_m[i_surf] = m_fin
    # aero_node[working_node:working_node + num_node_fin] = True
    # chord[working_node:working_node + num_node_fin] = fin_chord
    # global end_of_fuselage_node
    # twist[end_of_fuselage_node] = 0
    # twist[working_node:] = 0
    # elastic_axis[working_node:working_node + num_node_main] = fin_ea
    # working_elem += num_elem_fin
    # working_node += num_node_fin
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(x, y, s=None, c=surface_distribution)
    # plt.show()

    with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
        airfoils_group = h5file.create_group('airfoils')
        # add one airfoil
        naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
                                generate_naca_camber(P=main_airfoil_P, M=main_airfoil_M)))
        naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
            generate_naca_camber(P=tail_airfoil_P, M=tail_airfoil_M)))
        naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
            generate_naca_camber(P=0, M=0)))

        # chord
        chord_input = h5file.create_dataset('chord', data=chord)
        dim_attr = chord_input .attrs['units'] = 'm'

        # twist
        twist_input = h5file.create_dataset('twist', data=twist)
        dim_attr = twist_input.attrs['units'] = 'rad'

        # airfoil distribution
        airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

        surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
        surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
        m_distribution_input = h5file.create_dataset('m_distribution', data=m_distribution.encode('ascii', 'ignore'))

        aero_node_input = h5file.create_dataset('aero_node', data=aero_node)
        elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)


def generate_naca_camber(M=0, P=0):
    m = M*1e-2
    p = P*1e-1
    def naca(x, m, p):
        if x < 1e-6:
            return 0.0
        elif x < p:
            return m/(p*p)*(2*p*x - x*x)
        elif x > p and x < 1+1e-6:
            return m/((1-p)*(1-p))*(1 - 2*p + 2*p*x - x*x)

    x_vec = np.linspace(0, 1, 1000)
    y_vec = np.array([naca(x, m, p) for x in x_vec])
    return x_vec, y_vec


def generate_solver_file():
    file_name = route + '/' + case_name + '.solver.txt'
    config = configparser.ConfigParser()
    config['SHARPy'] = {'case': case_name,
                        'route': route,
                        'flow': 'StaticCoupled, BeamPlot, AeroGridPlot',
                        # 'flow': 'NonLinearStatic, BeamPlot',
                        # 'flow': 'StaticUvlm, AeroForcesSteadyCalculator, BeamPlot, AeroGridPlot',
                        'plot': 'on'}
    config['StaticCoupled'] = {'print_info': 'on',
                               'structural_solver': 'NonLinearStatic',
                               'aero_solver': 'StaticUvlm',
                               'max_iter': 40,
                               'n_load_steps': 1,
                               'tolerance': 1e-4,
                               'relaxation_factor': 0.,
                               'residual_plot': 'off'}
    config['StaticUvlm'] = {'print_info': 'on',
                            'M_distribution': 'uniform',
                            'Mstar': 1,
                            'rollup': 'off',
                            'aligned_grid': 'on',
                            'prescribed_wake': 'off'}
    config['NonLinearStatic'] = {'print_info': 'on',
                                 'out_b_frame': 'off',
                                 'out_a_frame': 'off',
                                 'elem_proj': 2,
                                 'max_iterations': 999,
                                 'num_load_steps': 25,
                                 'delta_curved': 1e-5,
                                 'min_delta': 1e-4,
                                 'newmark_damp': 0.000,
                                 'gravity_on': 'on',
                                 'gravity': 9.754,
                                 'gravity_dir':  (str(-np.sin(alpha_rad)) +
                                                 ', ' +
                                                 str(0.0) +
                                                 ', ' +
                                                 str(np.cos(alpha_rad)))
                                 }
    config['BeamPlot'] = {'route': './output',
                          'frame': '',
                          'applied_forces': 'on'}
    config['AeroGridPlot'] = {'route': './output'}
    config['AeroForcesSteadyCalculator'] = {'beams': '0, 1'}

    with open(file_name, 'w') as configfile:
        config.write(configfile)


def generate_flightcon_file():
    file_name = route + '/' + case_name + '.flightcon.txt'
    config = configparser.ConfigParser()
    config['FlightCon'] = {'u_inf': u_inf,
                           'alpha': alpha,
                           'beta': beta,
                           'rho_inf': rho,
                           'c_ref': c_ref,
                           'b_ref': b_ref}

    with open(file_name, 'w') as configfile:
        config.write(configfile)


if __name__ == '__main__':
    clean_test_files()
    generate_fem_file()
    generate_solver_file()
    generate_aero_file()
    generate_flightcon_file()








