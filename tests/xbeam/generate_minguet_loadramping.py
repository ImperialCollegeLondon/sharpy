"""
Test case of the paper:
Experiments and analysis for composite blades under large deflections. I - Static behavior
Pierre Minguet and John Dugundji
"""

import numpy as np
import os
import pdb
from enum import Enum

import cases.models_generator.gen_main as gm
import sharpy.utils.algebra as algebra
import sharpy.utils.sharpydir as sharpydir

def run(specimen_list, label):
    #specimen_list = 'c090, c450, c2070'

    spe = Enum('specimen', specimen_list)

    if 'c090' in specimen_list:
        spe.c090.E = np.zeros((1, 6, 6))
        spe.c090.E[0][0, 0] = 3.7e6
        spe.c090.E[0][1, 1] = 2.6e5
        spe.c090.E[0][2, 2] = 2.9e5
        spe.c090.E[0][3, 3] = 0.183
        spe.c090.E[0][4, 4] = 0.707
        spe.c090.E[0][5, 5] = 276
        spe.c090.thickness = 1.49e-3

    if 'c450' in specimen_list:
        spe.c450.E = np.zeros((1, 6, 6))
        spe.c450.E[0][0, 0] = 4e6
        spe.c450.E[0][1, 1] = 2.6e5
        spe.c450.E[0][2, 2] = 5.5e5
        spe.c450.E[0][3, 3] = 0.368
        spe.c450.E[0][4, 4] = 0.522
        spe.c450.E[0][5, 5] = 298
        spe.c450.E[0][0, 1] = spe.c450.E[0][1, 0] = 2.7e5
        spe.c450.E[0][3, 4] = spe.c450.E[0][4, 3] = 0.102
        spe.c450.thickness = 1.46e-3

    if 'c2070' in specimen_list:
        spe.c2070.E = np.zeros((1, 6, 6))
        spe.c2070.E[0][0, 0] = 3.9e6
        spe.c2070.E[0][1, 1] = 1.1e5
        spe.c2070.E[0][2, 2] = 1.2e5
        spe.c2070.E[0][3, 3] = 1.18
        spe.c2070.E[0][4, 4] = 0.983
        spe.c2070.E[0][5, 5] = 290
        spe.c2070.E[0][0, 3] = spe.c2070.E[0][3, 0] = 522
        spe.c2070.thickness = 1.92e-3

    #import pdb; pdb.set_trace();

    spe.num_nodes1 = num_nodes
    spe.deltaL = int(510/(spe.num_nodes1-1))*1e-3
    spe.L1 = 560e-3
    num_nodes2 = int(spe.L1/spe.deltaL)+1
    if num_nodes2%2 == 1:
        spe.num_nodes2 = num_nodes2
    else:
        spe.num_nodes2 = num_nodes2 - 1
    #assert spe.num_nodes2%2 == 1, 'even number of nodes'
    spe.L2 = (spe.num_nodes2-1)*spe.deltaL
    spe.width = 30e-3
    spe.density = 1580

    for ci in spe:
        ci.area = ci.thickness*ci.width
        ci.mass = spe.density*ci.area
        ci.Iyy = spe.density*(ci.width*ci.thickness**3/12)
        ci.Izz = spe.density*(ci.thickness*ci.width**3/12)
        ci.J = ci.Iyy + ci.Izz
        ci.M = np.zeros((1, 6, 6))
        ci.M[0, :, :] = np.diag([ci.mass, ci.mass, ci.mass,
                                 ci.J, ci.Iyy, ci.Izz])
    #spe.num_nodes2 = 31
    #spe.L2 = 550e-3
    # load cases
    #lumped_m = [[100e-3], [150e-3], [200e-3], [250e-3], [300e-3], [350e-3], [400e-3]]

    components_sett = dict()
    components_sett['workflow'] = ['create_structure']
    components_sett['geometry'] = {'length': spe.L2,
                                   'num_node':spe.num_nodes2
                                  }
    E = getattr(spe,label).E
    M = getattr(spe,label).M
    components_sett['fem'] = {'stiffness_db':E,
                              'mass_db':M,
                              'frame_of_reference_delta':[-1.,0.,0.],
                              'lumped_mass': lumped_m[0],
                              'lumped_mass_inertia': [np.zeros((3, 3))],
                              'lumped_mass_nodes': [num_nodes-1],
                              'lumped_mass_position': [np.zeros(3)]
                              }

    components_sett = {'beam1': components_sett}

    
    model_sett = {'model_name':'minguet_'+label+'_',
                  'model_route':sharpydir.SharpyDir+ '/tests/xbeam/minguet_'+label,
                  'iterate_type': 'Full_Factorial',
                  'iterate_vars': {'beam1*fem-lumped_mass': lumped_m
                  },
                  'iterate_labels': {'label_type':'number',
                                     'print_name_var':0},
                  'write_iterate_vars':True}

    simulation_sett = {'simulation_input':None,
                       'default_module':'sharpy.routines.static',
                       'default_solution':'sol_101',
                       'default_solution_vars': {'gravity':9.81,
                                                 'gravity_on':True,
                                                 's_maxiter':100,
                                                 's_tolerance':1e-7,
                                                 's_delta_curved':1e-3,
                                                 's_load_steps':5,
                                                 's_relaxation':0.2,
                                                 'l_ramp':True,
                                                 'print_info':True,
                                                 'add2_flow': [['NonLinearStatic',
                                                               ['BeamPlot',
                                                                'WriteVariablesTime']]],
                                                 'modify_settings':{'WriteVariablesTime':
                                                                    {'structure_variables':
                                                                     ['pos', 'psi'],
                                                                     'structure_nodes':list(range(spe.num_nodes2)),
                                                                     'cleanup_old_solution': 'on'}}},
                       'default_sharpy':{},
                       'model_route':None}
    
    simulation_sett = {'sharpy': simulation_sett}

    minguet_model = gm.Model('sharpy', ['sharpy'],
                                 model_dict=model_sett,
                                 components_dict=components_sett,
                                 simulation_dict=simulation_sett)
    minguet_model.run()

def read_minguet():

    #list_materials=['c2070']
    list_materials =  ['c450', 'c090', 'c2070']
    minguet_res = Enum('minguet', list_materials)
    for ci in list_materials:
        d1 = dict(); d1ex = dict()
        for xi in ['u', 'v', 'w']:
            path = sharpydir.SharpyDir + '/tests/xbeam/minguet_data/%s_%s.dat'%(ci, xi)
            pathex = sharpydir.SharpyDir + '/tests/xbeam/minguet_data/%sex_%s.dat'%(ci, xi)
            d1[xi] = np.genfromtxt(path,skip_header=1).T
            d1ex[xi] = np.genfromtxt(pathex,skip_header=1).T
            
        setattr(getattr(minguet_res, ci), 'comp', d1)
        setattr(getattr(minguet_res, ci), 'experiment', d1ex)
    return minguet_res

def read_sharpy(label):

    dir1 = sharpydir.SharpyDir+ '/tests/xbeam/minguet_'+label
    pos_data = []
    for i in range(len(lumped_m)):
        write_vars_dir = (dir1+'/minguet_'+label+'_%s'%i+
                          '/minguet_'+label+'_'+'/WriteVariablesTime')
        pos_data_i = np.atleast_2d(np.genfromtxt(write_vars_dir + '/struct_pos_node%s.dat'%(55), usecols=[1,2,3]))
        pos_data.append(pos_data_i)

    return pos_data

def plot_results(pos, minguet=None, label=None):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    p0 =  pos[0][0]# [0., 510e-3, 0.]#
    len_y = len(lumped_m)
    x = [i[0] for i in lumped_m[0:]]
    y_0 = [abs(pos[i][0][0]- p0[0]) for i in range(0, len_y)]
    y_1 = [abs(pos[i][0][1]- p0[1]) for i in range(0, len_y)]
    y_2 = [abs(pos[i][0][2]- p0[2]) for i in range(0, len_y)]
    ax.plot(x, y_0, label='sharpy (x-component)')
    ax.plot(x, y_1, label='sharpy (y-component)')
    ax.plot(x, y_2, label='sharpy (z-component)')
    if minguet is not None:
        res = getattr(minguet, label)
        ax.plot(1e-3*np.array(res.experiment['u'][0]),
                1e-3*np.array(res.experiment['u'][1]), 'o', label='Minguet experiments')
        ax.plot(1e-3*np.array(res.experiment['v'][0]),
                1e-3*np.array(res.experiment['v'][1]), 'o')
        ax.plot(1e-3*np.array(res.experiment['w'][0]),
                1e-3*np.array(res.experiment['w'][1]), 'o')
        ax.plot(1e-3*np.array(res.comp['u'][0]),
                1e-3*np.array(res.comp['u'][1]), '--', label='Minguet computational')
        ax.plot(1e-3*np.array(res.comp['v'][0]),
                1e-3*np.array(res.comp['v'][1]), '--')
        ax.plot(1e-3*np.array(res.comp['w'][0]),
                1e-3*np.array(res.comp['w'][1]), '--')
    ax.set_xlabel('Tip mass [Kg]')
    ax.set_ylabel('Monitoring-station displacement [m]')
    #plt.ion()
    plt.legend()
    plt.show()

    
num_nodes = 52
#success_loads = [0, 20, 61, 75, 109, 121, 134, 159, 171, 200] # tip loads for analysis (g) 
success_loads = [0, 15., 40., 80., 120., 180., 220., 280., 320., 360., 400.]
#success_loads = [0, 400.]
lumped_m = [[i*1e-3] for i in success_loads]
label=['c450',
       'c090']

pos = dict()
minguet_res = read_minguet()
for label_i in label:
    run(specimen_list = label_i, label=label_i)
    pos[label_i] = read_sharpy(label_i)
    plot_results(pos[label_i], minguet=minguet_res, label=label_i)

success_loads = [0, 15., 40., 80., 120., 180., 220., 280., 320., 360., 400., 435, 470, 500]
#success_loads = [0, 400.]
lumped_m = [[i*1e-3] for i in success_loads]
label=['c2070']

pos = dict()
minguet_res = read_minguet()
for label_i in label:
    run(specimen_list = label_i, label=label_i)
    pos[label_i] = read_sharpy(label_i)
    plot_results(pos[label_i], minguet=minguet_res, label=label_i)

