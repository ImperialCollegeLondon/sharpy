import numpy as np
import os
import sys
import pdb
import importlib
import cases.models_generator.gen_main as gm
from cases.models_generator.gen_utils import node2aero, get_chord
import sharpy.utils.algebra as algebra
importlib.reload(gm)

split_path = [sys.path[i].split('/') for i in range(len(sys.path))]
for i in range(len(split_path)):
    if 'sharpy' in split_path[i]:
        ind = i
        break
sharpy_dir = sys.path[ind]
# aeroelasticity parameters
main_ea = 0.33
main_cg = 0.43
sigma = 1

# other
c_ref = 1.8288

ea, ga = 1e9, 1e9
gj = 0.987581e6
eiy = 9.77221e6
eiz = 1e2 * eiy
base_stiffness = np.diag([ea, ga, ga, sigma * gj, sigma * eiy, eiz])
stiffness1 = np.zeros((1, 6, 6))
stiffness1[0] = base_stiffness
m_unit = 35.71
j_tors = 8.64
pos_cg_b = np.array([0., c_ref * (main_cg - main_ea), 0.])
m_chi_cg = algebra.skew(m_unit * pos_cg_b)
mass1 = np.zeros((1, 6, 6))
mass1[0, :, :] = np.diag([m_unit, m_unit, m_unit,
                         j_tors, .1 * j_tors, .9 * j_tors])
mass1[0, :3, 3:] = m_chi_cg
mass1[0, 3:, :3] = -m_chi_cg

stiffness2 = stiffness1*0.9
mass2 = mass1*0.9

chord01 = 6.5
ea01 = 0.4
sweep1 = np.pi/180*20
num_node1 = 7
lenght1 = 6.
dl1 = lenght1/num_node1
ledge01 = np.array([-chord01*ea01,0.,0.])
ledge1 = [np.array([dl1*i*np.sin(sweep1), dl1*i*np.cos(sweep1), 0.]) for i in range(num_node1)]
ledge1 = np.array(ledge1) + ledge01
beam1 = [np.array([dl1*i*np.sin(sweep1), dl1*i*np.cos(sweep1), 0.]) for i in range(num_node1)]
beam1 = np.array(beam1)
tedge1 = [np.array([(1.-ea01)*chord01, dl1*i*np.cos(sweep1), 0.]) for i in range(num_node1)]
tedge1 = np.array(tedge1)
ea1, chord1 = get_chord(ledge1, tedge1, beam1)
ea1, chord1 = node2aero(ea1,chord1)
g1c = dict()

g1c['wing_r1'] = {'workflow':['create_structure', 'create_aero'],
                  'fem': {'stiffness_db':stiffness1,
                          'mass_db':mass1,
                          'coordinates':beam1,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':chord1,
                           'elastic_axis':ea1,
                           'surface_m':8}
                 }
g1c['wing_r2'] = {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':14.,
                               'num_node':15,
                               'direction':None,
                               'node0':[0., 0., 0.],
                               'sweep':sweep1,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness2,
                          'mass_db':mass2,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':[chord1[-1,1],1.6],
                           'elastic_axis':ea1[-1,1],
                           'surface_m':8}
                 }

g1c['wing_l1'] = {'symmetric': {'component':'wing_r1'}}
g1c['wing_l2'] = {'symmetric': {'component':'wing_r2'}}

g1mm = {'model_name':'A320mock',
        'model_route':sharpy_dir+'/cases/models_generator/examples/wing_plantform',
        'iterate_type': 'DoE', # Full_Factorial
        'iterate_vars': {'wing_r2*geometry-sweep':np.linspace(sweep1,sweep1*1.5,2),
                         'wing_r2*geometry-length':np.linspace(14,18.,2)},
        'iterate_labels': {'label_type':'number',
                           'print_name_var':0},
        'assembly': {'include_aero':1,
                     'default_settings': 1, #beam_number and aero surface and
                                            #surface_distribution
                                            #selected by default one
                                            #per component
                     'wing_r1':{'upstream_component':'',
                               'node_in_upstream':0},
                     'wing_r2':{'boundary_connection':0,
                               'upstream_component':'wing_r1',
                                'node_in_upstream':num_node1-1},
                     'wing_l1':{'upstream_component':'wing_r1',
                                'node_in_upstream':0},
                     'wing_l2':{'boundary_connection':0,
                               'upstream_component':'wing_l1',
                                'node_in_upstream':num_node1-1},
                     }
        }

g1sm = {'sharpy': {'simulation_input':None,
                   'default_module':'sharpy.routines.roms',
                   'default_solution':'sol_500',
                   'default_solution_vars': {'panels_wake':8*3,
                                             'num_modes':20,
                                             'rho': 1.225,
                                             'u_inf':1.,
                                             'c_ref':3
                                             },
                   'default_sharpy':{},
                   'model_route':None}}

g1 = gm.Model('sharpy',['sharpy'], model_dict=g1mm, components_dict=g1c,
               simulation_dict=g1sm)
#g1.build()
g1.run()
                                
