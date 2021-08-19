import numpy as np
import os
import importlib
import cases.models_generator.gen_main as gm
import sharpy.utils.algebra as algebra
importlib.reload(gm)
import sys

split_path = [sys.path[i].split('/') for i in [0,1,2]]
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
stiffness = np.zeros((1, 6, 6))
stiffness[0] = base_stiffness
m_unit = 35.71
j_tors = 8.64
pos_cg_b = np.array([0., c_ref * (main_cg - main_ea), 0.])
m_chi_cg = algebra.skew(m_unit * pos_cg_b)
mass = np.zeros((1, 6, 6))
mass[0, :, :] = np.diag([m_unit, m_unit, m_unit,
                         j_tors, .1 * j_tors, .9 * j_tors])
mass[0, :3, 3:] = m_chi_cg
mass[0, 3:, :3] = -m_chi_cg


g1c = dict()
g1c['fuselage_front'] = {'workflow':['create_structure','create_aero0'],
                  'geometry': {'length':14,
                               'num_node':11,
                               'direction':[-1.,0.,0.],
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[0,-1.,0.]}
}
g1c['fuselage_rear'] = {'workflow':['create_structure','create_aero0'],
                  'geometry': {'length':37-14,
                               'num_node':11,
                               'direction':[1.,0.,0.],
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[0,1.,0.]}
}

g1c['wing_r'] = {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':20,
                               'num_node':11,
                               'direction':None,
                               'node0':[0., 0., 0.],
                               'sweep':20.,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':[6,1.6],
                           'elastic_axis':0.33,
                           'surface_m':16}
                 }
g1c['wing_l'] = {'symmetric': {'component':'wing_r'}}
g1c['tail_con'] = {'workflow':['create_structure', 'create_aero0'],
                  'geometry': {'length':2,
                               'num_node':3,
                               'direction':[1.,0.,3.],
                               'node0':[0., 0., 0.]},
                  'fem': {'stiffness_db':10*stiffness,
                          'mass_db':mass/10,
                          'frame_of_reference_delta':[-1, 0., 0.]}
                  }
g1c['vertical_tail'] = {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':6,
                               'num_node':11,
                               'direction':[1,0,5],
                               'node0':[0., 0., 0.],
                               'sweep':0.,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':[3.3,1.5],
                           'elastic_axis':0.5,
                           'surface_m':16}
                 }
g1c['horizontal_tail_right'] = {'workflow':['create_structure', 'create_aero'],
                  'geometry': {'length':6.5,
                               'num_node':11,
                               'direction':None,
                               'node0':[0., 0., 0.],
                               'sweep':20.*np.pi/180,
                               'dihedral':0.},
                  'fem': {'stiffness_db':stiffness,
                          'mass_db':mass,
                          'frame_of_reference_delta':[-1, 0., 0.]},
                  'aero': {'chord':[3.3,1.25],
                           'elastic_axis':0.4,
                           'surface_m':16}
                 }
g1c['horizontal_tail_left'] = {'symmetric': {'component':'horizontal_tail_right'}}

g1mm = {'model_name':'A320mock',
        'model_route':sharpy_dir+'/cases/models_generator/examples/aircraft0',
        'iterate_type': 'Full_Factorial',
        'iterate_vars': {'wing_r*geometry-sweep':np.pi/180*np.array([0,20,40]),
                         'wing_r*geometry-length':np.linspace(14,18.,3)},
        'iterate_labels': {'label_type':'number',
                           'print_name_var':0},
        'assembly': {'include_aero':1,
                     'default_settings': 1, # beam_number and aero surface and
                                            # surface_distribution
                                            # selected by default one
                                            # per component
                     'fuselage_front':{'node2add':0,
                                 'upstream_component':'',
                                 'node_in_upstream':0},
                     'fuselage_rear':{'node2add':0,
                                 'upstream_component':'fuselage_front',
                                 'node_in_upstream':0},
                     'wing_r':{'keep_aero_node':1,
                               'upstream_component':'fuselage_front',
                               'node_in_upstream':0},
                     'wing_l':{'upstream_component':'fuselage_front',
                               'node_in_upstream':0},
                     'tail_con':{'node2add':0,
                               'upstream_component':'fuselage_rear',
                                 'node_in_upstream':9},
                     'vertical_tail':{'node2add':0,
                               'upstream_component':'tail_con',
                                      'node_in_upstream':2},
                     'horizontal_tail_right':{'node2add':0,
                               'upstream_component':'tail_con',
                                              'node_in_upstream':2},
                     'horizontal_tail_left':{'node2add':0,
                               'upstream_component':'tail_con',
                                      'node_in_upstream':2}          
                     }
        }

g1sm = {'sharpy': {'simulation_input':None,
                   'default_module':'sharpy.routines.basic',
                   'default_solution':'sol_0',
                   'default_solution_vars': {'panels_wake':16*5,
                                             'AerogridPlot':{'folder':'./runs',
                                                             'include_rbm': 'off',
                                                             'include_applied_forces': 'off',
                                                             'minus_m_star': 0},
                                             'BeamPlot' : {'folder': './runs',
                                                           'include_rbm': 'off',
                                                           'include_applied_forces': 'off'}},
                   'default_sharpy':{},
                   'model_route':None}}

g1 = gm.Model('sharpy',['sharpy'], model_dict=g1mm, components_dict=g1c,
               simulation_dict=g1sm)
#g1.build()
g1.run()
                                
