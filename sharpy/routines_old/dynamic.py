import numpy as np
import copy
import sharpy.routines_old.basic as basic

def sol_400(num_modes,
            forA=[0.,0.,0.],
            beam_orientation=[1., 0, 0, 0],
            rigid_body_modes=False,
            rigid_modes_cg=False,
            use_undamped_modes=True,
            flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in the reference configuration
    """

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader','NoAero','Modal']
    for k in flow:
        settings_new[k] = {}

    settings_new['BeamLoader']['for_pos'] = forA
    settings_new['BeamLoader']['orientation'] = beam_orientation
    settings_new['BeamLoader']['unsteady'] = False
    settings_new['Modal']['NumLambda'] = num_modes
    settings_new['Modal']['rigid_body_modes'] = rigid_body_modes
    settings_new['Modal']['rigid_modes_cg'] = rigid_modes_cg
    settings_new['Modal']['use_undamped_modes'] = use_undamped_modes
    settings_new['Modal']['print_matrices'] = True
    settings_new['Modal']['write_modes_vtk'] = True
    
    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new

def sol_402(flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in a deformed structural or aeroelastic configuration
    """

    settings_new = dict()
    if flow == []:
        flow = []
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}      

    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new

def sol_402(flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in a deformed  configuration after trim analysis
    """

    settings_new = dict()
    if flow == []:
        flow = []
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}      

    settings_new = basic.update_dic(settings_new, settings)        
    return flow, settings_new
