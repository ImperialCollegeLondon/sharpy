import numpy as np
import copy
import json
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.rom_interface as rom_interface
from cases.models_generator.gen_utils import update_dic

def print_solver_defaults(solver, indent=4):
    solvers = copy.deepcopy(solver_interface.dictionary_of_solvers())
    print(json.dumps(solvers[solver], indent=indent))

def print_solver_var(solver, indent=4):
    solver1 = solver_interface.solver_from_string(solver)
    print(json.dumps(solver1.settings_description, indent=indent))

def print_solver_names():
    solvers = copy.deepcopy(solver_interface.dictionary_of_solvers())
    for k in solvers.keys():
        print(k)
    
def print_solver(solver, indent=4):
    solvers = copy.deepcopy(solver_interface.dictionary_of_solvers())
    solver1 = solver_interface.solver_from_string(solver)
    dic1 = dict()
    for k in solvers[solver].keys():
        dic1[k] = solver1.settings_description[k], solvers[solver][k]
    print(json.dumps(dic1, indent=indent,sort_keys=True))

def print_rom_defaults(rom, indent=4):
    roms = copy.deepcopy(rom_interface.dictionary_of_solvers())
    print(json.dumps(roms[rom], indent=indent))

def print_rom_var(rom, indent=4):
    rom1 = rom_interface.rom_from_string(rom)
    print(json.dumps(rom1.settings_description, indent=indent))

def print_rom_names():
    roms = copy.deepcopy(rom_interface.dictionary_of_solvers())
    for k in roms.keys():
        print(k)
    
def print_rom(rom, indent=4):
    roms = copy.deepcopy(rom_interface.dictionary_of_solvers())
    rom1 = rom_interface.rom_from_string(rom)
    dic1 = dict()
    for k in roms[rom].keys():
        dic1[k] = rom1.settings_description[k], roms[rom][k]
    print(json.dumps(dic1, indent=indent,sort_keys=True))

    
def sol_0(panels_wake,
            flow=[],
            **settings):
    """
    Solution to plot the reference configuration
    """

    settings_new = dict()
    if flow == []:
        flow = ['BeamLoader', 'AerogridLoader',
                'AerogridPlot', 'BeamPlot']
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}        
    settings_new['BeamLoader']['usteady'] = 'off'
    settings_new['AerogridLoader'] = {
        'unsteady': 'off',
        'aligned_grid': 'on',
        'mstar': 1,
        'freestream_dir': [1.,0.,0.],
        'wake_shape_generator': 'StraightWake',
        'wake_shape_generator_input': {'u_inf': 0.,
                                       'u_inf_direction': [1.,0.,0.],
                                       'dt': 0.1}}
    settings_new['AerogridPlot'] = {'folder':'./runs',
                                    'include_rbm': 'off',
                                    'include_applied_forces': 'off',
                                    'minus_m_star': 0},
    settings_new['BeamPlot'] = {'folder': './runs',
                                'include_rbm': 'off'}
    
    settings_new = update_dic(settings_new, settings)        
    return flow, settings_new
