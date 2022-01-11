import numpy as np
import copy
import json
import sharpy.utils.solver_interface as solver_interface
import sharpy.linear.utils.ss_interface as ss_interface
import sharpy.utils.rom_interface as rom_interface
import sharpy.rom.balanced as balanced


# solvers = copy.deepcopy(solver_interface.dictionary_of_solvers())     
# linear_solvers = copy.deepcopy(ss_interface.dictionary_of_systems())
# roms = copy.deepcopy(rom_interface.dictionary_of_solvers())

def print_solver_names():
    """
    Print list with all solvers and postprocessors
    """
    
    for k in solvers.keys():
        print(k)
    
def print_solver(solver, indent=4):
    """
    Print solver variables description and default values
    """

    solver1 = solver_interface.solver_from_string(solver)
    dic1 = dict()
    for k in solvers[solver].keys():
        try:
            dic1[k] = solver1.settings_description[k] +'->%s'%solvers[solver][k]
        except KeyError:
            print('No settings_description for %s'%k)
    print(json.dumps(dic1, indent=indent,sort_keys=True))

def print_linear_names():
    """
    Print list of linear solutions of LinearAssembler solver
    """

    for k in linear_solvers.keys():
        print(k)
    
def print_linear(system, indent=4):
    """
    Print variables description and default values of the linear solution
    """

    linear1 = ss_interface.sys_from_string(system)
    dic1 = dict()
    for k in linear_solvers[system].keys():
        dic1[k] = linear1.settings_description[k]+'->%s'%linear_solvers[system][k]
    print(json.dumps(dic1, indent=indent,sort_keys=True))
    
def print_rom_names():
    """
    Print list of Reduced Order models for the linear solution
    """

    for k in roms.keys():
        print(k)
    
def print_rom(rom, indent=4):
    """
    Print variables description and default values of the rom solvers
    """

    rom1 = rom_interface.rom_from_string(rom)
    dic1 = dict()
    for k in roms[rom].keys():
        dic1[k] = rom1.settings_description[k]+'->%s'%roms[rom][k]
    if rom == 'Balanced':
        list_balanced = []
        ki = 1
        for k in balanced.dict_of_balancing_roms.keys():
            dic2 = dict()
            
            dic2['0%salgorithm'%ki] = k 
            for k2,v in balanced.dict_of_balancing_roms[k].settings_description.items():
                dic2[k2] = v + '->%s'% balanced.dict_of_balancing_roms[k].settings_default[k2]
            list_balanced.append(dic2)
            ki += 1

        print(json.dumps(dic1, indent=indent,sort_keys=True))
        for di in list_balanced:
            print(json.dumps(di, indent=indent,sort_keys=True))
    else:
        print(json.dumps(dic1, indent=indent,sort_keys=True))
        
def check_solvers_defaults():
    solvers = copy.deepcopy(solver_interface.dictionary_of_solvers())
    dic1 = dict()
    for s in solvers.keys():
        solver1 = solver_interface.solver_from_string(s)
        for k in solvers[s].keys():
            try:
                dic1[k] = solver1.settings_description[k], solvers[s][k]
            except KeyError as err:
                print(s)
                print(err)
                print('\n')

def update_dic(dic, dic_new):
    """ 
    Updates dic with dic_new in a recursive manner 
    """
    
    for k, v in dic_new.items():
        if not isinstance(v,dict):
                dic.update({k:v})
        else:
            if k in dic.keys():
                if isinstance(dic[k],dict):
                    update_dic(dic[k],v)
                else:
                    dic[k] = v
            else:
                dic[k] = v
    return dic

    
def sol_0(aero=1,
          panels_wake=10,
          flow=[],
          **settings):
    """
    Solution to plot the reference configuration
    """

    settings_new = dict()
    if flow == []:
        if aero: 
            flow = ['BeamLoader', 'AerogridLoader',
                    'AerogridPlot', 'BeamPlot']
        else:
            flow = ['BeamLoader','BeamPlot']

    for k in flow:
        settings_new[k] = {}
            
    settings_new['BeamLoader']['unsteady'] = 'off'
    settings_new['AerogridLoader']['mstar'] = panels_wake
    settings_new['AerogridLoader']['wake_shape_generator'] = 'StraightWake'
    settings_new['AerogridLoader']['wake_shape_generator_input'] = \
                                  {'u_inf': 0.,
                                   'u_inf_direction': [1.,0.,0.],
                                   'dt': 0.1}
    settings_new['BeamPlot']['include_rbm'] = 'off'
    settings_new['AerogridPlot']['include_rbm'] = 'off'
    settings_new['AerogridPlot']['include_applied_forces'] = 'off'
    
    settings_new = update_dic(settings_new, settings) 
    return flow, settings_new

def write_sharpy(case_route, case_name, settings, flow):

    import configobj    
    config = configobj.ConfigObj()
    file_name = case_route + '/' + case_name + '.sharpy'
    config.filename = file_name
    settings['SHARPy'] = {'case':case_name,
                          'route': case_route,
                          'flow': flow,
                          'write_screen': 'on',
                          'write_log': 'on',
                          'log_folder': case_route,
                          'log_file': case_name + '.log',
                          'save_settings': 'on'}
    
    for k, v in settings.items():
        config[k] = v
    config.write()
