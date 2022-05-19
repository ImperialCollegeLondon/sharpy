import numpy as np
import copy
import json
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.generator_interface as generator_interface
import sharpy.linear.utils.ss_interface as ss_interface
import sharpy.utils.rom_interface as rom_interface
import sharpy.rom.balanced as balanced
import sharpy.utils.algebra as algebra
import sharpy.solvers
import sharpy.generators

class FixedDict(object):
        def __init__(self, dictionary):
            self._dictionary = dictionary
        def __setitem__(self, key, item):
                if key not in self._dictionary:
                    raise KeyError("The key {} is not defined.".format(key))
                self._dictionary[key] = item
        def __getitem__(self, key):
            return self._dictionary[key]

class Basic:

    def __init__(self, **kwargs):
        self.settings_new = dict()
        self.flow = []
        self.constants = dict()
        #self.infosol =InfoSolutions()
        
    def get_solver_sett(self, solver_name, **kwargs):
        """
        updates default settings of solver_name with values in kwargs
        """
        
        solver = solver_interface.solver_from_string(solver_name)
        default_settings = solver.settings_default
        new_settings = update_dic(default_settings, kwargs)
        return new_settings
    
    def get_linear_sett(self, system_name, **kwargs):
        """
        updates default settings of linear_system with values in kwargs
        """
            
        system = ss_interface.sys_from_string(system_name)
        default_settings = system.settings_default
        new_settings = update_dic(default_settings, kwargs)
        return new_settings
        
    def set_constants(self, **kwargs):

        self.constants['num_cores'] = kwargs.get('num_cores', 1)
        self.constants['u_inf_direction'] = kwargs.get('u_inf_direction', [1., 0., 0.])
        self.constants['gravity_dir'] = kwargs.get('gravity_dir', [0., 0., 1.])
        self.constants['gravity'] = kwargs.get('gravity', 9.807)
        self.constants['forA'] = kwargs.get('forA', [0., 0., 0.])

    def set_struct_loader(self,
                          unsteady=False,
                          rotationA=[1.0, 0., 0., 0.],
                          **kwargs):
        
        if len(rotationA) == 3:
            rotationA = algebra.euler2quat(np.array(rotationA)) # [roll,alpha,beta]
        self.settings_new['BeamLoader']['for_pos'] = self.constants['forA']
        self.settings_new['BeamLoader']['orientation'] = rotationA 
        self.settings_new['BeamLoader']['unsteady'] = unsteady

    def set_loaders(self,
                    panels_wake,
                    u_inf,
                    dt,
                    rotationA=[1.0, 0, 0, 0],
                    unsteady=False,
                    aligned_grid=True,
                    wake_shape_generator='StraightWake',
                    control_surface_deflection=[],
                    control_surface_deflection_generator_settings={},
                    shift_panels=True,                    
                    **kwargs):

        self.set_struct_loader(unsteady,
                               rotationA,
                               **kwargs)
        
        self.settings_new['AerogridLoader']['freestream_dir'] = \
            self.constants['u_inf_direction']
        self.settings_new['AerogridLoader']['unsteady'] = unsteady
        self.settings_new['AerogridLoader']['mstar'] = panels_wake
        self.settings_new['AerogridLoader']['aligned_grid'] = aligned_grid
        self.settings_new['AerogridLoader']['control_surface_deflection'] = \
            control_surface_deflection
        self.settings_new['AerogridLoader']['control_surface_deflection_generator_settings'] = \
            control_surface_deflection_generator_settings
        self.settings_new['AerogridLoader']['wake_shape_generator'] = wake_shape_generator
        self.settings_new['AerogridLoader']['wake_shape_generator_input'] = \
                                      {'u_inf': u_inf,
                                       'u_inf_direction': self.constants['u_inf_direction'],
                                       'dt': dt}
        self.settings_new['AerogridLoader']["shift_panels"] = shift_panels
        
    def set_flow(self,
                 predefined_flow,
                 new_flow=[],
                 add2_flow=[],
                 **kwargs):
        
        if new_flow == []:
            self.flow = predefined_flow
        else:
            self.flow = new_flow
        for i, fi in enumerate(add2_flow):
            assert fi[0] in self.flow, "%s in add2_flow not in flow variable"%fi[0]
            index = self.flow.index(fi[0])+1            
            if isinstance(fi[1], list):
                count = 0
                for j, fj in enumerate(fi[1]):
                    if fj.lower() == 'plot':
                        self.flow.insert(index + count, 'BeamPlot')
                        self.flow.insert(index + count + 1, 'AerogridPlot')
                        count += 2
                    else:
                        self.flow.insert(index + count, fj)
                        count += 1
            elif fi[1].lower() == 'plot':
                self.flow.insert(index, 'BeamPlot')
                self.flow.insert(index + 1, 'AerogridPlot')
            elif isinstance(fi[1], str):
                self.flow.insert(index, fi[1])
            
        for k in self.flow:
            self.settings_new[k] = {}

    def set_plot(self,
                 u_inf=1.,
                 dt=0.1,
                 **kwargs):
        """
        Set the settings for paraview plotting
        """
        for fi in self.flow:
            if fi == 'AerogridPlot':
                self.set_aerogrid_plot(u_inf,
                                       dt,
                                       **kwargs)
            elif fi == 'BeamPlot':
                self.set_beam_plot(**kwargs)
        
    def set_aerogrid_plot(self,
                          u_inf,
                          dt,
                          name_prefix="",
                          include_forward_motion=False,
                          include_incidence_angle=False,
                          include_rbm=True,
                          include_unsteady_applied_forces=False,
                          include_velocities=False,
                          **kwargs):
        
        self.settings_new['AerogridPlot']['u_inf'] = u_inf
        self.settings_new['AerogridPlot']['dt']    = dt
        self.settings_new['AerogridPlot']['include_forward_motion'] = \
            include_forward_motion
        self.settings_new['AerogridPlot']['include_incidence_angle'] = \
            include_incidence_angle
        self.settings_new['AerogridPlot']['include_rbm'] = include_rbm
        self.settings_new['AerogridPlot']['include_unsteady_applied_forces'] = \
            include_unsteady_applied_forces
        self.settings_new['AerogridPlot']['include_velocities'] = \
            include_velocities
        self.settings_new['AerogridPlot']["name_prefix"] = name_prefix
                
    def set_beam_plot(self,
                      include_FoR=False,
                      include_rbm=True,
                      name_prefix="",
                      output_rbm=False,
                      **kwargs):

        self.settings_new['BeamPlot']['include_FoR'] = include_FoR
        self.settings_new['BeamPlot']['name_prefix'] = name_prefix
        self.settings_new['BeamPlot']['include_rbm'] = include_rbm
        self.settings_new['BeamPlot']['output_rbm'] = output_rbm
        
    def modify_settings(self, flow, **kwargs):

        dic2update = {k:v for k, v in kwargs.items() if k in flow}
        if len(dic2update) > 0:
            self.settings_new = update_dic(self.settings_new, dic2update)
        
    def sol_0(self,
              aero=1,
              u_inf=1.,
              dt=1.,
              panels_wake=10,
              rotationA=[0, 0, 0],
              **kwargs):
        """
        Solution to plot the reference configuration
        """

        self.set_constants(**kwargs)
        if aero:
            predefined_flow = ['BeamLoader', 'AerogridLoader',
                               'BeamPlot', 'AerogridPlot']
            self.set_flow(predefined_flow, **kwargs)
            self.set_loaders(panels_wake,
                             u_inf,
                             dt,
                             rotationA,
                             unsteady=False,
                             **kwargs)
        else:
            predefined_flow = ['BeamLoader','BeamPlot']
            self.set_flow(predefined_flow, **kwargs)
            self.set_struct_loader(False,
                                   rotationA,
                                   **kwargs)
        self.set_plot(**kwargs)
        self.modify_settings(self.flow, **kwargs)
        return self.flow, self.settings_new

    def write_sharpy(self,
                     case_route,
                     case_name,
                     write_screen='on',
                     write_log='on',
                     save_settings='on'):

        import configobj    
        config = configobj.ConfigObj()
        file_name = case_route + '/' + case_name + '.sharpy'
        config.filename = file_name
        self.settings_new['SHARPy'] = {'case':case_name,
                                       'route': case_route,
                                       'flow': self.flow,
                                       'write_screen': write_screen,
                                       'write_log': write_log,
                                       'log_folder': case_route,
                                       'log_file': case_name + '.log',
                                       'save_settings': save_settings}

        for k, v in self.settings_new.items():
            config[k] = v
        config.write()

class InfoSolutions:

    def __init__(self):
        self.solvers = copy.deepcopy(solver_interface.dictionary_of_solvers(print_info=False))     
        self.linear_solvers = copy.deepcopy(ss_interface.dictionary_of_systems())
        self.roms = copy.deepcopy(rom_interface.dictionary_of_solvers())
        self.generators = copy.deepcopy(generator_interface.dictionary_of_generators())

    def print_solver_names(self):
        """
        Print list with all solvers and postprocessors
        """

        for k in self.solvers.keys():
            print(k)

    def print_solver(self, solver, indent=4):
        """
        Print solver variables description and default values
        """

        solver1 = solver_interface.solver_from_string(solver)
        dic1 = dict()
        for k in self.solvers[solver].keys():
            try:
                dic1[k] = solver1.settings_description[k] +'->%s'%self.solvers[solver][k]
            except KeyError:
                print('No settings_description for %s'%k)
        print(json.dumps(dic1, indent=indent,sort_keys=True))

    def print_generator_names(self):
        """
        Print list with all generators
        """

        for k in self.generators.keys():
            print(k)

    def print_generator(self, generator, indent=4):
        """
        Print generator variables description and default values
        """

        generator1 = generator_interface.generator_from_string(generator)
        dic1 = dict()
        for k in self.generators[generator].keys():
            try:
                dic1[k] = (generator1.settings_description[k] +'->%s'
                           % self.generators[generator][k])
            except KeyError:
                print('No settings_description for %s'%k)
        print(json.dumps(dic1, indent=indent,sort_keys=True))


    def print_linear_names(self):
        """
        Print list of linear solutions of LinearAssembler solver
        """

        for k in self.linear_solvers.keys():
            print(k)

    def print_linear(self, system, indent=4):
        """
        Print variables description and default values of the linear solution
        """

        linear1 = ss_interface.sys_from_string(system)
        dic1 = dict()
        for k in self.linear_solvers[system].keys():
            dic1[k] = (linear1.settings_description[k]+'->%s'
                       %self.linear_solvers[system][k])
        print(json.dumps(dic1, indent=indent,sort_keys=True))

    def print_rom_names(self):
        """
        Print list of Reduced Order models for the linear solution
        """

        for k in self.roms.keys():
            print(k)

    def print_rom(self, rom, indent=4):
        """
        Print variables description and default values of the rom solvers
        """

        rom1 = rom_interface.rom_from_string(rom)
        dic1 = dict()
        for k in self.roms[rom].keys():
            dic1[k] = rom1.settings_description[k]+'->%s'% self.roms[rom][k]
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

    def check_solvers_defaults(self):

        dic1 = dict()
        for s in self.solvers.keys():
            solver1 = solver_interface.solver_from_string(s)
            for k in self.solvers[s].keys():
                try:
                    dic1[k] = solver1.settings_description[k], self.solvers[s][k]
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
