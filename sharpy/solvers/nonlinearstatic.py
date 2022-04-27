import numpy as np

import sharpy.structure.utils.xbeamlib as xbeamlib
import sharpy.utils.cout_utils as cout
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver, solver_from_string
import sharpy.utils.algebra as algebra

_BaseStructural = solver_from_string('_BaseStructural')

@solver
class NonLinearStatic(_BaseStructural):
    """
    Structural solver used for the static simulation of free-flying structures.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every k-th step of the FSI iteration.

    This solver can be called as part of a standalone structural simulation or as the structural solver of a coupled
    static aeroelastic simulation.

    """
    solver_id = 'NonLinearStatic'
    solver_classification = 'structural'

    # settings list
    settings_types = _BaseStructural.settings_types.copy()
    settings_default = _BaseStructural.settings_default.copy()
    settings_description = _BaseStructural.settings_description.copy()

    # additional setting for load ramping - disabled by default, only enable with structural only simulations
    settings_types['load_ramping'] = 'bool'
    settings_default['load_ramping'] = False
    settings_description['load_ramping'] = 'Flag to enable load ramping'
    
    # additional parameter for load ramping - only used if load ramping enabled to decide ramp steps
    # initial value does not matter -> value will be set and modified at runtime
    settings_types['load_ramping_conv'] = 'bool'
    settings_default['load_ramping_conv'] = False
    settings_description['load_ramping_conv'] = 'Flag for convergence and subsequent ramping'
    # end of load ramping edit

    settings_types['initial_position'] = 'list(float)'
    settings_default['initial_position'] = np.array([0.0, 0.0, 0.0])

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

    def run(self):
        # print(self.settings)
        # load ramping edits - unchanged if flag = False for compatibility with FSI simulations
        if not self.settings['load_ramping']:
            self.data.structure.timestep_info[self.data.ts].for_pos[0:3] = self.settings['initial_position']
            xbeamlib.cbeam3_solv_nlnstatic(self.data.structure, self.settings, self.data.ts)
            self.extract_resultants()
            return self.data
        else:
            self.data.ts = 0
            self.settings['load_ramping_conv'] = 0
            load_ramping_marker = 0
            while not self.settings['load_ramping_conv']:
                
                if load_ramping_marker >= 5:
                    print('Maximum load step doubling reached (5 times). Check your case to determine cause. Exiting...')
                    break

                load_ramping_marker += 1
                
                print('Solution running with %d load step(s). ' % self.settings['num_load_steps'])
                
                for i_step in range(self.settings['num_load_steps'] + 1):
#                   print(i_step)
                    if (i_step == self.settings['num_load_steps'] and
                    self.settings['num_load_steps'] > 0):
                        break
                
                
                    # load step coefficient
                    if not self.settings['num_load_steps'] == 0:
                        load_step_multiplier = (i_step + 1.0)/self.settings['num_load_steps']
                    else:
                        load_step_multiplier = 1.0

                

                
                    # new storage every load step
                    if i_step > 0:
                        self.data.ts += 1
                        self.next_step()

                    

                    n_node, _ = self.data.structure.timestep_info[self.data.ts].pos.shape
                    struct_forces = np.full([n_node,6], 0)
                    # print(struct_forces)

                    # copy force in beam
                    old_g = self.settings['gravity']
                    self.settings['gravity'] = old_g*load_step_multiplier

#                   print(self.settings['gravity'])
#                   print(load_step_multiplier)

                    temp1 = load_step_multiplier*(struct_forces + self.data.structure.ini_info.steady_applied_forces)
                    self.data.structure.timestep_info[self.data.ts].steady_applied_forces[:] = temp1
                    # run beam
                    self.data.structure.timestep_info[self.data.ts].for_pos[0:3] = self.settings['initial_position']

#                   print('Before solver: '+str(self.settings['load_ramping_conv']))
#                   print('Before solver: '+str(self.data.structure.timestep_info[self.data.ts].pos))
                    xbeamlib.cbeam3_solv_nlnstatic(self.data.structure, self.settings, self.data.ts)
#                   print('After solver: '+str(self.settings['load_ramping_conv']))

#                   print('After solver: '+str(self.data.structure.timestep_info[self.data.ts].pos))
                    self.extract_resultants()
                    
                
                    self.settings['gravity'] = old_g
#                     (self.data.structure.timestep_info[self.data.ts].total_forces[0:3],
#                         self.data.structure.timestep_info[self.data.ts].total_forces[3:6]) = (
#                             self.extract_resultants(self.data.structure.timestep_info[self.data.ts]))


                    # convergence

                if self.settings['load_ramping_conv']:
                    self.cleanup_timestep_info()
                    print('Converged. Exiting...')
                elif not self.settings['load_ramping_conv']:        
                    print('Solution did not converge with %d load step(s). Doubling number of load steps...' % self.settings['num_load_steps'])
                    self.settings['num_load_steps'] *= 2  
                    self.del_timestep_info()
                    self.run()
                else:
                    self.settings['load_ramping_conv'] = True
                    self.cleanup_timestep_info()
                    print('Unknown error 14239, exiting...')
            return self.data





    def next_step(self):
        self.data.structure.next_step()

    def extract_resultants(self, tstep=None):
        if tstep is None:
            tstep = self.data.structure.timestep_info[self.data.ts]
        applied_forces = self.data.structure.nodal_b_for_2_a_for(tstep.steady_applied_forces,
                                                                 tstep)

        applied_forces_copy = applied_forces.copy()
        gravity_forces_copy = tstep.gravity_forces.copy()
        for i_node in range(self.data.structure.num_node):
            applied_forces_copy[i_node, 3:6] += algebra.cross3(tstep.pos[i_node, :],
                                                               applied_forces_copy[i_node, 0:3])
            gravity_forces_copy[i_node, 3:6] += algebra.cross3(tstep.pos[i_node, :],
                                                               gravity_forces_copy[i_node, 0:3])

        totals = np.sum(applied_forces_copy + gravity_forces_copy, axis=0)
        return totals[0:3], totals[3:6]

    def update(self, tstep=None):
        self.create_q_vector(tstep)

    def create_q_vector(self, tstep=None):
        import sharpy.structure.utils.xbeamlib as xb
        if tstep is None:
            tstep = self.data.structure.timestep_info[-1]

        xb.xbeam_solv_disp2state(self.data.structure, tstep)
                      
    def cleanup_timestep_info(self):
        if len(self.data.structure.timestep_info) > 1:
            # copy last info to first
            self.data.structure.timestep_info[0] = self.data.structure.timestep_info[-1].copy()
            # delete all the rest
            while len(self.data.structure.timestep_info) - 1:
                del self.data.structure.timestep_info[-1]
        self.data.ts = 0
        
    def del_timestep_info(self):
        if len(self.data.structure.timestep_info) >= 1:
            # delete all the rest
            while len(self.data.structure.timestep_info)-1 :
                del self.data.structure.timestep_info[-1]
        self.data.ts = 0
        self.data.structure.timestep_info[self.data.ts].for_pos[0:3] = self.settings['initial_position']



