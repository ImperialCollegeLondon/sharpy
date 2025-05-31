from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.cout_utils as cout

@solver
class _BaseStructural(BaseSolver):
    """
    Structural solver used for the dynamic simulation of free-flying structures.

    This solver provides an interface to the structural library (``xbeam``) and updates the structural parameters
    for every time step of the simulation.

    This solver is called as part of a standalone structural simulation.

    """
    solver_id = '_BaseStructural'
    solver_classification = 'structural'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Print output to screen'

    settings_types['max_iterations'] = 'int'
    settings_default['max_iterations'] = 100
    settings_description['max_iterations'] = 'Sets maximum number of iterations'

    settings_types['num_load_steps'] = 'int'
    settings_default['num_load_steps'] = 1

    settings_types['delta_curved'] = 'float'
    settings_default['delta_curved'] = 1e-2

    settings_types['min_delta'] = 'float'
    settings_default['min_delta'] = 1e-5
    settings_description['min_delta'] = 'Structural solver relative tolerance'

    settings_types['abs_threshold'] = 'float'
    settings_default['abs_threshold'] = 1e-13
    settings_description['abs_threshold'] = 'Structural solver absolute tolerance'

    settings_types['newmark_damp'] = 'float'
    settings_default['newmark_damp'] = 1e-4
    settings_description['newmark_damp'] = 'Sets the Newmark damping coefficient'

    settings_types['gravity_on'] = 'bool'
    settings_default['gravity_on'] = False
    settings_description['gravity_on'] = 'Flag to include gravitational forces'

    settings_types['gravity'] = 'float'
    settings_default['gravity'] = 9.81
    settings_description['gravity'] = 'Gravitational acceleration'

    settings_types['gravity_dir'] = 'list(float)'
    settings_default['gravity_dir'] = [0., 0., 1.]
    settings_description['gravity_dir'] = 'Direction in G where gravity applies'

    settings_types['relaxation_factor'] = 'float'
    settings_default['relaxation_factor'] = 0.3

    settings_types['dt'] = 'float'
    settings_default['dt'] = 0.01
    settings_description['dt'] = 'Time step increment'

    settings_types['num_steps'] = 'int'
    settings_default['num_steps'] = 500

    def initialise(self, data, restart=False):
        pass

    def run(self, **kwargs):
        pass
