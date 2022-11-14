import sharpy.utils.settings as settings_utils
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.cout_utils as cout
from sharpy.utils.constants import vortex_radius_def


@solver
class PrescribedUvlm(BaseSolver):
    """
    This class runs a prescribed rigid body motion simulation of a rigid
    aerodynamic body.
    """
    solver_id = 'PrescribedUvlm'
    solver_classification = 'Aero'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write status to screen'

    settings_types['structural_solver'] = 'str'
    settings_default['structural_solver'] = None
    settings_description['structural_solver'] = 'Structural solver to use in the coupled simulation'

    settings_types['structural_solver_settings'] = 'dict'
    settings_default['structural_solver_settings'] = None
    settings_description['structural_solver_settings'] = 'Dictionary of settings for the structural solver'

    settings_types['aero_solver'] = 'str'
    settings_default['aero_solver'] = None
    settings_description['aero_solver'] = 'Aerodynamic solver to use in the coupled simulation'

    settings_types['aero_solver_settings'] = 'dict'
    settings_default['aero_solver_settings'] = None
    settings_description['aero_solver_settings'] = 'Dictionary of settings for the aerodynamic solver'

    settings_types['n_time_steps'] = 'int'
    settings_default['n_time_steps'] = None
    settings_description['n_time_steps'] = 'Number of time steps for the simulation'

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step'

    settings_types['postprocessors'] = 'list(str)'
    settings_default['postprocessors'] = list()
    settings_description['postprocessors'] = 'List of the postprocessors to run at the end of every time step'

    settings_types['postprocessors_settings'] = 'dict'
    settings_default['postprocessors_settings'] = dict()
    settings_description['postprocessors_settings'] = 'Dictionary with the applicable settings for every ``postprocessor``. Every ``postprocessor`` needs its entry, even if empty'

    settings_types['cfl1'] = 'bool'
    settings_default['cfl1'] = True
    settings_description['cfl1'] = 'If it is ``True``, it assumes that the discretisation complies with CFL=1'

    settings_types['vortex_radius'] = 'float'
    settings_default['vortex_radius'] = vortex_radius_def
    settings_description['vortex_radius'] = 'Distance between points below which induction is not computed'

    settings_types['vortex_radius_wake_ind'] = 'float'
    settings_default['vortex_radius_wake_ind'] = vortex_radius_def
    settings_description['vortex_radius_wake_ind'] = 'Distance between points below which induction is not computed in the wake convection'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None

        self.previous_force = None

        self.dt = 0.
        self.postprocessors = dict()
        self.with_postprocessors = False

    def initialise(self, data, restart=False):
        self.data = data
        self.settings = data.settings[self.solver_id]
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt']

        self.aero_solver = solver_interface.initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.data, self.settings['aero_solver_settings'])
        self.data = self.aero_solver.data

        # if there's data in timestep_info[>0], copy the last one to
        # timestep_info[0] and remove the rest
        self.cleanup_timestep_info()

        # initialise postprocessors
        self.postprocessors = dict()
        if len(self.settings['postprocessors']) > 0:
            self.with_postprocessors = True
        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = solver_interface.initialise_solver(postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc], caller=self,
                restart=restart)

        self.residual_table = cout.TablePrinter(2, 14, ['g', 'f'])
        self.residual_table.field_length[0] = 6
        self.residual_table.field_length[1] = 6
        self.residual_table.print_header(['ts', 't'])

    def cleanup_timestep_info(self):
        if len(self.data.aero.timestep_info) > 1:
            # copy last info to first
            self.data.aero.timestep_info[0] = self.data.aero.timestep_info[-1]
            # delete all the rest
            while len(self.data.aero.timestep_info) - 1:
                del self.data.aero.timestep_info[-1]

        self.data.ts = 0

    def increase_ts(self):
        self.data.structure.next_step()
        self.aero_solver.add_step()

    def run(self, **kwargs):
        structural_kstep = self.data.structure.ini_info.copy()

        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        for self.data.ts in range(1, self.settings['n_time_steps'] + 1):
            aero_kstep = self.data.aero.timestep_info[-1].copy()
            structural_kstep = self.data.structure.timestep_info[-1].copy()
            ts = len(self.data.structure.timestep_info) - 1
            if ts > 0:
                self.data.structure.timestep_info[ts].for_vel[:] = self.data.structure.dynamic_input[ts - 1]['for_vel']
                self.data.structure.timestep_info[ts].for_acc[:] = self.data.structure.dynamic_input[ts - 1]['for_acc']


            # # # generate new grid (already rotated)
            # self.aero_solver.update_custom_grid(structural_kstep, aero_kstep)
            #
            # # run the solver
            # self.data = self.aero_solver.run(aero_kstep,
            #                                  structural_kstep,
            #                                  self.data.aero.timestep_info[-1],
            #                                  convect_wake=True)
            #
            # self.residual_table.print_line([self.data.ts,
            #                                 self.data.ts*self.dt])

            self.data.structure.next_step()
            self.data.structure.integrate_position(self.data.ts, self.settings['dt'])

            self.aero_solver.add_step()
            self.data.aero.timestep_info[-1] = aero_kstep.copy()
            self.aero_solver.update_custom_grid(self.data.structure.timestep_info[-1],
                                                self.data.aero.timestep_info[-1])
            # run the solver
            self.data = self.aero_solver.run(self.data.aero.timestep_info[-1],
                                             self.data.structure.timestep_info[-1],
                                             self.data.aero.timestep_info[-2],
                                             convect_wake=True)
            self.residual_table.print_line([self.data.ts,
                                            self.data.ts*self.dt])

            # run postprocessors
            if self.with_postprocessors:
                for postproc in self.postprocessors:
                    self.data = self.postprocessors[postproc].run(online=True)

        return self.data

#
#     def map_forces(self, aero_kstep, structural_kstep, unsteady_forces_coeff=1.0):
#         # set all forces to 0
#         structural_kstep.steady_applied_forces.fill(0.0)
#         structural_kstep.unsteady_applied_forces.fill(0.0)
#
#         # aero forces to structural forces
#         struct_forces = mapping.aero2struct_force_mapping(
#             aero_kstep.forces,
#             self.data.aero.struct2aero_mapping,
#             aero_kstep.zeta,
#             structural_kstep.pos,
#             structural_kstep.psi,
#             self.data.structure.node_master_elem,
#             self.data.structure.master,
#             structural_kstep.cag())
#         dynamic_struct_forces = unsteady_forces_coeff*mapping.aero2struct_force_mapping(
#             aero_kstep.dynamic_forces,
#             self.data.aero.struct2aero_mapping,
#             aero_kstep.zeta,
#             structural_kstep.pos,
#             structural_kstep.psi,
#             self.data.structure.node_master_elem,
#             self.data.structure.master,
#             structural_kstep.cag())
#
#         # prescribed forces + aero forces
#         structural_kstep.steady_applied_forces = (
#             (struct_forces + self.data.structure.ini_info.steady_applied_forces).
#                 astype(dtype=ct.c_double, order='F', copy=True))
#         structural_kstep.unsteady_applied_forces = (
#             (dynamic_struct_forces + self.data.structure.dynamic_input[max(self.data.ts - 1, 0)]['dynamic_forces']).
#                 astype(dtype=ct.c_double, order='F', copy=True))
#
#     def relaxation_factor(self, k):
#         initial = self.settings['relaxation_factor']
#         if not self.settings['dynamic_relaxation']:
#             return initial
#
#         final = self.settings['final_relaxation_factor']
#         if k >= self.settings['relaxation_steps']:
#             return final
#
#         value = initial + (final - initial)/self.settings['relaxation_steps']*k
#         return value
#
#
# def relax(beam, timestep, previous_timestep, coeff):
#     if coeff > 0.0:
#         timestep.steady_applied_forces[:] = ((1.0 - coeff)*timestep.steady_applied_forces
#                                              + coeff*previous_timestep.steady_applied_forces)
#         timestep.unsteady_applied_forces[:] = ((1.0 - coeff)*timestep.unsteady_applied_forces
#                                                + coeff*previous_timestep.unsteady_applied_forces)
#         # timestep.pos_dot[:] = (1.0 - coeff)*timestep.pos_dot + coeff*previous_timestep.pos_dot
#         # timestep.psi[:] = (1.0 - coeff)*timestep.psi + coeff*previous_timestep.psi
#         # timestep.psi_dot[:] = (1.0 - coeff)*timestep.psi_dot + coeff*previous_timestep.psi_dot
#
#         # normalise_quaternion(timestep)
#         # xbeam_solv_state2disp(beam, timestep)
#
#
#
#
#
