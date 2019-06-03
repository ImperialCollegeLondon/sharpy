import ctypes as ct
import time

import numpy as np

import sharpy.aero.utils.mapping as mapping
import sharpy.utils.cout_utils as cout
import sharpy.utils.solver_interface as solver_interface
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.structure.utils.xbeamlib as xbeam


@solver
class DynamicCoupled(BaseSolver):
    """
    The ``DynamicCoupled`` solver couples the aerodynamic and structural solvers of choice to march forward in time
    the aeroelastic system's solution.

    Using the ``DynamicCoupled`` solver requires that an instance of the ``StaticCoupled`` solver is called in the
    SHARPy solution ``flow`` when defining the problem case.

    Args:
        data(ProblemData): class containing the data of the problem
        custom_settings (dict): dictionary containing custom settings for the solver to use

    Attributes:
        settings (dict): Contains the solver's ``settings``. See below for acceptable values:

            =======================================  =============  =========================================================================  =========
            Name                                     Type           Description                                                                Default
            =======================================  =============  =========================================================================  =========
            ``print_info``                           ``bool``       Print modal calculations to terminal                                       ``True``
            ``structural_solver``                    ``str``        ``solver_id`` of desired structural solver                                 ``None``
            ``structural_solver_settings``           ``dict``       Dictionary containing the settings for the structural solver               ``None``
            ``aero_solver``                          ``str``        ``solver_id`` of desired aerodynamics solver                               ``None``
            ``aero_solver_settings``                 ``dict``       Dictionary containing the settings for the aerodynamic solver              ``None``
            ``n_time_steps``                         ``int``        Number of timesteps                                                        ``100``
            ``dt``                                   ``float``      Time increment between timesteps                                           ``0.05``
            ``fsi_substeps``                         ``int``        Desc                                                                       ``70``
            ``fsi_tolerance``                        ``float``      Fluid-structure interaction tolerance                                      ``1e-5``
            ``relaxation_factor``                    ``float``      Desc                                                                       ``0.2``
            ``final_relaxation_factor``              ``float``      Desc                                                                       ``0.0``
            ``minimum_steps``                        ``int``        Desc                                                                       ``3``
            ``relaxation_steps``                     ``int``        Desc                                                                       ``100``
            ``dynamic_relaxation``                   ``bool``       Desc                                                                       ``True``
            ``post_processors``                      ``list(str)``  List of ``solver_id`` of desired post-processors to use                    ``None``
            ``post_processor_settings``              ``dict``       Dictionary with post-processor settings                                    ``None``
            ``cleanup_previous_solution``            ``bool``       Remove all the previous timesteps except the last one                      ``True``
            ``include_unsteady_force_contribution``  ``bool``       Include forces that depend on ``gamma_dot``                                ``False``
            ``steps_without_unsteady_force``         ``int``        Time steps without the application of unsteady forces                      ``0``
            ``pseudosteps_ramp_unsteady_force``      ``int``        FSI iterations to progressively ramp the application of unsteady forces    ``0``
            =======================================  =============  =========================================================================  ========

        data (ProblemData): class containing solution information
        structuralsolver (BaseSolver): class of the structural solver
        aero_solver (BaseSolver): class of the aerodynamic solver
        res (float):
        res_dqdt (float):
        res_dqddt (float):
        previous_force:
        dt (float):
        predictor (bool):
        residual_table:
        postprocessors (dict):
        with_postprocessors (bool):


    """
    solver_id = 'DynamicCoupled'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['print_info'] = 'bool'
        self.settings_default['print_info'] = True

        self.settings_types['structural_solver'] = 'str'
        self.settings_default['structural_solver'] = None

        self.settings_types['structural_solver_settings'] = 'dict'
        self.settings_default['structural_solver_settings'] = None

        self.settings_types['aero_solver'] = 'str'
        self.settings_default['aero_solver'] = None

        self.settings_types['aero_solver_settings'] = 'dict'
        self.settings_default['aero_solver_settings'] = None

        self.settings_types['n_time_steps'] = 'int'
        self.settings_default['n_time_steps'] = 100

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.05

        self.settings_types['fsi_substeps'] = 'int'
        self.settings_default['fsi_substeps'] = 70

        self.settings_types['fsi_tolerance'] = 'float'
        self.settings_default['fsi_tolerance'] = 1e-5

        self.settings_types['relaxation_factor'] = 'float'
        self.settings_default['relaxation_factor'] = 0.2

        self.settings_types['final_relaxation_factor'] = 'float'
        self.settings_default['final_relaxation_factor'] = 0.0

        self.settings_types['minimum_steps'] = 'int'
        self.settings_default['minimum_steps'] = 3

        self.settings_types['relaxation_steps'] = 'int'
        self.settings_default['relaxation_steps'] = 100

        self.settings_types['dynamic_relaxation'] = 'bool'
        self.settings_default['dynamic_relaxation'] = True

        self.settings_types['postprocessors'] = 'list(str)'
        self.settings_default['postprocessors'] = list()

        self.settings_types['postprocessors_settings'] = 'dict'
        self.settings_default['postprocessors_settings'] = dict()

        self.settings_types['cleanup_previous_solution'] = 'bool'
        self.settings_default['cleanup_previous_solution'] = False

        self.settings_types['include_unsteady_force_contribution'] = 'bool'
        self.settings_default['include_unsteady_force_contribution'] = False

        self.settings_types['steps_without_unsteady_force'] = 'int'
        self.settings_default['steps_without_unsteady_force'] = 0

        self.settings_types['pseudosteps_ramp_unsteady_force'] = 'int'
        self.settings_default['pseudosteps_ramp_unsteady_force'] = 0

        self.data = None
        self.settings = None
        self.structural_solver = None
        self.aero_solver = None
        self.print_info = False

        self.res = 0.0
        self.res_dqdt = 0.0
        self.res_dqddt = 0.0

        self.previous_force = None

        self.dt = 0.

        self.predictor = False
        self.residual_table = None
        self.postprocessors = dict()
        self.with_postprocessors = False

        self.time_aero = 0.
        self.time_struc = 0.

    def get_g(self):
        return self.structural_solver.settings['gravity'].value

    def set_g(self, new_g):
        self.structural_solver.settings['gravity'] = ct.c_double(new_g)

    def get_rho(self):
        return self.aero_solver.settings['rho'].value

    def set_rho(self, new_rho):
        self.aero_solver.settings['rho'] = ct.c_double(new_rho)

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.dt = self.settings['dt']
        self.print_info = self.settings['print_info']
        if self.settings['cleanup_previous_solution']:
            # if there's data in timestep_info[>0], copy the last one to
            # timestep_info[0] and remove the rest
            self.cleanup_timestep_info()

        self.structural_solver = solver_interface.initialise_solver(self.settings['structural_solver'])
        self.structural_solver.initialise(self.data, self.settings['structural_solver_settings'])
        self.aero_solver = solver_interface.initialise_solver(self.settings['aero_solver'])
        self.aero_solver.initialise(self.structural_solver.data, self.settings['aero_solver_settings'])
        self.data = self.aero_solver.data

        # initialise postprocessors
        self.postprocessors = dict()
        if len(self.settings['postprocessors']) > 0:
            self.with_postprocessors = True
        for postproc in self.settings['postprocessors']:
            self.postprocessors[postproc] = solver_interface.initialise_solver(postproc)
            self.postprocessors[postproc].initialise(
                self.data, self.settings['postprocessors_settings'][postproc])

        # print information header
        if self.print_info:
            self.residual_table = cout.TablePrinter(8, 12, ['g', 'f', 'g', 'f', 'f', 'f', 'e', 'e'])
            self.residual_table.field_length[0] = 5
            self.residual_table.field_length[1] = 6
            self.residual_table.field_length[2] = 4
            self.residual_table.print_header(['ts', 't', 'iter', 'struc ratio', 'iter time', 'residual vel',
                                              'FoR_vel(x)', 'FoR_vel(z)'])


    def cleanup_timestep_info(self):
        if max(len(self.data.aero.timestep_info), len(self.data.structure.timestep_info)) > 1:
            # copy last info to first
            self.data.aero.timestep_info[0] = self.data.aero.timestep_info[-1]
            self.data.structure.timestep_info[0] = self.data.structure.timestep_info[-1]
            # delete all the rest
            while len(self.data.aero.timestep_info) - 1:
                del self.data.aero.timestep_info[-1]
            while len(self.data.structure.timestep_info) - 1:
                del self.data.structure.timestep_info[-1]

        self.data.ts = 0

    def run(self):
        # dynamic simulations start at tstep == 1, 0 is reserved for the initial state
        for self.data.ts in range(len(self.data.structure.timestep_info),
                                  self.settings['n_time_steps'].value + len(self.data.structure.timestep_info)):
            initial_time = time.perf_counter()
            structural_kstep = self.data.structure.timestep_info[-1].copy()
            self.time_aero = 0.0
            self.time_struc = 0.0

            k = 0
            for k in range(self.settings['fsi_substeps'].value + 1):
                if k == self.settings['fsi_substeps'].value and not self.settings['fsi_substeps'] == 0:
                    cout.cout_wrap('The FSI solver did not converge!!!')
                    break

                # generate new grid (already rotated)
                aero_kstep = self.data.aero.timestep_info[-1].copy()
                self.aero_solver.update_custom_grid(structural_kstep, aero_kstep)

                # compute unsteady contribution
                force_coeff = 0.0
                unsteady_contribution = False
                if self.settings['include_unsteady_force_contribution'].value:
                    if self.data.ts > self.settings['steps_without_unsteady_force'].value:
                        unsteady_contribution = True
                        if k < self.settings['pseudosteps_ramp_unsteady_force'].value:
                            force_coeff = k/self.settings['pseudosteps_ramp_unsteady_force'].value
                        else:
                            force_coeff = 1.

                # run the solver
                ini_time_aero = time.perf_counter()
                self.data = self.aero_solver.run(aero_kstep,
                                                 structural_kstep,
                                                 convect_wake=True,
                                                 unsteady_contribution=unsteady_contribution)
                self.time_aero += time.perf_counter() - ini_time_aero

                previous_kstep = structural_kstep.copy()
                structural_kstep = self.data.structure.timestep_info[-1].copy()
                # move the aerodynamic surface according the the structural one
                self.aero_solver.update_custom_grid(structural_kstep, aero_kstep)
                self.map_forces(aero_kstep,
                                structural_kstep,
                                force_coeff)

                # check if nan anywhere.
                # if yes, pdb.set_trace()
                if np.isnan(structural_kstep.steady_applied_forces).any():
                    print('NaN found in steady_applied_forces!')
                    import pdb; pdb.set_trace()
                if np.isnan(structural_kstep.unsteady_applied_forces).any():
                    print('NaN found in unsteady_applied_forces!')
                    import pdb; pdb.set_trace()

                # relaxation
                relax_factor = self.relaxation_factor(k)
                relax(self.data.structure,
                      structural_kstep,
                      previous_kstep,
                      relax_factor)

                if k > 0.9*self.settings['fsi_substeps'].value:
                    relax_factor = 0.3
                elif k > 0.8*self.settings['fsi_substeps'].value:
                    relax_factor = 0.8

                # run structural solver
                ini_time_struc = time.perf_counter()
                self.data = self.structural_solver.run(structural_step=structural_kstep)
                self.time_struc += time.perf_counter() - ini_time_struc

                # check convergence
                if self.convergence(k,
                                    structural_kstep,
                                    previous_kstep):
                    # move the aerodynamic surface according the the structural one
                    self.aero_solver.update_custom_grid(structural_kstep, aero_kstep)
                    break

            self.aero_solver.add_step()
            self.data.aero.timestep_info[-1] = aero_kstep.copy()

            self.structural_solver.add_step()
            self.data.structure.timestep_info[-1] = structural_kstep.copy()
            self.data.structure.integrate_position(-1, self.settings['dt'].value)

            final_time = time.perf_counter()

            if self.print_info:
                self.residual_table.print_line([self.data.ts,
                                                self.data.ts*self.dt.value,
                                                k,
                                                self.time_struc/(self.time_aero + self.time_struc),
                                                final_time - initial_time,
                                                np.log10(self.res_dqdt),
                                                structural_kstep.for_vel[0],
                                                structural_kstep.for_vel[2],
                                                np.sum(structural_kstep.steady_applied_forces[:, 0]),
                                                np.sum(structural_kstep.steady_applied_forces[:, 2])])
            self.structural_solver.extract_resultants()
            # run postprocessors
            if self.with_postprocessors:
                for postproc in self.postprocessors:
                    self.data = self.postprocessors[postproc].run(online=True)

        if self.print_info:
            cout.cout_wrap('...Finished', 1)
        return self.data

    def convergence(self, k, tstep, previous_tstep):
        # check for non-convergence
        if not all(np.isfinite(tstep.q)):
            import pdb; pdb.set_trace()
            raise Exception('***Not converged! There is a NaN value in the forces!')

        if not k:
            # save the value of the vectors for normalising later
            self.base_q = np.linalg.norm(tstep.q.copy())
            self.base_dqdt = np.linalg.norm(tstep.dqdt.copy())
            if self.base_dqdt == 0:
                self.base_dqdt = 1.
            return False

        # we don't want this to converge before introducing the gamma_dot forces!
        if self.settings['include_unsteady_force_contribution'].value:
            if k < self.settings['pseudosteps_ramp_unsteady_force'].value:
                return False

        # relative residuals
        self.res = (np.linalg.norm(tstep.q-
                                   previous_tstep.q)/
                    self.base_q)
        self.res_dqdt = (np.linalg.norm(tstep.dqdt-
                                        previous_tstep.dqdt)/
                         self.base_dqdt)

        # convergence
        if k > self.settings['minimum_steps'].value - 1:
            if self.res < self.settings['fsi_tolerance'].value:
                if self.res_dqdt < self.settings['fsi_tolerance'].value:
                    return True

        return False

    def map_forces(self, aero_kstep, structural_kstep, unsteady_forces_coeff=1.0):
        # set all forces to 0
        structural_kstep.steady_applied_forces.fill(0.0)
        structural_kstep.unsteady_applied_forces.fill(0.0)

        # aero forces to structural forces
        struct_forces = mapping.aero2struct_force_mapping(
            aero_kstep.forces,
            self.data.aero.struct2aero_mapping,
            aero_kstep.zeta,
            structural_kstep.pos,
            structural_kstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.connectivities,
            structural_kstep.cag())
        dynamic_struct_forces = unsteady_forces_coeff*mapping.aero2struct_force_mapping(
            aero_kstep.dynamic_forces,
            self.data.aero.struct2aero_mapping,
            aero_kstep.zeta,
            structural_kstep.pos,
            structural_kstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.connectivities,
            structural_kstep.cag())

        # prescribed forces + aero forces
        try:
            structural_kstep.steady_applied_forces = (
                (struct_forces + self.data.structure.ini_info.steady_applied_forces).
                astype(dtype=ct.c_double, order='F', copy=True))
            structural_kstep.unsteady_applied_forces = (
                (dynamic_struct_forces + self.data.structure.dynamic_input[max(self.data.ts - 1, 0)]['dynamic_forces']).
                astype(dtype=ct.c_double, order='F', copy=True))
        except KeyError:
            structural_kstep.steady_applied_forces = (
                (struct_forces + self.data.structure.ini_info.steady_applied_forces).
                astype(dtype=ct.c_double, order='F', copy=True))
            structural_kstep.unsteady_applied_forces = dynamic_struct_forces


    def relaxation_factor(self, k):
        initial = self.settings['relaxation_factor'].value
        if not self.settings['dynamic_relaxation'].value:
            return initial

        final = self.settings['final_relaxation_factor'].value
        if k >= self.settings['relaxation_steps'].value:
            return final

        value = initial + (final - initial)/self.settings['relaxation_steps'].value*k
        return value


def relax(beam, timestep, previous_timestep, coeff):
    timestep.steady_applied_forces[:] = ((1.0 - coeff)*timestep.steady_applied_forces +
            coeff*previous_timestep.steady_applied_forces)
    timestep.unsteady_applied_forces[:] = ((1.0 - coeff)*timestep.unsteady_applied_forces +
            coeff*previous_timestep.unsteady_applied_forces)


def normalise_quaternion(tstep):
    tstep.dqdt[-4:] = algebra.unit_vector(tstep.dqdt[-4:])
    tstep.quat = tstep.dqdt[-4:].astype(dtype=ct.c_double, order='F', copy=True)
