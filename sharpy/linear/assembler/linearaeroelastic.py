from sharpy.linear.utils.ss_interface import BaseElement, linear_system
import numpy as np
import sharpy.linear.src.lin_aeroelastic as lin_aeroelastic
import sharpy.linear.assembler.linearuvlm as linearuvlm
import sharpy.linear.assembler.linearbeam as linearbeam

@linear_system
class LinearAeroelastic(BaseElement):
    sys_id = 'LinearAeroelastic'

    def __init__(self):

        self.data = None
        self.sys = None  # The actual object
        self.ss = None  # The state space object

        self.settings = dict()
        self.state_variables = None

    def initialise(self, data):
        self.data = data

        try:
            self.settings = self.data.settings['LinearSpace'][self.sys_id]
        except KeyError:
            self.settings = None

        self.sys = lin_aeroelastic.LinAeroEla(data, custom_settings_linear=self.settings)

    def assemble(self):

        # Settings
        try:
            uvlm_settings = self.settings['aero_settings']
        except KeyError:
            uvlm_settings = None

        # Create Linear UVLM
        uvlm = linearuvlm.LinearUVLM()
        uvlm.initialise(self.data, custom_settings=uvlm_settings)
        uvlm.assemble()

        # Beam settings
        try:
            beam_settings = self.settings['beam_settings']
        except KeyError:
            beam_settings = None

        # Create beam
        beam = linearbeam.LinearBeam()
        beam.initialise(self.data, custom_settings=beam_settings)

        # Linearisation of the aerodynamic forces introduces stiffenning and damping terms into the beam matrices
        flex_nodes = self.sys.num_dof_flex
        rig_nodes = self.sys.num_dof_rig
        self.sys.get_gebm2uvlm_gains()

        stiff_aero = np.zeros_like(beam.sys.Kstr)
        stiff_aero[:flex_nodes, :flex_nodes] = self.sys.Kss
        stiff_aero[flex_nodes:, :flex_nodes] = self.sys.Krs
        beam.sys.Kstr += stiff_aero

        damping_aero = np.zeros_like(beam.sys.Cstr)
        damping_aero[:flex_nodes, flex_nodes:] = self.sys.Csr
        damping_aero[flex_nodes:, flex_nodes:] = self.sys.Crr
        damping_aero[flex_nodes:, :flex_nodes] = self.sys.Crr
        beam.sys.Cstr += damping_aero

        beam.assemble()

        Ksa = self.sys.Kforces  # maps aerodynamic grid forces to nodal forces

        # Map the nodal displacement and velocities onto the grid displacements and velocities
        Kas = np.block([[self.sys.Kdisp, np.zeros((3*uvlm.sys.Kzeta, beam.sys.num_dof))],
                        [self.sys.Kvel_disp, self.sys.Kvel_vel]])
        # TODO: gust inputs and how do these inputs play with the coupling
        # Idea: add gain only to certain inputs, because if we add zeros it will lose the inputs. The zeros matrix
        # for the gust inputs will appear in the coupling matrices

        uvlm.ss.addGain(Ksa, where='out')
        uvlm.ss.addGain(Kas, where='in')  # Won't work with the gusts.


        # TODO: Modal projection

