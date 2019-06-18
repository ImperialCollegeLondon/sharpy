from sharpy.linear.utils.ss_interface import BaseElement, linear_system
import numpy as np
import sharpy.linear.src.lin_aeroelastic as lin_aeroelastic
import sharpy.linear.assembler.linearuvlm as linearuvlm
import sharpy.linear.assembler.linearbeam as linearbeam
import sharpy.linear.src.libss as libss

@linear_system
class LinearAeroelastic(BaseElement):
    sys_id = 'LinearAeroelastic'

    def __init__(self):

        self.data = None
        self.sys = None  # The actual object
        self.ss = None  # The state space object
        self.lsys = None

        self.settings = dict()
        self.state_variables = None
        self.couplings = None

    def initialise(self, data):
        self.data = data

        try:
            self.settings = self.data.settings['LinearAssembler'][self.sys_id]
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
        damping_aero = np.zeros_like(beam.sys.Cstr)
        stiff_aero[:flex_nodes, :flex_nodes] = self.sys.Kss

        # Add if motion is not clamped
        rigid_dof = 0
        if beam.sys.Kstr.shape != self.sys.Kss.shape:
            rigid_dof = beam.sys.Kstr.shape[0]-self.sys.Kss.shape[0]
            stiff_aero[flex_nodes:, :flex_nodes] = self.sys.Krs

            damping_aero[:flex_nodes, flex_nodes:] = self.sys.Csr
            damping_aero[flex_nodes:, flex_nodes:] = self.sys.Crr
            damping_aero[flex_nodes:, :flex_nodes] = self.sys.Crs


        Ksa = self.sys.Kforces[:beam.sys.num_dof, :]  # maps aerodynamic grid forces to nodal forces

        # Map the nodal displacement and velocities onto the grid displacements and velocities
        Kas = np.block([[self.sys.Kdisp[:, :beam.sys.num_dof], self.sys.Kdisp_vel[:, :beam.sys.num_dof]],
                        [self.sys.Kvel_disp[:, :beam.sys.num_dof], self.sys.Kvel_vel[:, :beam.sys.num_dof]]])
        beam.sys.Cstr += damping_aero
        beam.sys.Kstr += stiff_aero

        beam.assemble()
        # TODO: gust inputs and how do these inputs play with the coupling
        # Idea: add gain only to certain inputs, because if we add zeros it will lose the inputs. The zeros matrix
        # for the gust inputs will appear in the coupling matrices

        uvlm.ss.addGain(Ksa, where='out')
        uvlm.ss.addGain(Kas, where='in')  # Won't work with the gusts.


        # TODO: Modal projection

        # Couple systems
        Tas = np.eye(beam.ss.inputs, uvlm.ss.outputs)
        Tsa = np.eye(uvlm.ss.inputs, beam.ss.outputs)

        self.ss = libss.couple(ss01=uvlm.ss, ss02=beam.ss, K12=Tsa, K21=Tas)
        self.aero_states = uvlm.ss.states
        self.beam_states = beam.ss.states
        self.lsys = {'LinearUVLM': uvlm,
                     'LinearBeam': beam}
        self.couplings = {'Ksa': Ksa,
                          'Kas': Kas}

        return self.data
