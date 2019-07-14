from sharpy.linear.utils.ss_interface import BaseElement, linear_system
import numpy as np
import sharpy.linear.src.lin_aeroelastic as lin_aeroelastic
import sharpy.linear.assembler.linearuvlm as linearuvlm
import sharpy.linear.assembler.linearbeam as linearbeam
import sharpy.linear.src.libss as libss
import matplotlib.pyplot as plt

@linear_system
class LinearAeroelastic(BaseElement):
    sys_id = 'LinearAeroelastic'

    def __init__(self):

        self.sys = None  # The actual object
        self.ss = None  # The state space object
        self.lsys = dict()  # Contains the underlying objects
        self.uvlm = None
        self.beam = None

        self.settings = dict()
        self.state_variables = None
        self.couplings = None

    def initialise(self, data):

        try:
            self.settings = data.settings['LinearAssembler'][self.sys_id]
        except KeyError:
            self.settings = None

        self.sys = lin_aeroelastic.LinAeroEla(data, custom_settings_linear=self.settings)

        # Initialise aerodynamic
        # Settings
        try:
            uvlm_settings = self.settings['aero_settings']
        except KeyError:
            uvlm_settings = None

        # Create Linear UVLM
        uvlm = linearuvlm.LinearUVLM()
        uvlm.initialise(data, custom_settings=uvlm_settings)
        uvlm.assemble()
        self.uvlm = uvlm

        # Beam settings
        try:
            beam_settings = self.settings['beam_settings']
        except KeyError:
            beam_settings = None

        # Create beam
        beam = linearbeam.LinearBeam()
        beam.initialise(data, custom_settings=beam_settings)
        self.beam = beam

    def assemble(self):

        uvlm = self.uvlm
        beam = self.beam

        # Linearisation of the aerodynamic forces introduces stiffenning and damping terms into the beam matrices
        flex_nodes = self.sys.num_dof_flex
        rig_nodes = self.sys.num_dof_rig
        self.sys.get_gebm2uvlm_gains()

        stiff_aero = np.zeros_like(beam.sys.Kstr)
        damping_aero = np.zeros_like(beam.sys.Cstr)
        stiff_aero[:flex_nodes, :flex_nodes] = self.sys.Kss

        if beam.sys.Kstr.shape != self.sys.Kss.shape:
            rigid_dof = beam.sys.Kstr.shape[0]-self.sys.Kss.shape[0]
            stiff_aero[flex_nodes:, :flex_nodes] = self.sys.Krs

            damping_aero[:flex_nodes, flex_nodes:] = self.sys.Csr
            damping_aero[flex_nodes:, flex_nodes:] = self.sys.Crr
            damping_aero[flex_nodes:, :flex_nodes] = self.sys.Crs

        beam.sys.Cstr += damping_aero
        beam.sys.Kstr += stiff_aero

        beam.assemble()

        # Coupling matrices
        Ksa = self.sys.Kforces[:beam.sys.num_dof, :]  # maps aerodynamic grid forces to nodal forces

        # Map the nodal displacement and velocities onto the grid displacements and velocities
        Kas = np.zeros((uvlm.ss.inputs, beam.ss.outputs + (uvlm.ss.inputs - 2*self.sys.Kdisp.shape[0])))
        Kas[:2*self.sys.Kdisp.shape[0], :beam.ss.outputs] = np.block([[self.sys.Kdisp[:, :beam.sys.num_dof], self.sys.Kdisp_vel[:, :beam.sys.num_dof]],
                        [self.sys.Kvel_disp[:, :beam.sys.num_dof], self.sys.Kvel_vel[:, :beam.sys.num_dof]]])

        # Retain other inputs
        Kas[2*self.sys.Kdisp.shape[0]:, beam.ss.outputs:] = np.eye(uvlm.ss.inputs - 2 * self.sys.Kdisp.shape[0])

        uvlm.ss.addGain(Ksa, where='out')
        uvlm.ss.addGain(Kas, where='in')

        Tas = np.eye(uvlm.ss.inputs, beam.ss.outputs)
        Tsa = np.eye(beam.ss.inputs, uvlm.ss.outputs)

        # TODO: Modal projection - try with ``in_out_coords= nodes``. Might work without further adjustments
        self.ss = libss.couple(ss01=uvlm.ss, ss02=beam.ss, K12=Tas, K21=Tsa)
        # self.aero_states = uvlm.ss.states
        # self.beam_states = beam.ss.states
        self.couplings = {'Ksa': Ksa,
                          'Kas': Kas}

        # TODO
        self.state_variables = {'aero': uvlm.ss.states,
                                'beam': beam.ss.states}






