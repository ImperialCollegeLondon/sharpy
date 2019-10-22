import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.linear.src.lin_aeroelastic as lin_aeroelastic
import sharpy.linear.src.libss as libss
import scipy.linalg as sclalg
import warnings
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout


@ss_interface.linear_system
class LinearAeroelastic(ss_interface.BaseElement):
    r"""
    Assemble a linearised aeroelastic system

    The aeroelastic system can be seen as the coupling between a linearised aerodynamic system (System 1) and
    a linearised beam system (System 2).

    The coupled system retains inputs and outputs from both systems such that

    .. math:: \mathbf{u} = [\mathbf{u}_1;\, \mathbf{u}_2]

    and the outputs are also ordered in a similar fashion

    .. math:: \mathbf{y} = [\mathbf{y}_1;\, \mathbf{y}_2]

    Reference the individual systems for the particular ordering of the respective input and output variables.
    """
    sys_id = 'LinearAeroelastic'

    settings_default = dict()
    settings_types = dict()
    settings_description = dict()

    settings_types['aero_settings'] = 'dict'
    settings_default['aero_settings'] = None
    settings_description['aero_settings'] = 'Linear UVLM settings'

    settings_types['beam_settings'] = 'dict'
    settings_default['beam_settings'] = None
    settings_description['beam_settings'] = 'Linear Beam settings'

    settings_types['uvlm_filename'] = 'str'
    settings_default['uvlm_filename'] = ''
    settings_description['uvlm_filename'] = 'Path to .data.h5 file containing UVLM/ROM state space to load'

    settings_types['track_body'] = 'bool'
    settings_default['track_body'] = True
    settings_description['track_body'] = 'UVLM inputs and outputs projected to coincide with lattice at linearisation'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.sys = None  # The actual object
        self.ss = None  # The state space object
        self.lsys = dict()  # Contains the underlying objects
        self.uvlm = None
        self.beam = None

        self.load_uvlm_from_file = False

        self.settings = dict()
        self.state_variables = None
        self.couplings = dict()

    def initialise(self, data):

        try:
            self.settings = data.settings['LinearAssembler']['linear_system_settings']
        except KeyError:
            self.settings = None
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.sys = lin_aeroelastic.LinAeroEla(data, custom_settings_linear=self.settings)

        # Create Linear UVLM
        self.uvlm = ss_interface.initialise_system('LinearUVLM')
        self.uvlm.initialise(data, custom_settings=self.settings['aero_settings'])
        if self.settings['uvlm_filename'] == '':
            self.uvlm.assemble()
        else:
            self.load_uvlm_from_file = True

        # Create beam
        self.beam = ss_interface.initialise_system('LinearBeam')
        self.beam.initialise(data, custom_settings=self.settings['beam_settings'])

    def assemble(self):
        r"""
        Assembly of the linearised aeroelastic system.

        The UVLM state-space system has already been assembled. Prior to assembling the beam's first order state-space,
        the damping and stiffness matrices have to be modified to include the damping and stiffenning terms that arise
        from the linearisation of the aeordynamic forces with respect to the A frame of reference. See
        :func:`sharpy.linear.src.lin_aeroela.get_gebm2uvlm_gains()` for details on the linearisation.

        Then the beam is assembled as per the given settings in normalised time if the aerodynamic system has been
        scaled. The discrete time systems of the UVLM and the beam must have the same time step.

        The UVLM inputs and outputs are then projected onto the structural degrees of freedom (obviously with the
        exception of external gusts and control surfaces). Hence, the gains :math:`\mathbf{K}_{sa}` and
        :math:`\mathbf{K}_{as}` are added to the output and input of the UVLM system, respectively. These gains perform
        the following relation:

        .. math:: \begin{bmatrix}\zeta \\ \zeta' \\ u_g \\ \delta \end{bmatrix} = \mathbf{K}_{as}
            \begin{bmatrix} \eta \\ \eta' \\ u_g \\ \delta \end{bmatrix} =

        .. math:: \mathbf{N}_{nodes} = \mathbf{K}_{sa} \mathbf{f}_{vertices}

        If the beam is expressed in modal form, the UVLM is further projected onto the beam's modes to have the
        following input/output structure:


        Returns:

        """
        uvlm = self.uvlm
        beam = self.beam

        # Linearisation of the aerodynamic forces introduces stiffenning and damping terms into the beam matrices
        flex_nodes = self.sys.num_dof_flex
        # rig_nodes = self.sys.num_dof_rig
        self.sys.get_gebm2uvlm_gains()

        stiff_aero = np.zeros_like(beam.sys.Kstr)
        damping_aero = np.zeros_like(beam.sys.Cstr)
        stiff_aero[:flex_nodes, :flex_nodes] = self.sys.Kss

        rigid_dof = beam.sys.Kstr.shape[0] - flex_nodes
        total_dof = flex_nodes + rigid_dof

        if rigid_dof > 0:
            rigid_dof = beam.sys.Kstr.shape[0]-self.sys.Kss.shape[0]
            stiff_aero[flex_nodes:, :flex_nodes] = self.sys.Krs

            damping_aero[:flex_nodes, flex_nodes:] = self.sys.Csr
            damping_aero[flex_nodes:, flex_nodes:] = self.sys.Crr
            damping_aero[flex_nodes:, :flex_nodes] = self.sys.Crs

        beam.sys.Cstr += damping_aero
        beam.sys.Kstr += stiff_aero

        beam.assemble(t_ref=uvlm.sys.ScalingFacts['time'])

        if not self.load_uvlm_from_file:
            # Projecting the UVLM inputs and outputs onto the structural degrees of freedom
            Ksa = self.sys.Kforces[:beam.sys.num_dof, :]  # maps aerodynamic grid forces to nodal forces

            # Map the nodal displacement and velocities onto the grid displacements and velocities
            Kas = np.zeros((uvlm.ss.inputs, 2*beam.sys.num_dof + (uvlm.ss.inputs - 2*self.sys.Kdisp.shape[0])))
            Kas[:2*self.sys.Kdisp.shape[0], :2*beam.sys.num_dof] = np.block([[self.sys.Kdisp[:, :beam.sys.num_dof], self.sys.Kdisp_vel[:, :beam.sys.num_dof]],
                            [self.sys.Kvel_disp[:, :beam.sys.num_dof], self.sys.Kvel_vel[:, :beam.sys.num_dof]]])

            # Retain other inputs
            Kas[2*self.sys.Kdisp.shape[0]:, 2*beam.sys.num_dof:] = np.eye(uvlm.ss.inputs - 2 * self.sys.Kdisp.shape[0])

            # Scaling
            if uvlm.scaled:
                Kas /= uvlm.sys.ScalingFacts['length']

            uvlm.ss.addGain(Ksa, where='out')
            uvlm.ss.addGain(Kas, where='in')

            self.couplings['Ksa'] = Ksa
            self.couplings['Kas'] = Kas

            if self.settings['beam_settings']['modal_projection'].value == True and \
                    self.settings['beam_settings']['inout_coords'] == 'modes':
                # Project UVLM onto modal space
                phi = beam.sys.U
                in_mode_matrix = np.zeros((uvlm.ss.inputs, beam.ss.outputs + (uvlm.ss.inputs - 2*beam.sys.num_dof)))
                in_mode_matrix[:2*beam.sys.num_dof, :2*beam.sys.num_modes] = sclalg.block_diag(phi, phi)
                in_mode_matrix[2*beam.sys.num_dof:, 2*beam.sys.num_modes:] = np.eye(uvlm.ss.inputs - 2*beam.sys.num_dof)
                out_mode_matrix = phi.T

                uvlm.ss.addGain(in_mode_matrix, where='in')
                uvlm.ss.addGain(out_mode_matrix, where='out')

            # Reduce uvlm projected onto structural coordinates
            if uvlm.rom is not None:
                for k, rom in uvlm.rom.items():
                    uvlm.ss = rom.run(uvlm.ss)

        else:
            uvlm.ss = self.load_uvlm(self.settings['uvlm_filename'])

        # Coupling matrices
        Tas = np.eye(uvlm.ss.inputs, beam.ss.outputs)
        Tsa = np.eye(beam.ss.inputs, uvlm.ss.outputs)

        # Scale coupling matrices
        if uvlm.scaled:
            Tsa *= uvlm.sys.ScalingFacts['force'] * uvlm.sys.ScalingFacts['time'] ** 2
        if rigid_dof > 0:
            raise NotImplementedError('Linearisation of problems with rigid body dynamics not yet implemented')
            warnings.warn('Time scaling for problems with rigid body motion under development.')
            Tas[:flex_nodes + 3, :flex_nodes + 3] /= uvlm.sys.ScalingFacts['length']
            Tas[total_dof: total_dof + flex_nodes + 3] /= uvlm.sys.ScalingFacts['length']
        else:
            if not self.settings['beam_settings']['modal_projection'].value:
                Tas /= uvlm.sys.ScalingFacts['length']

        ss = libss.couple(ss01=uvlm.ss, ss02=beam.ss, K12=Tas, K21=Tsa)
        self.couplings['Tas'] = Tas
        self.couplings['Tsa'] = Tsa
        Y_freq = uvlm.ss.freqresp(np.array([0]))[:, :, 0]
        self.state_variables = {'aero': uvlm.ss.states,
                                'beam': beam.ss.states}

        return ss

    def update(self, u_infty):
        """
        Updates the aeroelastic scaled system with the new reference velocity.

        Only the beam equations need updating since the only dependency in the forward flight velocity resides there.

        Args:
              u_infty (float): New reference velocity

        Returns:
            sharpy.linear.src.libss.ss: Updated aeroelastic state-space system

        """
        t_ref = self.uvlm.sys.ScalingFacts['length'] / u_infty

        self.beam.sys.update_matrices_time_scale(t_ref)
        self.beam.sys.assemble()
        self.beam.ss = self.beam.sys.SSdisc

        self.ss = libss.couple(ss01=self.uvlm.ss, ss02=self.beam.ss,
                               K12=self.couplings['Tas'], K21=self.couplings['Tsa'])

        return self.ss

    @staticmethod
    def load_uvlm(filename):
        import sharpy.utils.h5utils as h5
        cout.cout_wrap('Loading UVLM state space system projected onto structural DOFs from file')
        read_data = h5.readh5(filename).ss
        # uvlm_ss_read = read_data.linear.linear_system.uvlm.ss
        uvlm_ss_read = read_data
        return libss.ss(uvlm_ss_read.A, uvlm_ss_read.B, uvlm_ss_read.C, uvlm_ss_read.D, dt=uvlm_ss_read.dt)
