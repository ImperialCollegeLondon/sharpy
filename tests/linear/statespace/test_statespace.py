import copy
import unittest

import numpy as np

from sharpy.linear.src import libsparse as libsp
from sharpy.linear.src.libss import StateSpace, SSconv, compare_ss, scale_SS, Gain, random_ss, couple, join, disc2cont, series
from sharpy.linear.utils.ss_interface import LinearVector, InputVariable, StateVariable, OutputVariable


class Test_dlti(unittest.TestCase):
    """ Test methods into this module for DLTI systems """

    def setUp(self):
        # allocate some state-space model (dense and sparse)
        dt = 0.3
        Ny, Nx, Nu = 8, 3, 5
        A = np.random.rand(Nx, Nx)
        B = np.random.rand(Nx, Nu)
        C = np.random.rand(Ny, Nx)
        D = np.random.rand(Ny, Nu)
        self.SS = StateSpace(A, B, C, D, dt=dt)
        self.SSsp = StateSpace(libsp.csc_matrix(A), libsp.csc_matrix(B), C, D, dt=dt)

        self.SS.input_variables = LinearVector([InputVariable('input1', size=3, index=0),
                                                InputVariable('input2', size=2, index=1)])
        self.SS.state_variables = LinearVector([StateVariable('state1', size=3, index=0)])
        self.SS.output_variables = LinearVector([OutputVariable('output1', size=3, index=0),
                                                 OutputVariable('output2', size=5, index=1)])

        self.SSsp.input_variables = self.SS.input_variables
        self.SSsp.output_variables = self.SS.output_variables
        self.SSsp.state_variables = self.SS.state_variables

    def test_SSconv(self):

        SS = self.SS
        SSsp = self.SSsp
        Nu, Nx, Ny = SS.inputs, SS.states, SS.outputs
        A, B, C, D = SS.get_mats()

        # remove predictor: try different scenario
        B1 = np.random.rand(Nx, Nu)
        SSpr0 = StateSpace(*SSconv(A, B, B1, C, D), dt=0.3)
        SSpr1 = StateSpace(*SSconv(A, B, libsp.csc_matrix(B1), C, D), dt=0.3)
        SSpr2 = StateSpace(*SSconv(
            libsp.csc_matrix(A), B, libsp.csc_matrix(B1), C, D), dt=0.3)
        SSpr3 = StateSpace(*SSconv(
            libsp.csc_matrix(A), libsp.csc_matrix(B), B1, C, D), dt=0.3)
        SSpr4 = StateSpace(*SSconv(
            libsp.csc_matrix(A), libsp.csc_matrix(B), libsp.csc_matrix(B1), C, D), dt=0.3)
        compare_ss(SSpr0, SSpr1)
        compare_ss(SSpr0, SSpr2)
        compare_ss(SSpr0, SSpr3)
        compare_ss(SSpr0, SSpr4)

    def test_scale_SS(self):

        SS = self.SS
        SSsp = self.SSsp
        Nu, Nx, Ny = SS.inputs, SS.states, SS.outputs

        # scale (hard-copy)
        insc = np.random.rand(Nu)
        stsc = np.random.rand(Nx)
        outsc = np.random.rand(Ny)
        SSadim = scale_SS(SS, insc, outsc, stsc, byref=False)
        SSadim_sp = scale_SS(SSsp, insc, outsc, stsc, byref=False)
        compare_ss(SSadim, SSadim_sp)

        # scale (by reference)
        SS.scale(insc, outsc, stsc)
        SSsp.scale(insc, outsc, stsc)
        compare_ss(SS, SSsp)

    def test_addGain(self):

        SS = self.SS
        SSsp = self.SSsp
        Nu, Nx, Ny = SS.inputs, SS.states, SS.outputs

        # add gains
        Kin = np.random.rand(Nu, 5)
        Kout = np.random.rand(4, Ny)

        gain_in = Gain(Kin)
        gain_in.input_variables = LinearVector([InputVariable('input1', size=5, index=0)])
        gain_in.output_variables = LinearVector([OutputVariable('output1', size=Nu, index=0)])

        gain_out = Gain(Kout)
        gain_out.input_variables = LinearVector.transform(self.SS.output_variables, InputVariable)
        gain_out.output_variables = LinearVector([OutputVariable('final_output', size=gain_out.outputs, index=0)])

        SS.addGain(gain_in, 'in')
        SS.addGain(gain_out, 'out')
        SSsp.addGain(gain_in, 'in')
        SSsp.addGain(gain_out, 'out')
        compare_ss(SS, SSsp)

    def test_freqresp(self):
        # freq response: try different scenario

        SS = self.SS
        SSsp = self.SSsp
        Nu, Nx, Ny = SS.inputs, SS.states, SS.outputs

        kv = np.linspace(0, 1, 8)
        Y = SS.freqresp(kv)
        Ysp = SSsp.freqresp(kv)
        er = np.max(np.abs(Y - Ysp))
        assert er < 1e-10, 'Test on freqresp failed'

        SS.D = libsp.csc_matrix(SS.D)
        Y1 = SS.freqresp(kv)
        er = np.max(np.abs(Y - Y1))
        assert er < 1e-10, 'Test on freqresp failed'

    def test_couple(self):
        dt = .2
        Nx1, Nu1, Ny1 = 3, 4, 2
        Nx2, Nu2, Ny2 = 4, 3, 2
        K12 = np.random.rand(Nu1, Ny2)
        K21 = np.random.rand(Nu2, Ny1)
        SS1 = random_ss(Nx1, Nu1, Ny1, dt=.2)
        SS2 = random_ss(Nx2, Nu2, Ny2, dt=.2)

        SS1sp = StateSpace(libsp.csc_matrix(SS1.A),
                           libsp.csc_matrix(SS1.B),
                           libsp.csc_matrix(SS1.C),
                           libsp.csc_matrix(SS1.D), dt=dt)
        SS2sp = StateSpace(libsp.csc_matrix(SS2.A),
                           libsp.csc_matrix(SS2.B),
                           libsp.csc_matrix(SS2.C),
                           libsp.csc_matrix(SS2.D), dt=dt)
        K12sp = libsp.csc_matrix(K12)
        K21sp = libsp.csc_matrix(K21)

        # SCref=couple_full(SS1,SS2,K12,K21)
        SC0 = couple(SS1, SS2, K12, K21)
        # compare_ss(SCref,SC0)
        for SSa in [SS1, SS1sp]:
            for SSb in [SS2, SS2sp]:
                for k12 in [K12, K12sp]:
                    for k21 in [K21, K21sp]:
                        SChere = couple(SSa, SSb, k12, k21)
                        compare_ss(SC0, SChere)

    def test_join(self):

        Nx, Nu, Ny = 4, 3, 2
        SS_list = [random_ss(Nx, Nu, Ny, dt=.2) for ii in range(3)]

        wv = [.3, .5, .2]
        SSjoin = join(SS_list, wv)

        kv = np.array([0., 1., 3.])
        Yjoin = SSjoin.freqresp(kv)

        Yref = np.zeros_like(Yjoin)
        for ii in range(3):
            Yref += wv[ii] * SS_list[ii].freqresp(kv)

        er = np.max(np.abs(Yjoin - Yref))
        assert er < 1e-14, 'test_join error %.3e too large' % er

    def test_disc2cont(self):
        # not the best test given that eigenvalue comparison is not great with random systems. (error grows near
        # nyquist frequency)

        # this test is for execution purposes only.
        sys = copy.deepcopy(self.SS)
        self.SS.disc2cont()

        ct_sys = disc2cont(sys)

    def test_remove_inputs(self):
        dt = 0.3
        Ny, Nx, Nu = 4, 3, 10
        A = np.random.rand(Nx, Nx)
        B = np.random.rand(Nx, Nu)
        C = np.random.rand(Ny, Nx)
        D = np.random.rand(Ny, Nu)
        self.SS = StateSpace(A, B, C, D, dt=dt)
        self.SSsp = StateSpace(libsp.csc_matrix(A), libsp.csc_matrix(B), C, D, dt=dt)

        self.SS.input_variables = LinearVector([InputVariable('input1', size=3, index=0),
                                                InputVariable('input2', size=4, index=1),
                                                InputVariable('input3', size=2, index=2),
                                                InputVariable('input4', size=1, index=3)])
        self.SSsp.input_variables = self.SS.input_variables

        rows_loc = self.SS.input_variables.num_variables * [None]
        for ith, variable in enumerate(self.SS.input_variables):
            rows_loc[ith] = variable.rows_loc

        self.SS.remove_inputs('input2', 'input4')

        assert self.SS.B.shape == (Nx, self.SS.input_variables.size), 'B matrix not trimmed correctly'
        assert self.SS.D.shape == (Ny, self.SS.input_variables.size), 'D matrix not trimmed correctly'

        assert self.SS.input_variables[0].rows_loc == rows_loc[0], \
            'Rows of input 1 not retained correctly'
        assert self.SS.input_variables[1].rows_loc == rows_loc[2], \
            'Rows of input 3 not retained correctly'

        # sparse system
        self.SSsp.remove_inputs('input2', 'input4')
        assert self.SSsp.B.shape == (Nx, self.SSsp.input_variables.size), 'Bsp matrix not trimmed correctly'
        assert self.SSsp.D.shape == (Ny, self.SSsp.input_variables.size), 'Dsp matrix not trimmed correctly'

        assert self.SSsp.input_variables[0].rows_loc == rows_loc[0], \
            'Rows of input 1 not retained correctly in sparse system'
        assert self.SSsp.input_variables[1].rows_loc == rows_loc[2], \
            'Rows of input 3 not retained correctly in sparse system'

    def test_series(self):
        """
        Test Series connection

        input11 -- >>> SS2 ----> input1, input2 -----> self.SS ----> y
        """
        Nx2, Nu2, Ny2 = 4, 3, self.SS.inputs
        SS2 = random_ss(Nx2, Nu2, Ny2, dt=self.SS.dt)
        SS2.input_variables = LinearVector([InputVariable('input11', size=3, index=0)])
        SS2.state_variables = LinearVector([StateVariable('state11', size=4, index=0)])
        SS2.output_variables = LinearVector([OutputVariable('input1', size=3, index=0),
                                             OutputVariable('input2', size=2, index=1)])

        SSnew = series(SS2, self.SS)
        state_vars = LinearVector.merge(SS2.state_variables, self.SS.state_variables)

        LinearVector.check_same_vectors(SSnew.state_variables, state_vars)

        assert SSnew.outputs == self.SS.outputs, 'Number of outputs not the same as self.SS'
        assert SSnew.inputs == SS2.inputs, 'Number of inputs not the same as SS2'


class TestStateSpaceManipulation(unittest.TestCase):

    def setUp(self):
        Ny, Nx, Nu = 6, 3, 10
        A = np.random.rand(Nx, Nx)
        B = np.random.rand(Nx, Nu)
        C = np.random.rand(Ny, Nx)
        D = np.random.rand(Ny, Nu)
        self.ss = StateSpace(A, B, C, D)
        self.SSsp = StateSpace(libsp.csc_matrix(A), libsp.csc_matrix(B), C, D)

        self.ss.input_variables = LinearVector([InputVariable('input1', size=3, index=0),
                                                InputVariable('input2', size=4, index=1),
                                                InputVariable('input3', size=2, index=2),
                                                InputVariable('input4', size=1, index=3)])

        self.ss.state_variables = LinearVector([StateVariable('state1', size=Nx, index=0)])
        self.ss.output_variables = LinearVector([OutputVariable('output1', size=1, index=0),
                                                 OutputVariable('output2', size=3, index=1),
                                                 OutputVariable('output3', size=2, index=2)])

    def test_remove_outputs(self):
        self.ss.remove_outputs('output1')  # size 1
        self.ss.remove_outputs('output2')  # size 3

        assert self.ss.outputs == 2, 'Number of outputs not correct'

    def test_retain_channels(self):
        """
        Retain certain input and output channels only
        """
        retain_input_channels = [0, 7]
        self.ss.retain_inout_channels(retain_input_channels, where='in')

        assert self.ss.inputs == len(retain_input_channels), 'Number of inputs not correct'
        assert self.ss.input_variables[0].size == 1, 'input1 not correct size'
        assert self.ss.input_variables[1].name == 'input3', 'input2 not properly removed'

        retain_output_channels = [0, 5]
        self.ss.retain_inout_channels(retain_output_channels, where='out')
        assert self.ss.outputs == len(retain_output_channels), 'Number of outputs not correct'
        assert self.ss.output_variables[0].size == 1, 'output1 not correct size'
        assert self.ss.output_variables[1].name == 'output3', 'output2 not properly removed'
        assert self.ss.output_variables[1].size == 1, 'output3 not properly trimmed'
        assert self.ss.output_variables[1].index == 1, 'output3 not properly indexed'


class TestGains(unittest.TestCase):

    def test_dot(self):
        """
        u ---> K1 ----> K2 ---->y
        Returns:

        """
        m1 = 4
        p1 = 3
        k1 = np.random.rand(p1, m1)
        gain1 = Gain(k1,
                     input_vars=LinearVector([InputVariable('input', size=m1, index=0)]),
                     output_vars=LinearVector([OutputVariable('output1', size=p1, index=0)]))

        m2 = p1
        p2 = 5
        k2 = np.random.rand(p2, m2)
        gain2 = Gain(k2,
                     input_vars=LinearVector([InputVariable('output1', size=m2, index=0)]),
                     output_vars=LinearVector([OutputVariable('output', size=p2, index=0)]))

        gain = gain2.dot(gain1)
        np.testing.assert_array_almost_equal(gain.value, k2.dot(k1))

    @unittest.expectedFailure
    def test_fail_connection(self):
        """
        This one should fail because of dimension mismatch

        u ---> K1 ----> K2 ---->y

        """
        m1 = 4
        p1 = 3
        k1 = np.random.rand(p1, m1)
        gain1 = Gain(k1,
                     input_vars=LinearVector([InputVariable('input', size=m1, index=0)]),
                     output_vars=LinearVector([OutputVariable('output1', size=p1, index=0)]))

        m2 = 2
        p2 = 5
        k2 = np.random.rand(p2, m2)
        gain2 = Gain(k2,
                     input_vars=LinearVector([InputVariable('output1', size=m2, index=0)]),
                     output_vars=LinearVector([OutputVariable('output', size=p2, index=0)]))

        # check the check fails :)
        with self.subTest('ss_interface_check'):
            LinearVector.check_connection(gain1.output_variables, gain2.input_variables)

        with self.subTest('connection_check'):
            gain = gain2.dot(gain1)
            np.testing.assert_array_almost_equal(gain.value, k2.dot(k1))


if __name__ == '__main__':
    unittest.main()
