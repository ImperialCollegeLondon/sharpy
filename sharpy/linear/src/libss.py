"""
Linear Time Invariant systems
author: S. Maraniello
date: 15 Sep 2017 (still basement...)

Library of methods to build/manipulate state-space models. The module supports
the sparse arrays types defined in libsparse.

The module includes:

Classes:
- StateSpace: provides a class to build DLTI/LTI systems with full and/or sparse
	matrices and wraps many of the methods in these library. Methods include:
	- freqresp: wraps the freqresp function
	- addGain: adds gains in input/output. This is not a wrapper of addGain, as
	the system matrices are overwritten

Methods for state-space manipulation:
- couple: feedback coupling. Does not support sparsity
- freqresp: calculate frequency response. Supports sparsity.
- series: series connection between systems
- parallel: parallel connection between systems
- SSconv: convert state-space model with predictions and delays
- addGain: add gains to state-space model.
- join2: merge two state-space models into one.
- join: merge a list of state-space models into one.
- sum state-space models and/or gains
- scale_SS: scale state-space model
- simulate: simulates discrete time solution
- Hnorm_from_freq_resp: compute H norm of a frequency response
- adjust_phase: remove discontinuities from a frequency response

Special Models:
- SSderivative: produces DLTI of a numerical derivative scheme
- SSintegr: produces DLTI of an integration scheme
- build_SS_poly: build state-space model with polynomial terms.

Filtering:
- butter

Utilities:
- get_freq_from_eigs: clculate frequency corresponding to eigenvalues

Comments:
- the module supports sparse matrices hence relies on libsparse.

to do:
	- remove unnecessary coupling routines
	- couple function can handle sparse matrices but only outputs dense matrices
		- verify if typical coupled systems are sparse
		- update routine
		- add method to automatically determine whether to use sparse or dense?
"""

import copy
import warnings
import numpy as np
import scipy.signal as scsig
import scipy.linalg as scalg
from sharpy.linear.utils.ss_interface import LinearVector, StateVariable, InputVariable, OutputVariable
import scipy.interpolate as scint
import h5py
import sharpy.utils.h5utils as h5utils

# dependency
import sharpy.linear.src.libsparse as libsp


# ------------------------------------------------------------- Dedicated class

class StateSpace:
    """
    Wrap state-space models allocation into a single class and support both
    full and sparse matrices. The class emulates
        scipy.signal.ltisys.StateSpaceContinuous
        scipy.signal.ltisys.StateSpaceDiscrete
    but supports sparse matrices and other functionalities.

    Methods:
    - get_mats: return matrices as tuple
    - check_types: check matrices types are supported
    - freqresp: calculate frequency response over range.
    - addGain: project inputs/outputs
    - scale: allows scaling a system
    """

    def __init__(self, A, B, C, D, dt=None):
        """
        Allocate state-space model (A,B,C,D). If dt is not passed, a
        continuous-time system is assumed.
        """

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt
        self.check_types()

        # vector variable tracking
        self._input_variables = None  # type: LinearVector
        self._state_variables = None
        self._output_variables = None

        # verify dimensions
        assert self.A.shape == (self.states, self.states), 'A and B rows not matching'
        assert self.C.shape[1] == self.states, 'A and C columns not matching'
        assert self.D.shape[0] == self.outputs, 'C and D rows not matching'
        try:
            assert self.D.shape[1] == self.inputs, 'B and D columns not matching'
        except IndexError:
            assert self.inputs == 1, 'D shape does not match number of inputs'

    @property
    def inputs(self):
        """Number of inputs :math:`m` to the system."""
        if self.B.shape.__len__() == 1:
            return 1
        else:
            return self.B.shape[1]

    @property
    def outputs(self):
        """Number of outputs :math:`p` of the system."""
        return self.C.shape[0]

    @property
    def states(self):
        """Number of states :math:`n` of the system."""
        return self.A.shape[0]

    @property
    def input_variables(self):
        return self._input_variables

    @input_variables.setter
    def input_variables(self, variables):
        if variables.variable_class is not InputVariable:
            raise TypeError('LinearVector does not include InputVariable s')
        if variables.size != self.inputs:
            raise IndexError('Size of LinearVector of InputVariable s ({:g}) is not the same as the number of '
                             'inputs in the '
                             'system ({:g})'.format(variables.size, self.inputs))
        self._input_variables = variables

    @property
    def output_variables(self):
        return self._output_variables

    @output_variables.setter
    def output_variables(self, variables):
        if variables.variable_class is not OutputVariable:
            raise TypeError('LinearVector does not include OutputVariable s')
        if variables.size != self.outputs:
            raise IndexError('Size of LinearVector of OutputVariable s ({:g}) is not the same as the number of '
                             'outputs in the '
                             'system ({:g})'.format(variables.size, self.outputs))
        self._output_variables = variables

    @property
    def state_variables(self):
        return self._state_variables

    @state_variables.setter
    def state_variables(self, variables):
        if variables.variable_class is not StateVariable:
            raise TypeError('LinearVector does not include StateVariable s')
        if variables.size != self.states:
            raise IndexError('Size of LinearVector of StateVariable s ({:g}) is not the same as the number '
                             'of states in the '
                             'system ({:g})'.format(variables.size, self.states))
        self._state_variables = variables

    def initialise_variables(self, *variable_tuple, var_type='in'):
        if var_type == 'in' or var_type == 'input':
            var_class = InputVariable
        elif var_type == 'out' or var_type == 'output':
            var_class = OutputVariable
        elif var_type == 'state':
            var_class = StateVariable
        else:
            raise TypeError('Unknown variable type')

        list_of_variables = []
        for ith, var_dict in enumerate(variable_tuple):
            list_of_variables.append(var_class(name=var_dict['name'],
                                               size=var_dict['size'],
                                               index=var_dict.get('index', ith)))

        if var_type == 'in' or var_type == 'input':
            self._input_variables = LinearVector(list_of_variables)
        elif var_type == 'out' or var_type == 'output':
            self._output_variables = LinearVector(list_of_variables)
        elif var_type == 'state':
            self._state_variables = LinearVector(list_of_variables)

    def __repr__(self):
        str_out = ''
        str_out += 'State-space object\n'
        str_out += 'States: {:g}\n'.format(self.states)
        str_out += 'Inputs: {:g}\n'.format(self.inputs)
        str_out += 'Outputs: {:g}\n'.format(self.outputs)
        if self.dt is not None:
            str_out += 'dt: {:g}'.format(self.dt)

        if self.input_variables is not None:
            str_out += '\nInput Variables:\n' + str(self.input_variables)
        if self.state_variables is not None:
            str_out += 'State Variables:\n' + str(self.state_variables)
        if self.output_variables is not None:
            str_out += 'Output Variables:\n' + str(self.output_variables)

        return str_out

    def check_types(self):
        assert type(self.A) in libsp.SupportedTypes, \
            'Type of A matrix (%s) not supported' % type(self.A)
        assert type(self.B) in libsp.SupportedTypes, \
            'Type of B matrix (%s) not supported' % type(self.B)
        assert type(self.C) in libsp.SupportedTypes, \
            'Type of C matrix (%s) not supported' % type(self.C)
        assert type(self.D) in libsp.SupportedTypes, \
            'Type of D matrix (%s) not supported' % type(self.D)

    def get_mats(self):
        return self.A, self.B, self.C, self.D

    def freqresp(self, wv):
        """
        Calculate frequency response over frequencies wv

        Note: this wraps frequency response function.
        """
        dlti = True
        if self.dt is None:
            dlti = False
        return freqresp(self, wv, dlti=dlti)

    def addGain(self, K, where):
        """
        Projects input u or output y the state-space system through the gain
        matrix K. The input 'where' determines whether inputs or outputs are
        projected as:
            - where='in': inputs are projected such that:
                u_new -> u=K*u_new -> SS -> y  => u_new -> SSnew -> y
            - where='out': outputs are projected such that:
                 u -> SS -> y -> y_new=K*y => u -> SSnew -> ynew

        Args:
            K (np.array or Gain): gain matrix or Gain object
            where (str): ``in`` or ``out``

        Warning:
            This is not a wrapper of the addGain method in this module, as
            the state-space matrices are directly overwritten.
        """

        assert where in ['in', 'out'], \
            'Specify whether gains are added to input or output'

        with_vars = False
        if isinstance(K, Gain):
            gain = K
            K = K.value
            with_vars = True

        if where == 'in':
            self.B = libsp.dot(self.B, K)
            self.D = libsp.dot(self.D, K)
            if with_vars:
                self._input_variables = gain.input_variables

        if where == 'out':
            self.C = libsp.dot(K, self.C)
            self.D = libsp.dot(K, self.D)
            if with_vars:
                self._output_variables = gain.output_variables

    def scale(self, input_scal=1., output_scal=1., state_scal=1.):
        """
        Given a state-space system, scales the equations such that the original
        state, input and output, (x, u and y), are substituted by
            xad=x/state_scal
            uad=u/input_scal
            yad=y/output_scal
        The entries input_scal/output_scal/state_scal can be:
            - floats: in this case all input/output are scaled by the same value
            - lists/arrays of length Nin/Nout: in this case each dof will be scaled
            by a different factor

        If the original system has form:
            xnew=A*x+B*u
            y=C*x+D*u
        the transformation is such that:
            xnew=A*x+(B*uref/xref)*uad
            yad=1/yref( C*xref*x+D*uref*uad )
        """
        scale_SS(self, input_scal, output_scal, state_scal, byref=True)

    def project(self, wt, v):
        """
        Given 2 transformation matrices, ``(WT, V)`` of shapes ``(Nk, self.states)`` and
        ``(self.states, Nk)`` respectively, this routine projects the state space
        model states according to:


        .. math::
            Anew = WT A V \\
            Bnew = WT B \\
            Cnew = C V \\
            Dnew = D \\

        The projected model has the same number of inputs/outputs as the original
        one, but Nk states.

        Args:
            wt (Gain or np.ndarray): Left projection matrix
            v (Gain or np.ndarray): Righty projection matrix
        """
        if isinstance(wt, Gain) and isinstance(v, Gain):
            self.A = libsp.dot(wt.value, libsp.dot(self.A, v.value))
            self.B = libsp.dot(wt.value, self.B)
            self.C = libsp.dot(self.C, v.value)
            self.state_variables = LinearVector.transform(v.input_variables, to_type=StateVariable)
        else:
            self.A = libsp.dot(wt, libsp.dot(self.A, v))
            self.B = libsp.dot(wt, self.B)
            self.C = libsp.dot(self.C, v)

    def truncate(self, N):
        """ Retains only the first N states. """

        assert N > 0 and N <= self.states, 'N must be in [1,self.states]'

        self.A = self.A[:N, :N]
        self.B = self.B[:N, :]
        self.C = self.C[:, :N]
        # self.states = N  # No need to update, states is now a property. NG 26/3/19

    def max_eig(self):
        """
        Returns most unstable eigenvalue
        """

        ev = np.linalg.eigvals(self.A)

        if self.dt is None:
            return np.max(ev.real)
        else:
            return np.max(np.abs(ev))

    def eigvals(self):
        """
        Returns:
            np.ndarray: Eigenvalues of the system

        """
        if self.dt:
            return eigvals(self.A, dlti=True)
        else:
            return eigvals(self.A, dlti=False)

    def disc2cont(self):
        r"""
        Transform a discrete time system to a continuous time system using a bilinear (Tustin) transformation.

        Wrapper of :func:`~sharpy.linear.src.libss.disc2cont`

        """
        if self.dt:
            self = disc2cont(self)

    def retain_inout_channels(self, retain_channels, where):
        """
        Retain selected input or output channels only.

        Args:
            retain_channels (list): List of channels to retain
            where (str): ``in`` or ``out`` for input/output channels
        """
        retain_inout_channels(self, retain_channels, where)

    def summary(self):
        msg = 'State-space system\nStates: %g\nInputs: %g\nOutputs: %g\n' % (self.states, self.inputs, self.outputs)
        return msg

    def transfer_function_evaluation(self, s):
        r"""
        Returns the transfer function of the system evaluated at :math:`s\in\mathbb{C}`.

        Args:
            s (complex): Point in the complex plane at which to evaluate the transfer function.

        Returns:
            np.ndarray: Transfer function evaluated at :math:`s`.
        """
        a, b, c, d = self.get_mats()

        n = a.shape[0]

        return c.dot(scalg.inv(s * np.eye(n) - a)).dot(b) + d

    def save(self, path):
        """Save state-space object to h5 file"""
        with h5py.File(path, 'w') as f:
            f.create_dataset('a', data=self.A)
            f.create_dataset('b', data=self.B)
            f.create_dataset('c', data=self.C)
            f.create_dataset('d', data=self.D)
            if self.dt:
                f.create_dataset('dt', data=self.dt)

            if self.input_variables is not None:
                self.input_variables.add_to_h5_file(f)
                self.output_variables.add_to_h5_file(f)
                self.state_variables.add_to_h5_file(f)

    @classmethod
    def load_from_h5(cls, h5_file_name):
        """
        Loads a state-space object from an h5 file, including variable information

        Args:
            h5_file_name (str): Path to file

        Returns:
            StateSpace: loaded state-space from file
        """

        with h5py.File(h5_file_name, 'r') as f:
            data_dict = h5utils.load_h5_in_dict(f)

        new_ss = cls(data_dict['a'],
                     data_dict['b'],
                     data_dict['c'],
                     data_dict['d'],
                     dt=data_dict.get('dt'))

        input_variables = data_dict.get('InputVariable')
        if input_variables is not None:
            new_ss.input_variables = LinearVector.load_from_h5_file('InputVariable',
                                                                    data_dict['InputVariable'])
            new_ss.output_variables = LinearVector.load_from_h5_file('OutputVariable',
                                                                     data_dict['OutputVariable'])
            new_ss.state_variables = LinearVector.load_from_h5_file('StateVariable',
                                                                    data_dict['StateVariable'])

            return new_ss
        else:
            return new_ss

    def remove_inputs(self, *input_remove_list):
        """
        Removes inputs through their variable names.

        Needs that the ``StateSpace`` attribute ``input_variables`` is defined.

        Args:
            input_remove_list (list(str)): List of inputs to remove

        """
        if self.input_variables is None:
            raise AttributeError('No input variables have been defined for the current state-space object. Define '
                                 'some variables prior to using the remove_inputs() method.')

        self.input_variables.remove(*input_remove_list)

        i = 0
        retain_input_array = None
        for variable in self.input_variables:
            if i == 0:
                retain_input_array = variable.cols_loc
            else:
                retain_input_array = np.hstack((retain_input_array, variable.cols_loc))
            i += 1

        if retain_input_array is not None:
            if type(self.B) is libsp.csc_matrix:
                self.B = libsp.csc_matrix(self.B[:, retain_input_array])
                self.D = libsp.csc_matrix(self.D[:, retain_input_array])
            else:
                self.B = self.B[:, retain_input_array]
                self.D = self.D[:, retain_input_array]

        self.input_variables.update_locations()

    def remove_outputs(self, *output_remove_list):
        """
        Removes outputs through their variable names.

        Needs that the ``StateSpace`` attribute ``output_variables`` is defined.

        Args:
            output_remove_list (list(str)): List of outputs to remove

        """
        if self.output_variables is None:
            raise AttributeError('No output variables have been defined for the current state-space object. Define '
                                 'some variables prior to using the remove_outputs() method.')

        new_outputs = 0
        for variable in self.output_variables:
            if variable.name not in output_remove_list:
                new_outputs += variable.size

        out_gain = np.zeros((new_outputs, self.outputs))
        worked_outputs = 0
        for variable in self.output_variables:
            if variable.name not in output_remove_list:
                index = variable.rows_loc
                out_gain[worked_outputs:worked_outputs + variable.size, index] = np.eye(variable.size)
                worked_outputs += variable.size

        if new_outputs != self.outputs:
            if type(self.B) is libsp.csc_matrix:
                self.C = libsp.csc_matrix(out_gain.dot(self.C))
                self.D = libsp.csc_matrix(out_gain.dot(self.D))
            else:
                self.C = out_gain.dot(self.C)
                self.D = out_gain.dot(self.D)

        self.output_variables.remove(*output_remove_list)
        self.output_variables.update_locations()

    @classmethod
    def from_scipy(cls, scipy_ss):
        """
        Transforms a ``scipy.signal.lti`` or dlti into a StateSpace class

        Args:
            scipy_ss (scipy.signal.ltisys.StateSpaceContinous or scipy.signal.ltisys.StateSpaceDiscrete): Scipy
              State Space object.

        Returns:
            StateSpace: SHARPy state space object
        """
        a = scipy_ss.A
        b = scipy_ss.B
        c = scipy_ss.C
        d = scipy_ss.D

        return cls(a, b, c, d, dt=scipy_ss.dt)


class Gain:

    def __init__(self, value, input_vars=None, output_vars=None):
        self.value = value
        self._input_variables = None
        self._output_variables = None

        if input_vars is not None:
            self.input_variables = input_vars
        if output_vars is not None:
            self.output_variables = output_vars

    @property
    def input_variables(self):
        return self._input_variables

    @input_variables.setter
    def input_variables(self, variables):
        if variables.variable_class is not InputVariable:
            raise TypeError('LinearVector does not include InputVariable s')
        if variables.size != self.inputs:
            raise IndexError('Size of LinearVector of InputVariable s ({:g}) is not the same as the number of '
                             'inputs in the '
                             'system ({:g})'.format(variables.size, self.inputs))
        self._input_variables = variables

    @property
    def output_variables(self):
        return self._output_variables

    @output_variables.setter
    def output_variables(self, variables):
        if variables.variable_class is not OutputVariable:
            raise TypeError('LinearVector does not include OutputVariable s')
        if variables.size != self.outputs:
            raise IndexError('Size of LinearVector of OutputVariable s ({:g}) is not the same as the number of '
                             'outputs in the '
                             'system ({:g})'.format(variables.size, self.outputs))
        self._output_variables = variables

    @property
    def inputs(self):
        """Number of inputs :math:`m` to the system."""
        if self.value.shape.__len__() == 1:
            return 1
        else:
            return self.value.shape[1]

    @property
    def outputs(self):
        """Number of outputs :math:`p` of the gain."""
        return self.value.shape[0]

    def dot(self, elem):
        """
        Dot product of two Gains

        Args:
            elem (np.array or Gain):

        Returns:
            np.array or Gain: new matrix/Gain containing the dot product
        """
        if type(elem) is Gain:
            LinearVector.check_connection(elem.output_variables, self.input_variables)
            new_gain_value = libsp.dot(self.value, elem.value)
            return Gain(new_gain_value,
                        input_vars=elem.input_variables.copy(),
                        output_vars=self.output_variables.copy())
        else:
            return self.value.dot(elem)

    def __repr__(self):
        str_out = ''
        str_out += 'Gain object\n'
        str_out += 'Inputs: {:g}\n'.format(self.inputs)
        str_out += 'Outputs: {:g}\n'.format(self.outputs)

        if self.input_variables is not None:
            str_out += '\nInput Variables:\n' + str(self.input_variables)
        if self.output_variables is not None:
            str_out += 'Output Variables:\n' + str(self.output_variables)

        return str_out

    def transpose(self):
        """
        Transposes the gain, such that the inputs become the outputs and vice-versa.
        """

        if self.input_variables is not None:
            temp_input_var = self.input_variables.copy()
            input_variables = LinearVector.transform(self.output_variables,
                                                     to_type=InputVariable)
            output_variables = LinearVector.transform(temp_input_var,
                                                      to_type=OutputVariable)

            return Gain(self.value.T,
                        input_vars=input_variables,
                        output_vars=output_variables)
        else:
            return Gain(self.value.T)

    @property
    def T(self):
        return self.transpose()

    def copy(self):
        if self.input_variables is not None:
            return Gain(self.value, input_vars=self.input_variables.copy(), output_vars=self.output_variables.copy())
        else:
            return Gain(self.value)

    def save(self, path):
        """Save gain object to h5 file"""
        with h5py.File(path, 'w') as f:
            f.create_dataset('gain', data=self.value)

            if self.input_variables is not None:
                self.input_variables.add_to_h5_file(f)
                self.output_variables.add_to_h5_file(f)

    def add_as_group_to_h5(self, h5_file_handle, group_name):
        """
        Adds gain to an h5 file handle
        
        Args:
            h5_file_handle (h5py.File): writeable h5 file handle
            group_name (str): Desired group name to save gain in h5

        """
        gain_group = h5_file_handle.create_group(group_name)
        gain_group.create_dataset(name='gain', data=self.value)

        if self.input_variables is not None:
            self.input_variables.add_to_h5_file(gain_group)
            self.output_variables.add_to_h5_file(gain_group)

    @classmethod
    def load_from_h5(cls, h5_file_name):
        """
        Returns a gain object from an .h5 file

        Args:
            h5_file_name (str): Path to h5 file

        Returns:
            Gain: instance of a Gain
        """
        with h5py.File(h5_file_name, 'r') as f:
            data_dict = h5utils.load_h5_in_dict(f)

        return cls.load_from_dict(data_dict)

    @classmethod
    def load_from_dict(cls, data_dict):
        """

        Returns a Gain from a dictionary of data, useful for loading from a group of gains in a single
        .h5 file

        Args:
            data_dict (dict): Dictionary with keys: ``gain`` and (if available) ``InputVariable``
              and ``OutputVariable``.

        Returns:
            Gain: instance of Gain
        """
        input_variables = data_dict.get('InputVariable')
        if input_variables is not None:
            input_variables = LinearVector.load_from_h5_file('InputVariable',
                                                             data_dict['InputVariable'])
            output_variables = LinearVector.load_from_h5_file('OutputVariable',
                                                              data_dict['OutputVariable'])

            return cls(data_dict['gain'], input_vars=input_variables,
                       output_vars=output_variables)
        else:
            return cls(data_dict['gain'])

    @classmethod
    def save_multiple_gains(cls, h5_file_name, *gains_names_tuple):
        """
        Saves multiple gains to a single h5 file

        Args:
            h5_file_name (str): Path to h5 file
            *gains_names_tuple (tuple): ``(gain_name (str), gain(Gain))`` tuples to save. The gain name will be the name
              given on the h5 file

        """
        with h5py.File(h5_file_name, 'w') as f:
            for name, gain in gains_names_tuple:
                gain.add_as_group_to_h5(f, name)

    @classmethod
    def load_multiple_gains(cls, h5_file_name):
        """
        Loads multiple gains from a single h5 file

        Args:
            h5_file_name (str): Path to h5 file

        Returns:
            dict: Dictionary of loaded gains in a gain_name: Gain dictionary
        """
        with h5py.File(h5_file_name, 'r') as f:
            data_dict = h5utils.load_h5_in_dict(f)

        out_gains = {}
        for gain_name, gain_data in data_dict.items():
            out_gains[gain_name] = cls.load_from_dict(gain_data)

        return out_gains


class ss_block():
    """
    State-space model in block form. This class has the same purpose as "StateSpace",
    but the A, B, C, D are allocated in the form of nested lists. The format is
    similar to the one used in numpy.block but:
        1. Block matrices can contain both dense and sparse matrices
        2. Empty blocks are defined through None type

    Methods:
    - remove_block: drop one of the blocks from the s-s model
    - addGain: project inputs/outputs
    - project: project state
    """

    def __init__(self, A, B, C, D, S_states, S_inputs, S_outputs, dt=None):
        """
        Allocate state-space model (A,B,C,D) in block form starting from nested
        lists of full/sparse matrices (as per numpy.block).

        Input:
        - A, B, C, D: lists of matrices defining the state-space model.
        - S_states, S_inputs, S_outputs: lists with dimensions of of each block
        representing the states, inputs and outputs of the model.
        - dt: time-step. In None, a continuous-time system is assumed.
        """

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt

        self.S_u = S_inputs
        self.S_y = S_outputs
        self.S_x = S_states

        # determine number of blocks
        self.blocks_u = len(S_inputs)
        self.blocks_y = len(S_outputs)
        self.blocks_x = len(S_states)

        # determine inputs/outputs/states
        self.inputs = sum(S_inputs)
        self.outputs = sum(S_outputs)
        self.states = sum(S_states)

        self.check_sizes()

    def check_sizes(self):
        pass

    def remove_block(self, where, index):
        """
        Remove a block from either inputs or outputs.

        Inputs:
        - where = {'in', 'out'}: determined whether to remove inputs or outputs
        - index: index of block to remove
        """

        assert where in ['in', 'out'], "'where' must be equal to {'in', 'out'}"

        if where == 'in':
            for ii in range(self.blocks_x):
                del self.B[ii][index]
            for ii in range(self.blocks_y):
                del self.D[ii][index]

        if where == 'out':
            for ii in range(self.blocks_y):
                del self.C[ii]
                del self.D[ii]

    def addGain(self, K, where):
        """
        Projects input u or output y the state-space system through the gain
        block matrix K. The input 'where' determines whether inputs or outputs
        are projected as:
            - where='in': inputs are projected such that:
                u_new -> u=K*u_new -> SS -> y  => u_new -> SSnew -> y
            - where='out': outputs are projected such that:
                 u -> SS -> y -> y_new=K*y => u -> SSnew -> ynew

        Input: K must be a list of list of matrices. The size of K must be
        compatible with either B or C for block matrix product.
        """

        assert where in ['in', 'out'], \
            'Specify whether gains are added to input or output'

        rows, cols = self.get_sizes(K)

        if where == 'in':
            self.B = libsp.block_dot(self.B, K)
            self.D = libsp.block_dot(self.D, K)
            self.S_u = cols
            self.blocks_u = len(cols)
            self.inputs = sum(cols)

        if where == 'out':
            self.C = libsp.block_dot(K, self.C)
            self.D = libsp.block_dot(K, self.D)
            self.S_y = rows
            self.blocks_y = len(rows)
            self.outputs = sum(rows)

    def get_sizes(self, M):
        """
        Get the size of each block in M.
        """

        rM, cM = len(M), len(M[0])
        rows = rM * [None]
        cols = cM * [None]

        for ii in range(rM):
            for jj in range(cM):
                if M[ii][jj] is not None:
                    rhere, chere = M[ii][jj].shape

                    if rows[ii] is None:  # allocate
                        rows[ii] = rhere
                    else:  # check
                        assert rows[ii] == rhere, \
                            'Block (%d,%d) has inconsistent size with other in same row!' % (ii, jj)

                    if cols[jj] is None:  # allocate
                        cols[jj] = chere
                    else:  # check
                        assert cols[jj] == chere, \
                            'Block (%d,%d) has inconsistent size with other in same column!' % (ii, jj)

        return rows, cols

    def project(self, WT, V, by_arrays=True, overwrite=False):
        """
        Given 2 transformation matrices, (W,V) of shape (Nk,self.states), this
        routine projects the state space model states according to:

          Anew = W^T A V
          Bnew = W^T B
          Cnew = C V
          Dnew = D

        The projected model has the same number of inputs/outputs as the original
        one, but Nk states.

        Inputs:
        - WT = W^T
        - V = V
        - by_arrays: if True, W, V are either numpy.array or sparse matrices. If
          False, they are block matrices.
        - overwrite: if True, overwrites the A, B, C matrices
        """

        if by_arrays:  # transform to block structures

            II0 = 0
            Vblock = []
            WTblock = [[]]
            for ii in range(self.blocks_x):
                iivec = range(II0, II0 + self.S_x[ii])
                Vblock.append([V[iivec, :]])
                WTblock[0].append(WT[:, iivec])
                II0 += self.S_x[ii]
        else:
            Vblock = V
            WTblock = WT

        if overwrite:
            self.A = libsp.block_dot(WTblock, libsp.block_dot(self.A, Vblock))
            self.B = libsp.block_dot(WTblock, self.B)
            self.C = libsp.block_dot(self.C, Vblock)
        else:
            return (libsp.block_dot(WTblock, libsp.block_dot(self.A, Vblock)),
                    libsp.block_dot(WTblock, self.B),
                    libsp.block_dot(self.C, Vblock))

    def solve_step(self, xn, un):

        # TODO: add options about predictor ...
        xn1 = libsp.block_sum(libsp.block_dot(self.A, xn), libsp.block_dot(self.B, un))
        yn = libsp.block_sum(libsp.block_dot(self.C, xn), libsp.block_dot(self.D, un))

        return xn1, yn


    def get_mats(self):
        
        A = np.zeros((self.states, self.states))
        B = np.zeros((self.states, self.inputs))
        C = np.zeros((self.outputs, self.states))
        D = np.zeros((self.outputs, self.inputs))
        
        iloc = 0
        for i in range(self.blocks_x):
            jloc = 0
            for j in range(self.blocks_x):
                if not self.A[i][j] is None:
                    if type(self.A[i][j]) == libsp.csc_matrix:
                        A[iloc:iloc+self.S_x[i], jloc:jloc+self.S_x[j]] = self.A[i][j].todense()
                    else:
                        A[iloc:iloc+self.S_x[i], jloc:jloc+self.S_x[j]] = self.A[i][j].copy()
                jloc += self.S_x[j]
            iloc += self.S_x[i]
        
        iloc = 0
        for i in range(self.blocks_x):
            jloc = 0
            for j in range(self.blocks_u):
                if not self.B[i][j] is None:
                    # print(i, j, iloc, jloc, self.S_x[i], self.S_u[j], self.B[i][j].shape)
                    # print(iloc, iloc+self.S_x[i], jloc, jloc+self.S_u[j])
                    if type(self.B[i][j]) == libsp.csc_matrix:
                        B[iloc:iloc+self.S_x[i], jloc:jloc+self.S_u[j]] = self.B[i][j].todense()
                    else:
                        B[iloc:iloc+self.S_x[i], jloc:jloc+self.S_u[j]] = self.B[i][j].copy()
                jloc += self.S_u[j]
            iloc += self.S_x[i]

        iloc = 0
        for i in range(self.blocks_y):
            jloc = 0
            for j in range(self.blocks_x):
                if not self.C[i][j] is None:
                    if type(self.C[i][j]) == libsp.csc_matrix:
                        C[iloc:iloc+self.S_y[i], jloc:jloc+self.S_x[j]] = self.C[i][j].todense()
                    else:
                        C[iloc:iloc+self.S_y[i], jloc:jloc+self.S_x[j]] = self.C[i][j].copy()
                jloc += self.S_x[j]
            iloc += self.S_y[i]
        
        iloc = 0
        for i in range(self.blocks_y):
            jloc = 0
            for j in range(self.blocks_u):
                if not self.D[i][j] is None:
                    if type(self.D[i][j]) == libsp.csc_matrix:
                        D[iloc:iloc+self.S_y[i], jloc:jloc+self.S_u[j]] = self.D[i][j].todense()
                    else:
                        D[iloc:iloc+self.S_y[i], jloc:jloc+self.S_u[j]] = self.D[i][j].copy()
                jloc += self.S_u[j]
            iloc += self.S_y[i]

        return A, B, C, D

# ---------------------------------------- Methods for state-space manipulation
def project(ss_here, WT, V):
    """
    Given 2 transformation matrices, (WT,V) of shapes (Nk,self.states) and
    (self.states,Nk) respectively, this routine returns a projection of the
    state space ss_here according to:

        Anew = WT A V
        Bnew = WT B
        Cnew = C V
        Dnew = D

    The projected model has the same number of inputs/outputs as the original
    one, but Nk states.
    """

    Ap = libsp.dot(WT, libsp.dot(ss_here.A, V))
    Bp = libsp.dot(WT, ss_here.B)
    Cp = libsp.dot(ss_here.C, V)

    return StateSpace(Ap, Bp, Cp, ss_here.D, ss_here.dt)


def couple(ss01, ss02, K12, K21, out_sparse=False):
    """
    Couples 2 dlti systems ss01 and ss02 through the gains K12 and K21, where
    K12 transforms the output of ss02 into an input of ss01.

    Other inputs:
        - out_sparse: if True, the output system is stored as sparse (not recommended)
    """
    if ss01.dt is None and ss02.dt is None:
        pass
    else:
        try:
            assert np.abs(ss01.dt - ss02.dt) < 1e-10 * ss01.dt, 'Time-steps not matching!'
        except TypeError:
            raise TypeError('One of the systems to couple is discrete and the other continuous')

    if ss01.input_variables is not None and ss02.input_variables is not None \
            and isinstance(K12, Gain) and isinstance(K21, Gain):
        with_enhanced_vars = True
        LinearVector.check_connection(K12.output_variables, ss01.input_variables)
        LinearVector.check_connection(ss02.output_variables, K12.input_variables)

        LinearVector.check_connection(K21.output_variables, ss02.input_variables)
        LinearVector.check_connection(ss01.output_variables, K21.input_variables)
        K21 = K21.value
        K12 = K12.value
    else:
        with_enhanced_vars = False
        assert K12.shape == (ss01.inputs, ss02.outputs), \
            'Gain K12 shape not matching with systems number of inputs/outputs'
        assert K21.shape == (ss02.inputs, ss01.outputs), \
            'Gain K21 shape not matching with systems number of inputs/outputs'

    A1, B1, C1, D1 = ss01.get_mats()
    A2, B2, C2, D2 = ss02.get_mats()

    # extract size
    Nx1, Nu1 = B1.shape
    Ny1 = C1.shape[0]
    Nx2, Nu2 = B2.shape
    Ny2 = C2.shape[0]

    #  terms to invert
    maxD1 = np.max(np.abs(D1))
    maxD2 = np.max(np.abs(D2))
    if maxD1 < 1e-32:
        pass
    if maxD2 < 1e-32:
        pass

    # compute self-influence gains
    K11 = libsp.dot(K12, libsp.dot(D2, K21))
    K22 = libsp.dot(K21, libsp.dot(D1, K12))

    # left hand side terms
    L1 = libsp.dot(-K11, D1)
    L2 = libsp.dot(-K22, D2)
    L1 += libsp.eye_as(L1)
    L2 += libsp.eye_as(L2)

    # coupling terms
    cpl_12 = libsp.solve(L1, K12)
    cpl_21 = libsp.solve(L2, K21)

    cpl_11 = libsp.dot(cpl_12, libsp.dot(D2, K21))
    cpl_22 = libsp.dot(cpl_21, libsp.dot(D1, K12))

    # Build coupled system
    if out_sparse:
        raise NameError('out_sparse=True not supported yet (verify if worth it first).')
    else:
        A = np.block([
            [libsp.dense(A1 + libsp.dot(libsp.dot(B1, cpl_11), C1)), libsp.dense(libsp.dot(libsp.dot(B1, cpl_12), C2))],
            [libsp.dense(libsp.dot(libsp.dot(B2, cpl_21), C1)),
             libsp.dense(A2 + libsp.dot(libsp.dot(B2, cpl_22), C2))]])

        C = np.block([
            [libsp.dense(C1 + libsp.dot(libsp.dot(D1, cpl_11), C1)), libsp.dense(libsp.dot(libsp.dot(D1, cpl_12), C2))],
            [libsp.dense(libsp.dot(libsp.dot(D2, cpl_21), C1)),
             libsp.dense(C2 + libsp.dot(libsp.dot(D2, cpl_22), C2))]])

        B = np.block([
            [libsp.dense(B1 + libsp.dot(libsp.dot(B1, cpl_11), D1)), libsp.dense(libsp.dot(libsp.dot(B1, cpl_12), D2))],
            [libsp.dense(libsp.dot(libsp.dot(B2, cpl_21), D1)),
             libsp.dense(B2 + libsp.dot(libsp.dot(B2, cpl_22), D2))]])

        D = np.block([
            [libsp.dense(D1 + libsp.dot(libsp.dot(D1, cpl_11), D1)), libsp.dense(libsp.dot(libsp.dot(D1, cpl_12), D2))],
            [libsp.dense(libsp.dot(libsp.dot(D2, cpl_21), D1)),
             libsp.dense(D2 + libsp.dot(libsp.dot(D2, cpl_22), D2))]])

    coupled_ss = StateSpace(A, B, C, D, dt=ss01.dt)
    if with_enhanced_vars:
        coupled_ss.state_variables = LinearVector.merge(ss01.state_variables, ss02.state_variables)
        coupled_ss.input_variables = LinearVector.merge(ss01.input_variables, ss02.input_variables)
        coupled_ss.output_variables = LinearVector.merge(ss01.output_variables, ss02.output_variables)

    return coupled_ss


def disc2cont(sys):
    r"""
    Transform a discrete time system to a continuous time system using a bilinear (Tustin) transformation.

    Given a discrete time system with time step :math:`\Delta T`, the equivalent continuous time system is given
    by:

    .. math::
        \bar{A} &= \omega_0(A-I)(I + A)^{-1}  \\
        \bar{B} &= \sqrt{2\omega_0}(I+A)^{-1}B  \\
        \bar{C} &= \sqrt{2\omega_0}C(I+A)^{-1}  \\
        \bar{D} &= D - C(I+A)^{-1}B

    where :math:`\omega_0 = \frac{2}{\Delta T}`.

    References:
        MIT OCW 6.245

    Args:
        sys (libss.StateSpace): SHARPy discrete-time state-space object.

    Returns:
        libss.StateSpace: Converted continuous-time state-space object.
    """

    assert sys.dt is not None, 'System to transform is not a discrete-time system.'

    n = sys.A.shape[0]
    eye = np.eye(n)
    eye_a_inv = np.linalg.inv(sys.A + eye)
    omega_0 = 2 / sys.dt

    a = omega_0 * (sys.A - eye).dot(eye_a_inv)
    b = np.sqrt(2 * omega_0) * eye_a_inv.dot(sys.B)
    c = np.sqrt(2 * omega_0) * sys.C.dot(eye_a_inv)
    d = sys.D - sys.C.dot(eye_a_inv.dot(sys.B))

    sys_ct = StateSpace(a, b, c, d)

    if sys.input_variables is not None:
        sys_ct.input_variables = sys.input_variables
        sys_ct.state_variables = sys.state_variables
        sys_ct.output_variables = sys.output_variables

    return sys_ct


def retain_inout_channels(sys, retain_channels, where):
    """
    Retain selected input or output channels only.

    Args:
        retain_channels (list): List of channels to retain
        where (str): ``in`` or ``out`` for input/output channels

    Returns:
        StateSpace: Updated state-space object
    """
    retain_m = len(retain_channels)  # new number of in/out

    if where == 'in':
        m = sys.inputs  # current number of in/out
        gain_input_vars = sys.input_variables
        gain_output_vars = LinearVector.transform(sys.input_variables, to_type=OutputVariable)
    elif where == 'out':
        m = sys.outputs
        gain_input_vars = LinearVector.transform(sys.output_variables, to_type=InputVariable)
        gain_output_vars = sys.output_variables.copy()
    else:
        raise NameError('Argument ``where`` can only be ``in`` or ``out``.')

    gain_matrix = np.zeros((retain_m, m))
    for ith, channel in enumerate(retain_channels):
        gain_matrix[ith, channel] = 1

    # Go through variables...
    for var in gain_input_vars:
        n_vars = np.sum(
            (np.array(retain_channels) < var.end_position) * (np.array(retain_channels) >= var.first_position))

        if n_vars == 0:
            gain_output_vars.remove(var.name)
        else:
            gain_output_vars.modify(var.name, size=n_vars)

    gain_output_vars.update_indices()
    gain_output_vars.update_locations()

    gain_matrix = Gain(gain_matrix,
                       input_vars=gain_input_vars,
                       output_vars=gain_output_vars)

    if where == 'in':
        sys.addGain(gain_matrix.transpose(), where='in')
    elif where == 'out':
        sys.addGain(gain_matrix, where='out')
    else:
        raise NameError('Argument ``where`` can only be ``in`` or ``out``.')

    return sys


def freqresp(SS, wv, dlti=True):
    """
    In-house frequency response function supporting dense/sparse types

    Inputs:
    - SS: instance of StateSpace class, or scipy.signal.StateSpace*
    - wv: frequency range
    - dlti: True if discrete-time system is considered.

    Outputs:
    - Yfreq[outputs,inputs,len(wv)]: frequency response over wv

    Warnings:
    -  This function may not be very efficient for dense matrices (as A is not
    reduced to upper Hessenberg form), but can exploit sparsity in the state-space
    matrices.
    """

    assert type(SS) == StateSpace, \
        'Type %s of state-space model not supported. Use libss.StateSpace instead!' % type(SS)
    SS.check_types()

    if hasattr(SS, 'dt') and dlti:
        Ts = SS.dt
        wTs = Ts * wv
        zv = np.cos(wTs) + 1.j * np.sin(wTs)
    else:
        # print('Assuming a continuous time system')
        zv = 1.j * wv

    Nx = SS.A.shape[0]
    Ny = SS.D.shape[0]
    try:
        Nu = SS.B.shape[1]
    except IndexError:
        Nu = 1

    Nw = len(wv)

    Yfreq = np.empty((Ny, Nu, Nw,), dtype=np.complex_)
    Eye = libsp.eye_as(SS.A)
    for ii in range(Nw):
        sol_cplx = libsp.solve(zv[ii] * Eye - SS.A, SS.B)
        Yfreq[:, :, ii] = libsp.dot(SS.C, sol_cplx, type_out=np.ndarray) + SS.D

    return Yfreq


def series(SS01, SS02):
    r"""
    Connects two state-space blocks in series. If these are instances of DLTI
    state-space systems, they need to have the same type and time-step. If the input systems are sparse, they are
    converted to dense.

    The connection is such that:

    .. math::
        u \rightarrow \mathsf{SS01} \rightarrow \mathsf{SS02} \rightarrow y \Longrightarrow
        u \rightarrow \mathsf{SStot} \rightarrow y

    where the state vector :math:`x` is :math:`[x_1, x_2]`.

    Args:
        SS01 (libss.StateSpace): State Space 1 instance. Can be DLTI/CLTI, dense or sparse.
        SS02 (libss.StateSpace): State Space 2 instance. Can be DLTI/CLTI, dense or sparse.

    Returns
        libss.StateSpace: Combined state space system in series in dense format.
    """

    if type(SS01) is not type(SS02):
        raise TypeError('The two input systems are not of the same type')
    if SS01.dt != SS02.dt:
        raise NameError('DLTI systems do not have the same time-step. SS01 dt={:f}, SS02 dt={:f}'.format(
            SS01.dt, SS02.dt))

    # check series connection
    if SS01.output_variables is not None and SS02.input_variables is not None:
        LinearVector.check_connection(SS01.output_variables, SS02.input_variables)
        # for i_var in range(SS01.output_variables.num_variables):
        #     out1 = SS01.output_variables[i_var]
        #     in2 = SS02.input_variables[i_var]
        #     if out1.name != in2.name:
        #         raise NameError('Series coupling outputs1 and inputs2 have different names')
        #     if not (out1.rows_loc == in2.cols_loc).all:
        #         raise IndexError('Series coupling. Output1 channels do not line up with input2 channels.')

    # determine size of total system
    Nst01, Nst02 = SS01.states, SS02.states
    Nst = Nst01 + Nst02
    Nin = SS01.inputs
    Nout = SS02.outputs

    if SS01.outputs != SS02.inputs:
        raise ValueError('SS01 outputs not equal to SS02 inputs,\nSS01={:s}\nSS02={:s}'.format(str(SS01), str(SS02)))

    # Build A matrix
    A = np.zeros((Nst, Nst))

    A[:Nst01, :Nst01] = libsp.dense(SS01.A)
    A[Nst01:, Nst01:] = libsp.dense(SS02.A)
    A[Nst01:, :Nst01] = libsp.dense(libsp.dot(SS02.B, SS01.C))

    # Build the rest
    B = np.concatenate((libsp.dense(SS01.B), libsp.dense(libsp.dot(SS02.B, SS01.D))), axis=0)
    C = np.concatenate((libsp.dense(libsp.dot(SS02.D, SS01.C)), libsp.dense(SS02.C)), axis=1)
    D = libsp.dense(libsp.dot(SS02.D, SS01.D))

    SStot = StateSpace(A, B, C, D, dt=SS01.dt)

    SStot.input_variables = SS01.input_variables
    try:
        SStot.state_variables = LinearVector.merge(SS01.state_variables, SS02.state_variables)
    except AttributeError:
        SStot.state_variables = None
    SStot.output_variables = SS02.output_variables

    return SStot


def parallel(SS01, SS02):
    """
    Returns the sum (or parallel connection of two systems). Given two state-space
    models with the same output, but different input:
        u1 --> SS01 --> y
        u2 --> SS02 --> y

    """

    if type(SS01) is not type(SS02):
        raise NameError('The two input systems need to have the same size!')
    if SS01.dt != SS02.dt:
        raise NameError('DLTI systems do not have the same time-step!')
    Nout = SS02.outputs
    if Nout != SS01.outputs:
        raise NameError('DLTI systems need to have the same number of output!')

    # if type(SS01) is control.statesp.StateSpace:
    # 	SStot=control.parallel(SS01,SS02)
    # else:

    # determine size of total system
    Nst01, Nst02 = SS01.states, SS02.states
    Nst = Nst01 + Nst02
    Nin01, Nin02 = SS01.inputs, SS02.inputs
    Nin = Nin01 + Nin02

    # Build A,B matrix
    A = np.zeros((Nst, Nst))
    A[:Nst01, :Nst01] = SS01.A
    A[Nst01:, Nst01:] = SS02.A
    B = np.zeros((Nst, Nin))
    B[:Nst01, :Nin01] = SS01.B
    B[Nst01:, Nin01:] = SS02.B

    # Build the rest
    C = np.block([SS01.C, SS02.C])
    D = np.block([SS01.D, SS02.D])

    SStot = scsig.dlti(A, B, C, D, dt=SS01.dt)

    return SStot


def SSconv(A, B0, B1, C, D, Bm1=None):
    r"""
    Convert a DLTI system with prediction and delay of the form:

        .. math::
            \mathbf{x}_{n+1} &= \mathbf{A\,x}_n + \mathbf{B_0\,u}_n + \mathbf{B_1\,u}_{n+1} + \mathbf{B_{m1}\,u}_{n-1} \\
            \mathbf{y}_n &= \mathbf{C\,x}_n + \mathbf{D\,u}_n

    into the state-space form:

        .. math::
            \mathbf{h}_{n+1} &= \mathbf{A_h\,h}_n + \mathbf{B_h\,u}_n \\
            \mathbf{y}_n &= \mathbf{C_h\,h}_n + \mathbf{D_h\,u}_n

    If :math:`\mathbf{B_{m1}}` is ``None``, the original state is retrieved through

        .. math:: \mathbf{x}_n = \mathbf{h}_n + \mathbf{B_1\,u}_n

    and only the :math:`\mathbf{B}` and :math:`\mathbf{D}` matrices are modified.


    If :math:`\mathbf{B_{m1}}` is not ``None``, the SS is augmented with the new state

        .. math:: \mathbf{g}_{n} = \mathbf{u}_{n-1}

    or, equivalently, with the equation

        .. math:: \mathbf{g}_{n+1} = \mathbf{u}_n

    leading to the new form

        .. math::
            \mathbf{H}_{n+1} &= \mathbf{A_A\,H}_{n} + \mathbf{B_B\,u}_n \\
            \mathbf{y}_n &= \mathbf{C_C\,H}_{n} + \mathbf{D_D\,u}_n

    where :math:`\mathbf{H} = (\mathbf{x},\,\mathbf{g})`.

    Args:
        A (np.ndarray): dynamics matrix
        B0 (np.ndarray): input matrix for input at current time step ``n``. Set to None if this is zero.
        B1 (np.ndarray): input matrix for input at time step ``n+1`` (predictor term)
        C (np.ndarray): output matrix
        D (np.ndarray): direct matrix
        Bm1 (np.ndarray): input matrix for input at time step ``n-1`` (delay term)

    Returns:
        tuple: tuple packed with the state-space matrices :math:`\mathbf{A},\,\mathbf{B},\,\mathbf{C}` and :math:`\mathbf{D}`.

    References:
        Franklin, GF and Powell, JD. Digital Control of Dynamic Systems, Addison-Wesley Publishing Company, 1980

    Warnings:
        functions untested for delays (Bm1 != 0)
    """

    # Account for u^{n+1} terms (prediction)
    if B0 is None:
        Bh = libsp.dot(A, B1)
    else:
        Bh = B0 + libsp.dot(A, B1)
    Dh = D + libsp.dot(C, B1)

    # Account for u^{n-1} terms (delay)
    if Bm1 is None:
        outs = (A, Bh, C, Dh)
    else:
        warnings.warn('Function untested when Bm1!=None')

        Nx, Nu, Ny = A.shape[0], Bh.shape[1], C.shape[0]
        AA = np.block([[A, Bm1],
                       [np.zeros((Nu, Nx)), np.zeros((Nu, Nu))]])
        BB = np.block([[Bh], [np.eye(Nu)]])
        CC = np.block([C, np.zeros((Ny, Nu))])
        DD = Dh
        outs = (AA, BB, CC, DD)

    return outs


def addGain(SShere, Kmat, where):
    """
    Convert input u or output y of a SS DLTI system through gain matrix K. We
    have the following transformations:
    - where='in': the input dof of the state-space are changed
        u_new -> Kmat*u -> SS -> y  => u_new -> SSnew -> y
    - where='out': the output dof of the state-space are changed
         u -> SS -> y -> Kmat*u -> ynew => u -> SSnew -> ynew
    - where='parallel': the input dofs are changed, but not the output
         -
        {u_1 -> SS -> y_1
       { u_2 -> y_2= Kmat*u_2    =>    u_new=(u_1,u_2) -> SSnew -> y=y_1+y_2
        {y = y_1+y_2
         -
    Warning: function not tested for Kmat stored in sparse format
    """

    assert where in ['in', 'out', 'parallel-down', 'parallel-up'], \
        'Specify whether gains are added to input or output'

    if where == 'in':
        A = SShere.A
        B = SShere.B.dot(Kmat)
        C = SShere.C
        D = SShere.D.dot(Kmat)

    if where == 'out':
        A = SShere.A
        B = SShere.B
        C = Kmat.dot(SShere.C)
        D = Kmat.dot(SShere.D)

    if where == 'parallel-down':
        A = SShere.A
        C = SShere.C
        B = np.block([SShere.B, np.zeros((SShere.B.shape[0], Kmat.shape[1]))])
        D = np.block([SShere.D, Kmat])

    if where == 'parallel-up':
        A = SShere.A
        C = SShere.C
        B = np.block([np.zeros((SShere.B.shape[0], Kmat.shape[1])), SShere.B])
        D = np.block([Kmat, SShere.D])

    if SShere.dt == None:
        SSnew = StateSpace(A, B, C, D)
    else:
        SSnew = StateSpace(A, B, C, D, dt=SShere.dt)

    return SSnew


def join2(SS1, SS2):
    r"""
    Join two state-spaces or gain matrices such that, given:

        .. math::
            \mathbf{u}_1 \longrightarrow &\mathbf{SS}_1 \longrightarrow \mathbf{y}_1 \\
            \mathbf{u}_2 \longrightarrow &\mathbf{SS}_2 \longrightarrow \mathbf{y}_2

    we obtain:

        .. math::
            \mathbf{u} \longrightarrow \mathbf{SS}_{TOT} \longrightarrow \mathbf{y}

    with :math:`\mathbf{u}=(\mathbf{u}_1,\mathbf{u}_2)^T` and :math:`\mathbf{y}=(\mathbf{y}_1,\mathbf{y}_2)^T`.

    The output :math:`\mathbf{SS}_{TOT}` is either a gain matrix or a state-space system according
    to the input :math:`\mathbf{SS}_1` and :math:`\mathbf{SS}_2`

    Args:
        SS1 (scsig.StateSpace or np.ndarray): State space 1 or gain 1
        SS2 (scsig.StateSpace or np.ndarray): State space 2 or gain 2

    Returns:
        scsig.StateSpace or np.ndarray: combined state space or gain matrix

    """
    type_dlti = scsig.ltisys.StateSpaceDiscrete

    if isinstance(SS1, np.ndarray) and isinstance(SS2, np.ndarray):

        Nin01, Nin02 = SS1.shape[1], SS2.shape[1]
        Nout01, Nout02 = SS1.shape[0], SS2.shape[0]
        SStot = np.block([[SS1, np.zeros((Nout01, Nin02))],
                          [np.zeros((Nout02, Nin01)), SS2]])

    elif isinstance(SS1, np.ndarray) and isinstance(SS2, type_dlti):

        Nin01, Nout01 = SS1.shape[1], SS1.shape[0]
        Nin02, Nout02 = SS2.inputs, SS2.outputs
        Nx02 = SS2.A.shape[0]

        A = SS2.A
        B = np.block([np.zeros((Nx02, Nin01)), SS2.B])
        C = np.block([[np.zeros((Nout01, Nx02))],
                      [SS2.C]])
        D = np.block([[SS1, np.zeros((Nout01, Nin02))],
                      [np.zeros((Nout02, Nin01)), SS2.D]])

        SStot = scsig.StateSpace(A, B, C, D, dt=SS2.dt)

    elif isinstance(SS1, type_dlti) and isinstance(SS2, np.ndarray):

        Nin01, Nout01 = SS1.inputs, SS1.outputs
        Nin02, Nout02 = SS2.shape[1], SS2.shape[0]
        Nx01 = SS1.A.shape[0]

        A = SS1.A
        B = np.block([SS1.B, np.zeros((Nx01, Nin02))])
        C = np.block([[SS1.C],
                      [np.zeros((Nout02, Nx01))]])
        D = np.block([[SS1.D, np.zeros((Nout01, Nin02))],
                      [np.zeros((Nout02, Nin01)), SS2]])

        SStot = scsig.StateSpace(A, B, C, D, dt=SS1.dt)

    elif isinstance(SS1, type_dlti) and isinstance(SS2, type_dlti):

        assert SS1.dt == SS2.dt, 'State-space models must have the same time-step'

        Nin01, Nout01 = SS1.inputs, SS1.outputs
        Nin02, Nout02 = SS2.inputs, SS2.outputs
        Nx01, Nx02 = SS1.A.shape[0], SS2.A.shape[0]

        A = np.block([[SS1.A, np.zeros((Nx01, Nx02))],
                      [np.zeros((Nx02, Nx01)), SS2.A]])
        B = np.block([[SS1.B, np.zeros((Nx01, Nin02))],
                      [np.zeros((Nx02, Nin01)), SS2.B]])
        C = np.block([[SS1.C, np.zeros((Nout01, Nx02))],
                      [np.zeros((Nout02, Nx01)), SS2.C]])
        D = np.block([[SS1.D, np.zeros((Nout01, Nin02))],
                      [np.zeros((Nout02, Nin01)), SS2.D]])
        SStot = scsig.StateSpace(A, B, C, D, dt=SS1.dt)

    else:
        raise NameError('Input types not recognised in any implemented option!')

    return SStot


def join(SS_list, wv=None):
    """
    Given a list of state-space models belonging to the StateSpace class, creates a
    joined system whose output is the sum of the state-space outputs. If wv is
    not None, this is a list of weights, such that the output is:

        y = sum( wv[ii] y_ii )

    Ref: equation (4.22) of
    Benner, P., Gugercin, S. & Willcox, K., 2015. A Survey of Projection-Based
    Model Reduction Methods for Parametric Dynamical Systems. SIAM Review, 57(4),
    pp.483–531.

    Warnings:
        - system matrices must be numpy arrays
        - the function does not perform any check!
    """

    N = len(SS_list)
    if wv is not None:
        assert N == len(wv), "'weights input should have'"

    A = scalg.block_diag(*[getattr(ss, 'A') for ss in SS_list])
    B = np.block([[getattr(ss, 'B')] for ss in SS_list])

    if wv is None:
        C = np.block([getattr(ss, 'C') for ss in SS_list])
    else:
        C = np.block([ww * getattr(ss, 'C') for ww, ss in zip(wv, SS_list)])

    D = np.zeros_like(SS_list[0].D)
    for ii in range(N):
        if wv is None:
            D += SS_list[ii].D
        else:
            D += wv[ii] * SS_list[ii].D

    return StateSpace(A, B, C, D, SS_list[0].dt)


def sum_ss(SS1, SS2, negative=False):
    """
    Given 2 systems or gain matrices (or a combination of the two) having the
    same amount of input/output, the function returns a gain or state space
    model summing the two. Namely, given:
        u -> SS1 -> y1
        u -> SS2 -> y2
    we obtain:
        u -> SStot -> y1+y2 	if negative=False
    """
    type_dlti = scsig.ltisys.StateSpaceDiscrete

    if isinstance(SS1, np.ndarray) and isinstance(SS2, np.ndarray):
        SStot = SS1 + SS2

    elif isinstance(SS1, np.ndarray) and isinstance(SS2, type_dlti):
        Kmat = SS1
        A = SS2.A
        B = SS2.B
        C = SS2.C
        D = SS2.D + Kmat
        SStot = scsig.StateSpace(A, B, C, D, dt=SS2.dt)

    elif isinstance(SS1, type_dlti) and isinstance(SS2, np.ndarray):
        Kmat = SS2
        A = SS1.A
        B = SS1.B
        C = SS1.C
        D = SS1.D + Kmat

        SStot = scsig.StateSpace(A, B, C, D, dt=SS2.dt)

    elif isinstance(SS1, type_dlti) and isinstance(SS2, type_dlti):

        assert np.abs(1. - SS1.dt / SS2.dt) < 1e-13, \
            'State-space models must have the same time-step'

        Nin01, Nout01 = SS1.inputs, SS1.outputs
        Nin02, Nout02 = SS2.inputs, SS2.outputs
        Nx01, Nx02 = SS1.A.shape[0], SS2.A.shape[0]

        A = np.block([[SS1.A, np.zeros((Nx01, Nx02))],
                      [np.zeros((Nx02, Nx01)), SS2.A]])
        B = np.block([[SS1.B, ],
                      [SS2.B]])
        C = np.block([SS1.C, SS2.C])
        D = SS1.D + SS2.D

        SStot = scsig.StateSpace(A, B, C, D, dt=SS1.dt)


    else:
        raise NameError('Input types not recognised in any implemented option!')

    return SStot


def scale_SS(SSin, input_scal=1., output_scal=1., state_scal=1., byref=True):
    r"""
    Given a state-space system, scales the equations such that the original
    input and output, :math:`u` and :math:`y`, are substituted by :math:`u_{AD}=\frac{u}{u_{ref}}`
    and :math:`y_{AD}=\frac{y}{y_{ref}}`.

    If the original system has form:

        .. math::
                \mathbf{x}^{n+1} &= \mathbf{A\,x}^n + \mathbf{B\,u}^n \\
                \mathbf{y}^{n} &= \mathbf{C\,x}^{n} + \mathbf{D\,u}^n

    the transformation is such that:

        .. math::
                \mathbf{x}^{n+1} &= \mathbf{A\,x}^n + \mathbf{B}\,\frac{u_{ref}}{x_{ref}}\mathbf{u_{AD}}^n \\
                \mathbf{y_{AD}}^{n+1} &= \frac{1}{y_{ref}}(\mathbf{C}\,x_{ref}\,\mathbf{x}^{n+1} + \mathbf{D}\,u_{ref}\,\mathbf{u_{AD}}^n)

    By default, the state-space model is manipulated by reference (``byref=True``)

    Args:
        SSin (scsig.dlti): original state-space formulation
        input_scal (float or np.ndarray): input scaling factor :math:`u_{ref}`. It can be a float or an array, in which
                                          case the each element of the input vector will be scaled by a different
                                          factor.
        output_scal (float or np.ndarray): output scaling factor :math:`y_{ref}`. It can be a float or an array, in which
                                           case the each element of the output vector will be scaled by a different
                                           factor.
        state_scal (float or np.ndarray): state scaling factor :math:`x_{ref}`. It can be a float or an array, in which
                                          case the each element of the state vector will be scaled by a different
                                          factor.
        byref (bool): state space manipulation order

    Returns:
          scsig.dlti: scaled state space formulation
    """

    # check input:
    Nin, Nout = SSin.inputs, SSin.outputs
    Nstates = SSin.A.shape[0]

    if isinstance(input_scal, (list, np.ndarray)):
        assert len(input_scal) == Nin, \
            'Length of input_scal not matching number of state-space inputs!'
    else:
        input_scal = Nin * [input_scal]

    if isinstance(output_scal, (list, np.ndarray)):
        assert len(output_scal) == Nout, \
            'Length of output_scal not matching number of state-space outputs!'
    else:
        output_scal = Nout * [output_scal]

    if isinstance(state_scal, (list, np.ndarray)):
        assert len(state_scal) == Nstates, \
            'Length of state_scal not matching number of state-space states!'
    else:
        state_scal = Nstates * [state_scal]

    if byref:
        SS = SSin
    else:
        print('deep-copying state-space model before scaling')
        SS = copy.deepcopy(SSin)

    # update input related matrices
    for ii in range(Nin):
        SS.B[:, ii] = SS.B[:, ii] * input_scal[ii]
        SS.D[:, ii] = SS.D[:, ii] * input_scal[ii]
        # SS.B[:,ii]*=input_scal[ii]
        # SS.D[:,ii]*=input_scal[ii]

    # update output related matrices
    for ii in range(Nout):
        SS.C[ii, :] = SS.C[ii, :] / output_scal[ii]
        SS.D[ii, :] = SS.D[ii, :] / output_scal[ii]
        # SS.C[ii,:]/=output_scal[ii]
        # SS.D[ii,:]/=output_scal[ii]

    # update state related matrices
    for ii in range(Nstates):
        SS.B[ii, :] = SS.B[ii, :] / state_scal[ii]
        SS.C[:, ii] = SS.C[:, ii] * state_scal[ii]
        # SS.B[ii,:]/=state_scal[ii]
        # SS.C[:,ii]*=state_scal[ii]

    return SS


def simulate(SShere, U, x0=None):
    """
    Routine to simulate response to generic input.

    Warnings:
        This routine is for testing and may lack of robustness. Use
        scipy.signal instead.
    """

    A, B, C, D = SShere.A, SShere.B, SShere.C, SShere.D

    NT = U.shape[0]
    Nx = A.shape[0]
    Ny = C.shape[0]

    X = np.zeros((NT, Nx))
    Y = np.zeros((NT, Ny))

    if x0 is not None: X[0] = x0
    if len(U.shape) == 1:
        U = U.reshape((NT, 1))

    Y[0] = libsp.dot(C, X[0]) + libsp.dot(D, U[0])

    for ii in range(1, NT):
        X[ii] = libsp.dot(A, X[ii - 1]) + libsp.dot(B, U[ii - 1])
        Y[ii] = libsp.dot(C, X[ii]) + libsp.dot(D, U[ii])

    return Y, X


def Hnorm_from_freq_resp(gv, method):
    """
    Given a frequency response over a domain kv, this funcion computes the
    H norms through numerical integration.

    Note that if kv[-1]<np.pi/dt, the method assumed gv=0 for each frequency
    kv[-1]<k<np.pi/dt.

    Warning: only use for SISO systems! For MIMO definitions are different
    """

    if method == 'H2':
        Nk = len(gv)
        gvsq = gv * gv.conj()
        Gnorm = np.sqrt(np.trapz(gvsq / (Nk - 1.)))

    elif method == 'Hinf':
        Gnorm = np.linalg.norm(gv, np.inf)

    if np.abs(Gnorm.imag / Gnorm.real) > 1e-16:
        raise NameError('Norm is not a real number. Verify data/algorithm!')

    return Gnorm


def adjust_phase(y, deg=True):
    """
    Modify the phase y of a frequency response to remove discontinuities.
    """

    if deg is True:
        shift = 360.
    else:
        shift = 2. * np.pi

    dymax = 0.0

    N = len(y)
    for ii in range(N - 1):
        dy = y[ii + 1] - y[ii]
        if np.abs(dy) > dymax: dymax = np.abs(dy)
        if dy > 0.97 * shift:
            print('Subtracting shift to frequency response phase diagram!')
            y[ii + 1::] = y[ii + 1::] - shift

        elif dy < -0.97 * shift:
            y[ii + 1::] = y[ii + 1::] + shift
            print('Adding shift to frequency response phase diagram!')

    return y


# -------------------------------------------------------------- Special Models


def SSderivative(ds):
    """
    Given a time-step ds, and an single input time history u, this SS model
    returns the output y=[u,du/ds], where du/dt is computed with second order
    accuracy.
    """

    A = np.array([[0]])
    Bm1 = np.array([0.5 / ds])
    B0 = np.array([[-2 / ds]])
    B1 = np.array([[1.5 / ds]])
    C = np.array([[0], [1]])
    D = np.array([[1], [0]])

    # change state
    Aout, Bout, Cout, Dout = SSconv(A, B0, B1, C, D, Bm1)

    return Aout, Bout, Cout, Dout


def SSintegr(ds, method='trap'):
    """
    Builds a state-space model of an integrator.

    - method: Numerical scheme. Available options are:
        - 1tay: 1st order Taylor (fwd)
                I[ii+1,:]=I[ii,:] + ds*F[ii,:]
        - trap: I[ii+1,:]=I[ii,:] + 0.5*dx*(F[ii,:]+F[ii+1,:])

        Note: other option can be constructured if information on derivative of
        F is available  (for e.g.)
    """

    A = np.array([[1]])
    C = np.array([[1.]])
    D = np.array([[0.]])

    if method == '1tay':
        Bm1 = np.array([0.])
        B0 = np.array([[ds]])
        B1 = np.array([[0.]])
        Aout, Bout, Cout, Dout = A, B0, C, D

    elif method == 'trap':
        Bm1 = np.array([0.])
        B0 = np.array([[.5 * ds]])
        B1 = np.array([[.5 * ds]])
        Aout, Bout, Cout, Dout = SSconv(A, B0, B1, C, D, Bm1=None)

    else:
        raise NameError('Method %s not available!' % method)

    # change state

    return Aout, Bout, Cout, Dout


def build_SS_poly(Acf, ds, negative=False):
    """
    Builds a discrete-time state-space representation of a polynomial system
    whose frequency response has from:
        Ypoly[oo,ii](k) = -A2[oo,ii] D2(k) - A1[oo,ii] D1(k) - A0[oo,ii]
    where C1,D2 are discrete-time models of first and second derivatives, ds is
    the time-step and the coefficient matrices are such that:
        A{nn}=Acf[oo,ii,nn]
    """

    Nout, Nin, Ncf = Acf.shape
    assert Ncf == 3, 'Acf input last dimension must be equal to 3!'

    Ader, Bder, Cder, Dder = SSderivative(ds)
    SSder = scsig.dlti(Ader, Bder, Cder, Dder, dt=ds)
    SSder02 = series(SSder, join2(np.array([[1]]), SSder))

    SSder_all = copy.deepcopy(SSder02)
    for ii in range(Nin - 1):
        SSder_all = join2(SSder_all, SSder02)

    # Build polynomial forcing terms
    sign = 1.0
    if negative == True: sign = -1.0

    A0 = Acf[:, :, 0]
    A1 = Acf[:, :, 1]
    A2 = Acf[:, :, 2]
    Kforce = np.zeros((Nout, 3 * Nin))
    for ii in range(Nin):
        Kforce[:, 3 * ii] = sign * (A0[:, ii])
        Kforce[:, 3 * ii + 1] = sign * (A1[:, ii])
        Kforce[:, 3 * ii + 2] = sign * (A2[:, ii])
    SSpoly_neg = addGain(SSder_all, Kforce, where='out')

    return SSpoly_neg


def butter(order, Wn, N=1, btype='lowpass'):
    """
    build MIMO butterworth filter of order ord and cut-off freq over Nyquist
    freq ratio Wn.
    The filter will have N input and N output and N*ord states.

    Note: the state-space form of the digital filter does not depend on the
    sampling time, but only on the Wn ratio.
    As a result, this function only returns the A,B,C,D matrices of the filter
    state-space form.
    """

    # build DLTI SISO
    num, den = scsig.butter(order, Wn, btype=btype, analog=False, output='ba')
    Af, Bf, Cf, Df = scsig.tf2ss(num, den)
    SSf = scsig.dlti(Af, Bf, Cf, Df, dt=1.0)

    SStot = SSf
    for ii in range(1, N):
        SStot = join2(SStot, SSf)

    return SStot.A, SStot.B, SStot.C, SStot.D


# ----------------------------------------------------------------------- Utils

def get_freq_from_eigs(eigs, dlti=True):
    """
    Compute natural freq corresponding to eigenvalues, eigs, of a continuous or
    discrete-time (dlti=True) systems.

    Note: if dlti=True, the frequency is normalised by (1./dt), where dt is the
    DLTI time-step - i.e. the frequency in Hertz is obtained by multiplying fn by
    (1./dt).
    """
    if dlti:
        fn = 0.5 * np.angle(eigs) / np.pi
    else:
        fn = np.abs(eigs.imag)
    return fn


def eigvals(a, dlti=False):
    """
    Ordered eigenvalaues of a matrix.

    Args:
        a (np.ndarray): Matrix.
        dlti (bool): If true, the eigenvalues are ordered by modulus, else by real part.

    Returns:
        np.ndarray: ordered set of eigenvalues.
    """
    eigs = np.linalg.eigvals(a)

    if dlti:
        order = np.argsort(np.abs(eigs))
    else:
        order = np.argsort(eigs.real)

    return eigs[order]


# --------------------------------------------------------------------- Testing


def random_ss(Nx, Nu, Ny, dt=None, use_sparse=False, stable=True):
    """
    Define random system from number of states (Nx), inputs (Nu) and output (Ny).

    Args:
        Nx (int): Number of states
        Nu (int): Number of inputs
        Ny (int): Number of outputs
        dt (float (optional)): Time step for discrete systems
        use_sparse (bool): Use sparse matrices
        stable (bool): Ensure the system is stable

    Returns:
        StateSpace: State space object
    """

    A = np.random.rand(Nx, Nx)
    if stable:
        ev, U = np.linalg.eig(A)
        evabs = np.abs(ev)

        for ee in range(len(ev)):
            if evabs[ee] > 0.99:
                ev[ee] /= 1.1 * evabs[ee]
        A = np.dot(U * ev, np.linalg.inv(U)).real
    B = np.random.rand(Nx, Nu)
    C = np.random.rand(Ny, Nx)
    D = np.random.rand(Ny, Nu)

    if use_sparse:
        ss = StateSpace(libsp.csc_matrix(A),
                        libsp.csc_matrix(B),
                        libsp.csc_matrix(C),
                        libsp.csc_matrix(D),
                        dt=dt)
    else:
        ss = StateSpace(A, B, C, D, dt=dt)

    ss.initialise_variables(({'name': 'input_variable', 'size': Nu}), var_type='in')
    ss.initialise_variables(({'name': 'output_variable', 'size': Ny}), var_type='out')
    ss.initialise_variables(({'name': 'state_variable', 'size': Nx}), var_type='state')

    return ss


def compare_ss(SS1, SS2, tol=1e-10, Print=False):
    """
    Assert matrices of state-space models are identical
    """

    era = np.max(np.abs(libsp.dense(SS1.A) - libsp.dense(SS2.A)))
    if Print: print('Max. error A: %.3e' % era)

    erb = np.max(np.abs(libsp.dense(SS1.B) - libsp.dense(SS2.B)))
    if Print: print('Max. error B: %.3e' % erb)

    erc = np.max(np.abs(libsp.dense(SS1.C) - libsp.dense(SS2.C)))
    if Print: print('Max. error C: %.3e' % erc)

    erd = np.max(np.abs(libsp.dense(SS1.D) - libsp.dense(SS2.D)))
    if Print: print('Max. error D: %.3e' % erd)

    assert era < tol, 'Error A matrix %.2e>%.2e' % (era, tol)
    assert erb < tol, 'Error B matrix %.2e>%.2e' % (erb, tol)
    assert erc < tol, 'Error C matrix %.2e>%.2e' % (erc, tol)
    assert erd < tol, 'Error D matrix %.2e>%.2e' % (erd, tol)

    # print('System matrices identical within tolerance %.2e'%tol)
    return (era, erb, erc, erd)


# -----------------------------------------------------------------------------


def ss_to_scipy(ss):
    """
    Converts to a scipy.signal linear time invariant system

    Args:
        ss (libss.StateSpace): SHARPy state space object

    Returns:
        scipy.signal.dlti
    """

    if ss.dt == None:
        sys = scsig.lti(ss.A, ss.B, ss.C, ss.D)
    else:
        sys = scsig.dlti(ss.A, ss.B, ss.C, ss.D, dt=ss.dt)

    return sys
