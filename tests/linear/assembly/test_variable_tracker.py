import unittest

from sharpy.linear.utils.ss_interface import InputVariable, LinearVector, OutputVariable


class TestVariables(unittest.TestCase):

    def setUp(self):
        # initialising with index out of order
        Kzeta = 4
        input_variables_list = [InputVariable('zeta', size=3 * Kzeta, index=0),
                                InputVariable('zeta_dot', size=3 * Kzeta, index=1),  # this should be one
                                InputVariable('u_gust', size=3 * Kzeta, index=2)]

        self.input_variables = LinearVector(input_variables_list)

    def test_initialisation(self):
        Kzeta = 4
        input_variables_list = [InputVariable('zeta', size=3 * Kzeta, index=0),
                                InputVariable('u_gust', size=3 * Kzeta, index=2),
                                InputVariable('zeta_dot', size=3 * Kzeta, index=1)]

        input_variables = LinearVector(input_variables_list)

        sorted_indices = [variable.index for variable in input_variables]
        variable_names = [variable.name for variable in input_variables]
        assert sorted_indices == [0, 1, 2], 'Error sorting indices in initialisation'
        assert variable_names == ['zeta', 'zeta_dot', 'u_gust'], 'Error sorting indices in initialisation'

    def test_remove(self):

        self.input_variables.remove('zeta_dot')

        sorted_indices = [variable.index for variable in self.input_variables]
        assert sorted_indices == [0, 1], 'Error removing variable'

    def test_modify(self):

        new_values = {'name': 'new_u_gust',
                      'size': 5,
                      }

        self.input_variables.modify('u_gust', **new_values)

        assert self.input_variables.get_variable_from_name('new_u_gust').size == 5, 'Variable not modified correctly'

    def test_add(self):

        new_variable = InputVariable('control_surface', size=2, index=3)

        self.input_variables.add(new_variable)
        self.input_variables.update_locations()

        # test it's properly included - it will raise an error internally if not
        included_var = self.input_variables.get_variable_from_name('control_surface')

        new_variable_repeated_index = InputVariable('control_surface_2', size=2, index=3)
        try:
            self.input_variables.add(new_variable_repeated_index)
        except IndexError:
            # correct behaviour
            pass
        else:
            raise IndexError('Variable was added when it should have been rejected because of repeated index')

        new_variable_inbetween = InputVariable('first_variable', size=2, index=-1)
        self.input_variables.add(new_variable_inbetween)
        self.input_variables.update_locations()

        assert self.input_variables[0].name == new_variable_inbetween.name, 'New variable not added at the start of' \
                                                                            ' the list'

        self.input_variables.add('var_from_string', size=3, index=10)
        assert self.input_variables[-1].name == 'var_from_string', 'Variable from string not added correctly'

        self.input_variables.update_indices()

    def test_append(self):
        self.input_variables.append('last_var', size=3)
        self.input_variables.update_indices()

        assert self.input_variables[-1].name == 'last_var', 'Variable not properly appended'

        # test adding an incorrect input
        out_var = OutputVariable('out_var', size=1, index=4)

        try:
            self.input_variables.append(out_var)
        except TypeError:
            pass  # this is what should happen
        else:
            raise TypeError('Error. Able to append an output variable to an input variable')

        self.input_variables.append(InputVariable(name='last_last_var', size=2, index=0))

        assert self.input_variables[-1].name == 'last_last_var', 'Variable not properly appended'

    def test_merge(self):
        second_variables_list = [InputVariable('eta', size=3, index=0),
                                 InputVariable('eta_dot', size=3, index=1)]

        second_vector = LinearVector(second_variables_list)

        merged_vector = LinearVector.merge(self.input_variables, second_vector)

        assert merged_vector[-1].name == 'eta_dot', 'Vectors not coupled properly'
        assert merged_vector[2].name == 'u_gust', 'Vectors not coupled properly'

    def test_copy(self):
        vec1 = self.input_variables
        vec2 = vec1.copy()

        assert vec1 is not vec2, 'Object not deep copied'

    def test_transform(self):

        new_vector = LinearVector.transform(self.input_variables, to_type=OutputVariable)

        LinearVector.check_same_vectors(new_vector, self.input_variables)
        assert new_vector.num_variables == self.input_variables.num_variables, 'Number of variables not equal'
        assert new_vector.size == self.input_variables.size, 'Size not equal'

        for i_var in range(new_vector.num_variables):
            assert new_vector[i_var].name == self.input_variables[i_var].name, 'Name does not match'
            assert new_vector[i_var].index == self.input_variables[i_var].index, 'Index does not match'
            assert new_vector[i_var].size == self.input_variables[i_var].size, 'Size does not match'

        new_vector.append(OutputVariable('last_var', size=2, index=0))

        assert new_vector.num_variables == self.input_variables.num_variables + 1, 'Number of variables got ' \
                                                                                   'updated in the original object'
        assert new_vector[-1].name == 'last_var', 'Variable not correctly appended'
        assert new_vector[-1].index == new_vector.num_variables - 1, 'Variable not correctly appended'


if __name__ == '__main__':
    unittest.main()
