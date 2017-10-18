import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exceptions
import sharpy.utils.cout_utils as cout
import numpy as np
import unittest


class TestSettings(unittest.TestCase):
    """
    Tests the settings utilities module
    """

    def test_settings_to_custom_types(self):
        cout.cout_quiet()
        in_dict = dict()
        default_dict = dict()

        types_dict = dict()
        default_dict = dict()
        in_dict['integer_var'] = '1234'
        types_dict['integer_var'] = 'int'
        default_dict['integer_var'] = 0

        in_dict['float_var'] = '1.234'
        types_dict['float_var'] = 'float'
        default_dict['float_var'] = 0.0

        in_dict['str_var'] = 'aaaa'
        types_dict['str_var'] = 'str'
        default_dict['str_var'] = 'default_string'

        in_dict['bool_var'] = 'on'
        types_dict['bool_var'] = 'bool'
        default_dict['bool_var'] = False

        # assigned values test
        result = settings.to_custom_types(in_dict, types_dict, default_dict)
        # integer variable
        self.assertEqual(in_dict['integer_var'].value, 1234, 'Integer test for assigned values not passed')
        # float variable
        self.assertEqual(in_dict['float_var'].value, 1.234, 'Float test for assigned values not passed')
        # string variable
        self.assertEqual(in_dict['str_var'], 'aaaa', 'String test for assigned values not passed')
        # bool variable
        self.assertEqual(in_dict['bool_var'].value, True, 'Bool test for assigned values not passed')

        # default values test
        in_default_dict = dict()
        result = settings.to_custom_types(in_default_dict, types_dict, default_dict)
        # integer variable
        self.assertEqual(in_default_dict['integer_var'].value, default_dict['integer_var'],
                         'Integer test for default values not passed')
        # float variable
        self.assertEqual(in_default_dict['float_var'].value, default_dict['float_var'],
                         'Float test for default values not passed')
        # string variable
        self.assertEqual(in_default_dict['str_var'], default_dict['str_var'],
                         'String test for default values not passed')
        # bool variable
        self.assertEqual(in_default_dict['bool_var'].value, default_dict['bool_var'],
                         'Bool test for default values not passed')

        # non-existant default values
        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['integer_var'] = None

            # remove value in in_dict
            temp_in_dict = in_dict.copy()
            del temp_in_dict['integer_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['float_var'] = None

            # remove value in in_dict
            temp_in_dict = in_dict.copy()
            del temp_in_dict['float_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['str_var'] = None

            # remove value in in_dict
            temp_in_dict = in_dict.copy()
            del temp_in_dict['str_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['bool_var'] = None

            # remove value in in_dict
            temp_in_dict = in_dict.copy()
            del temp_in_dict['bool_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        cout.cout_talk()

