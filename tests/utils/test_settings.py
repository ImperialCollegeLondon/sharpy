import sharpy.utils.settings as settings
import sharpy.utils.exceptions as exceptions
import sharpy.utils.cout_utils as cout
from copy import deepcopy
import numpy as np
import unittest
import tests.coupled.static.pazy.generate_pazy as gp
import os


class TestSettings(unittest.TestCase):
    """
    Tests the settings utilities module
    """

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    
    def setUp(self):
        cout.start_writer()

    def tearDown(self):
        cout.finish_writer()

    def test_settings_to_custom_types(self):
        in_dict = dict()
        default_dict = dict()
        types_dict = dict()


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

        in_dict['list_var'] = ['aa', 'bb', '11', 'ss']
        types_dict['list_var'] = 'list(str)'
        default_dict['list_var'] = ['a', 'b']
        split_list = ['aa', 'bb', '11', 'ss']

        in_dict['float_list_var'] = ['1.1', '2.2', '3.3']
        types_dict['float_list_var'] = 'list(float)'
        default_dict['float_list_var'] = np.array([0.0, -1.1])
        split_float_list = np.array([1.1, 2.2, 3.3])

        original_dict = deepcopy(in_dict)

        # assigned values test
        result = settings.to_custom_types(in_dict, types_dict, default_dict)
        # integer variable
        self.assertEqual(in_dict['integer_var'], 1234, 'Integer test for assigned values not passed')
        # float variable
        self.assertEqual(in_dict['float_var'], 1.234, 'Float test for assigned values not passed')
        # string variable
        self.assertEqual(in_dict['str_var'], 'aaaa', 'String test for assigned values not passed')
        # bool variable
        self.assertEqual(in_dict['bool_var'], True, 'Bool test for assigned values not passed')
        # list variable
        for i in range(4):
            self.assertEqual(in_dict['list_var'][i], split_list[i], 'List test for assigned values not passed')
        # float list variable
        for i in range(3):
            self.assertEqual(in_dict['float_list_var'][i], split_float_list[i], 'Floating point list test for assigned values not passed')

        # default values test
        in_default_dict = dict()
        result = settings.to_custom_types(in_default_dict, types_dict, default_dict)
        # integer variable
        self.assertEqual(in_default_dict['integer_var'], default_dict['integer_var'],
                         'Integer test for default values not passed')
        # float variable
        self.assertEqual(in_default_dict['float_var'], default_dict['float_var'],
                         'Float test for default values not passed')
        # string variable
        self.assertEqual(in_default_dict['str_var'], default_dict['str_var'],
                         'String test for default values not passed')
        # bool variable
        self.assertEqual(in_default_dict['bool_var'], default_dict['bool_var'],
                         'Bool test for default values not passed')
        # list(str) variable
        for i in range(2):
            self.assertEqual(in_default_dict['list_var'][i], default_dict['list_var'][i],
                             'String list test for default values not passed')
        # list(float) variable
        for i in range(2):
            self.assertEqual(in_default_dict['float_list_var'][i], default_dict['float_list_var'][i],
                             'float list test for default values not passed')

        # non-existant default values
        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['integer_var'] = None

            # remove value in in_dict
            temp_in_dict = deepcopy(original_dict)
            del temp_in_dict['integer_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['float_var'] = None

            # remove value in in_dict
            temp_in_dict = deepcopy(original_dict)
            del temp_in_dict['float_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['str_var'] = None

            # remove value in in_dict
            temp_in_dict = deepcopy(original_dict)
            del temp_in_dict['str_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['bool_var'] = None

            # remove value in in_dict
            temp_in_dict = deepcopy(original_dict)
            del temp_in_dict['bool_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['list_var'] = None

            # remove value in in_dict
            temp_in_dict = deepcopy(original_dict)
            del temp_in_dict['list_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

        with self.assertRaises(exceptions.NoDefaultValueException):
            temp_default_dict = default_dict.copy()
            temp_default_dict['float_list_var'] = None

            # remove value in in_dict
            temp_in_dict = deepcopy(original_dict)
            del temp_in_dict['float_list_var']
            result = settings.to_custom_types(temp_in_dict, types_dict, temp_default_dict)

    def test_multiple_setting_option(self):
        u_inf = 50
        alpha = 7
        case_name = 'pazy_uinf{:04g}_alpha{:04g}'.format(u_inf * 10, alpha * 10)

        M = 16
        N = 64
        Msf = 1

        cases_folder = self.route_test_dir + '/pazy/cases/'
        output_folder = self.route_test_dir + '/pazy/cases/'
        # run case
        gp.generate_pazy(u_inf, case_name, output_folder, cases_folder,
                         alpha=alpha,
                         M=M,
                         N=N,
                         Msf=Msf,
                         test_multiple_inputs=True)

    def tearDown(self):
        cases_folder = self.route_test_dir + '/pazy/cases/'

        if os.path.isdir(cases_folder):
            import shutil
            shutil.rmtree(cases_folder)
