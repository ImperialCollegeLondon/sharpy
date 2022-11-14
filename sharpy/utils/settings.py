"""
Settings Generator Utilities
"""
import configparser
import ctypes as ct
import numpy as np
import sharpy.utils.exceptions as exceptions
import sharpy.utils.cout_utils as cout
import ast


class DictConfigParser(configparser.ConfigParser):
    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
        return d


def cast(k, v, pytype, ctype, default):
    try:
        # if default is None:
        #     raise TypeError
        val = ctype(pytype(v))
    except KeyError:
        val = ctype(default)
        cout.cout_wrap("--- The variable " + k + " has no given value, using the default " + default, 2)
    except TypeError:
        raise exceptions.NoDefaultValueException(k)
    except ValueError:
        val = ctype(v.value)
    return val


def to_custom_types(dictionary, types, default, options=dict(), no_ctype=True):
    for k, v in types.items():
        if type(v) != list:
            data_type = v
        else:
            if k in dictionary:
                data_type = get_data_type_for_several_options(dictionary[k], v, k)
            else:
                # Choose first data type  in list for default value
                data_type = v[0]
        dictionary[k] = get_custom_type(dictionary, data_type, k, default, no_ctype)
      
    check_settings_in_options(dictionary, types, options)

    unrecognised_settings = []
    for k in dictionary.keys():
        if k not in list(types.keys()):
            unrecognised_settings.append(exceptions.NotRecognisedSetting(k))

    for setting in unrecognised_settings:
        cout.cout_wrap(repr(setting), 4)

    if unrecognised_settings:
        raise Exception(unrecognised_settings)


def get_data_type_for_several_options(dict_value, list_settings_types, setting_name):
    """
    Checks the data type of the setting input in case of several data type options. 
    Only a scalar or list can be the case for these cases.  

    Args:
        dict_values: Dictionary value of processed settings
        list_settings_types (list): Possible setting type options for this setting

    Raises:
        exception.NotValidSetting: if the setting is not allowed.
    """
    for data_type in list_settings_types:
        if 'list' in data_type and (type(dict_value) == list or not np.isscalar(dict_value)):
                return data_type
        elif 'list' not in data_type and np.isscalar(dict_value):
                return data_type
    exceptions.NotValidSettingType(setting_name, dict_value, list_settings_types)

def get_default_value(default_value, k, v, data_type = None, py_type = None):
    if default_value is None:
        raise exceptions.NoDefaultValueException(k)
    if v in ['float', 'int', 'bool']:
        converted_value = cast(k, default_value, py_type, data_type, default_value)
    elif v == 'str':
        converted_value = cast(k, default_value, eval(v), eval(v), default_value)
    else:
        converted_value = default_value.copy()
    notify_default_value(k, converted_value)
    return converted_value

def get_custom_type(dictionary, v, k, default, no_ctype):
    if v == 'int':
        if no_ctype:
            data_type = int
        else:
            data_type = ct.c_int
        try:
            dictionary[k] = cast(k, dictionary[k], int, data_type, default[k])
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v, data_type=data_type, py_type=int)

    elif v == 'float':
        if no_ctype:
            data_type = float
        else:
            data_type = ct.c_double
        try:
            dictionary[k] = cast(k, dictionary[k], float, data_type, default[k])
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v, data_type=data_type, py_type=float)

    elif v == 'str':
        try:
            dictionary[k] = cast(k, dictionary[k], str, str, default[k])
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v)

    elif v == 'bool':
        if no_ctype:
            data_type = bool
        else:
            data_type = ct.c_bool
        try:
            dictionary[k] = cast(k, dictionary[k], str2bool, data_type, default[k])
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v, data_type=data_type, py_type=str2bool)

    elif v == 'list(str)':
        try:
            # if isinstance(dictionary[k], list):
            #     continue
            # dictionary[k] = dictionary[k].split(',')
            # getting rid of leading and trailing spaces
            dictionary[k] = list(map(lambda x: x.strip(), dictionary[k]))
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v)

    elif v == 'list(dict)':
        try:
            # if isinstance(dictionary[k], list):
            #     continue
            # dictionary[k] = dictionary[k].split(',')
            # getting rid of leading and trailing spaces
            for i in range(len(dictionary[k])):
                dictionary[k][i] = ast.literal_eval(dictionary[k][i])
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v)

    elif v == 'list(float)':
        try:
            dictionary[k]
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v)

        if isinstance(dictionary[k], np.ndarray):
            return dictionary[k]
        if isinstance(dictionary[k], list):
            for i in range(len(dictionary[k])):
                dictionary[k][i] = float(dictionary[k][i])
            dictionary[k] = np.array(dictionary[k])
            return dictionary[k]
        # dictionary[k] = dictionary[k].split(',')
        # # getting rid of leading and trailing spaces
        # dictionary[k] = list(map(lambda x: x.strip(), dictionary[k]))
        if dictionary[k].find(',') < 0:
            dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=' ', dtype=ct.c_double)
        else:
            dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=',', dtype=ct.c_double)

    elif v == 'list(int)':
        try:
            dictionary[k]
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v)

        if isinstance(dictionary[k], np.ndarray):
            return dictionary[k]
        if isinstance(dictionary[k], list):
            for i in range(len(dictionary[k])):
                dictionary[k][i] = int(dictionary[k][i])
            dictionary[k] = np.array(dictionary[k])
            return dictionary[k]
        # dictionary[k] = dictionary[k].split(',')
        # # getting rid of leading and trailing spaces
        # dictionary[k] = list(map(lambda x: x.strip(), dictionary[k]))
        if dictionary[k].find(',') < 0:
            dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=' ').astype(ct.c_int)
        else:
            dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=',').astype(ct.c_int)

    elif v == 'list(complex)':
        try:
            dictionary[k]
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v)

        if isinstance(dictionary[k], np.ndarray):
            return dictionary[k]
        if isinstance(dictionary[k], list):
            for i in range(len(dictionary[k])):
                dictionary[k][i] = complex(dictionary[k][i])
            dictionary[k] = np.array(dictionary[k])
            return dictionary[k]
        # dictionary[k] = dictionary[k].split(',')
        # # getting rid of leading and trailing spaces
        # dictionary[k] = list(map(lambda x: x.strip(), dictionary[k]))
        if dictionary[k].find(',') < 0:
            dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=' ').astype(complex)
        else:
            dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=',').astype(complex)

    elif v == 'dict':
        try:
            if not isinstance(dictionary[k], dict):
                raise TypeError('Setting for {:s} is not a dictionary'.format(k))
        except KeyError:
            dictionary[k] = get_default_value(default[k], k, v)
    else:
        raise TypeError('Variable %s has an unknown type (%s) that cannot be casted' % (k, v))
    return dictionary[k]

def check_settings_in_options(settings, settings_types, settings_options):
    """
    Checks that settings given a type ``str`` or ``int`` and allowable options are indeed valid.

    Args:
        settings (dict): Dictionary of processed settings
        settings_types (dict): Dictionary of settings types
        settings_options (dict): Dictionary of options (may be empty)

    Raises:
        exception.NotValidSetting: if the setting is not allowed.
    """
    for k in settings_options:
        if settings_types[k] == 'int':
            try:
                value = settings[k].value
            except AttributeError:
                value = settings[k]
            if value not in settings_options[k]:
                raise exceptions.NotValidSetting(k, value, settings_options[k])

        elif settings_types[k] == 'str':
            value = settings[k]
            if value not in settings_options[k] and value:
                # checks that the value is within the options and that it is not an empty string.
                raise exceptions.NotValidSetting(k, value, settings_options[k])

        elif settings_types[k] == 'list(str)':
            for item in settings[k]:
                if item not in settings_options[k] and item:
                    raise exceptions.NotValidSetting(k, item, settings_options[k])

        else:
            pass  # no other checks implemented / required


def load_config_file(file_name: str) -> dict:
    """This function reads the flight condition and solver input files.

    Args:
        file_name (str): contains the path and file name of the file to be read by the ``configparser``
            reader.

    Returns:
        config (dict): a ``ConfigParser`` object that behaves like a dictionary
    """
    # config = DictConfigParser()
    # config.read(file_name)
    # dict_config = config.as_dict()
    import configobj
    dict_config = configobj.ConfigObj(file_name)
    return dict_config


def str2bool(string):
    false_list = ['false', 'off', '0', 'no']
    if isinstance(string, bool):
        return string
    if isinstance(string, ct.c_bool):
        return string.value

    if not string:
        return False
    elif string.lower() in false_list:
        return False
    else:
        return True


def notify_default_value(k, v):
    cout.cout_wrap('Variable ' + k + ' has no assigned value in the settings file.')
    cout.cout_wrap('    will default to the value: ' + str(v), 1)


class SettingsTable:
    """
    Generates the documentation's setting table at runtime.

    Sphinx is our chosen documentation manager and takes docstrings in reStructuredText format. Given that the SHARPy
    solvers contain several settings, this class produces a table in reStructuredText format with the solver's settings
    and adds it to the solver's docstring.

    This table will then be printed alongside the remaining docstrings.

    To generate the table, parse the setting's description to a solver dictionary named ``settings_description``, in a
    similar fashion to what is done with ``settings_types`` and ``settings_default``. If no description is given it will
    be left blank.

    Then, add at the end of the solver's class declaration method an instance of the ``SettingsTable`` class and a call
    to the ``SettingsTable.generate()`` method.

    Examples:
        The end of the solver's class declaration should contain

        .. code-block:: python

            # Generate documentation table
            settings_table = settings.SettingsTable()
            __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

        to generate the settings table.

    """
    def __init__(self):
        self.n_fields = 4
        self.n_settings = 0
        self.field_length = [0] * self.n_fields
        self.titles = ['Name', 'Type', 'Description', 'Default']

        self.settings_types = dict()
        self.settings_description = dict()
        self.settings_default = dict()
        self.settings_options = dict()
        self.settings_options_strings = dict()

        self.line_format = ''

        self.table_string = ''

    def generate(self, settings_types, settings_default, settings_description, settings_options=dict(), header_line=None):
        """
        Returns a rst-format table with the settings' names, types, description and default values

        Args:
            settings_types (dict): Setting types.
            settings_default (dict): Settings default value.
            settings_description (dict): Setting description.

            header_line (str): Header line description (optional)

        Returns:
            str: .rst formatted string with a table containing the settings' information.
        """
        self.settings_types = settings_types
        self.settings_default = settings_default
        self.n_settings = len(self.settings_types)
        #

        if header_line is None:
            header_line = 'The settings that this solver accepts are given by a dictionary, ' \
                          'with the following key-value pairs:'
        else:
            assert type(header_line) == str, 'header_line not a string, verify order of arguments'

        if type(settings_options) != dict:
            raise TypeError('settings_options is not a dictionary')

        if settings_options:
            # if settings_options are provided
            self.settings_options = settings_options
            self.n_fields += 1
            self.field_length.append(0)
            self.titles.append('Options')
            self.process_options()

        try:
            self.settings_description = settings_description
        except AttributeError:
            pass

        self.set_field_length()
        self.line_format = self.setting_line_format()

        table_string = '\n    ' + header_line + '\n'
        table_string += '\n    ' + self.print_divider_line()
        table_string += '    ' + self.print_header()
        table_string += '    ' + self.print_divider_line()
        for setting in self.settings_types:
            table_string += '    ' + self.print_setting(setting)
        table_string += '    ' + self.print_divider_line()

        self.table_string = table_string

        return table_string

    def process_options(self):
        self.settings_options_strings = self.settings_options.copy()
        for k, v in self.settings_options.items():
            opts = ''
            for option in v:
                opts += ' ``%s``,' %str(option)
            self.settings_options_strings[k] = opts[1:-1]  # removes the initial whitespace and final comma

    def set_field_length(self):

        field_lengths = [[] for i in range(self.n_fields)]
        for setting in self.settings_types:
            stype = str(self.settings_types.get(setting, ''))
            description = self.settings_description.get(setting, '')
            default = str(self.settings_default.get(setting, ''))
            option = str(self.settings_options_strings.get(setting, ''))

            field_lengths[0].append(len(setting) + 4)  # length of name
            field_lengths[1].append(len(stype) + 4)  # length of type + 4 for the rst ``X``
            field_lengths[2].append(len(description))  # length of type
            field_lengths[3].append(len(default) + 4)  # length of type + 4 for the rst ``X``

            if self.settings_options:
                field_lengths[4].append(len(option))

        for i_field in range(self.n_fields):
            field_lengths[i_field].append(len(self.titles[i_field]))
            self.field_length[i_field] = max(field_lengths[i_field]) + 2  # add the two spaces as column dividers

    def print_divider_line(self):
        divider = ''
        for i_field in range(self.n_fields):
            divider += '='*(self.field_length[i_field]-2) + '  '
        divider += '\n'
        return divider

    def print_setting(self, setting):
        type = '``' + str(self.settings_types.get(setting, '')) + '``'
        description = self.settings_description.get(setting, '')
        default = '``' + str(self.settings_default.get(setting, '')) + '``'
        if self.settings_options:
            option = self.settings_options_strings.get(setting, '')
            line = self.line_format.format(['``' + str(setting) + '``', type, description, default, option]) + '\n'
        else:
            line = self.line_format.format(['``' + str(setting) + '``', type, description, default]) + '\n'
        return line

    def print_header(self):
        header = self.line_format.format(self.titles) + '\n'
        return header

    def setting_line_format(self):
        string = ''
        for i_field in range(self.n_fields):
            string += '{0[' + str(i_field) + ']:<' + str(self.field_length[i_field]) + '}'
        return string


def set_value_or_default(dictionary, key, default_val):
    try:
        value = dictionary[key]
    except KeyError:
        value = default_val
    return value

