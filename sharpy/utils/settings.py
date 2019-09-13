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


def to_custom_types(dictionary, types, default):
    for k, v in types.items():
        if v == 'int':
            try:
                dictionary[k] = cast(k, dictionary[k], int, ct.c_int, default[k])
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = cast(k, default[k], int, ct.c_int, default[k])
                notify_default_value(k, dictionary[k])

        elif v == 'float':
            try:
                dictionary[k] = cast(k, dictionary[k], float, ct.c_double, default[k])
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = cast(k, default[k], float, ct.c_double, default[k])
                notify_default_value(k, dictionary[k])

        elif v == 'str':
            try:
                dictionary[k] = cast(k, dictionary[k], str, str, default[k])
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = cast(k, default[k], eval(v), eval(v), default[k])
                notify_default_value(k, dictionary[k])

        elif v == 'bool':
            try:
                dictionary[k] = cast(k, dictionary[k], str2bool, ct.c_bool, default[k])
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = cast(k, default[k], str2bool, ct.c_bool, default[k])
                notify_default_value(k, dictionary[k])

        elif v == 'list(str)':
            try:
                # if isinstance(dictionary[k], list):
                #     continue
                # dictionary[k] = dictionary[k].split(',')
                # getting rid of leading and trailing spaces
                dictionary[k] = list(map(lambda x: x.strip(), dictionary[k]))
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = default[k].copy()
                notify_default_value(k, dictionary[k])

        elif v == 'list(dict)':
            try:
                # if isinstance(dictionary[k], list):
                #     continue
                # dictionary[k] = dictionary[k].split(',')
                # getting rid of leading and trailing spaces
                for i in range(len(dictionary[k])):
                    dictionary[k][i] = ast.literal_eval(dictionary[k][i])
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = default[k].copy()
                notify_default_value(k, dictionary[k])

        elif v == 'list(float)':
            try:
                if isinstance(dictionary[k], np.ndarray):
                    continue
                if isinstance(dictionary[k], list):
                    for i in range(len(dictionary[k])):
                        dictionary[k][i] = float(dictionary[k][i])
                    dictionary[k] = np.array(dictionary[k])
                    continue
                # dictionary[k] = dictionary[k].split(',')
                # # getting rid of leading and trailing spaces
                # dictionary[k] = list(map(lambda x: x.strip(), dictionary[k]))
                if dictionary[k].find(',') < 0:
                    dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=' ', dtype=ct.c_double)
                else:
                    dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=',', dtype=ct.c_double)
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = default[k].copy()
                notify_default_value(k, dictionary[k])

        elif v == 'list(int)':
            try:
                if isinstance(dictionary[k], np.ndarray):
                    continue
                if isinstance(dictionary[k], list):
                    for i in range(len(dictionary[k])):
                        dictionary[k][i] = int(dictionary[k][i])
                    dictionary[k] = np.array(dictionary[k])
                    continue
                # dictionary[k] = dictionary[k].split(',')
                # # getting rid of leading and trailing spaces
                # dictionary[k] = list(map(lambda x: x.strip(), dictionary[k]))
                if dictionary[k].find(',') < 0:
                    dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=' ').astype(ct.c_int)
                else:
                    dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=',').astype(ct.c_int)
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = default[k].copy()
                notify_default_value(k, dictionary[k])

        elif v == 'list(complex)':
            try:
                if isinstance(dictionary[k], np.ndarray):
                    continue
                if isinstance(dictionary[k], list):
                    for i in range(len(dictionary[k])):
                        dictionary[k][i] = float(dictionary[k][i])
                    dictionary[k] = np.array(dictionary[k])
                    continue
                # dictionary[k] = dictionary[k].split(',')
                # # getting rid of leading and trailing spaces
                # dictionary[k] = list(map(lambda x: x.strip(), dictionary[k]))
                if dictionary[k].find(',') < 0:
                    dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=' ').astype(complex)
                else:
                    dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=',').astype(complex)
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = default[k].copy()
                notify_default_value(k, dictionary[k])

        elif v == 'dict':
            try:
                if not isinstance(dictionary[k], dict):
                    raise TypeError
            except KeyError:
                if default[k] is None:
                    raise exceptions.NoDefaultValueException(k)
                dictionary[k] = default[k].copy()
                notify_default_value(k, dictionary[k])



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
    # NG 9/4/19 - when using these methods after SHARPy it will raise an error since the log file would have been closed
    # at the end of SHARPy
    try:
        cout.cout_wrap('Variable ' + k + ' has no assigned value in the settings file.')
        cout.cout_wrap('    will default to the value: ' + str(v), 1)
    except ValueError:
        pass


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

        self.line_format = ''

        self.table_string = ''

    def generate(self, settings_types, settings_default, settings_description):
        """
        Returns a rst-format table with the settings' names, types, description and default values

        Args:
            settings_types (dict): Setting types.
            settings_default (dict): Settings default value.
            settings_description (dict): Setting description.

        Returns:
            str: .rst formatted string with a table containing the settings' information.
        """
        self.settings_types = settings_types
        self.settings_default = settings_default
        self.n_settings = len(self.settings_types)
        #
        try:
            self.settings_description = settings_description
        except AttributeError:
            pass

        self.set_field_length()
        self.line_format = self.setting_line_format()

        table_string = '\n    ' + 'The settings that this solver accepts are given by a dictionary, with the following key-value pairs:\n'
        table_string += '\n    ' + self.print_divider_line()
        table_string += '    ' + self.print_header()
        table_string += '    ' + self.print_divider_line()
        for setting in self.settings_types:
            table_string += '    ' + self.print_setting(setting)
        table_string += '    ' + self.print_divider_line()

        self.table_string = table_string

        return table_string

    def set_field_length(self):

        field_lengths = [[], [], [], []]
        for setting in self.settings_types:
            stype = str(self.settings_types.get(setting, ''))
            description = self.settings_description.get(setting, '')
            default = str(self.settings_default.get(setting, ''))
            field_lengths[0].append(len(setting) + 4)  # length of name
            field_lengths[1].append(len(stype) + 4)  # length of type + 4 for the rst ``X``
            field_lengths[2].append(len(description))  # length of type + 4 for the rst ``X``
            field_lengths[3].append(len(default) + 4)  # length of type + 4 for the rst ``X``

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
