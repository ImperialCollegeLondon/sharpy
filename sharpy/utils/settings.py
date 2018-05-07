import configparser
import ctypes as ct
import numpy as np
import sharpy.utils.exceptions as exceptions
import sharpy.utils.cout_utils as cout
import sharpy.utils.cout_utils as cout

false_list = ['false', 'off', '0', 'no']


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
                    dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=' ', dtype=ct.c_int)
                else:
                    dictionary[k] = np.fromstring(dictionary[k].strip('[]'), sep=',', dtype=ct.c_int)
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
