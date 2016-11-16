import configparser

false_list = ['false', 'off', '0', 'no']

class DictConfigParser(configparser.ConfigParser):

    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
        return d


def load_config_file(file_name: str) -> dict:
    """This function reads the flight condition and solver input files.

    Args:
        file_name (str): contains the path and file name of the file to be read by the ``configparser``
            reader.

    Returns:
        config (dict): a ``ConfigParser`` object that behaves like a dictionary
    """
    config = DictConfigParser()
    config.read(file_name)
    dict_config = config.as_dict()
    return dict_config


def str2bool(string):
    if not string:
        return False
    elif string.lower() in false_list:
        return False
    else:
        return True