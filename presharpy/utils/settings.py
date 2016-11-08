import configparser


def load_config_file(file_name: str) -> dict:
    """This function reads the flight condition and solver input files.

    Args:
        file_name (str): contains the path and file name of the file to be read by the ``configparser``
            reader.

    Returns:
        config (dict): a ``ConfigParser`` object that behaves like a dictionary
    """
    config = configparser.ConfigParser()
    config.read(file_name)
    return config
