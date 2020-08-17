import logging


def load_logger_settings(**kwargs):

    log_name = kwargs.get('log_name', './sharpy_network.log')

    file_level = get_logger_level(kwargs.get('file_level', 'debug'))
    console_level = get_logger_level(kwargs.get('console_level', 'info'))

    logger = logging.getLogger()  # Modify root logger settings
    logger.setLevel(logging.DEBUG)
    # # create file handler which logs even debug messages
    fh = logging.FileHandler(log_name, 'w+')
    fh.setLevel(file_level)
    # # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    # # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def get_logger_level(level):
    if level == 'debug':
        mode = logging.DEBUG
    elif level == 'info':
        mode = logging.INFO
    elif level == 'warning':
        mode = logging.WARNING
    elif level == 'error':
        mode = logging.ERROR
    else:
        raise NameError('Unknown mode for logging module')

    return mode
