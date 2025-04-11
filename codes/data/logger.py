import logging
import logging.config
import os

DEFAULT_NAME = 'dhyper_fol'


def get_logger(name=DEFAULT_NAME, filename=None, verbose=True):
    # get logger for the given name
    level = logging.INFO if verbose else logging.ERROR
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s]: %(message)s'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)

    shell_handler = logging.StreamHandler()
    shell_handler.setFormatter(formatter)
    logger.addHandler(shell_handler)

    if filename is None:
        return logger

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    file_handler = logging.FileHandler(filename, 'a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


GENERAL_SHELL_LOGGER = get_logger(DEFAULT_NAME)