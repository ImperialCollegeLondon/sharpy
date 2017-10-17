from abc import ABCMeta, abstractmethod
from sharpy.utils.cout_utils import cout_wrap
import os

generators = {}


# decorator for generators
def generator(arg):
    global generators
    try:
        arg.generator_type
    except AttributeError:
        raise AttributeError('Class defined as generator has no generator_type attribute')

    try:
        generators[arg.generator_type]
    except KeyError:
        generators[arg.generator_type] = arg

    return arg


class BaseGenerator(metaclass=ABCMeta):
    @property
    def generator_type(self):
        raise NotImplementedError

    def __init__(self, dict_in):
        pass


def instance_generator(dict_in):
    return generators[dict_in['type']](dict_in)



