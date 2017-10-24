from abc import ABCMeta, abstractmethod
import sharpy.utils.cout_utils as cout
import os


class BaseStructure(metaclass=ABCMeta):
    @abstractmethod
    def generate(self, in_data, settings):
        pass

