import sharpy.utils.cout_utils as cout


class NoDefaultValueException(Exception):
    def __init__(self, variable, message=''):
        super().__init__(message)
        cout.cout_wrap(cout.separator, 3)
        cout.cout_wrap("The variable " + variable + " has no default value, please indicate one", 3)
        cout.cout_wrap(cout.separator, 3)

