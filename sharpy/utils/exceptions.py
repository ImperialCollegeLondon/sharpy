import sharpy.utils.cout_utils as cout


class DefaultValueBaseException(Exception):
    def __init__(self, variable, value, message=''):
        super().__init__(message)


class NoDefaultValueException(DefaultValueBaseException):
    def __init__(self, variable, value=None, message=''):
        super().__init__(message, value)
        cout.cout_wrap(cout.separator, 3)
        cout.cout_wrap("The variable " + variable + " has no default value, please indicate one", 3)
        cout.cout_wrap(cout.separator, 3)


class NotValidInputFile(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotImplementedSolver(Exception):
    def __init__(self, solver_name, message=''):
        super().__init__(message)
        cout.cout_wrap("The solver " + solver_name + " is not implemented. Check the list of available solvers when starting SHARPy", 3)
