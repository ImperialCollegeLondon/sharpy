"""SHARPy Exception Classes
"""
import sharpy.utils.cout_utils as cout


class DefaultValueBaseException(Exception):
    def __init__(self, variable, value, message=''):
        super().__init__(message)
        
    def output_message(self, message, color_id = 3):
        if cout.cout_wrap is None:
            print(message)
        else:
            cout.cout_wrap.print_separator(3)
            cout.cout_wrap(message, color_id)
            cout.cout_wrap.print_separator(3)


class NoDefaultValueException(DefaultValueBaseException):
    def __init__(self, variable, value=None, message=''):
        super().__init__(message, value)        
        self.output_message("The variable " + variable + " has no default value, please indicate one") 


class NotValidInputFile(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotImplementedSolver(Exception):
    def __init__(self, solver_name, message=''):
        super().__init__(message)
        if cout.cout_wrap is None:
            print("The solver " + solver_name + " is not implemented. Check the list of available solvers when starting SHARPy")
        else:
            cout.cout_wrap("The solver " + solver_name + " is not implemented. Check the list of available solvers when starting SHARPy", 3)


class NotConvergedStructuralSolver(Exception):
    def __init__(self, solver_name, n_iter=None, message=''):
        super().__init__(message)
        cout.cout_wrap("The solver " + solver_name + " did not converge in " + str(n_iter) + " iterations.", 3)


class DocumentationError(Exception):
    """
    Error in documentation
    """
    try:
        cout.cout_wrap('Documentation for module has been given no title')
    except ValueError:
        pass


class NotConvergedSolver(Exception):
    """
    To be raised when the solver does not converge. Before this, SHARPy
    would add a pdb trace, but this causes problems when using SHARPy
    as a black box.
    """
    pass


class NotValidSetting(DefaultValueBaseException):
    """
    Raised when a user gives a setting an invalid value
    """

    def __init__(self, setting, variable, options, value=None, message=''):
        message = 'The setting %s with entry %s is not one of the valid options: %s' % (setting, variable, options)
        super().__init__(variable, value, message=message)
        self.output_message(message, color_id = 4)
        
    
class NotValidSettingType(DefaultValueBaseException):
    """
    Raised when a user gives a setting with an invalid type
    """

    def __init__(self, setting, variable,  data_types, value=None, message=''):
        message = 'The setting %s with entry %s is not one of the valid types: %s' % (setting, variable, data_types)
        super().__init__(variable, value, message=message)
        self.output_message(message, color_id = 4)


class SolverNotFound(Exception):
    def __init__(self, solver_name):
        message = 'The solver %s cannot be found in the list of solvers. Ensure you have spelt the solver name ' \
                  'correctly.' % solver_name
        super().__init__(message)


class NotRecognisedSetting(DefaultValueBaseException):
    """
    Raised when a setting is not recognised
    """
    def __init__(self, setting, value=None, message=''):
        message = 'Unrecognised setting {:s}. Please check input file and/or documentation'.format(setting)
        super().__init__(variable=None, value=None, message=message)
        self.output_message(message, color_id = 4)
