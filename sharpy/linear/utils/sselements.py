"""
Linear State Space Element Class

"""

class Element(object):
    """
    State space member
    """

    def __init__(self):

        self.sys_id = str()  # A string with the name of the element

        self.sys = None  # The actual object
        self.ss = None  # The state space object

        self.settings = dict()

    def initialise(self, data, sys_id):

        self.sys_id = sys_id
        settings = data.linear.settings[sys_id]  # Load settings, the settings should be stored in data.linear.settings
        # data.linear.settings should be created in the class above containing the entire set up

        # Get the actual class object (like lingebm) from a dictionary in the same way that it is done for the solvers
        # in sharpy
        # sys = sys_from_string(sys_id)
        # To use the decorator idea we would first need to instantiate the class. Need to see how this is done with NL
        # SHARPy


    def assemble(self):
        pass