from sharpy.utils.solver_interface import solver
import sharpy.aero.models.nonliftingbodygrid as nonliftingbodygrid
import sharpy.utils.settings as settings_utils
from sharpy.solvers.gridloader import GridLoader


@solver
class NonliftingbodygridLoader(GridLoader):
    """
    ``NonliftingbodygridLoader`` class, inherited from ``GridLoader``

    Generates aerodynamic grid for nonlifting bodies based on the input data

    Args:
        data (PreSharpy): ``ProblemData`` class structure

    Attributes:
        settings (dict): Name-value pair of the settings employed by the aerodynamic solver
        settings_types (dict): Acceptable types for the values in ``settings``
        settings_default (dict): Name-value pair of default values for the aerodynamic settings
        data (ProblemData): class structure
        file_name (str): name of the ``.nonlifting_body.h5`` HDF5 file
        aero: empty attribute
        aero_data_dict (dict): key-value pairs of aerodynamic data


    """
    solver_id = 'NonliftingbodygridLoader'
    solver_classification = 'loader'

    def __init__(self):
        super().__init__
        self.file_name = '.nonlifting_body.h5'

        # nonlifting_body storage
        self.nonlifting_body = None

    def run(self, **kwargs):
        self.data.nonlifting_body = nonliftingbodygrid.NonliftingBodyGrid()
        self.data.nonlifting_body.generate(self.data_dict,
                                           self.data.structure,
                                           self.settings,
                                           self.data.ts)
        return self.data
