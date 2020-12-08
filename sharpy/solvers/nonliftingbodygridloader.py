from sharpy.utils.solver_interface import solver
import sharpy.aero.models.nonlifting_body_grid as nonlifting_body_grid
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

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['unsteady'] = 'bool'
    settings_default['unsteady'] = False
    settings_description['unsteady'] = 'Unsteady effects'

    settings_types['aligned_grid'] = 'bool'
    settings_default['aligned_grid'] = True
    settings_description['aligned_grid'] = 'Align grid'

    settings_types['freestream_dir'] = 'list(float)'
    settings_default['freestream_dir'] = [1.0, 0.0, 0.0]
    settings_description['freestream_dir'] = 'Free stream flow direction'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)
    def __init__(self):
        super().__init__()
        self.file_name = '.nonlifting_body.h5'

        # nonlifting_body storage
        self.nonlifting_body = None

    def run(self):
        self.data.nonlifting_body = nonlifting_body_grid.Nonlifting_body_grid()
        self.data.nonlifting_body.generate(self.data_dict,
                                           self.data.structure,
                                           self.settings,
                                           self.data.ts)
        return self.data
