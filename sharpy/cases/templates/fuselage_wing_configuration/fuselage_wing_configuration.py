import configobj
from .fwc_structure import FWC_Structure
from .fwc_aero import FWC_Aero
from .fwc_fuselage import FWC_Fuselage
import os
import sharpy.sharpy_main


class Fuselage_Wing_Configuration:
    """
        Fuselage_Wing_Configuration is a template class to create an 
        aircraft model of a simple wing-fuselage configuration within
        SHARPy. 
       
    """

    def __init__(self, case_name, case_route, output_route):
        self.case_name = case_name
        self.case_route = case_route
        self.output_route = output_route

        self.structure = None
        self.aero = None
        self.fuselage = None

        self.settings = None

    def init_aeroelastic(self, **kwargs):
        self.clean()
        self.init_structure(**kwargs)
        self.init_aero(**kwargs)
        if not kwargs.get('lifting_only', True):
            self.init_fuselage(**kwargs)

    def init_structure(self, **kwargs):
        self.structure = FWC_Structure(self.case_name, self.case_route, **kwargs)

    def init_aero(self, **kwargs):
        self.aero = FWC_Aero(self.structure, self.case_name, self.case_route,**kwargs)

    def init_fuselage(self, **kwargs):
        self.fuselage = FWC_Fuselage(self.structure, self.case_name, self.case_route,**kwargs)
    def generate(self):
        if not os.path.isdir(self.case_route):
            os.makedirs(self.case_route)
        self.structure.generate()
        self.aero.generate()
        if self.fuselage is not None:
            self.fuselage.generate()

    def create_settings(self, settings):
        file_name = self.case_route + '/' + self.case_name + '.sharpy'
        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()
        self.settings = settings

    def clean(self):
        list_files = ['.fem.h5', '.aero.h5', '.nonlifting_body.h5', '.dyn.h5', '.mb.h5', '.sharpy', '.flightcon.txt']
        for file in list_files:
            path_file = self.case_route + '/' + self.case_name + file
            if os.path.isfile(path_file):
                os.remove(path_file)

    def run(self):
        sharpy.sharpy_main.main(['', self.case_route + '/' + self.case_name + '.sharpy'])

