import os
import numpy as np
import pickle
import pathlib
from tvtk.api import tvtk, write_data

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.structure.utils.xbeamlib as xbeamlib


@solver
class CreateSnapshot(BaseSolver):
    solver_id = 'CreateSnapshot'

    def __init__(self):
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['frequency'] = 'int'
        self.settings_default['frequency'] = 5

        self.settings_types['keep'] = 'int'
        self.settings_default['keep'] = 2

        self.settings_types['compression'] = 'str'
        self.settings_default['compression'] = ''
        # TODO not yet implemented

        self.settings_types['folder'] = 'str'
        self.settings_default['folder'] = './snapshots/'

        self.settings_types['symlink'] = 'bool'
        self.settings_default['symlink'] = True

        self.settings = None
        self.data = None
        self.ts = None

        self.filename = None

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # create folder for containing files if necessary
        if not os.path.exists(self.settings['folder']):
            os.makedirs(self.settings['folder'])

        # snapshot prefix
        self.filename = (self.settings['folder'] + '/' +
                         self.data.settings['SHARPy']['case'] +
                         '.snapshot')

    def snap_name(self, ts=None):
        if ts is None:
            ts = self.ts
        return "%s.%06d" % (self.filename, ts)

    def run(self, online=True):
        self.ts = self.data.ts
        if self.ts % self.settings['frequency'].value == 0:
            # clean older files
            if self.settings['keep'].value:
                self.delete_previous_snapshots()

            # create file
            file = self.snap_name()
            with open(file, 'wb') as f:
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # update symlink
            if self.settings['symlink']:
                try:
                    os.unlink(self.filename)
                except FileNotFoundError:
            pass
        os.symlink(os.path.abspath(file), self.filename)

        return self.data

    def delete_previous_snapshots(self):
        n_keep = self.settings['keep'].value - 1

        # get list of files in directory
        files = [f for f in os.listdir(self.settings['folder'])
                 if os.path.isfile(os.path.join(self.settings['folder'], f))]

        # arrange by name
        files.sort()

        # make sure the symlink is kept (so out of the list)
        files = [a for a in files if '.snapshot.' in a]

        if len(files) <= n_keep:
            return

        # delete from the list the snapshots we want to keep
        del files[len(files) - n_keep:]

        for file in files:
            os.unlink(os.path.abspath(self.settings['folder'] + '/' + file))


