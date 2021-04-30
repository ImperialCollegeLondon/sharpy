import os
import pickle

from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings

"""CreateSnapshot Solver documentation.

CreateSnapshot stores all the necessary data to restart a simulation
when the execution has been halted.

The *.snapshot.<ts> file contains the current version of the self.data structure.
The data structure is packaged with pickle (https://docs.python.org/3/library/pickle.html), and
it is totally seamless to extract it. The current usage of this is to call this as an inline post processor.
For example, the DynamicCoupled settings would look like:
    {
    'postprocessors': ['BeamLoads', '...', 'CreateSnapshot'],
    'postprocessors_settings': {'BeamLoads': {},
                                'CreateSnapshot': {}}
    }

It has been tested with DynamicCoupled when no settings are modified.

In order to restart the simulation, one has to do:
{
    sharpy <path to the solver.txt> -r <path to the snapshot>
}
It is important to note that the flow setting has to be modified so that
the previously run solvers are not re-run. For example, a standard simulation would have a flow such that:
{
flow = [
        'BeamLoader',
        'AerogridLoader',
        'StaticCoupled',
        'BeamLoads',
        'AerogridPlot',
        'BeamPlot',
        'DynamicCoupled',
        ]
}
Before restarting the solution, we need to comment everything up to DynamicCoupled (not included).
DynamicCoupled will restart at the last stored timestep.

Todo:
    * No tests have been conducted about modifying the settings (for example number of time steps, or
    relaxation factors...)
    * No other solvers have been tested yet.
"""
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

        self.settings_types['symlink'] = 'bool'
        self.settings_default['symlink'] = True

        self.settings = None
        self.data = None
        self.ts = None

        self.filename = None
        self.caller = None
        self.folder = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # create folder for containing files if necessary
        self.folder = data.output_folder + '/aero/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # snapshot prefix
        self.filename = (self.folder + '/' +
                         self.data.settings['SHARPy']['case'] +
                         '.snapshot')
        self.caller = caller

    def snap_name(self, ts=None):
        if ts is None:
            ts = self.ts
        return "%s.%06d" % (self.filename, ts)

    def run(self, online=True):
        self.ts = self.data.ts
        if self.ts % self.settings['frequency'] == 0:
            # clean older files
            if self.settings['keep']:
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
        n_keep = self.settings['keep'] - 1

        # get list of files in directory
        files = [f for f in os.listdir(self.folder)
                 if os.path.isfile(os.path.join(self.folder, f))]

        # arrange by name
        files.sort()

        # make sure the symlink is kept (so out of the list)
        files = [a for a in files if '.snapshot.' in a]

        if len(files) <= n_keep:
            return

        # delete from the list the snapshots we want to keep
        del files[len(files) - n_keep:]

        for file in files:
            os.unlink(os.path.abspath(self.folder + '/' + file))
