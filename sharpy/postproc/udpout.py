from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.io.network_interface as network_interface
import sharpy.utils.cout_utils as cout


@solver
class UDPout(BaseSolver):
    """
    Send output data via UDP

    This post-processor is in essence a wrapper of the :class:`~sharpy.io.network_interface.NetworkLoader`
    where only an output network adapter is created.

    """
    solver_id = 'UDPout'
    solver_classification = 'post-processor'

    settings_types = network_interface.NetworkLoader.settings_types.copy()
    settings_default = network_interface.NetworkLoader.settings_default.copy()
    settings_description = network_interface.NetworkLoader.settings_description.copy()

    # Remove unnecessary settings from NetworkLoader (all related to the inputs)
    del settings_default['input_network_settings']
    del settings_types['input_network_settings']
    del settings_description['input_network_settings']

    del settings_default['received_data_filename']
    del settings_types['received_data_filename']
    del settings_description['received_data_filename']

    del settings_types['send_output_to_all_clients']
    del settings_default['send_output_to_all_clients']
    del settings_description['send_output_to_all_clients']

    table = settings_utils.SettingsTable()
    __doc__ += table.generate(settings_types, settings_default, settings_description,
                              header_line='This post-processor takes in the following settings, for a more '
                                          'detailed description see '
                                          ':class:`~sharpy.io.network_interface.NetworkLoader`')

    def __init__(self):
        self.settings = None
        self.data = None

        self.network_loader = None
        self.out_network = None
        self.set_of_variables = None

        self.ts_max = 0

        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings

        self.network_loader = network_interface.NetworkLoader()
        self.network_loader.initialise(in_settings=self.settings)

        self.set_of_variables = self.network_loader.get_inout_variables()

        self.out_network = self.network_loader.get_networks(networks='out')

        self.ts_max = self.data.ts + 1

        self.caller = caller

    def run(self, **kwargs):
        
        online = settings_utils.set_value_or_default(kwargs, 'online', False)

        if online:
            self.set_of_variables.get_value(self.data)
            msg = self.set_of_variables.encode()
            self.out_network.send(msg, self.out_network.clients)
        else:
            for ts_index in range(self.ts_max):
                self.set_of_variables.get_value(self.data, timestep_index=ts_index)
                msg = self.set_of_variables.encode()
                self.out_network.send(msg, self.out_network.clients)

            cout.cout_wrap('...Finished', 1)
            self.shutdown()

        return self.data

    def shutdown(self):
        self.out_network.close()
