import socket
import os
import selectors
import logging
import numpy as np

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.udp_utils as udp

sel = selectors.DefaultSelector()

@solver
class UDPout(BaseSolver):
    """
    Send output data via UDP

    SHARPy listens on port 65432
    """
    solver_id = 'UDPout'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['receiver_hostnames'] = 'list(str)'
    settings_default['receiver_hostnames'] = None
    settings_description['receiver_hostnames'] = 'List of addresses for the receiver hosts.'

    settings_types['receiver_port'] = 'list(int)'
    settings_default['receiver_port'] = None
    settings_description['receiver_port'] = 'Corresponding port number for the receiver hosts.'

    settings_types['structure_variables'] = 'list(str)'
    settings_default['structure_variables'] = ['']
    settings_description['structure_variables'] = 'Variables of ``StructTimeStepInfo`` associated to the frame of ' \
                                                  'reference to be writen'

    settings_types['structure_nodes'] = 'list(int)'
    settings_default['structure_nodes'] = [-1]
    settings_description['structure_nodes'] = 'Number of the nodes to be writen'

    table = settings.SettingsTable()
    __doc__ += table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None
        self.data = None  # type: sharpy.presharpy.PreSharpy

        self.HOST = '127.0.0.1'
        self.PORT = 65432

        self.destinations = []

        self.s = None  # type: socket.socket()

        self.logger = None

        self.messages = []

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 no_ctype=True)

        logfile = self.data.settings['SHARPy']['log_folder'] + '/udp_log.txt'
        logging.basicConfig(filename=logfile,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=0)
        self.logger = logging.getLogger(__name__)
        logging.info('UDP output post-processor initialised')
        logging.info('SHARPy running at {}'.format((self.HOST, self.PORT)))

        n_hosts = len(self.settings['receiver_hostnames'])
        n_ports = len(self.settings['receiver_port'])

        assert n_hosts == n_ports, 'Number of receiving hosts not equal to number of ports provided'

        # Bind local port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logging.info('Binding local SHARPy port at {}'.format((self.HOST, self.PORT)))
        self.s.bind((self.HOST, self.PORT))
        logging.info('Local UDP port binded')

        # Register port in selector for reading and writing
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        sel.register(self.s, events, data=None)

        for receiver_i in range(n_hosts):
            self.destinations.append((self.settings['receiver_hostnames'][receiver_i],
                                      self.settings['receiver_port'][receiver_i]))
            logging.info('Ready to set-up a receiving client at {}'.format(self.destinations[receiver_i]))
            self.messages.append(udp.Message(sel, self.s, self.destinations[receiver_i]))

    def run(self, online=True):

        # SEND TO CLIENTS
        aero_tsteps = len(self.data.aero.timestep_info) - 1
        struct_tsteps = len(self.data.structure.timestep_info) - 1
        ts = max(aero_tsteps, struct_tsteps)

        values_to_send = []  # raw list of all values to be sent
        for var_name in self.settings['structure_variables']:
            var_value = getattr(self.data.structure.timestep_info[ts], var_name)
            for inode in self.settings['structure_nodes']:
                values_to_send.extend(list(var_value[inode]))

        full_msgs, unfilled_msg = np.divmod(len(values_to_send), 7)
        completed_messages = []  # list of lists. Each member list has a maximum of 7 entries
        worked_values = 0
        for i in range(full_msgs):
            completed_messages.append([ts, *values_to_send[worked_values:worked_values+7]])
            worked_values += 7

        if unfilled_msg != 0:
            completed_messages.append([ts, *values_to_send[worked_values:]])

        for message_content in completed_messages:
            for msg in self.messages:
                msg.send(1, *message_content)  # index (the first value) fixed at 1 for now
                logging.info('Sent packet to {}'.format(msg.addr))

        return self.data

    def shutdown(self):
        self.s.close()
        cout.cout_wrap('Closed socket {}'.format((self.HOST, self.PORT)))
        logging.info('Closed socket {}'.format((self.HOST, self.PORT)))
