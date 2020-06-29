import socket
import selectors
import logging
import sharpy.io.message_interface as message_interface
import sharpy.io.inout_variables as inout_variables
import sharpy.utils.settings as settings
import sharpy.io.logger_utils as logger_utils

sel = selectors.DefaultSelector()

# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     level=20)
logger = logging.getLogger(__name__)

client_list = list()

class NetworkLoader:
    """
    Settings for each input or output port

    See Also:
        Endianness: https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['variables_filename'] = 'str'
    settings_default['variables_filename'] = None
    settings_description['variables_filename'] = 'Path to YAML file containing input/output variables'

    settings_types['byte_ordering'] = 'str'
    settings_default['byte_ordering'] = 'little'
    settings_description['byte_ordering'] = 'Desired endianness byte ordering'
    settings_options['byte_ordering'] = ['little', 'big']

    settings_types['input_network_settings'] = 'dict'
    settings_default['input_network_settings'] = dict()
    settings_description['input_network_settings'] = 'Settings for the input network.'

    settings_types['output_network_settings'] = 'dict'
    settings_default['output_network_settings'] = dict()
    settings_description['output_network_settings'] = 'Settings for the output network.'

    settings_types['send_output_to_all_clients'] = 'bool'
    settings_default['send_output_to_all_clients'] = False
    settings_description['send_output_to_all_clients'] = 'Send output to all clients, including those from where the ' \
                                                         'input is received.'

    settings_types['received_data_filename'] = 'str'
    settings_default['received_data_filename'] = ''
    settings_description['received_data_filename'] = 'If not empty, writes received input data to the specified file.'

    settings_types['log_name'] = 'str'
    settings_default['log_name'] = './network_output.log'
    settings_description['log_name'] = 'Network log name'

    settings_types['console_log_level'] = 'str'
    settings_default['console_log_level'] = 'info'
    settings_description['console_log_level'] = 'Minimum logging level in console.'
    settings_options['console_log_level'] = ['debug', 'info', 'warning', 'error']

    settings_types['file_log_level'] = 'str'
    settings_default['file_log_level'] = 'debug'
    settings_description['file_log_level'] = 'Minimum logging level in log file.'
    settings_options['file_log_level'] = ['debug', 'info', 'warning', 'error']

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.settings = None

        self.byte_ordering = '<'

    def initialise(self, in_settings):
        self.settings = in_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 no_ctype=True, options=self.settings_options)

        if self.settings['byte_ordering'] == 'little':
            self.byte_ordering = '<'
        elif self.settings['byte_ordering'] == 'big':
            self.byte_ordering = '>'
        else:
            raise KeyError('Unknown byte ordering {}'.format(self.settings['byte_ordering']))

        logger_utils.load_logger_settings(log_name=self.settings['log_name'],
                                          file_level=self.settings['file_log_level'],
                                          console_level=self.settings['console_log_level'])
        logger.info('Initialising Network Interface. Local host name: {}'.format(socket.gethostname()))

    def get_inout_variables(self):
        set_of_variables = inout_variables.SetOfVariables()
        set_of_variables.load_variables_from_yaml(self.settings['variables_filename'])
        set_of_variables.set_byte_ordering(self.byte_ordering)

        if self.settings['received_data_filename'] != '':
            set_of_variables.set_input_file(self.settings['received_data_filename'])

        return set_of_variables

    def get_networks(self):

        logger.info('Initialising output network')
        out_network = OutNetwork()
        out_network.initialise('w', in_settings=self.settings['output_network_settings'])
        out_network.set_byte_ordering(self.byte_ordering)

        logger.info('Initialising input network')
        in_network = InNetwork()
        in_network.initialise('r', in_settings=self.settings['input_network_settings'])
        in_network.set_byte_ordering(self.byte_ordering)

        if self.settings['send_output_to_all_clients']:
            out_network.set_client_list(client_list)
            in_network.set_client_list(client_list)

        return out_network, in_network


class Network:

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['address'] = 'str'
    settings_default['address'] = '127.0.0.1'
    settings_description['address'] = 'Own network address.'

    settings_types['port'] = 'int'
    settings_default['port'] = 65000
    settings_description['port'] = 'Own port.'

    def __init__(self, host=None, port=None):  # remove args when this is tested

        self.addr = (host, port)  # own address

        self.sock = None
        self.sel = sel

        self.clients = list()

        self.queue = None  # queue object

        self.settings = None

        self._byte_ordering = '<'

    def set_byte_ordering(self, value):
        self._byte_ordering = value

    def set_client_list(self, list_of_clients):
        """
        Set a client list for network.

        Args:
            client_list (list): List of tuples containing ``(HOST, PORT)``
        """
        own_clients = self.clients.copy()  # make a copy of own clients prior to setting the common list
        self.clients = list_of_clients
        self.add_client(own_clients)

    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        events = get_events(mode)
        logger.debug('Modifying selector to {}'.format(mode))
        sel.modify(self.sock, events, data=self)

    def initialise(self, mode, in_settings):
        self.settings = in_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default,
                                 no_ctype=True)
        self.addr = (self.settings['address'], self.settings['port'])

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        logger.info('Binded socket to {}'.format(self.addr))
        events = get_events(mode)
        self.sock.setblocking(False)
        sel.register(self.sock, events, data=self)

    def send(self, msg, dest_addr):
        if type(dest_addr) is list:
            for dest in dest_addr:
                self._sendto(msg, dest)
        elif type(dest_addr) is tuple:
            self._sendto(msg, dest_addr)

    def set_queue(self, queue):
        self.queue = queue

    def _sendto(self, msg, address):
        logger.debug('Network - Sending')
        self.sock.sendto(msg, address)
        logger.debug('Network - Sent data packet to {}'.format(address))

    def receive(self, msg_length=1024):
        r_msg, client_addr = self.sock.recvfrom(msg_length)  # adapt message length
        logger.info('Received a {}-byte long data packet from {}'.format(len(r_msg), client_addr))
        self.add_client(client_addr)
        # r_msg = struct.unpack('f', r_msg)  # need to move decoding to dedicated message processing
        return r_msg
        # return recv_data

    def process_events(self, mask):  # should only have the relevant queue
        logger.debug('Should not be here')
        pass

    def add_client(self, client_addr):
        if type(client_addr) is tuple:
            self._add_client(client_addr)
        elif type(client_addr) is list:
            for client in client_addr:
                self._add_client(client)

    def _add_client(self, client_addr):
        if client_addr not in self.clients:
            self.clients.append(client_addr)
            logger.info('Added new client to list {}'.format(client_addr))

    def close(self):
        self.sel.unregister(self.sock)
        logger.info('Unregistered socket from selectors')
        self.sock.close()
        logger.info('Closed socket')


class OutNetwork(Network):

    settings_types = Network.settings_types.copy()
    settings_default = Network.settings_default.copy()
    settings_description = Network.settings_description.copy()

    settings_types['port'] = 'int'
    settings_default['port'] = 65000
    settings_description['port'] = 'Own port for output network'

    settings_types['send_on_demand'] = 'bool'
    settings_default['send_on_demand'] = True
    settings_description['send_on_demand'] = 'Waits for a signal demanding the output data. Else, sends to destination' \
                                             ' buffer'

    settings_types['destination_address'] = 'list(str)'
    settings_default['destination_address'] = list()  # add check to raise error if send_on_demand false and this is empty
    settings_description['destination_address'] = 'List of addresses to send output data. If ``send_on_demand`` is ' \
                                                  '``False`` this is a required setting.'

    settings_types['destination_ports'] = 'list(int)'
    settings_default['destination_ports'] = list()
    settings_description['destination_ports'] = 'List of ports number for the destination addresses.'

    def initialise(self, mode, in_settings):
        super().initialise(mode, in_settings)

        if self.settings['send_on_demand'] is False and len(self.settings['destination_address']) == 0:
            logger.warning('No destination host address provided')

        clients = list(zip(self.settings['destination_address'], self.settings['destination_ports']))
        self.add_client(clients)

    def process_events(self, mask):

        if mask and selectors.EVENT_READ and not self.queue.empty():
            if self.settings['send_on_demand']:
                logger.info('Out Network - waiting for request for data')
                msg = self.receive()
                # get variable that has been demanded, this would be easy if a SetOfVariables was sent in the queue
                # logger.info('Received request for data {}'.format(msg))
                logger.debug('Received request for data')
        if mask and selectors.EVENT_WRITE and not self.queue.empty():
            logger.debug('Out Network ready to receive from the queue')
            # value = self.queue.get()  # check that it waits for the queue not to be empty
            set_of_vars = self.queue.get()  # always gets latest time step info
            logger.debug('Out Network - got message from queue')
            # for out_idx in set_of_vars.out_variables:
            #     value = set_of_vars[out_idx].value
            value = set_of_vars.encode()
            logger.info('Message of length {} bytes ready to send'.format(len(value)))
            self.send(value, self.clients)
                # self.send(value, self.clients)


class InNetwork(Network):

    settings_types = Network.settings_types.copy()
    settings_default = Network.settings_default.copy()
    settings_description = Network.settings_description.copy()

    settings_types['port'] = 'int'
    settings_default['port'] = 65001
    settings_description['port'] = 'Own port for input network'

    def __init__(self):
        super().__init__()
        self._in_message_length = 1024
        self._recv_buffer = b''

    def set_message_length(self, value):
        self._in_message_length = value
        logger.debug('Set input signal message size to {} bytes'.format(self._in_message_length))

    def process_events(self, mask):
        self.sock.setblocking(False)
        if mask and selectors.EVENT_READ:
            logger.info('In Network - waiting for input data of size {} bytes'.format(self._in_message_length))
            msg = self.receive(self._in_message_length)
            self._recv_buffer += msg
            # any required processing
            # send list of tuples
            if len(self._recv_buffer) == self._in_message_length:
                logger.info('In Network - {}/{} bytes read'.format(len(self._recv_buffer), self._in_message_length))
                list_of_variables = message_interface.decoder(self._recv_buffer, byte_ordering=self._byte_ordering)
                self.queue.put(list_of_variables)
                logger.debug('In Network - put data in the queue')
                self._recv_buffer = b''  # clean up


def get_events(mode):
    if mode == "r":
        events = selectors.EVENT_READ
    elif mode == "w":
        events = selectors.EVENT_WRITE
    elif mode == "rw":
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
    else:
        raise ValueError(f"Invalid events mask mode {repr(mode)}.")

    return events
