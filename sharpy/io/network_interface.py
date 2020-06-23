import socket
import selectors
import logging
import struct

sel = selectors.DefaultSelector()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=20)
logger = logging.getLogger(__name__)


class Network:

    def __init__(self, host, port):

        self.addr = (host, port)

        self.sock = None
        self.sel = sel

        self.clients = list()

        self.queue = None # queue object

    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        events = get_events(mode)
        self.sel.modify(self.sock, events, data=self)

    def initialise(self, mode):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        logger.info('Binded socket to {}'.format(self.addr))
        events = get_events(mode)
        self.sock.setblocking(False)
        self.sel.register(self.sock, events, data=self)

    def send(self, msg, dest_addr):
        if type(dest_addr) is list:
            for dest in dest_addr:
                self._sendto(msg, dest)
        elif type(dest_addr) is tuple:
            self._sendto(msg, dest_addr)

    def set_queue(self, queue):
        self.queue = queue

    def _sendto(self, msg, address):
        logger.info('Network - Sending')
        msg = struct.pack('f', msg) # need to move encoding to dedicated message processing
        self.sock.sendto(msg, address)
        logger.info('Network - Sent data packet to {}'.format(address))

    def receive(self):
        r_msg, client_addr = self.sock.recvfrom(413)  # adapt message length
        logger.info('Received data packet from {}'.format(client_addr))
        self.add_client(client_addr)
        # r_msg = struct.unpack('f', r_msg)  # need to move decoding to dedicated message processing
        return r_msg

    def process_events(self, mask):  # should only have the relevant queue
        logger.info('should not be here')
        pass
        # if mask and selectors.EVENT_READ:
        #     logger.info('Network - Receiving')
        #     msg = self.receive()
        #     # would need to process msg beforehand
        #     in_queue.put(msg)
        #     logger.info('Network - Placed message in the queue')
        #
        # if mask and selectors.EVENT_WRITE:
        #     msg = out_queue.get()
        #     logger.info('Network - Got message from the queue')
        #     self.send(msg, self.clients)

        # return in_queue, out_queue: not needed, processing done on original objects

    def add_client(self, client_addr):
        if client_addr not in self.clients:
            self.clients.append(client_addr)
            logger.info('Added new client to list {}'.format(client_addr))

    def close(self):
        self.sel.unregister(self.sock)
        logger.info('Unregistered socket from selectors')
        self.sock.close()
        logger.info('Closed socket')


class OutNetwork(Network):

    def process_events(self, mask):
        value = None
        self.sock.setblocking(False)
        if mask and selectors.EVENT_READ:
            logger.info('Out Network - waiting for request for data')
            msg = self.receive()
            # get variable that has been demanded, this would be easy if a SetOfVariables was sent in the queue
            # logger.info('Received request for data {}'.format(msg))
            logger.info('Received request for data')
            # if mask and selectors.EVENT_WRITE:
            logger.info('Out Network ready to receive from the queue')
            value = self.queue.get()  # check that it waits for the queue not to be empty
            logger.info('Out Network - got message from queue')

            self.send(value, self.clients)


class InNetwork(Network):

    def process_events(self, mask):
        self.sock.setblocking(False)
        if mask and selectors.EVENT_READ:
            logger.info('In Network - waiting for input data')
            msg = self.receive()
            # any required processing
            self.queue.put(msg)
            logger.info('In Network - put data in the queue')


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
