import socket
import selectors
import logging
import struct

sel = selectors.DefaultSelector()

class Network:

    def __init__(self, host, port):

        self.addr = (host, port)

        self.sock = None
        self.sel = sel

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=20)
        self.logger = logging.getLogger(__name__)

        self.clients = list()

    def initialise(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.addr)
        logging.info('Binded socket to {}'.format(self.addr))
        self.sel.register(self.sock, selectors.EVENT_READ | selectors.EVENT_WRITE)

    def send(self, msg, dest_addr):
        logging.info('Network - sending interface')
        if type(dest_addr) is list:
            for dest in dest_addr:
                self._sendto(msg, dest)
        elif type(dest_addr) is tuple:
            self._sendto(msg, dest_addr)

    def _sendto(self, msg, address):
        logging.info('Network - Sending')
        msg = struct.pack('f', msg) # need to move encoding to dedicated message processing
        self.sock.sendto(msg, address)
        logging.info('Network - Sent data packet to {}'.format(address))

    def receive(self):
        r_msg, client_addr = self.sock.recvfrom(41)
        logging.info('Received data packet from {}'.format(client_addr))
        self.add_client(client_addr)
        r_msg = struct.unpack('f', r_msg)  # need to move decoding to dedicated message processing
        return r_msg

    def process_events(self, mask, in_queue, out_queue):
        if mask and selectors.EVENT_READ:
            logging.info('Network - Receiving')
            msg = self.receive()
            # would need to process msg beforehand
            in_queue.put(msg)
            logging.info('Network - Placed message in the queue')

        if mask and selectors.EVENT_WRITE:
            msg = out_queue.get()
            logging.info('Network - Got message from the queue')
            self.send(msg, self.clients)

        # return in_queue, out_queue: not needed, processing done on original objects

    def add_client(self, client_addr):
        if client_addr not in self.clients:
            self.clients.append(client_addr)

    def close(self):
        self.sel.unregister(self.sock)
        logging.info('Unregistered socket from selectors')
        self.sock.close()
        logging.info('Closed socket')
