import socket
import select
import time
import logging
import struct
import sharpy.io.message_interface as message_interface
import numpy as np
"""
This is not a test but is to be used as client when testing the development of the input
output capabilities of sharpy.

Run this script as client.

Run ``python generate_pazy_test_io_local.py`` as server
"""

# sel = selectors.DefaultSelector()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=20)
logger = logging.getLogger(__name__)

sharpy_incoming = ('127.0.0.1', 64011)  # control side socket
sharpy_outgoing = ('127.0.0.1', 64010)  # output side socket

own_control = ('127.0.0.1', 64000)
own_receive = ('127.0.0.1', 64001)

in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
in_sock.bind(own_control)

out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
out_sock.bind(own_receive)

# from https://stackoverflow.com/questions/2719017/how-to-set-timeout-on-pythons-socket-recv-method
# ready_to_read = select.select([out_sock], [], [], 2)
out_sock.settimeout(10)

tsteps = 401
t = np.linspace(0, 0.2, tsteps)
cs_deflection = 10 * np.sin(2 * t * np.pi / 0.2) * np.pi/180
# cs_deflection = np.linspace(0, )
# cs_deflection = np.array([0, 0, 90, 90, 90, 90]) * np.pi / 180
curr_ts = 0
wingtip_deflection = []
mid_wing_deflection = []
while True:
    if curr_ts > tsteps:
        break
    # send control input to sharpy
    ctrl_value = struct.pack('<5sif', b'RREF0', 0, cs_deflection[curr_ts])
    logger.info('Sending control input of size {} bytes'.format(len(ctrl_value)))
    in_sock.sendto(ctrl_value, sharpy_incoming)
    logger.info('Sent control input to {}'.format(sharpy_incoming))

    # time.sleep(2)
    # input('Continue loop')

    # receive output data
    # req_message = b'I want data'  # this would be the RREF0 value
    # out_sock.sendto(req_message, sharpy_outgoing)
    # if ready_to_read[0]:
    try:
        msg, conn = out_sock.recvfrom(21)
    except socket.timeout:
        logger.info('Socket time out')
        break
    logger.info('Received {} data from {}'.format(msg, conn))
    logger.info('Received data is {} bytes long'.format(len(msg)))
    # else:
    #     break
    # decoding
    values = message_interface.decoder(msg)
    logger.info('Received {}'.format(values))
    wingtip_deflection.append(values[0][1])
    mid_wing_deflection.append(values[1][1])
    # input('Next time step')
    curr_ts += 1


out_sock.close()
in_sock.close()
logger.info('Closed input and output sockets')
