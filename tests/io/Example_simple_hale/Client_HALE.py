import socket
import select
import time
import logging
import struct
import sharpy.io.message_interface as message_interface
import numpy as np

"""
This is not a test but is to be used as client when testing the development of the input
output capabilities of sharpy. It will just give back the initial control surface deflection.

Run this script as client.

Run ``python generate_hale_io.py`` as server from the folder tests/io/Example_simple_hale
"""

# sel = selectors.DefaultSelector()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=20
)
logger = logging.getLogger(__name__)

sharpy_incoming = ("127.0.0.1", 64011)  # control side socket
sharpy_outgoing = ("127.0.0.1", 64010)  # output side socket

own_control = ("127.0.0.1", 64000)
own_receive = ("127.0.0.1", 64001)

in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
in_sock.bind(own_control)

out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
out_sock.bind(own_receive)

# from https://stackoverflow.com/questions/2719017/how-to-set-timeout-on-pythons-socket-recv-method
# ready_to_read = select.select([out_sock], [], [], 2)
out_sock.settimeout(300)

# initial values
cs_deflection = [-2.08 * np.pi / 180, 0]
thrust = 6.16
dt = 0.025
n_tstep = 0  # Counter for time

x_pos = []
y_pos = []
z_pos = []

x_vel = []
y_vel = []
z_vel = []
p = []
q = []
r = []

root_OOP_moment = []
root_OOP_strain = []

time_vec = []

while True:
    if n_tstep > 401:
        break

    # send control input to sharpy
    ctrl_value = struct.pack("<5sif", b"RREF0", 0, cs_deflection[0])
    ctrl_value += struct.pack("if", 1, cs_deflection[1])
    ctrl_value += struct.pack("if", 2, thrust)
    logger.info("Sending control input of size {} bytes".format(len(ctrl_value)))
    in_sock.sendto(ctrl_value, sharpy_incoming)
    logger.info("Sent control input to {}".format(sharpy_incoming))

    # time.sleep(2)
    # input('Continue loop')

    # receive output data. set msg_len to whatever length SHARPy is sending
    msg_len = 93
    try:
        msg, conn = out_sock.recvfrom(msg_len)
    except socket.timeout:
        logger.info("Socket time out")
        break
    logger.info("Received {} data from {}".format(msg, conn))
    logger.info("Received data is {} bytes long".format(len(msg)))
    # else:
    #     break
    # decoding
    values = message_interface.decoder(msg)

    # Add the received values to the variables
    x_pos.append(-values[0][1])
    y_pos.append(values[1][1])
    z_pos.append(values[2][1])

    x_vel.append(-values[3][1])
    y_vel.append(values[4][1])
    z_vel.append(values[5][1])
    p.append(values[6][1] * 180 / np.pi)
    q.append(values[7][1] * 180 / np.pi)
    r.append(values[8][1] * 180 / np.pi)

    root_OOP_moment.append(values[9][1])
    root_OOP_strain.append(values[10][1])

    time_vec.append(dt * n_tstep)

    n_tstep += 1

##

out_sock.close()
in_sock.close()
logger.info("Closed input and output sockets")
