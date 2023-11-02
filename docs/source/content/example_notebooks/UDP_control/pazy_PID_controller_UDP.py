import socket
import logging
import struct
import sharpy.io.message_interface as message_interface
import numpy as np
from pid_controller import PID_Controller
import json
import os

"""
This script establishes a UDP connection to SHARPy from which it receives sensor measurements. 
Based on these measurements, a PID-Controller computes an actuator input that is feedback 
to SHARPy again using UDP. 

The user just needs to specify the name of the case name (bottom of the script) and run the main
function 'run_controller(case_name)'. The case name is needed to find the appropriate json input 
file for all required information for the following steps:
1) Load input settings 
2) Establish a connection to the UDP i/o network created by SHARPy (server)
3) Initialise PID Controller
4) Enter the time loop including: 
    a) Sending a control input (starting with the initial value loaded from the init file)
        to SHARPy;
    b) The resulting sensor measurement for the next simulation time step is sent back to 
        this client.
    c) Based on the error to the defined target value, the PID controller computes a control
       input.
    d) Both the received sensor measurement and generated control input are saved in .txt-files.
    a) Next step, would be a) again. But now, we send the generated control input.
5) Closes UDP connections
"""

def create_and_bind_client(ip_addr, port):
    """
        Creates and bind the client to the server socket using UDP.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip_addr, port))
    return sock

def setup_udp_network(dict_parameters):
    """
        This function connects the present client to two sockets, specified within the 
        information (i.e. ip addresses and ports). Further, a timeout of 120 s are set
        for SHARPy's output network, i.e. if new sensor measurements are not received
        (normally caused by an error), the network connection is closed.
    """
    server_ip_addr = dict_parameters["server_ip_addr"] # SHARPy
    client_ip_addr =  dict_parameters["client_ip_addr"] # controller
    port_in_network =  dict_parameters["port_in_network"] 
    port_out_network_client = dict_parameters["port_out_network_client"] 
    port_in_network_client = port_out_network_client - 8 # needs to be different from port of network client

    sharpy_in_network= (server_ip_addr, port_in_network)

    in_sock = create_and_bind_client(client_ip_addr, port_in_network_client)

    out_sock = create_and_bind_client(client_ip_addr, port_out_network_client)
    out_sock.settimeout(120)

    return sharpy_in_network, in_sock, out_sock

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=20)
    return logging.getLogger(__name__)

def write_data_to_file(data, file_path, init_file=False):
    if init_file:
        file_mode = "w"
    else:
        file_mode = "ab"

    with open(file_path, file_mode) as f:
        np.savetxt(f, data, delimiter=',', newline='\r\n')

def receive_sensor_measurements(logger, out_sock, msg_len, num_sensors):
    """
        UDP socket waits and receives sensor measurements from SHARPy. After,
        the message is decoded and unpacked in an appropriate format and returned
        to the main function.
    """
    try:
        msg, conn = out_sock.recvfrom(msg_len)
    except socket.timeout:
        logger.info('Socket time out')
        raise
    logger.info('Received {}-byte long data "{}" from {}'.format(len(msg), msg, conn))

    values = message_interface.decoder(msg)
    logger.info('Received {}'.format(values))
    
    results = np.zeros((num_sensors,))
    for isensor in range(num_sensors):
        results[isensor] = values[isensor][1]
    return results

def send_control_input(logger, in_sock, sharpy_in_network, control_input_to_send):
    """
        The generated control input is packed into a binary message readable from SHARPy.
          Note that, this message includes the control input twice, since the ailerons 
          on right and left wing have different IDs. 
    """
    ctrl_value = struct.pack('<5sifif', b'RREF0', 0, control_input_to_send, 1, control_input_to_send)
    logger.info('Sending control input of size {} bytes.'.format(len(ctrl_value)))
    in_sock.sendto(ctrl_value, sharpy_in_network)
    logger.info('Sent control input {} to {}.'.format(control_input_to_send, sharpy_in_network))

def run_controller(case_name):
    
    # 1) Load input settings
    route_file_dir = os.path.abspath('')
    with open(route_file_dir + '/parameter_UDP_control_{}.json'.format(case_name), 'r') as fp:
        dict_simulation_parameters = json.load(fp)
    
    # define simulation settings
    curr_ts = 0
    num_sensors = dict_simulation_parameters["num_sensors"]
    msg_len = 5 + 8 * num_sensors
    dt = dict_simulation_parameters["dt"]
    number_timesteps = int(dict_simulation_parameters["simulation_time"] / dt)
    output_folder = dict_simulation_parameters["output_folder"]

    init_cs_deflection = float(dict_simulation_parameters["initial_cs_deflection"])
    reference_deflection= dict_simulation_parameters["reference_deflection"]
    
    # 2) Establish UDP connections
    sharpy_in_network, in_sock, out_sock = setup_udp_network(dict_simulation_parameters)

    # setup network and data output
    logger = init_logger()
    file_path_sensor =  os.path.join(output_folder, case_name, "{}_{}".format(case_name, "sensor_measurement"))
    file_path_control = os.path.join(output_folder, case_name, "{}_{}".format(case_name, "control_input"))

    # 3) Initialise controller and control input
    PID =  PID_Controller(10., 0., 0., target=reference_deflection)
    control_input = init_cs_deflection

    # 4) Enter time control loop
    while curr_ts < number_timesteps:
        # a)
        send_control_input(logger, 
                           in_sock, 
                           sharpy_in_network, 
                           control_input)

        # b)
        sensor_data = receive_sensor_measurements(logger, 
                                                  out_sock, 
                                                  msg_len,
                                                  num_sensors)
        

        # c) Generate control input with PID controller
        if curr_ts >= 5:
            control_input = PID.generate_control_input(sensor_data, dt)

        # d) Store sensor and control input to .txt-file
        write_data_to_file(sensor_data,
                           file_path_sensor, 
                           init_file= bool(curr_ts==0))

        write_data_to_file([control_input],
                           file_path_control,
                           init_file= bool(curr_ts==0))

        curr_ts += 1

    # 5) Close sockets
    out_sock.close()
    in_sock.close()
    logger.info('Closed input and output sockets')

if __name__ == '__main__':
    case_name = "pazy_udp_closed_loop_gust_response"
    run_controller(case_name)