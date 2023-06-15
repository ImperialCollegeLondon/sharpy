import socket
import logging
import struct
import sharpy.io.message_interface as message_interface
import numpy as np
from pid_controller import PID_Controller
import json
import os
"""
"""

def create_and_bind_client(ip_addr, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip_addr, port))
    return sock

def setup_udp_network(dict_parameters):
    server_ip_addr = dict_parameters["server_ip_addr"] #'127.0.0.1' # SHARPy
    client_ip_addr =  dict_parameters["client_ip_addr"] #'127.0.0.1' # controller
    port_in_network =  dict_parameters["port_in_network"] #64019
    port_out_network_client = dict_parameters["port_out_network_client"] # 59009 
    port_in_network_client = port_out_network_client - 8 # needs to be different from port out network client

    sharpy_in_network= (server_ip_addr, port_in_network)

    in_sock = create_and_bind_client(client_ip_addr, port_in_network_client)

    out_sock = create_and_bind_client(client_ip_addr, port_out_network_client)
    out_sock.settimeout(120)

    return sharpy_in_network, in_sock, out_sock

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=20)
    return logging.getLogger(__name__)

def write_sensor_measurements_to_file(data, file_path, init_file=False):
    
    if init_file:
        file_mode = "w"
    else:
        file_mode = "ab"

    with open(file_path, file_mode) as f:
        np.savetxt(f, data, delimiter=',', newline='\r\n')

def receive_sensor_measurements(logger, out_sock, msg_len, num_sensors):
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
    ctrl_value = struct.pack('<5sif', b'RREF0', 0, control_input_to_send)
    logger.info('Sending control input of size {} bytes.'.format(len(ctrl_value)))
    in_sock.sendto(ctrl_value, sharpy_in_network)
    logger.info('Sent control input {} to {}.'.format(control_input_to_send, sharpy_in_network))

def run_controller(case_name):

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

    init_cs_deflection = dict_simulation_parameters["initial_cs_deflection"]
    reference_deflection= dict_simulation_parameters["reference_deflection"] #.020137935876846313 #0.09681699577946704
    
    # setup udp network components
    sharpy_in_network, in_sock, out_sock = setup_udp_network(dict_simulation_parameters)

    # setup network and data output
    logger = init_logger()
    file_path_sensor =  os.path.join(output_folder, case_name, "{}_{}".format(case_name, "sensor_measurement"))
    file_path_control = os.path.join(output_folder, case_name, "{}_{}".format(case_name, "control_input"))

    PID =  PID_Controller(10., 0., 0., target=reference_deflection)
    control_input = init_cs_deflection
    while curr_ts < number_timesteps:
        send_control_input(logger, 
                           in_sock, 
                           sharpy_in_network, 
                           control_input)

        sensor_data = receive_sensor_measurements(logger, 
                                                  out_sock, 
                                                  msg_len,
                                                  num_sensors)
        

        # PID controller
        if curr_ts < 5:
            control_input = 0
        else:
            control_input = PID.generate_control_input(sensor_data, dt)

        
        write_sensor_measurements_to_file(sensor_data,
                                          file_path_sensor, 
                                          init_file= bool(curr_ts==0))

        write_sensor_measurements_to_file([control_input],
                                          file_path_control,
                                          init_file= bool(curr_ts==0))

        curr_ts += 1

    # Close sockets
    out_sock.close()
    in_sock.close()
    logger.info('Closed input and output sockets')

if __name__ == '__main__':
    case_name = "pazy_udp_closed_loop_gust_response"
    run_controller(case_name)