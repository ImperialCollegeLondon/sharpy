# Adapted from below, which was meant for receiving data
# https://github.com/charlylima/XPlaneUDP/blob/master/XPlaneUdp.py

# Method adapted to send data to XPlane instead of receive

import socket
import struct
import binascii
import pickle
import os
import numpy as np

import sharpy.utils.solver_interface as solver_interface
import sharpy.postproc
from sharpy.utils.algebra import quat2euler

import matplotlib.pyplot as plt


class XPlaneIpNotFound(Exception):
    args = "Could not find any running XPlane instance in network."


class XPlaneTimeout(Exception):
    args = "XPlane timeout."


class XPlaneUdp:

    '''
        Class that co-ordinates the sending of data via UDP to X-Plane
        It uses SHARPy UDPout postprocessor 
    '''

    # Discover X-Plane via Beacon
    # This uses multicast
    # constants
    UDP_PORT = 49000
    MCAST_GRP = "239.255.1.1"
    MCAST_PORT = 49707  # (MCAST_PORT was 49000 for XPlane10)
    EARTH_RADIUS = 6378.137*10 ** 3  # Equatorial radius in [m]

    MAX_DIHEDRAL = 180  # Degrees
    MAX_SWEEP = 90  # Degrees

    HEIGHT_OFFSET = 100  # Meters
    DEFLECTION_ANGLE_MULTIPLICATOR = 10

    def __init__(self, pklFile):
        # Find location of X-Plane
        # Line below is what should be used
        self.BeaconData = self.find_ip_xplane()
        # These are for debuggin so I don't need to run xplane in the
        # self.BeaconData = dict()
        # self.BeaconData['IP'] = "127.0.0.1"
        # self.BeaconData['Port'] = 49000

        self._byte_ordering = '<'

        _host, _port = self.find_local_ip()

        settings = dict()
        # Ensure that varaibles.yaml is in the order of 3 pos vars (x,y,z),
        # ##6 vel vars (x,y,z,pdot,qdot,rdot), 4 quat vars
        # and 4 quat vars
        settings['UDPout'] = {
            'output_network_settings': {
                'destination_address': [self.BeaconData['IP']],
                'destination_ports': [self.BeaconData['Port']],
                'address': _host,
                'port': _port,
            },
            'variables_filename': os.getcwd() + '/variables.yaml',
        }

        self.timestep_counter = 0
        self.data = pklFile

        self.udpSolver = solver_interface.initialise_solver('UDPout')
        self.udpSolver.initialise(data, custom_settings=settings['UDPout'])

        self.locLat = 51.470020  # 0
        self.locLon = -0.454295  # 0
        self.prevX = 0
        self.prevY = 0

        self.drefPaths = {}
        self.drefPaths['overrideCS'] = "sim/operation/override/override_control_surfaces"

        ### Ailerons ###
        self.drefPaths['leftAileronDef'] = "sim/flightmodel/controls/wing1l_ail1def"
        self.drefPaths['rightAileronDef'] = "sim/flightmodel/controls/wing1r_ail1def"

        ### Stabilisers ###
        self.drefPaths['elevatorDef'] = "sim/flightmodel/controls/hstab1_elv1def"
        self.drefPaths['rudderDef'] = "sim/flightmodel/controls/vstab1_rud1def"

        # Indicator (potential way for displaying aeroelastic info)
        # Dref for: The indicated pitch on the panel for the first vacuum instrument
        self.drefPaths['vertIndicator'] = "sim/cockpit/gyros/the_vac_ind_deg"
        # Dref for: The indicated roll on the panel for the first vacuum instrument
        self.drefPaths['horzIndicator'] = "sim/cockpit/gyros/phi_vac_ind_deg"

        # The flightmodel2 section contains data about how the actual aircraft is being drawn
        # Actual sweep in ratio (0=> no sweep, 1 => max sweep) [float, ratio]
        self.drefPaths['variableSweep'] = "sim/flightmodel2/controls/wingsweep_ratio"
        # Acutal dihedral [float, ratio]
        self.drefPaths['variableDihedral'] = "sim/flightmodel2/controls/dihedral_ratio"
        # Actual incidence [float, ratio]
        self.drefPaths['variableIncidence'] = "sim/flightmodel2/controls/incidence_ratio"
        # Datarefs above achieve the same
        # self.drefPaths['variableSweep'] = "sim/flightmodel/controls/swdi"
        # self.drefPaths['variableDihedral'] = "sim/flightmodel/controls/dihed_rat"
        # self.drefPaths['variableIncidence'] = "sim/flightmodel/controls/incid_ratio"

    def run(self):
        # Get values for a given time step
        vals = self.get_struct_value(
            self.data, timestep_index=self.timestep_counter)
        # Encode location information
        dvisMsg = self.encode_dvis(vals)
        # Send encoded information
        self.udpSolver.out_network.send(
            dvisMsg, self.udpSolver.out_network.clients)

        # Aeroelastic Information
        vals = self.convert_tip_disp_to_angle(vals, [7, 8], 16)
        indicatorMsg = self.encode_dref(
            self.drefPaths['vertIndicator'], 7, lstOfVals=vals)
        self.udpSolver.out_network.send(
            indicatorMsg, self.udpSolver.out_network.clients)

        indicatorMsg2 = self.encode_dref(
            self.drefPaths['horzIndicator'], 8, lstOfVals=vals)
        self.udpSolver.out_network.send(
            indicatorMsg2, self.udpSolver.out_network.clients)

        vals = np.append(vals, np.asarray([-2.08, +2.08, 0])*self.DEFLECTION_ANGLE_MULTIPLICATOR)

        # Send control information
        overrideMsg = self.encode_dref(self.drefPaths['overrideCS'], True)
        self.udpSolver.out_network.send(
            overrideMsg, self.udpSolver.out_network.clients)
        # Deflection angle is in radians

        leftAilMsg = self.encode_dref(
            self.drefPaths['leftAileronDef'], 9, lstOfVals=vals)
        self.udpSolver.out_network.send(
            leftAilMsg, self.udpSolver.out_network.clients)

        rightAilMsg = self.encode_dref(
            self.drefPaths['rightAileronDef'], 10, lstOfVals=vals)
        self.udpSolver.out_network.send(
            rightAilMsg, self.udpSolver.out_network.clients)

        elevatorMsg = self.encode_dref(
            self.drefPaths['elevatorDef'], 11, lstOfVals=vals)
        self.udpSolver.out_network.send(
            elevatorMsg, self.udpSolver.out_network.clients)

        vals = self.find_deflection_ratio(
            vals, [7, 8], [self.MAX_DIHEDRAL, self.MAX_SWEEP])
        dihedralMsg = self.encode_dref(
            self.drefPaths['variableDihedral'], 12, lstOfVals=vals)
        self.udpSolver.out_network.send(
            dihedralMsg, self.udpSolver.out_network.clients)

        sweepMsg = self.encode_dref(
            self.drefPaths['variableSweep'], 13, lstOfVals=vals)
        self.udpSolver.out_network.send(
            sweepMsg, self.udpSolver.out_network.clients)

        # incidenceMsg = self.encode_dref(
        #     self.drefPaths['variableIncidence'], 14, lstOfVals=vals)
        # self.udpSolver.out_network.send(
        #     incidenceMsg, self.udpSolver.out_network.clients)

        # # Incrementing the counter so it accesses the next time step values
        self.timestep_counter += 1

    def get_struct_value(self, data, timestep_index=-1):
        numVars = len(self.udpSolver.set_of_variables.variables)
        values = np.zeros(numVars)
        for i in range(numVars):
            try:
                values[i] = self.udpSolver.set_of_variables.variables[i].get_variable_value(
                    data, timestep_index)
            except:
                # This is very dangerous!!
                # Created to handle the fact that the first control surface deflection angle is
                # not stored in the pickle (list is empty)
                values[i] = 0
        return values

    def encode_dref(self, drefPath, idx, lstOfVals=np.asarray([])):
        # Tested in command line only
        '''
            Set any dataref to a given value via UDP

            The datarefs used by X-Plane can be found at:
                https://developer.x-plane.com/datarefs/

            struct dref_struct:
            {
                xflt var;
                xchar dref_path[strDIM];
            }

            N.B. Null termination means the null character must be placed at
            the end of dref_path. 

            Therefore, typical message looks like,
            DREF0 + (4byte value) + dref_path + 0 + spaces to complte the 
            whole message to 509 bytes. (N.B. + => append. Don't include
            them in the actual message)

            E.G. to switch on anti-ice swithc
            DREF0 + (4byte value of 1) + sim/cockpit/switches/anti_ice_surf_heat_left + 
            0 + spaces to complte to 509 bytes 
        '''
        msg = struct.pack('{}5s'.format(self._byte_ordering), b'DREF0')

        if lstOfVals.size == 0 and type(idx) == bool:
            # Based on the information from XPlane dataref website, the type should be
            # int but when int is used for the format, XPlane always reads the value as
            # 0 instead of what the actual value is. Therefore, the type for packing 
            # needs to be f (float) instead of int
            msg += struct.pack('{}f'.format(self._byte_ordering), int(idx))
        elif lstOfVals.size == 0:
            raise ValueError(
                "If list of values is empty, idx must be a boolean to indicate override")
        else:
            msg += struct.pack('{}f'.format(self._byte_ordering),
                               lstOfVals[idx])

        # Convert dref path which is a string to bytes
        drefPath = bytes(drefPath, 'utf-8')
        # Add dreft path and null terminate it
        msg += struct.pack("{}s".format(len(drefPath)), drefPath)
        # Find the size of the message
        msgSize = struct.calcsize('5sf' + "{}s".format(len(drefPath)))
        # Determine how many blank spaces to include
        remainder = 512 - msgSize
        # Append the black spaces to complete to 509 bytes
        msg += struct.pack('{}x'.format(remainder))

        return msg

    def encode_dvis(self, lstOfVals):
        '''
            For X-Plane to turn off it's own flight engine, send in 5
            chars "DVIS" (null-terminated) followed by the struct:

            struct dvis_struct:
            {
                xdob lat_lon_ele[3]; // double precision for lattitude,
                                        longitude & elevation above MSL (m)
                xdob psi_the_phi[3]; // True heading, pitch up, roll right 
                                        in degrees 
            }
            Double precision is used to avoid byte-spacing confusion 

            Note: This should be sent at a higher frame rate than X-Plane
            for smooth animcation 
        '''
        msg = struct.pack('{}5s'.format(self._byte_ordering), b'DVIS0')
        lat, lon = self.convert_to_lat_lon(lstOfVals)
        msg += struct.pack("{}ddd".format(self._byte_ordering),
                           lat, lon, lstOfVals[2]+self.HEIGHT_OFFSET)
        # Information from pickle is in quaternion form. Therefore, this needs
        # to be converted to euler angles
        eulerAngles = quat2euler(lstOfVals[3:7])
        # Above conversion results in radians but X-Plane requires degrees
        eulerAngles = np.rad2deg(eulerAngles)
        # Individual naming for clarity
        roll, pitch, yaw = eulerAngles
        msg += struct.pack('{}ddd'.format(self._byte_ordering),
                           yaw, pitch, roll)

        return msg

    def convert_to_lat_lon(self, values):
        '''
            This takes in the x,y positions from the simulation and 
            converts it to latitude and longitude (degrees)

            This method is taken from: 
            https://stackoverflow.com/questions/10122055/calculate-longitude-from-distance
        '''
        x = values[0]
        dx = x - self.prevX
        y = values[1]
        dy = y - self.prevY

        deltaLon = dx/self.EARTH_RADIUS
        deltaLat = dx/self.EARTH_RADIUS

        # Hard code local latitude and longitude for now
        # In future, get XPlan to output initial latitude and longitude
        self.locLat = self.locLat + np.sign(dy)*np.rad2deg(deltaLat)
        self.locLon = self.locLon + np.sign(dx)*np.rad2deg(deltaLon*1.195)

        return self.locLat, self.locLon

    def convert_tip_disp_to_angle(self, lstOfVals, indices, arm):
        '''
            indices (iterable) refers to the indices that need conversion
            arm is the length which the conversion is done on? 


            N.B.:   Positive vertical displacmenet => up
                    Positive horizontal displacement => aft

            returns updated matrix
        '''
        for index in indices:
            angle = np.rad2deg(np.arctan(float(lstOfVals[index])/arm))
            lstOfVals[index] = angle

        return lstOfVals

    def find_deflection_ratio(self, lstOfVals, indices, limits):
        '''
            indices: get the location within origin array of vals
            limits: what the value is normalised by
            N.B: This step MUST take place after the tip displacement 
                 has been converted to angles (degree) 
        '''
        temp = np.zeros(len(indices))
        for i, index in enumerate(indices):
            temp[i] = float(lstOfVals[index])/limits[i]

        lstOfVals = np.append(lstOfVals, temp)

        return lstOfVals

    def find_local_ip(self):
        '''
            This returns a single IP which is the default route 
            This function has been adapted from:
                https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
        '''
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP, port = s.getsockname()
        except Exception:
            IP = '127.0.0.1'
            # This is a randomly chosen number
            port = 50585
        finally:
            s.close()
        return IP, port

    def find_ip_xplane(self):
        '''
            Find the IP of XPlane Host in Network.
            It takes the first one it can find.
            This function is wholesale from:
                https://github.com/charlylima/XPlaneUDP/blob/master/XPlaneUdp.py
        '''

        self.BeaconData = {}

        # open socket for multicast group.
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.MCAST_GRP, self.MCAST_PORT))
        mreq = struct.pack("=4sl", socket.inet_aton(
            self.MCAST_GRP), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(3.0)

        while not self.BeaconData:

            # receive data
            try:
                packet, sender = sock.recvfrom(15000)

                # decode data
                # * Header
                header = packet[0:5]
                if header != b"BECN\x00":
                    print("Unknown packet from "+sender[0])
                    print(str(len(packet)) + " bytes")
                    print(packet)
                    print(binascii.hexlify(packet))

                else:
                    # * Data
                    data = packet[5:21]
                    # struct becn_struct
                    # {
                    # 	uchar beacon_major_version;		// 1 at the time of X-Plane 10.40
                    # 	uchar beacon_minor_version;		// 1 at the time of X-Plane 10.40
                    # 	xint application_host_id;		// 1 for X-Plane, 2 for PlaneMaker
                    # 	xint version_number;			// 104014 for X-Plane 10.40b14
                    # 	uint role;						// 1 for master, 2 for extern visual, 3 for IOS
                    # 	ushort port;					// port number X-Plane is listening on
                    # 	xchr	computer_name[strDIM];	// the hostname of the computer
                    # };
                    beacon_major_version = 0
                    beacon_minor_version = 0
                    application_host_id = 0
                    xplane_version_number = 0
                    role = 0
                    port = 0
                    (
                        beacon_major_version,  # 1 at the time of X-Plane 10.40
                        beacon_minor_version,  # 1 at the time of X-Plane 10.40
                        application_host_id,   # 1 for X-Plane, 2 for PlaneMaker
                        xplane_version_number,  # 104014 for X-Plane 10.40b14
                        role,                  # 1 for master, 2 for extern visual, 3 for IOS
                        port,                  # port number X-Plane is listening on
                    ) = struct.unpack("<BBiiIH", data)
                    computer_name = packet[21:-1]
                    if beacon_major_version == 1 \
                       and beacon_minor_version == 1 \
                       and application_host_id == 1:
                        self.BeaconData["IP"] = sender[0]
                        self.BeaconData["Port"] = port
                        self.BeaconData["hostname"] = computer_name.decode()
                        self.BeaconData["XPlaneVersion"] = xplane_version_number
                        self.BeaconData["role"] = role

            except socket.timeout:
                raise XPlaneIpNotFound()

        sock.close()
        return self.BeaconData


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="""This is the executable for Simulation of High Aspect Ratio Planes.\n
            Imperial College London 2020""")
    parser.add_argument('-r', '--restart', help='restart the solution with a given snapshot', type=str,
                        default=None, required=True)

    args = parser.parse_args()

    try:
        with open(args.restart, 'rb') as restart_file:
            data = pickle.load(restart_file)
    except FileNotFoundError:
        raise FileNotFoundError('The file specified for the snapshot \
            restart (-r) does not exist. Please check.')

    xp = XPlaneUdp(data)

    for _ in range(data.ts + 1):
        xp.run()
        time.sleep(0.03)
