import struct


class Message:
    def __init__(self, sel, sock, addr):
        self.sel = sel
        self.sock = sock
        self.addr = addr

        self.header = [68, 65, 84, 65, 60]  # DATA in ASCII

    def write_msg(self, index, *values):
        n_vals = len(values)
        if n_vals > 8:
            raise IndexError('Maximum allowed is 8 values')

        # -999 for unused value
        msg = b''

        for i in range(len(self.header)):
            msg += struct.pack('b', self.header[i])

        msg += struct.pack('i', index)

        for i in range(n_vals):
            msg += struct.pack('f', values[i])

        for i in range(n_vals, 8):
            # print('Encoding blank - {}'.format(i))
            msg += struct.pack('i', -999)

        return msg

    def send(self, index, *values):
        msg = self.write_msg(index, *values)
        self.sock.sendto(msg, self.addr)

    @staticmethod
    def read_msg(msg):
        index = struct.unpack('i', msg[5:9])
        print('Index:')
        print(index)
        values = []
        for data_field_i in range(9, 41, 4):
            values.append(struct.unpack('f', msg[data_field_i:data_field_i+4]))
        print(values)

if __name__ == '__main__':
    m = Message(None, None, None)

    a = m.write_msg(12, 9.81, 19.1, -1.2)
    print(a)
    print(len(a))
    m.read_msg(a)
