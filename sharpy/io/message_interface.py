import struct
import logging

# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     level=20)
logger = logging.getLogger(__name__)


def decoder(msg, byte_ordering='<'):
    n_bytes = len(msg)

    header = msg[:5]
    if header != b'RREF0':
        logger.error('Error, header is not RREF0')

    len_values = int(n_bytes - 5)
    if divmod(len_values, 8)[1] != 0:  # remainder equal 0
        logger.error('Error in decoding message. Length of values field not a multiple of 8')

    n_values = int(len_values // 8)
    values = []
    for i_val in range(n_values):
        current_signal = msg[5 + i_val * 8: 5 + i_val * 8 + 8]
        sig_index = struct.unpack('{}if'.format(byte_ordering), current_signal)
        values.append(sig_index)

    return values
