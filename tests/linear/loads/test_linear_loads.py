import unittest
import os
import generate_wing as gw

class TestLinearLoads(unittest.TestCase):

    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    def test_linear_loads(self):
        flow = ['BeamLoader', 'AerogridLoader']
        gw.generate_wing('test', self.route_test_dir + '/cases/', self.route_test_dir + '/output', flow=flow)


if __name__ == '__main__':
    unittest.main()
