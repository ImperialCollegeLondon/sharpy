import unittest

# import your test modules
import tests
# import scenario
# import thing


def load_tests(loader, tests, pattern):
    # initialize the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # add tests to the test suite
    suite.addTests(loader.loadTestsFromModule(tests))
    # suite.addTests(loader.loadTestsFromModule(scenario))
    # suite.addTests(loader.loadTestsFromModule(thing))
    return suite


if __name__ == '__main__':
    suite = load_tests()
    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)
