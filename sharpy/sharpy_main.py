# import os
# import time
#
# import sharpy.utils.cout_utils as cout
# import sharpy.utils.input_arg as input_arg
# import sharpy.utils.sharpydir as sharpydir
# import sharpy.utils.solver_interface as solver_interface
# from sharpy.presharpy.presharpy import PreSharpy
#
# from sharpy.presharpy.presharpy import PreSharpy
# # Loading solvers and postprocessors
# import sharpy.solvers
# import sharpy.postproc
# import sharpy.generators
# # ------------
import sharpy.utils.cout_utils as cout
import sys
import pickle


def main(args=None):
    """
    Main ``SHARPy`` routine

    This is the main ``SHARPy`` routine.
    It starts the solution process by reading the settings that are included in the ``.solver.txt`` file that is parsed
    as an argument.
    It reads the solvers specific settings and runs them in order

    Args:
        args (str): ``.solver.txt`` file with the problem information and settings

    Returns:
        ``PreSharpy`` class object

    """
    import time
    import argparse

    import sharpy.utils.input_arg as input_arg
    import sharpy.utils.solver_interface as solver_interface
    from sharpy.presharpy.presharpy import PreSharpy
    from sharpy.utils.cout_utils import start_writer, finish_writer
    # Loading solvers and postprocessors
    import sharpy.solvers
    import sharpy.postproc
    import sharpy.generators
    # ------------

    # output writer
    start_writer()
    # timing
    t = time.process_time()
    t0_wall = time.perf_counter()

    parser = argparse.ArgumentParser(prog='SHARPy', description=
    """This is the executable for Simulation of High Aspect Ratio Planes.\n
    Imperial College London 2018""")
    parser.add_argument('input_filename', help='path to the *.solver.txt input file', type=str)
    parser.add_argument('-r', '--restart', help='restart the solution with a given snapshot', type=str, default=None)
    if args is not None:
        args = parser.parse_args(args[1:])
    else:
        args = parser.parse_args()

    settings = input_arg.read_settings(args)

    if args.restart is None:
        # run preSHARPy
        data = PreSharpy(settings)
    else:
        try:
            with open(args.restart, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError('The file specified for the snapshot restart (-r) does not exist. Please check.')

        # update the settings
        data.update_settings(settings)

    # Loop for the solvers specified in *.solver.txt['SHARPy']['flow']
    for solver_name in settings['SHARPy']['flow']:
        solver = solver_interface.initialise_solver(solver_name)
        solver.initialise(data)
        data = solver.run()

    CPU_time = time.process_time() - t
    wall_time = time.perf_counter() - t0_wall
    cout.cout_wrap('FINISHED - Elapsed time = %f6 seconds' % wall_time, 2)
    cout.cout_wrap('FINISHED - CPU process time = %f6 seconds' % CPU_time, 2)
    finish_writer()
    return data
