import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import tests.linear.inputsimulation.src.generate_hale_nonlinear_forces as generate_hale
import tests.linear.inputsimulation.src.generate_hale_rom as generate_hale_rom

class TestJupyter(unittest.TestCase):


    test_dir = os.path.abspath(os.path.dirname(__file__))
    sharpy_root = os.path.abspath(test_dir + '../../../../')

    def setUp(self):

        nonlinear_results_file = self.test_dir + '/nonlinear_results.txt'
        with open(nonlinear_results_file, 'w') as f:
            f.write('alpha,\tFxG,\tFyG,\tFzG\n')

        generate_hale.execute_sharpy_simulations(self.test_dir)

        #run linear systems
        for alpha_lin in [2., 4.]:
            generate_hale_rom.generate_linearised_system(alpha_lin)


    def test_execute_jupyter(self):

        notebook_filename = os.path.abspath(self.sharpy_root + '/docs/source/content/example_notebooks/linear_system_inputs.ipynb')
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        os.makedirs(self.test_dir + '/notebooks/', exist_ok=True)
        ep.preprocess(nb, {'metadata': {'path': self.test_dir + '/notebooks/'}})

        with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        # add matplotlib exception 