r"""
Tests that the VTK files written for Paraview carry the file extension that matches the type of dataset they
contain. Paraview chooses the reader from the file extension, and each VTK XML reader refuses a file holding a
different dataset type (e.g. the ``.vtu`` unstructured grid reader refuses a structured grid), so a file with a
mismatched extension cannot be opened.
"""
import glob
import os
import shutil
import tempfile
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import sharpy.cases.templates.flying_wings as wings
import sharpy.sharpy_main
from sharpy.structure.utils.modalutils import write_zeta_vtk

# VTK XML reader associated with each file extension (the association Paraview uses to pick a reader)
readers = {'.vts': vtk.vtkXMLStructuredGridReader,
           '.vtu': vtk.vtkXMLUnstructuredGridReader,
           '.vtp': vtk.vtkXMLPolyDataReader}


def readable_by_extension(filename):
    r"""
    Checks whether the VTK XML reader associated with the file extension accepts the file.

    Args:
        filename (str): path to the VTK file

    Returns:
        bool: ``True`` if the reader for the extension can read the file
    """
    extension = os.path.splitext(filename)[1]
    return extension in readers and bool(readers[extension]().CanReadFile(filename))


class TestVtkFileExtensions(unittest.TestCase):
    """
    Tests the file extensions of the VTK output used for visualisation in Paraview
    """

    def setUp(self):
        self.route = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.route)

    def test_write_zeta_vtk(self):
        """
        Lattices written by ``write_zeta_vtk`` (the modal and stability mode shapes) are structured grids and
        must be saved as ``.vts`` files
        """
        m, n = 2, 4
        x, y = np.meshgrid(np.linspace(0., 1., m + 1), np.linspace(0., 2., n + 1), indexing='ij')
        zeta_ref = [np.stack((x, y, np.zeros_like(x)))]  # [3, m + 1, n + 1]
        zeta = [zeta_ref[0] + np.array([0., 0., 0.1]).reshape(3, 1, 1)]  # uniform heave of 0.1

        write_zeta_vtk(zeta, zeta_ref, os.path.join(self.route, 'mode'))

        filename = os.path.join(self.route, 'mode_00.vts')
        self.assertTrue(os.path.isfile(filename))
        self.assertTrue(readable_by_extension(filename))

        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
        self.assertEqual(grid.GetNumberOfPoints(), (m + 1) * (n + 1))
        self.assertEqual(grid.GetNumberOfCells(), m * n)
        displacement = vtk_to_numpy(grid.GetPointData().GetArray('point_displacement_magnitude'))
        np.testing.assert_allclose(displacement, 0.1)

    def test_case_output(self):
        """
        Every VTK file written by ``AerogridPlot``, ``BeamPlot`` and ``Modal`` can be opened with the reader
        associated with its extension
        """
        ws = wings.Goland(M=4,
                          N=8,
                          Mstar_fact=2,
                          u_inf=10.,
                          alpha=0.,
                          route=self.route + '/cases',
                          case_name='goland_vtk')
        ws.clean_test_files()
        ws.update_derived_params()
        ws.set_default_config_dict()
        ws.generate_aero_file()
        ws.generate_fem_file()

        output_folder = self.route + '/output/'
        ws.config['SHARPy'] = {'flow': ['BeamLoader', 'AerogridLoader', 'AerogridPlot', 'BeamPlot', 'Modal'],
                               'case': ws.case_name,
                               'route': ws.route,
                               'write_screen': 'off',
                               'write_log': 'off',
                               'log_folder': output_folder}
        ws.config['BeamLoader'] = {'unsteady': 'off',
                                   'orientation': ws.quat}
        ws.config['AerogridLoader'] = {'unsteady': 'off',
                                       'aligned_grid': 'on',
                                       'mstar': ws.Mstar_fact * ws.M,
                                       'freestream_dir': ws.u_inf_direction,
                                       'wake_shape_generator': 'StraightWake',
                                       'wake_shape_generator_input': {'u_inf': ws.u_inf,
                                                                      'u_inf_direction': ws.u_inf_direction,
                                                                      'dt': ws.dt}}
        ws.config['AerogridPlot'] = {'include_rbm': 'off',
                                     'include_applied_forces': 'off',
                                     'minus_m_star': 0}
        ws.config['BeamPlot'] = {'include_rbm': 'off',
                                 'include_applied_forces': 'off'}
        ws.config['Modal'] = {'NumLambda': 4,
                              'rigid_body_modes': 'off',
                              'print_matrices': 'off',
                              'save_data': 'off',
                              'continuous_eigenvalues': 'off',
                              'dt': 0,
                              'plot_eigenvalues': False,
                              'write_modes_vtk': True,
                              'use_undamped_modes': True}
        ws.config.write()

        sharpy.sharpy_main.main(['', ws.route + '/' + ws.case_name + '.sharpy'])

        files = glob.glob(output_folder + '**/*.vt?', recursive=True)
        file_names = [os.path.basename(f) for f in files]
        # the aerodynamic body and wake, the beam and the mode shapes must all have been written
        for prefix in ['body_', 'wake_', 'beam_', 'tstep000000_ModalShape_']:
            self.assertTrue(any(name.startswith(prefix) for name in file_names),
                            f"No VTK output file starting with {prefix} was written")

        for filename in files:
            self.assertTrue(readable_by_extension(filename),
                            f"{os.path.basename(filename)} cannot be read by the reader of its extension")


if __name__ == '__main__':
    unittest.main()
