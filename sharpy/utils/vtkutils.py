from typing import TYPE_CHECKING

import vtk

if TYPE_CHECKING:
    from vtk import vtkDataSet
    from numpy.typing import NDArray


def write_data(vtk_data: "vtkDataSet", fname: str) -> None:
    if vtk_data.IsA("vtkUnstructuredGrid"):
        writer = vtk.vtkXMLUnstructuredGridWriter()
    elif vtk_data.IsA("vtkRectilinearGrid"):
        writer = vtk.vtkRectilinearGridWriter()
    else:
        raise NotImplementedError("Unsupported vtkDataSet")

    writer.SetFileName(fname)
    writer.SetInputData(vtk_data)
    is_successful_write: int = writer.Write()

    if not is_successful_write:
        raise RuntimeError("Failed to write data")


def create_points(coordinates: "NDArray") -> vtk.vtkPoints:
    assert coordinates.shape[1] == 3, "Each entry for the coordinate must have 3 values"

    points: vtk.vtkPoints = vtk.vtkPoints()
    for point in coordinates:
        points.InsertNextPoint(*point)

    return points


def create_quad_cells(connections: list[list[int]]) -> vtk.vtkCellArray:
    assert connections and len(connections[0]) == 4, "List must not be empty and nested list must have 4 elements"

    cells: vtk.vtkCellArray = vtk.vtkCellArray()
    for quad in connections:
        cell: vtk.vtkQuad = vtk.vtkQuad()
        for j in range(4):
            cell.GetPointIds().SetId(j, quad[j])
        cells.InsertNextCell(cell)

    return cells

