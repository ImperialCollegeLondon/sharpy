from typing import TYPE_CHECKING, Literal, Callable

import numpy as np
import vtk

if TYPE_CHECKING:
    from vtk import vtkDataSet, vtkDataSetAttributes, vtkDataArray
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


def add_array(vtk_data: "vtkDataSet", target_attr: Literal["point", "cell"], name: str, data_to_add: "NDArray",
              num_components: int, attribute_array: Literal["scalars", "vectors"] | None = None,
              num_points: int | None = None) -> None:
    array: vtk.vtkDataArray = _create_vtk_data_array(data_to_add)
    array.SetName(name)
    array.SetNumberOfComponents(num_components)
    num_tuples: int = num_points if num_points is not None else len(data_to_add)
    array.SetNumberOfTuples(num_tuples)

    array_setter: Callable = array.SetValue if num_components == 1 else array.SetTuple
    for i, val in enumerate(data_to_add):
        array_setter(i, val)

    dataset_attribute_data: "vtkDataSetAttributes" = _get_dataset_attribute(target_attr, vtk_data)
    if attribute_array is None:
        dataset_attribute_data.AddArray(array)
    elif attribute_array == "scalars":
        dataset_attribute_data.SetScalars(array)
    elif attribute_array == "vectors":
        dataset_attribute_data.SetVectors(array)
    else:
        raise NotImplementedError("Unsupported attribute_array")


def _get_dataset_attribute(target_attr: Literal["point", "cell"], vtk_data: "vtkDataSet") -> "vtkDataSetAttributes":
    dataset_attribute_data: "vtkDataSetAttributes"
    if target_attr == "point":
        dataset_attribute_data = vtk_data.GetPointData()
    elif target_attr == "cell":
        dataset_attribute_data = vtk_data.GetCellData()
    else:
        raise NotImplementedError("Unsupported data type")
    return dataset_attribute_data


def _create_vtk_data_array(data_to_add: "NDArray") -> "vtkDataArray":
    array: vtk.vtkDataArray
    if np.issubdtype(data_to_add.dtype, np.integer):
        array = vtk.vtkIntArray()
    elif np.issubdtype(data_to_add.dtype, np.double):
        array: vtk.vtkDoubleArray = vtk.vtkDoubleArray()
    else:
        raise NotImplementedError(f"Unsupported dtype ({data_to_add.dtype}) of data_to_add")
    return array
