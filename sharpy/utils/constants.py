import ctypes as ct
import numpy as np

NDIM = int(3)
ct_NDIM = ct.c_uint(NDIM)
deg2rad = np.pi/180.
vortex_radius_def = 1e-6

cfact_biot = 0.25 / np.pi
