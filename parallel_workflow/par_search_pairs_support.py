import cupy as cp
from math import atan2, degrees, radians, cos, sin, sqrt, tan
from numba import cuda
import numpy as np

@cuda.jit(device=True)
def par_calculate_azimuth_3d(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    azimuth = atan2(dx, dy)
    azimuth = degrees(azimuth)
    azimuth = (azimuth + 360) % 360
    return azimuth

@cuda.jit(device=True)
def par_calculate_dip_3d(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    horizontal_distance = sqrt(dx**2 + dy**2)
    dip = atan2(dz, horizontal_distance)
    dip = degrees(dip)
    return dip

@cuda.jit(device=True)
def par_horizontal_length_difference(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    horizontal_distance = sqrt(dx**2 + dy**2)
    return horizontal_distance

@cuda.jit(device=True)
def par_vertical_length_associated_with_dip(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    horizontal_distance = sqrt(dx**2 + dy**2)
    projected_vertical_length = horizontal_distance * tan(atan2(dz, horizontal_distance))
    return projected_vertical_length

@cuda.jit(device=True)
def par_point_position_before_or_equal_plane(x1, y1, z1, x2, y2, z2, length, azimuth, dip):
    azimuth_rad = radians(azimuth)
    dip_rad = radians(dip)
    dx = cos(dip_rad) * sin(azimuth_rad)
    dy = cos(dip_rad) * cos(azimuth_rad)
    dz = sin(dip_rad)
    x_end = x1 + length * dx
    y_end = y1 + length * dy
    z_end = z1 + length * dz
    normal_vector = cuda.local.array(3, dtype=np.float32)
    normal_vector[0] = dx
    normal_vector[1] = dy
    normal_vector[2] = dz
    vector_to_point = cuda.local.array(3, dtype=np.float32)
    vector_to_point[0] = x2 - x_end
    vector_to_point[1] = y2 - y_end
    vector_to_point[2] = z2 - z_end
    dot_product = (normal_vector[0] * vector_to_point[0] +
                   normal_vector[1] * vector_to_point[1] +
                   normal_vector[2] * vector_to_point[2])
    return dot_product <= 0

@cuda.jit(device=True)
def par_point_position_after_or_equal_plane(x1, y1, z1, x2, y2, z2, length, azimuth, dip):
    azimuth_rad = radians(azimuth)
    dip_rad = radians(dip)
    dx = cos(dip_rad) * sin(azimuth_rad)
    dy = cos(dip_rad) * cos(azimuth_rad)
    dz = sin(dip_rad)
    x_end = x1 + length * dx
    y_end = y1 + length * dy
    z_end = z1 + length * dz
    normal_vector = cuda.local.array(3, dtype=np.float32)
    normal_vector[0] = dx
    normal_vector[1] = dy
    normal_vector[2] = dz
    vector_to_point = cuda.local.array(3, dtype=np.float32)
    vector_to_point[0] = x2 - x_end
    vector_to_point[1] = y2 - y_end
    vector_to_point[2] = z2 - z_end
    dot_product = (normal_vector[0] * vector_to_point[0] +
                   normal_vector[1] * vector_to_point[1] +
                   normal_vector[2] * vector_to_point[2])
    return dot_product >= 0