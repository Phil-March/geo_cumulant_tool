import cupy as cp
from math import atan2, degrees, radians, cos, sin, sqrt, tan
from numba import cuda
import numpy as np

@cuda.jit(device=True)
def par_calculate_azimuth_3d(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    azimuth = atan2(dy, dx) 
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
def par_distance_along_horizontal_bandwidth(x1, y1, z1, x2, y2, z2, azimuth, dip):
    azimuth_rad = radians(azimuth)
    dip_rad = radians(dip)

    dx = cos(dip_rad) * cos(azimuth_rad)
    dy = cos(dip_rad) * sin(azimuth_rad)
    dz = sin(dip_rad)
    direction_vector = cuda.local.array(3, dtype=np.float64)
    direction_vector[0] = dx
    direction_vector[1] = dy
    direction_vector[2] = dz

    vector_to_point = cuda.local.array(3, dtype=np.float64)
    vector_to_point[0] = x2 - x1
    vector_to_point[1] = y2 - y1
    vector_to_point[2] = z2 - z1

    projection_length = (vector_to_point[0] * direction_vector[0] +
                         vector_to_point[1] * direction_vector[1] +
                         vector_to_point[2] * direction_vector[2])
    
    projection_vector = cuda.local.array(3, dtype=np.float64)
    projection_vector[0] = projection_length * direction_vector[0]
    projection_vector[1] = projection_length * direction_vector[1]
    projection_vector[2] = projection_length * direction_vector[2]

    perpendicular_vector = cuda.local.array(3, dtype=np.float64)
    perpendicular_vector[0] = vector_to_point[0] - projection_vector[0]
    perpendicular_vector[1] = vector_to_point[1] - projection_vector[1]
    perpendicular_vector[2] = vector_to_point[2] - projection_vector[2]

    horizontal_distance = sqrt(perpendicular_vector[0]**2 + perpendicular_vector[1]**2)
    return horizontal_distance

@cuda.jit(device=True)
def par_distance_along_vertical_bandwidth(x1, y1, z1, x2, y2, z2, azimuth, dip):
    azimuth_rad = radians(azimuth)
    dip_rad = radians(dip)

    dx = cos(dip_rad) * cos(azimuth_rad)
    dy = cos(dip_rad) * sin(azimuth_rad)
    dz = sin(dip_rad)
    direction_vector = cuda.local.array(3, dtype=np.float64)
    direction_vector[0] = dx
    direction_vector[1] = dy
    direction_vector[2] = dz

    vector_to_point = cuda.local.array(3, dtype=np.float64)
    vector_to_point[0] = x2 - x1
    vector_to_point[1] = y2 - y1
    vector_to_point[2] = z2 - z1

    projection_length = (vector_to_point[0] * direction_vector[0] +
                         vector_to_point[1] * direction_vector[1] +
                         vector_to_point[2] * direction_vector[2])

    projection_vector = cuda.local.array(3, dtype=np.float64)
    projection_vector[0] = projection_length * direction_vector[0]
    projection_vector[1] = projection_length * direction_vector[1]
    projection_vector[2] = projection_length * direction_vector[2]

    perpendicular_vector = cuda.local.array(3, dtype=np.float64)
    perpendicular_vector[0] = vector_to_point[0] - projection_vector[0]
    perpendicular_vector[1] = vector_to_point[1] - projection_vector[1]
    perpendicular_vector[2] = vector_to_point[2] - projection_vector[2]

    vertical_distance = abs(perpendicular_vector[2])
    return vertical_distance

@cuda.jit(device=True)
def par_point_distance_to_shifted_plane(x1, y1, z1, x2, y2, z2, lag, azimuth, dip):
    """
    Calculate the distance from point (x2, y2, z2) to the plane defined by the initial point (x1, y1, z1),
    azimuth, and dip, and shifted by the lag length.
    """
    # Convert azimuth and dip from degrees to radians
    azimuth_rad = radians(azimuth)
    dip_rad = radians(dip)

    # Calculate the normal vector to the plane based on azimuth and dip
    nx = cos(dip_rad) * cos(azimuth_rad)
    ny = cos(dip_rad) * sin(azimuth_rad)
    nz = sin(dip_rad)
    
    normal_vector = cuda.local.array(3, dtype=np.float64)
    normal_vector[0] = nx
    normal_vector[1] = ny
    normal_vector[2] = nz

    # Shift the initial point along the azimuth and dip vector by the lag length
    x_shifted = x1 + lag * nx
    y_shifted = y1 + lag * ny
    z_shifted = z1 + lag * nz

    # Calculate the vector from the shifted point to the second point
    vector_to_point = cuda.local.array(3, dtype=np.float64)
    vector_to_point[0] = x2 - x_shifted
    vector_to_point[1] = y2 - y_shifted
    vector_to_point[2] = z2 - z_shifted

    # Calculate the dot product of vector_to_point and normal_vector
    dot_product = (vector_to_point[0] * normal_vector[0] +
                   vector_to_point[1] * normal_vector[1] +
                   vector_to_point[2] * normal_vector[2])

    # Calculate the magnitude of the normal vector
    normal_magnitude = sqrt(normal_vector[0]**2 + normal_vector[1]**2 + normal_vector[2]**2)

    # Calculate the distance from the point to the shifted plane
    distance = dot_product / normal_magnitude

    return distance