import numpy as np

def seq_calculate_azimuth_3d(x1, y1, z1, x2, y2, z2):
    """
    Calculate the azimuth between two points in 3D space.

    """
    # Calculate the differences in coordinates
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the azimuth
    azimuth = np.arctan2(dy, dx)

    # Convert from radians to degrees
    azimuth_degrees = np.degrees(azimuth)

    # Normalize the azimuth to be between 0 and 360 degrees
    if (azimuth_degrees < 0):
        azimuth_degrees += 360

    return azimuth_degrees

def seq_calculate_dip_3d(x1, y1, z1, x2, y2, z2):
    """
    Calculate the dip (elevation angle) between two points (x1, y1, z1) and (x2, y2, z2) in 3D space
    """
    # Calculate the differences in coordinates
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Calculate the horizontal distance between the two points
    horizontal_distance = np.sqrt(dx**2 + dy**2)

    # Calculate the dip in radians
    dip = np.arctan2(dz, horizontal_distance)

    # Convert dip from radians to degrees
    dip = np.degrees(dip)

    return dip

def distance_along_horizontal_bandwidth(x1, y1, z1, x2, y2, z2, azimuth, dip):
    """
    Calculate the horizontal distance perpendicular to the plane defined by azimuth and dip 
    from the initial point (x1, y1, z1) to the point (x2, y2, z2).
    """
    # Convert azimuth and dip from degrees to radians
    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # Calculate the direction vector based on azimuth and dip
    dx = np.cos(dip_rad) * np.cos(azimuth_rad)
    dy = np.cos(dip_rad) * np.sin(azimuth_rad)
    dz = np.sin(dip_rad)
    direction_vector = np.array([dx, dy, dz])

    # Vector from the initial point to the second point
    vector_to_point = np.array([x2 - x1, y2 - y1, z2 - z1])

    # Project the vector_to_point onto the direction_vector
    projection_length = np.dot(vector_to_point, direction_vector)
    projection_vector = projection_length * direction_vector

    # Calculate the perpendicular (horizontal) distance from the initial point to the projection
    perpendicular_vector = vector_to_point - projection_vector
    horizontal_distance = np.linalg.norm(perpendicular_vector)

    return horizontal_distance


def distance_along_vertical_bandwidth(x1, y1, z1, x2, y2, z2, azimuth, dip):
    """
    Calculate the vertical distance perpendicular to the plane defined by azimuth and dip 
    from the initial point (x1, y1, z1) to the point (x2, y2, z2).
    """
    # Convert azimuth and dip from degrees to radians
    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # Calculate the direction vector based on azimuth and dip
    dx = np.cos(dip_rad) * np.cos(azimuth_rad)
    dy = np.cos(dip_rad) * np.sin(azimuth_rad)
    dz = np.sin(dip_rad)
    direction_vector = np.array([dx, dy, dz])

    # Vector from the initial point to the second point
    vector_to_point = np.array([x2 - x1, y2 - y1, z2 - z1])

    # Project the vector_to_point onto the direction_vector
    projection_length = np.dot(vector_to_point, direction_vector)
    projection_vector = projection_length * direction_vector

    # Calculate the perpendicular (vertical) distance from the initial point to the projection
    perpendicular_vector = vector_to_point - projection_vector
    vertical_distance = np.abs(perpendicular_vector[2])

    return vertical_distance

def seq_point_distance_to_shifted_plane(x1, y1, z1, x2, y2, z2, lag, azimuth, dip):
    """
    Calculate the distance from point (x2, y2, z2) to the plane defined by the initial point (x1, y1, z1),
    azimuth, and dip, and shifted by the lag length.
    """
    # Convert azimuth and dip from degrees to radians
    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # Calculate the normal vector to the plane based on azimuth and dip
    nx = np.cos(dip_rad) * np.cos(azimuth_rad)
    ny = np.cos(dip_rad) * np.sin(azimuth_rad)
    nz = np.sin(dip_rad)
    normal_vector = np.array([nx, ny, nz])

    # Shift the initial point along the azimuth and dip vector by the lag length
    x_shifted = x1 + lag * nx
    y_shifted = y1 + lag * ny
    z_shifted = z1 + lag * nz

    # Calculate the vector from the shifted point to the second point
    vector_to_point = np.array([x2 - x_shifted, y2 - y_shifted, z2 - z_shifted])

    # Calculate the distance from the point to the shifted plane using the dot product
    distance = np.dot(vector_to_point, normal_vector) / np.linalg.norm(normal_vector)

    return distance