import numpy as np

def seq_calculate_azimuth_3d(x1, y1, z1, x2, y2, z2):
    """
    Calculate the azimuth between two points (x1, y1, z1) and (x2, y2, z2) in 3D space
    """
    # Calculate the differences in x and y coordinates
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the azimuth in radians
    azimuth = np.arctan2(dx, dy)

    # Convert azimuth from radians to degrees
    azimuth = np.degrees(azimuth)

    # Normalize the azimuth to the range [0, 360) degrees
    azimuth = (azimuth + 360) % 360

    return azimuth

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

def seq_horizontal_length_difference(x1, y1, z1, x2, y2, z2):
    """
    Calculate the horizontal length difference associated with the azimuth between two 3D points
    """
    # Calculate the differences in x and y coordinates
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the horizontal distance
    horizontal_distance = np.sqrt(dx**2 + dy**2)

    return horizontal_distance


def seq_vertical_length_associated_with_dip(x1, y1, z1, x2, y2, z2):
    """
    Calculate the vertical length associated with the dip between two 3D points
    """
    # Calculate the differences in coordinates
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Calculate the horizontal distance between the two points
    horizontal_distance = np.sqrt(dx**2 + dy**2)

    # Calculate the projected vertical length using the dip angle
    projected_vertical_length = horizontal_distance * np.tan(np.arctan2(dz, horizontal_distance))

    return projected_vertical_length

def seq_point_position_before_or_equal_plane(x1, y1, z1, x2, y2, z2, length, azimuth, dip):
    """
    Check if the point (x2, y2, z2) is before, on, or after the plane defined by the initial point,
    length, azimuth, and dip.
    """
    # Convert azimuth and dip from degrees to radians
    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # Calculate the direction vector
    dx = np.cos(dip_rad) * np.sin(azimuth_rad)
    dy = np.cos(dip_rad) * np.cos(azimuth_rad)
    dz = np.sin(dip_rad)
    direction = np.array([dx, dy, dz])

    # Calculate the end point of the line segment
    x_end = x1 + length * dx
    y_end = y1 + length * dy
    z_end = z1 + length * dz

    # Calculate the vector from the initial point to the end point (normal to the plane)
    normal_vector = direction

    # Calculate the vector from the initial point to the point to check
    vector_to_point = np.array([x2 - x_end, y2 - y_end, z2 - z_end])

    # Calculate the dot product
    dot_product = np.dot(normal_vector, vector_to_point)

    if dot_product <= 0:
        return True  # Point is before or on the plane
    return False  # Point is after the plane

def seq_point_position_after_or_equal_plane(x1, y1, z1, x2, y2, z2, length, azimuth, dip):
    """
    Check if the point (x2, y2, z2) is before, on, or after the plane defined by the initial point,
    length, azimuth, and dip.
    """
    # Convert azimuth and dip from degrees to radians
    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # Calculate the direction vector
    dx = np.cos(dip_rad) * np.sin(azimuth_rad)
    dy = np.cos(dip_rad) * np.cos(azimuth_rad)
    dz = np.sin(dip_rad)
    direction = np.array([dx, dy, dz])

    # Calculate the end point of the line segment
    x_end = x1 + length * dx
    y_end = y1 + length * dy
    z_end = z1 + length * dz

    # Calculate the vector from the initial point to the end point (normal to the plane)
    normal_vector = direction

    # Calculate the vector from the initial point to the point to check
    vector_to_point = np.array([x2 - x_end, y2 - y_end, z2 - z_end])

    # Calculate the dot product
    dot_product = np.dot(normal_vector, vector_to_point)

    if dot_product >= 0:
        return True  # Point is after or on the plane
    return False  # Point is before the plane