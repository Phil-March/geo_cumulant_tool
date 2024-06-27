import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parameters
azimuth = 45  # in degrees
azimuth_tolerance = 22.5  # in degrees
bandwh = 1  # horizontal opening in length units

dip = 30  # in degrees
dip_tolerance = 5  # in degrees
bandwv = 5  # vertical opening in length units

lag = 20  # in units
lag_tolerance = 1  # in units

# Convert angles to radians
azimuth_rad = np.radians(azimuth)
azimuth_tolerance_rad = np.radians(azimuth_tolerance)

dip_rad = np.radians(dip)
dip_tolerance_rad = np.radians(dip_tolerance)

# Calculate azimuth and dip ranges
def limit_tolerance(angle, tolerance_rad, bandw):
    tolerance_length = lag * np.tan(tolerance_rad)
    if tolerance_length > bandw:
        tolerance_length = bandw
        tolerance_rad = np.arctan(bandw / lag)
    return angle - tolerance_rad, angle + tolerance_rad

azimuth_min, azimuth_max = limit_tolerance(azimuth_rad, azimuth_tolerance_rad, bandwh)
dip_min, dip_max = limit_tolerance(dip_rad, dip_tolerance_rad, bandwv)

# Define the vertices of the solid based on tolerances
def create_vertices(az_min, az_max, dip_min, dip_max, lag, lag_tol):
    vertices = []
    for az in [az_min, az_max]:
        for dip in [dip_min, dip_max]:
            for l in [lag - lag_tol, lag + lag_tol]:
                x = l * np.cos(dip) * np.cos(az)
                y = l * np.cos(dip) * np.sin(az)
                z = l * np.sin(dip)
                vertices.append([x, y, z])
    return np.array(vertices)

vertices = create_vertices(azimuth_min, azimuth_max, dip_min, dip_max, lag, lag_tolerance)

# Define the faces of the solid
faces = [
    [vertices[j] for j in [0, 1, 3, 2]],  # bottom face
    [vertices[j] for j in [4, 5, 7, 6]],  # top face
    [vertices[j] for j in [0, 1, 5, 4]],  # side face
    [vertices[j] for j in [2, 3, 7, 6]],  # side face
    [vertices[j] for j in [0, 2, 6, 4]],  # front face
    [vertices[j] for j in [1, 3, 7, 5]],  # back face
]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the solid
poly3d = [[tuple(vertex) for vertex in face] for face in faces]
ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

cubic_limit = 30
# Set the plot limits
ax.set_xlim([-cubic_limit, cubic_limit])
ax.set_ylim([-cubic_limit, cubic_limit])
ax.set_zlim([-cubic_limit, cubic_limit])

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
