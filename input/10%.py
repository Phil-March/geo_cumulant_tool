import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file
df = pd.read_csv('input/walker_lake.csv')

# Filter rows where 'Y', 'X', and 'Z' are within specified ranges
df_filtered = df[(df['Y'] >= 40) & (df['Y'] <= 100) & (df['X'] >= 0) & (df['X'] <= 60) & (df['Z'] >= 20) & (df['Z'] <= 50)]

# Group the points into 10x10x10 cubes
df_filtered['X_group'] = (df_filtered['X'] // 10).astype(int)
df_filtered['Y_group'] = (df_filtered['Y'] // 10).astype(int)
df_filtered['Z_group'] = (df_filtered['Z'] // 10).astype(int)

# Function to keep one point per cube unless GRADE is above 400
def filter_by_cube(group):
    # Check if any values in the group are above 400
    above_400 = group[group['GRADE'] > 400]
    
    if len(above_400) > 0:
        # If any point has a GRADE above 400, keep all of them
        return above_400
    else:
        # Otherwise, keep the point with the highest GRADE within the cube
        return group.nlargest(1, 'GRADE')

# Apply the filtering function for each cube (grouped by X_group, Y_group, and Z_group)
df_filtered = df_filtered.groupby(['X_group', 'Y_group', 'Z_group'], group_keys=False).apply(filter_by_cube)

# Drop the group columns used for filtering
df_filtered = df_filtered.drop(columns=['X_group', 'Y_group', 'Z_group'])

# Save the reduced dataset to a new CSV file
df_filtered.to_csv('input/3d_walker_lake.csv', index=False)

# Create a 3D scatter plot of the filtered data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
scat = ax.scatter(df_filtered['X'], df_filtered['Y'], df_filtered['Z'], 
                  c=df_filtered['GRADE'], cmap='viridis', s=50)

# Add color bar to indicate the 'GRADE' values
cbar = plt.colorbar(scat, ax=ax)
cbar.set_label('GRADE')

# Set axis labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

# Set title for the plot
plt.title('3D Scatter Plot of Points with GRADE Values as Color')

# Show the plot
plt.show()
