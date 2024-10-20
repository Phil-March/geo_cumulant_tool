import pandas as pd

# Load the CSV file
df = pd.read_csv('input/walker_lake.csv')

# Randomly sample 10% of the rows (which leaves 90% removed)
df_sampled = df.sample(frac=0.10, random_state=42)  # Use 'frac=0.10' to keep 10% of the data

# Save the reduced dataset to a new CSV file
df_sampled.to_csv('input/3d_walker_lake.csv', index=False)