import pandas as pd

# Load the CSV file
df = pd.read_csv('input/walker_lake.csv')

# Filter rows where 'Y' and 'X' are between 0 and 50
df_filtered = df[(df['Y'] >= 0) & (df['Y'] <= 50) & (df['X'] >= 0) & (df['X'] <= 50)]

# Randomly sample 50% of the remaining rows
df_sampled = df_filtered.sample(frac=0.50, random_state=42)

# Save the reduced dataset to a new CSV file
df_sampled.to_csv('input/3d_walker_lake.csv', index=False)
