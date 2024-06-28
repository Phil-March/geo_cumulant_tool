import json
import numpy as np
import pandas as pd
import time

# Load initial data
df_data = pd.read_csv('3d_data.csv')

# Add the point_id column as the first column
df_data.insert(0, 'point_id', range(1, len(df_data) + 1))

# Load the JSON pairs data into a DataFrame
df_pairs = pd.read_json('par_pairs.json')

# Merge df_pairs with df_data to get the values for point_id
df_pairs = df_pairs.merge(df_data[['point_id', 'GRADE']], on='point_id', how='left')
df_pairs = df_pairs.rename(columns={'GRADE': 'point_id_value'})

# Merge df_pairs with df_data to get the values for paired_point_id
df_pairs = df_pairs.merge(df_data[['point_id', 'GRADE']], left_on='paired_point_id', right_on='point_id', how='left', suffixes=('', '_paired_id'))
df_pairs = df_pairs.rename(columns={'GRADE': 'paired_point_id_value'})

# Drop the extra point_id column from the second merge
df_pairs = df_pairs.drop(columns=['point_id_paired_id'])

# Display the updated DataFrame
print(df_pairs.columns)

# Function to compute cumulants
def compute_cumulants(df, order):
    # Extract values and calculate the mean
    values = df[['point_id_value', 'paired_point_id_value']].values
    mean_value = np.mean(values)

    # Center the data
    centered_values = values - mean_value

    # Initialize lists for cumulants
    second_order_cumulants = []
    third_order_cumulants = []

    if order >= 2:
        # Compute second-order cumulant (covariance)
        for index, row in df.iterrows():
            key = (row['dim_id'], row['n'])
            cumulant_value = centered_values[index, 0] * centered_values[index, 1]
            second_order_cumulants.append([key[0], key[1], cumulant_value])
        
        # Create DataFrame and normalize
        second_order_cumulants_df = pd.DataFrame(second_order_cumulants, columns=['dim_id', 'n', 'cumulant'])
        second_order_cumulants_df['cumulant'] /= len(df)
    else:
        second_order_cumulants_df = pd.DataFrame(columns=['dim_id', 'n', 'cumulant'])

    if order >= 3:
        # Compute third-order cumulant (skewness)
        for index1, row1 in df.iterrows():
            for index2, row2 in df.iterrows():
                if index1 != index2:
                    key = (row1['dim_id'], row1['n'], row2['dim_id'], row2['n'])
                    cumulant_value = centered_values[index1, 0] * centered_values[index1, 1] * centered_values[index2, 1]
                    third_order_cumulants.append([key[0], key[1], key[2], key[3], cumulant_value])
        
        # Create DataFrame and normalize
        third_order_cumulants_df = pd.DataFrame(third_order_cumulants, columns=['dim_id1', 'n1', 'dim_id2', 'n2', 'cumulant'])
        third_order_cumulants_df['cumulant'] /= len(df)
    else:
        third_order_cumulants_df = pd.DataFrame(columns=['dim_id1', 'n1', 'dim_id2', 'n2', 'cumulant'])

    return second_order_cumulants_df, third_order_cumulants_df

order = 3
second_order, third_order = compute_cumulants(df_pairs, order)

print(second_order)
print(third_order)
