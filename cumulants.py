import json
import numpy as np
import pandas as pd
import time

#Load initial data
df_data = pd.read_csv('3d_data.csv')

# Add the point_id column as the first column
df_data.insert(0, 'point_id', range(1, len(df_data) + 1))

# Load the JSON pairs data into a DataFrame
df_pairs = pd.read_json('par_pairs.json')

# Merge df_pairs with df_data to get the GRADE for point_id
df_pairs = df_pairs.merge(df_data[['point_id', 'GRADE']], left_on='point_id', right_on='point_id', how='left')
df_pairs = df_pairs.rename(columns={'GRADE': 'point_id_value'})

# Merge df_pairs with df_data to get the GRADE for paired_point_id
df_pairs = df_pairs.merge(df_data[['point_id', 'GRADE']], left_on='paired_point_id', right_on='point_id', how='left')
df_pairs = df_pairs.rename(columns={'GRADE': 'paired_point_id_value'})

# Drop the extra point_id column from the second merge
df_pairs = df_pairs.drop(columns=['point_id_y'])
df_pairs = df_pairs.rename(columns={'point_id_x': 'point_id'})

# Display the updated DataFrame
print(df_pairs.head())

def compute_cumulants(df, order):
    # Extract values and calculate the mean
    values = df[['point_id_value', 'paired_point_id_value']].values
    mean_value = np.mean(values)

    # Center the data
    centered_values = values - mean_value

    # Initialize cumulants dictionaries
    second_order_cumulants = {}
    third_order_cumulants = {}
    fourth_order_cumulants = {}

    if order >= 2:
        # Compute second-order cumulant (covariance)
        for index, row in df.iterrows():
            key = (row['dim_id'], row['n'])
            if key not in second_order_cumulants:
                second_order_cumulants[key] = 0
            second_order_cumulants[key] += centered_values[index, 0] * centered_values[index, 1]
        
        # Normalize
        for key in second_order_cumulants:
            second_order_cumulants[key] /= len(df)
    
    if order >= 3:
        # Compute third-order cumulant (skewness)
        for index1, row1 in df.iterrows():
            for index2, row2 in df.iterrows():
                if index1 != index2:
                    key = (row1['dim_id'], row1['n'], row2['dim_id'], row2['n'])
                    if key not in third_order_cumulants:
                        third_order_cumulants[key] = 0
                    third_order_cumulants[key] += centered_values[index1, 0] * centered_values[index1, 1] * centered_values[index2, 1]
        
        # Normalize
        for key in third_order_cumulants:
            third_order_cumulants[key] /= len(df)
    
    if order >= 4:
        # Compute fourth-order cumulant (kurtosis)
        for index1, row1 in df.iterrows():
            for index2, row2 in df.iterrows():
                for index3, row3 in df.iterrows():
                    for index4, row4 in df.iterrows():
                        if index1 != index2 and index1 != index3 and index1 != index4 and index2 != index3 and index2 != index4 and index3 != index4:
                            key = (row1['dim_id'], row1['n'], row2['dim_id'], row2['n'], row3['dim_id'], row3['n'], row4['dim_id'], row4['n'])
                            if key not in fourth_order_cumulants:
                                fourth_order_cumulants[key] = 0
                            fourth_order_cumulants[key] += centered_values[index1, 0] * centered_values[index1, 1] * centered_values[index2, 1] * centered_values[index3, 1] * centered_values[index4, 1]
        
        # Normalize
        for key in fourth_order_cumulants:
            fourth_order_cumulants[key] /= len(df)
    
    return second_order_cumulants, third_order_cumulants, fourth_order_cumulants