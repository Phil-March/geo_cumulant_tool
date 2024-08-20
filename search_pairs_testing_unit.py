import json
import numpy as np
import pandas as pd
import time
from seq_search_pairs import seq_search_pairs_gen
# from par_search_pairs import par_search_pairs_gen

# Load the CSV file into a DataFrame
df = pd.read_csv('3d_data.csv')

# Add the point_id column as the first column
df.insert(0, 'point_id', range(1, len(df) + 1))

# Convert the DataFrame to a NumPy array
data_vector = df.to_numpy()
# print(data_vector)

file_name = 'parameters.json'
ndir = 3

def load_parameters(file_name, ndir):
    # Load JSON file
    with open(file_name, 'r') as file:
        data = json.load(file)
    
    # Extract parameters from JSON based on ndir
    params = data[f'ndir{ndir}']

    # Initialize parameters
    dim = [i for i in range(ndir)]
    nlag = [params.get(f'nlag{i+1}', 0) for i in range(ndir)]
    lag = [params.get(f'lag{i+1}', 0.0) for i in range(ndir)]
    lag_tol = [params.get(f'lagtol{i+1}', 0.0) for i in range(ndir)]
    azm = [params.get(f'az{i+1}', 0.0) for i in range(ndir)]
    azm_tol = [params.get(f'aztol{i+1}', 0.0) for i in range(ndir)]
    bandwh = [params.get(f'bandh{i+1}', 0.0) for i in range(ndir)]
    dip = [params.get(f'dip{i+1}', 0.0) for i in range(ndir)]
    dip_tol = [params.get(f'dtol{i+1}', 0.0) for i in range(ndir)]
    bandwv = [params.get(f'bandv{i+1}', 0.0) for i in range(ndir)]

    return dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv

# Load parameters
dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv = load_parameters(file_name, ndir)


# # Get the pairs with parallel
# start_time_par = time.time()
# par_pairs = par_search_pairs_gen(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv)
# end_time_par = time.time()
# time_par = end_time_par - start_time_par
# print(f"Parallel function call completed in {time_par} seconds.")

# par_pairs_df = pd.DataFrame(par_pairs, columns=["point_id", "dim_id", "n", "paired_point_id"])
# par_pairs_df.to_json('par_pairs.json', orient='records', indent=4)


# Get the pairs with sequential
start_time_seq = time.time()
seq_pairs = seq_search_pairs_gen(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv)
end_time_seq = time.time()
time_seq = end_time_seq - start_time_seq
print(f"Sequential function call completed in {time_seq} seconds.")

# Convert pairs to a DataFrame and save to JSON file
seq_pairs_df = pd.DataFrame(seq_pairs, columns=["point_id", "dim_id", "n", "paired_point_id"])
seq_pairs_df.to_json('seq_pairs.json', orient='records', indent=4)

# Computation time efficiency increase

# Merge the DataFrames to find matching pairs
# merged_df = pd.merge(par_pairs_df, seq_pairs_df, on=["point_id", "dim_id", "n", "paired_point_id"], how="inner")

# # Count the total number of pairs in the sequential DataFrame
# total_pairs = len(seq_pairs_df)

# # Count the number of matching pairs
# matching_pairs = len(merged_df)

# # Calculate the compliance percentage
# compliance_percentage = (matching_pairs / total_pairs) * 100

# # Calculate the computational time reduction
# time_reduction = ((time_seq - time_par) / time_seq) * 100

# print(f"Compliance percentage: {compliance_percentage:.2f}%")
# print(f"Computational time reduction: {time_reduction:.2f}%")