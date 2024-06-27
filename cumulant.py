import json
import numpy as np
import pandas as pd
from seq_search_pairs import seq_search_pairs_gen

# Load the CSV file into a DataFrame
df = pd.read_csv('3d_data.csv')

# Add the point_id column as the first column
df.insert(0, 'point_id', range(1, len(df) + 1))

# Convert the DataFrame to a NumPy array
data_vector = df.to_numpy()

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

# Get the pairs
pairs = seq_search_pairs_gen(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv)

# Convert pairs to a DataFrame and save to JSON file
# pairs_df = pd.DataFrame(pairs, columns=["point_id", "dim_id", "n", "paired_point_id"])
# pairs_df.to_json('pairs.json', orient='records', indent=4)

# Convert pairs to a DataFrame
pairs_df = pd.DataFrame(pairs, columns=["point_id", "dim_id", "n", "paired_point_id"])

# Group by point_id, dim_id, and n, and aggregate the paired_point_id into lists
grouped_pairs_df = pairs_df.groupby(["point_id", "dim_id", "n"])["paired_point_id"].apply(list).reset_index()

# Convert the DataFrame to a dictionary
pairs_list = grouped_pairs_df.to_dict(orient="records")

# Write pairs to a JSON file
with open('pairs.json', 'w') as json_file:
    json.dump(pairs_list, json_file, indent=4)
