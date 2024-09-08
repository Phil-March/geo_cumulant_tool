import json
import numpy as np
import pandas as pd
import time
import os
from seq_search_pairs import seq_search_pairs_gen
from seq_cumulants import (center_grades, associate_grade, compute_3rd_order_cumulant, compute_4th_order_cumulant)

def load_parameters(ndir, file_name):
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

def compute_pairs():
    # Prompt user for the input ndir
    ndir = int(input("Please select the value for ndir: "))

    # Define the path to the search_parameters.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    search_parameters_path = os.path.join(parent_dir, 'search_parameters.json')

    # Load parameters
    dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv = load_parameters(ndir, file_name=search_parameters_path)

    # Path to the input directory
    input_dir = os.path.join(parent_dir, 'input')

    # List all files in the input directory
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    print("Available input files:")
    for i, file_name in enumerate(input_files):
        print(f"{i + 1}: {file_name}")

    # Prompt user to select a file
    file_index = int(input("Please select a file by entering the corresponding number: ")) - 1

    # Ensure the selected index is within range
    if file_index < 0 or file_index >= len(input_files):
        raise ValueError("Invalid file selection. Please select a valid file number.")

    # Get the selected file's path
    selected_file_name = input_files[file_index]
    selected_file_path = os.path.join(input_dir, selected_file_name)
    print(f"Selected file: {selected_file_path}")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(selected_file_path)

    # Add the point_id column as the first column
    df.insert(0, 'point_id', range(1, len(df) + 1))

    # Convert the DataFrame to a NumPy array
    data_vector = df.to_numpy()

    # Get the pairs with sequential
    start_time_seq = time.time()
    seq_pairs = seq_search_pairs_gen(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv)
    end_time_seq = time.time()
    time_seq = end_time_seq - start_time_seq
    print(f"Sequential function call completed in {time_seq} seconds.")

    # Convert pairs to a DataFrame
    seq_pairs_df = pd.DataFrame(seq_pairs, columns=["point_id", "dim_id", "n", "paired_point_id"])

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(parent_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file name
    output_file_name = f"seq_pairs_{os.path.splitext(selected_file_name)[0]}.json"
    output_file_path = os.path.join(output_dir, output_file_name)

    # Save the output to a JSON file in the output directory
    seq_pairs_df.to_json(output_file_path, orient='records', indent=4)

    print(f"Output saved to: {output_file_path}")

def compute_cumulants():
    # Define the path to the input and output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    input_dir = os.path.join(parent_dir, 'input')
    output_dir = os.path.join(parent_dir, 'output')

    # List all files in the input directory
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    print("Available input files:")
    for i, file_name in enumerate(input_files):
        print(f"{i + 1}: {file_name}")

    # Prompt user to select a data file
    data_file_index = int(input("Please select a data file by entering the corresponding number: ")) - 1

    # Ensure the selected index is within range
    if data_file_index < 0 or data_file_index >= len(input_files):
        raise ValueError("Invalid file selection. Please select a valid file number.")

    # Get the selected data file's path
    selected_data_file_name = input_files[data_file_index]
    selected_data_file_path = os.path.join(input_dir, selected_data_file_name)
    print(f"Selected data file: {selected_data_file_path}")

    # List all files in the output directory
    # List all JSON files in the output directory
    output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)) and f.endswith('.json')]
    print("Available output files:")
    for i, file_name in enumerate(output_files):
        print(f"{i + 1}: {file_name}")

    # Prompt user to select a pair file
    pair_file_index = int(input("Please select a pair file by entering the corresponding number: ")) - 1

    # Ensure the selected index is within range
    if pair_file_index < 0 or pair_file_index >= len(output_files):
        raise ValueError("Invalid file selection. Please select a valid file number.")

    # Get the selected pair file's path
    selected_pair_file_name = output_files[pair_file_index]
    selected_pair_file_path = os.path.join(output_dir, selected_pair_file_name)
    print(f"Selected pair file: {selected_pair_file_path}")

    # Prompt user to enter the number of chunks for the merging operation
    num_chunks = int(input("Please enter the number of chunks for the merging operation(limit memory usage): "))

    # Measure cumulative time for the entire process
    start_time = time.time()

    # Center grades
    df_centered = center_grades(selected_data_file_path)

    # Associate grades with pairs
    df_associated = associate_grade(df_centered, selected_pair_file_path)

    # Determine the number of dimensions in the pairs file
    num_dimensions = df_associated['dim_id'].nunique()

    if num_dimensions == 2:
        print("Computing 3rd-order cumulant...")
        cumulant_result = compute_3rd_order_cumulant(df_associated, num_chunks)
        output_cumulant_file_name = f"seq_cum_3rd_{os.path.splitext(selected_data_file_name)[0]}.csv"
    elif num_dimensions == 3:
        print("Computing 4th-order cumulant...")
        cumulant_result = compute_4th_order_cumulant(df_associated, num_chunks)
        output_cumulant_file_name = f"seq_cum_4th_{os.path.splitext(selected_data_file_name)[0]}.csv"
    else:
        raise ValueError(f"Unsupported number of dimensions: {num_dimensions}")

    # Save the cumulant result to a CSV file
    output_cumulant_file_path = os.path.join(output_dir, output_cumulant_file_name)
    cumulant_result.to_csv(output_cumulant_file_path, index=False)

    # End time for the cumulative process
    end_time = time.time()
    print(f"Total time for computing cumulants: {end_time - start_time:.2f} seconds.")
    print(f"Cumulant results saved to: {output_cumulant_file_path}")

def main():
    while True:
        print("\nMenu:")
        print("1. Compute pairs")
        print("2. Compute cumulants from pairs")
        print("3. Close")

        choice = input("Please select an option (1, 2, or 3): ")

        if choice == '1':
            compute_pairs()
        elif choice == '2':
            compute_cumulants()
        elif choice == '3':
            print("Closing the program.")
            break
        else:
            print("Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()
