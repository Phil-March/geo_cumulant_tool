import os
import pandas as pd

def compare_json_dfs(file1, file2):
    # Load the JSON files into DataFrames
    df1 = pd.read_json(file1)
    df2 = pd.read_json(file2)
    
    # Remove duplicates from both DataFrames
    df1 = df1.drop_duplicates()
    df2 = df2.drop_duplicates()
    
    # Find rows in df1 that are not in df2
    missing_in_df2 = df1[~df1.apply(tuple, 1).isin(df2.apply(tuple, 1))]
    
    # Find rows in df2 that are not in df1
    missing_in_df1 = df2[~df2.apply(tuple, 1).isin(df1.apply(tuple, 1))]

    return missing_in_df2, missing_in_df1

def list_missing_rows(file1, file2, output_folder):
    # Create the output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    missing_in_df2, missing_in_df1 = compare_json_dfs(file1, file2)
    
    # Save the missing rows to JSON files
    missing_in_df2_file = os.path.join(output_folder, "missing_in_df2.json")
    missing_in_df1_file = os.path.join(output_folder, "missing_in_df1.json")
    
    missing_in_df2.to_json(missing_in_df2_file, orient='records', lines=True)
    missing_in_df1.to_json(missing_in_df1_file, orient='records', lines=True)
    
    print(f"Missing rows from {file1} (not in {file2}) saved to: {missing_in_df2_file}")
    print(f"Missing rows from {file2} (not in {file1}) saved to: {missing_in_df1_file}")
    
    
# File paths
file1 = r"C:\Users\p_m12\Desktop\geo_cumulant_tool\output\par_pairs_2d_grid_test_data.json"
file2 = r"C:\Users\p_m12\Desktop\geo_cumulant_tool\output\seq_pairs_2d_grid_test_data.json"
output_folder = r"C:\Users\p_m12\Desktop\geo_cumulant_tool\temp_output"

# Call the function to list missing rows
list_missing_rows(file1, file2, output_folder)
