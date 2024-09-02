import os
import json
import pandas as pd

def list_output_files(file_type):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    output_folder = os.path.join(parent_dir, 'output')
    
    if not os.path.exists(output_folder):
        print("Output folder does not exist.")
        return []
    
    # Filter files based on file type
    if file_type == 'json':
        files = [f for f in os.listdir(output_folder) if f.endswith('.json')]
    elif file_type == 'csv':
        files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]
    else:
        files = []
    
    return files, output_folder

def display_files(files):
    print("Select the files by number:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

def compare_json_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Convert JSON data to DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Merge DataFrames on specific columns
    merged_df = pd.merge(df1, df2, on=["point_id", "dim_id", "n", "paired_point_id"], how="inner")

    # Calculate compliance percentage
    total_pairs = len(df2)
    matching_pairs = len(merged_df)
    compliance_percentage = (matching_pairs / total_pairs) * 100

    return compliance_percentage

def compare_csv_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Sort the rows and reset index to ensure matching rows regardless of order
    df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)
    
    # Merge DataFrames on all columns to check for matching rows
    merged_df = pd.merge(df1_sorted, df2_sorted, on=df1.columns.tolist(), how="inner")
    
    # Calculate compliance percentage
    total_rows = len(df2_sorted)
    matching_rows = len(merged_df)
    compliance_percentage = (matching_rows / total_rows) * 100

    return compliance_percentage

def main():
    while True:
        choice = input("Do you want to check:\n1. pairs (JSON) files\n2. cumulants (CSV) files\n3. Exit\nEnter 1, 2, or 3: ").strip()
        
        if choice == '3':
            print("Exiting the program.")
            break
        
        file_type = 'json' if choice == '1' else 'csv' if choice == '2' else None
        if not file_type:
            print("Invalid choice. Please choose 1 for pairs, 2 for cumulants, or 3 to exit.")
            continue
        
        files, output_folder = list_output_files(file_type)
        
        if not files:
            print(f"No {file_type.upper()} files found in the output folder.")
            continue
        
        display_files(files)
        
        file1_index = int(input("Enter the number for the first file: ").strip()) - 1
        file2_index = int(input("Enter the number for the second file: ").strip()) - 1
        
        file1_path = os.path.join(output_folder, files[file1_index])
        file2_path = os.path.join(output_folder, files[file2_index])

        if choice == '1':
            compliance_percentage = compare_json_files(file1_path, file2_path)
            print(f"Compliance percentage: {compliance_percentage:.2f}%")
            if compliance_percentage == 100:
                print("The JSON files are fully compliant (100% match).")
            else:
                print("The JSON files are not fully compliant.") 
        elif choice == '2':
            compliance_percentage = compare_csv_files(file1_path, file2_path)
            print(f"Compliance percentage: {compliance_percentage:.2f}%")
            if compliance_percentage == 100:
                print("The CSV files are fully compliant (100% match).")
            else:
                print("The CSV files are not fully compliant.")

if __name__ == "__main__":
    main()
