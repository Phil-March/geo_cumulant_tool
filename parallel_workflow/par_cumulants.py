import cupy as cp
import pandas as pd
import itertools
from numba import cuda

# Function to center grades using CuPy and Numba for GPU acceleration
@cuda.jit
def center_grades_kernel(grades, mean_grade, grades_centered):
    i = cuda.grid(1)
    if i < grades.size:
        grades_centered[i] = grades[i] - mean_grade[0]  # Access the first element of the array

def center_grades(data_file):
    # Load the data with Pandas
    df_data = pd.read_csv(data_file)
    
    # Convert to CuPy arrays
    grades = cp.array(df_data['GRADE'].values)
    
    # Compute the mean of the GRADE column and convert it to a CuPy array
    mean_grade = cp.array([cp.mean(grades)])
    
    # Create an array to hold the centered grades
    grades_centered = cp.empty_like(grades)
    
    # Launch the kernel
    threads_per_block = 128
    blocks_per_grid = (grades.size + threads_per_block - 1) // threads_per_block
    center_grades_kernel[blocks_per_grid, threads_per_block](grades, mean_grade, grades_centered)
    
    # Convert back to DataFrame
    df_data['GRADE_centered'] = cp.asnumpy(grades_centered)
    
    return df_data

# Function to merge and associate grades using CuPy
def associate_grade(df_data, pairs_file):
    # Add the point_id column as the first column
    df_data.insert(0, 'point_id', range(1, len(df_data) + 1))

    # Load the JSON pairs data with Pandas
    df_pairs = pd.read_json(pairs_file)

    # Convert to CuPy arrays
    point_ids = cp.array(df_data['point_id'].values)
    grades = cp.array(df_data['GRADE'].values)
    
    # Create dictionaries to map point IDs to grades
    point_id_map = dict(zip(point_ids.get(), grades.get()))
    df_pairs['point_id_value'] = df_pairs['point_id'].map(point_id_map)
    df_pairs['paired_point_id_value'] = df_pairs['paired_point_id'].map(point_id_map)

    return df_pairs

# Function to compute the 3rd order cumulant using CuPy and Numba
@cuda.jit
def compute_3rd_order_cumulant_kernel(point_id_values, paired_point_id_values_dir_0, paired_point_id_values_dir_1, result):
    i = cuda.grid(1)
    if i < point_id_values.size:
        result[i] = point_id_values[i] * paired_point_id_values_dir_0[i] * paired_point_id_values_dir_1[i]

def compute_3rd_order_cumulant(df_pairs):
    # Convert to CuPy arrays
    dim_id = cp.array(df_pairs['dim_id'].values)
    n_values = cp.array(df_pairs['n'].values)
    
    # Determine the maximum values of n for each dim_id (0 and 1)
    max_n_dim_0 = int(cp.max(n_values[dim_id == 0]))  # Convert to Python int
    max_n_dim_1 = int(cp.max(n_values[dim_id == 1]))  # Convert to Python int
    
    # Generate all combinations of values for each column
    combinations = list(itertools.product(range(1, max_n_dim_0 + 1), range(1, max_n_dim_1 + 1)))
    df_generated = pd.DataFrame(combinations, columns=['dir_0_nlag', 'dir_1_nlag'])

    # Merge for direction 0
    merged_0 = df_generated.merge(df_pairs[df_pairs['dim_id'] == 0], left_on='dir_0_nlag', right_on='n', how='left')

    # Merge for direction 1
    result = merged_0.merge(df_pairs[df_pairs['dim_id'] == 1], left_on='dir_1_nlag', right_on='n', how='left', suffixes=('', '_dir_1'))

    # Filter rows where all point_id_dir_0 and point_id_dir_1 match
    mask = (result['point_id'] == result['point_id_dir_1'])
    result = result[mask]
    
    # Allocate memory for the result
    E = cp.empty(result.shape[0], dtype=cp.float64)
    
    # Launch the kernel to compute E
    threads_per_block = 128
    blocks_per_grid = (E.size + threads_per_block - 1) // threads_per_block
    compute_3rd_order_cumulant_kernel[blocks_per_grid, threads_per_block](
        cp.array(result['point_id_value'].values),
        cp.array(result['paired_point_id_value'].values),
        cp.array(result['paired_point_id_value_dir_1'].values),
        E
    )
    
    # Compute the average of E and group by the nlag columns
    result['E'] = cp.asnumpy(E)
    final_result = result.groupby(['dir_0_nlag', 'dir_1_nlag'])['E'].mean().reset_index()
    final_result = final_result.rename(columns={'E': 'k_3'})

    return final_result

@cuda.jit
def compute_4th_order_cumulant_kernel(point_id_values, paired_point_id_values_dir_0, paired_point_id_values_dir_1, paired_point_id_values_dir_2, E_0):
    i = cuda.grid(1)
    if i < point_id_values.size:
        E_0[i] = point_id_values[i] * paired_point_id_values_dir_0[i] * paired_point_id_values_dir_1[i] * paired_point_id_values_dir_2[i]

def compute_4th_order_cumulant(df_pairs):
    # Convert to CuPy arrays
    dim_id = cp.array(df_pairs['dim_id'].values)
    n_values = cp.array(df_pairs['n'].values)
    point_id_values = cp.array(df_pairs['point_id_value'].values)

    # Determine the maximum values of n for each dim_id
    max_n_dim_0 = int(cp.max(n_values[dim_id == 0]))
    max_n_dim_1 = int(cp.max(n_values[dim_id == 1]))
    max_n_dim_2 = int(cp.max(n_values[dim_id == 2]))

    # Generate all combinations of values for each column
    dir_0_nlag, dir_1_nlag, dir_2_nlag = cp.meshgrid(
        cp.arange(1, max_n_dim_0 + 1),
        cp.arange(1, max_n_dim_1 + 1),
        cp.arange(1, max_n_dim_2 + 1),
        indexing='ij'
    )
    dir_0_nlag = dir_0_nlag.ravel()
    dir_1_nlag = dir_1_nlag.ravel()
    dir_2_nlag = dir_2_nlag.ravel()

    # Prepare masks for each dimension
    mask_0 = (dim_id == 0)
    mask_1 = (dim_id == 1)
    mask_2 = (dim_id == 2)

    # Extract values for each dimension
    n_values_0 = n_values[mask_0]
    n_values_1 = n_values[mask_1]
    n_values_2 = n_values[mask_2]

    point_id_values_0 = point_id_values[mask_0]
    point_id_values_1 = point_id_values[mask_1]
    point_id_values_2 = point_id_values[mask_2]

    # Perform the merging operation using broadcasting
    valid_rows_0 = cp.isin(n_values_0, dir_0_nlag)
    valid_rows_1 = cp.isin(n_values_1, dir_1_nlag)
    valid_rows_2 = cp.isin(n_values_2, dir_2_nlag)

    # Filter point_id_values for valid rows, matching each dimension separately
    paired_point_id_values_dir_0 = point_id_values_0[valid_rows_0]
    paired_point_id_values_dir_1 = point_id_values_1[valid_rows_1]
    paired_point_id_values_dir_2 = point_id_values_2[valid_rows_2]

    # Compute E_1, E_2, E_3 means
    E_1 = (paired_point_id_values_dir_0 * paired_point_id_values_dir_1).mean() * \
          paired_point_id_values_dir_2.mean()

    E_2 = (paired_point_id_values_dir_0 * paired_point_id_values_dir_2).mean() * \
          paired_point_id_values_dir_1.mean()

    E_3 = (paired_point_id_values_dir_1 * paired_point_id_values_dir_2).mean() * \
          paired_point_id_values_dir_0.mean()

    # Prepare arrays for the kernel computation
    E_0 = cp.empty(len(dir_0_nlag), dtype=cp.float64)

    # Launch the kernel to compute E_0
    threads_per_block = 128
    blocks_per_grid = (E_0.size + threads_per_block - 1) // threads_per_block
    compute_4th_order_cumulant_kernel[blocks_per_grid, threads_per_block](
        point_id_values_0[valid_rows_0],  # Pass the valid values directly to the kernel
        paired_point_id_values_dir_0,
        paired_point_id_values_dir_1,
        paired_point_id_values_dir_2,
        E_0
    )

    # Compute the fourth-order cumulant: k_4 = E_0 - (E_1 + E_2 + E_3)
    k_4 = E_0 - (E_1 + E_2 + E_3)

    # Convert back to CPU for further processing
    dir_0_nlag = cp.asnumpy(dir_0_nlag)
    dir_1_nlag = cp.asnumpy(dir_1_nlag)
    dir_2_nlag = cp.asnumpy(dir_2_nlag)
    k_4 = cp.asnumpy(k_4)

    # Group by dir_0_nlag, dir_1_nlag, dir_2_nlag and average k_4
    df_result = pd.DataFrame({
        'dir_0_nlag': dir_0_nlag,
        'dir_1_nlag': dir_1_nlag,
        'dir_2_nlag': dir_2_nlag,
        'k_4': k_4
    })

    final_result = df_result.groupby(['dir_0_nlag', 'dir_1_nlag', 'dir_2_nlag']).mean().reset_index()

    return final_result