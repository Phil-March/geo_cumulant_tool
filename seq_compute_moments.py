import pandas as pd
import itertools

def center_grades(data_file):
    # Load the data
    df_data = pd.read_csv(data_file)
    
    # Compute the mean of the GRADE column
    mean_grade = df_data['GRADE'].mean()

    # Subtract the mean grade from each entry in the GRADE column
    df_data['GRADE_centered'] = df_data['GRADE'] - mean_grade
    
    return df_data

data_2d = center_grades('2d_data.csv')

def associate_grade(df_data, pairs_file):

    # Add the point_id column as the first column
    df_data.insert(0, 'point_id', range(1, len(df_data) + 1))

    # Load the JSON pairs data into a DataFrame
    df_pairs = pd.read_json(pairs_file)

    # Merge df_pairs with df_data to get the values for point_id
    df_pairs = df_pairs.merge(df_data[['point_id', 'GRADE']], on='point_id', how='left')
    df_pairs = df_pairs.rename(columns={'GRADE': 'point_id_value'})

    # Merge df_pairs with df_data to get the values for paired_point_id
    df_pairs = df_pairs.merge(df_data[['point_id', 'GRADE']], left_on='paired_point_id', right_on='point_id', how='left', suffixes=('', '_paired_id'))
    df_pairs = df_pairs.rename(columns={'GRADE': 'paired_point_id_value'})
    # Drop the extra point_id column from the second merge
    df_pairs = df_pairs.drop(columns=['point_id_paired_id'])

    return df_pairs

df_pairs = associate_grade(data_2d, 'seq_pairs.json')

def compute_3rd_order_cumulant(df_pairs, nlag_dir):
    # Generate column names for the two directions
    columns = ['dir_0_nlag', 'dir_1_nlag']

    # Generate all combinations of values for each column
    combinations = list(itertools.product(range(1, nlag_dir[0] + 1), range(1, nlag_dir[1] + 1)))

    # Create DataFrame with the generated combinations
    df_generated = pd.DataFrame(combinations, columns=columns)

    # Merge for direction 0
    result = df_generated.merge(
        df_pairs[df_pairs['dim_id'] == 0], 
        left_on='dir_0_nlag', 
        right_on='n', 
        how='left'
    )
    result = result.rename(columns={
        'point_id': 'point_id_dir_0',
        'paired_point_id': 'paired_point_id_dir_0',
        'paired_point_id_value': 'paired_point_id_value_dir_0'
    })

    # Merge for direction 1
    result = result.merge(
        df_pairs[df_pairs['dim_id'] == 1], 
        left_on='dir_1_nlag', 
        right_on='n', 
        how='left',
        suffixes=('', '_dir_1')
    )
    result = result.rename(columns={
        'point_id': 'point_id_dir_1',
        'paired_point_id': 'paired_point_id_dir_1',
        'paired_point_id_value': 'paired_point_id_value_dir_1'
    })

    # Filter rows where all point_id_dir_0 and point_id_dir_1 match
    result = result.dropna(subset=['point_id_dir_0', 'point_id_dir_1'])
    result = result[result['point_id_dir_0'] == result['point_id_dir_1']]

    # Select final columns
    final_columns = columns + ['point_id_value'] + [
        'paired_point_id_dir_0', 'paired_point_id_value_dir_0',
        'paired_point_id_dir_1', 'paired_point_id_value_dir_1'
    ]
    result = result[final_columns]

    # Add new column 'E' as the product of point_id_value and the paired_point_id_values
    result['E'] = result['point_id_value'] * result['paired_point_id_value_dir_0'] * result['paired_point_id_value_dir_1']

    # Group by dir_0_nlag and dir_1_nlag columns and average E
    final_result = result.groupby(columns)['E'].mean().reset_index()
    final_result = final_result.rename(columns={'E': 'average_E'})

    return final_result


nlag_dir = [10, 10]  # Adjust the number of lags for each dimension

final_result = compute_3rd_order_cumulant(df_pairs, nlag_dir)
print(final_result)