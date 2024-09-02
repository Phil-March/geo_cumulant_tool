import cudf
import itertools

def center_grades(data_file):
    # Load the data
    df_data = cudf.read_csv(data_file)
    
    # Compute the mean of the GRADE column
    mean_grade = df_data['GRADE'].mean()

    # Subtract the mean grade from each entry in the GRADE column
    df_data['GRADE_centered'] = df_data['GRADE'] - mean_grade
    
    return df_data


def associate_grade(df_data, pairs_file):
    # Add the point_id column as the first column
    df_data.insert(0, 'point_id', range(1, len(df_data) + 1))

    # Load the JSON pairs data into a DataFrame
    df_pairs = cudf.read_json(pairs_file)

    # Merge df_pairs with df_data to get the values for point_id
    df_pairs = df_pairs.merge(df_data[['point_id', 'GRADE']], on='point_id', how='left')
    df_pairs = df_pairs.rename(columns={'GRADE': 'point_id_value'})

    # Merge df_pairs with df_data to get the values for paired_point_id
    df_pairs = df_pairs.merge(df_data[['point_id', 'GRADE']], left_on='paired_point_id', right_on='point_id', how='left', suffixes=('', '_paired_id'))
    df_pairs = df_pairs.rename(columns={'GRADE': 'paired_point_id_value'})
    # Drop the extra point_id column from the second merge
    df_pairs = df_pairs.drop(columns=['point_id_paired_id'])

    return df_pairs


def compute_3rd_order_cumulant(df_pairs):
    # Determine the maximum values of n for each dim_id (0 and 1)
    max_n_dim_0 = df_pairs[df_pairs['dim_id'] == 0]['n'].max()
    max_n_dim_1 = df_pairs[df_pairs['dim_id'] == 1]['n'].max()
    
    # Use the maximum n values to define the nlag_dir
    nlag_dir = [max_n_dim_0, max_n_dim_1]

    # Generate column names for the two directions
    columns = ['dir_0_nlag', 'dir_1_nlag']

    # Generate all combinations of values for each column
    combinations = list(itertools.product(range(1, nlag_dir[0] + 1), range(1, nlag_dir[1] + 1)))

    # Create DataFrame with the generated combinations
    df_generated = cudf.DataFrame(combinations, columns=columns)

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
    final_result = final_result.rename(columns={'E': 'k_3'})

    return final_result


def compute_4th_order_cumulant(df_pairs):
    # Determine the maximum values of n for each dim_id
    max_n_dim_0 = df_pairs[df_pairs['dim_id'] == 0]['n'].max()
    max_n_dim_1 = df_pairs[df_pairs['dim_id'] == 1]['n'].max()
    max_n_dim_2 = df_pairs[df_pairs['dim_id'] == 2]['n'].max()
    
    # Use the maximum n values to define the nlag_dir
    nlag_dir = [max_n_dim_0, max_n_dim_1, max_n_dim_2]

    # Generate column names for the three directions
    columns = ['dir_0_nlag', 'dir_1_nlag', 'dir_2_nlag']

    # Generate all combinations of values for each column
    combinations = list(itertools.product(range(1, nlag_dir[0] + 1), range(1, nlag_dir[1] + 1), range(1, nlag_dir[2] + 1)))

    # Create DataFrame with the generated combinations
    df_generated = cudf.DataFrame(combinations, columns=columns)

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

    # Merge for direction 2
    result = result.merge(
        df_pairs[df_pairs['dim_id'] == 2], 
        left_on='dir_2_nlag', 
        right_on='n', 
        how='left',
        suffixes=('', '_dir_2')
    )
    result = result.rename(columns={
        'point_id': 'point_id_dir_2',
        'paired_point_id': 'paired_point_id_dir_2',
        'paired_point_id_value': 'paired_point_id_value_dir_2'
    })

    # Filter rows where all point_id_dir_0, point_id_dir_1, and point_id_dir_2 match
    result = result.dropna(subset=['point_id_dir_0', 'point_id_dir_1', 'point_id_dir_2'])
    result = result[(result['point_id_dir_0'] == result['point_id_dir_1']) & 
                    (result['point_id_dir_1'] == result['point_id_dir_2'])]

    # Select final columns
    final_columns = columns + ['point_id_value'] + [
        'paired_point_id_dir_0', 'paired_point_id_value_dir_0',
        'paired_point_id_dir_1', 'paired_point_id_value_dir_1',
        'paired_point_id_dir_2', 'paired_point_id_value_dir_2'
    ]
    result = result[final_columns]

    # Compute mu_4 = E[Z(u)⋅Z(u+h1)⋅Z(u+h2)⋅Z(u+h3)]
    result['E_0'] = (result['point_id_value'] * 
                     result['paired_point_id_value_dir_0'] * 
                     result['paired_point_id_value_dir_1'] * 
                     result['paired_point_id_value_dir_2'])

    # Compute E[Z(u),Z(u + h1)] ⋅ E[Z(u + h2),Z(u + h3)]
    result['E_1'] = (result['point_id_value'] * result['paired_point_id_value_dir_0']).mean() * \
                    (result['paired_point_id_value_dir_1'] * result['paired_point_id_value_dir_2']).mean()

    # Compute E[Z(u),Z(u + h2)] ⋅ E[Z(u + h1),Z(u + h3)]
    result['E_2'] = (result['point_id_value'] * result['paired_point_id_value_dir_1']).mean() * \
                    (result['paired_point_id_value_dir_0'] * result['paired_point_id_value_dir_2']).mean()

    # Compute E[Z(u),Z(u + h3)] ⋅ E[Z(u+ h1),Z(u + h2)]
    result['E_3'] = (result['point_id_value'] * result['paired_point_id_value_dir_2']).mean() * \
                    (result['paired_point_id_value_dir_0'] * result['paired_point_id_value_dir_1']).mean()

    # Compute the fourth-order cumulant: 
    # mu_4 - (E_1 + E_2 + E_3)
    result['k_4'] = result['E_0'] - (result['E_1'] + result['E_2'] + result['E_3'])

    # Group by dir_0_nlag, dir_1_nlag, and dir_2_nlag columns and average the cumulant
    final_result = result.groupby(columns)['k_4'].mean().reset_index()

    return final_result
