import pandas as pd
import itertools


def associate_grade(data_file, pairs_file):
    # Load initial data
    df_data = pd.read_csv(data_file)

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

df_pairs = associate_grade('2d_data.csv', 'seq_pairs.json')

def compute_moments_from_pairs(df_pairs, ndir, nlag_dir):
    # Generate column names
    columns = [f"dir_{i}_nlag" for i in range(len(ndir))]

    # Generate all combinations of values for each column
    combinations = list(itertools.product(*[range(1, nlag + 1) for nlag in nlag_dir]))

    # Create DataFrame
    df_generated = pd.DataFrame(combinations, columns=columns)

    # Perform merges
    result = df_generated.copy()
    for i in range(len(ndir)):
        suffix = f'_dir_{i}' if i > 0 else ''
        result = result.merge(
            df_pairs[df_pairs['dim_id'] == ndir[i]], 
            left_on=f'dir_{i}_nlag', 
            right_on='n', 
            how='left',
            suffixes=('', suffix)
        )
        
        # Rename columns for each merge
        result = result.rename(columns={
            'point_id': f'point_id_dir_{i}',
            'paired_point_id': f'paired_point_id_dir_{i}',
            'paired_point_id_value': f'paired_point_id_value_dir_{i}'
        })

    # Filter rows where all existing point_id_dir_X match
    point_id_columns = [col for col in result.columns if col.startswith('point_id_dir_')]
    if point_id_columns:
        result = result.dropna(subset=point_id_columns)
        result = result[result[point_id_columns].nunique(axis=1) == 1]

    # Select final columns
    final_columns = columns + ['point_id_value'] + [col for col in result.columns if col.startswith('paired_point_id_dir_') or col.startswith('paired_point_id_value_dir_')]
    result = result[final_columns]

    # Add new column 'E'
    result['E'] = result['point_id_value']
    for col in result.columns:
        if col.startswith('paired_point_id_value_dir_'):
            result['E'] *= result[col]

    # Group by all dir_X_nlag columns and average E
    group_columns = [f'dir_{i}_nlag' for i in range(len(ndir))]
    final_result = result.groupby(group_columns)['E'].mean().reset_index()
    final_result = final_result.rename(columns={'E': 'average_E'})

    return final_result

# Example usage:
ndir = [0, 1]  # You can add more dimensions here
nlag_dir = [10, 10]  # Adjust the number of lags for each dimension

final_result = compute_moments_from_pairs(df_pairs, ndir, nlag_dir)
print(final_result)