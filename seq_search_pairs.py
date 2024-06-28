
from seq_search_pairs_support import (
    seq_calculate_azimuth_3d,
    seq_calculate_dip_3d,
    seq_horizontal_length_difference,
    seq_vertical_length_associated_with_dip,
    seq_point_position_before_or_equal_plane,
    seq_point_position_after_or_equal_plane
)

def seq_search_pairs_gen(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv):
    pairs = []

    # for each points
    for p in data_vector:
        point_id = p[0]
        # for each dimension (direction)
        for dim_id in dim:
            # for each potential pair
            for n in range(1, nlag[dim_id] + 1):
                for potential_pair in data_vector:
                    # Select point_id
                    potential_pair_id = potential_pair[0]
                    
                    #Ensure potential pair point is not itself
                    if point_id == potential_pair_id:
                        continue

                    # Access the azimuth tolerence boundaries
                    cal_azimuth = seq_calculate_azimuth_3d(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_azimuth > azm[dim_id] + azm_tol[dim_id] or cal_azimuth < azm[dim_id] - azm_tol[dim_id]:
                        continue

                    # Access the dip tolerence boundaries
                    cal_dip = seq_calculate_dip_3d(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_dip > dip[dim_id] + dip_tol[dim_id] or cal_dip < dip[dim_id] - dip_tol[dim_id]:
                        continue
                    
                    #Access the horizontal bandwidth boundaries
                    cal_hor_length_diff = seq_horizontal_length_difference(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_hor_length_diff > bandwh[dim_id]:
                        continue
                    
                    #Access the vertical bandwidth boundaries
                    cal_ver_diff = seq_vertical_length_associated_with_dip(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_ver_diff > bandwv[dim_id]:
                        continue
                    
                    #Access within the lag tolerance
                    cal_within_max_lag_tol = seq_point_position_before_or_equal_plane(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], (n * lag[dim_id]) + lag_tol[dim_id], cal_azimuth, cal_dip)
                    if not cal_within_max_lag_tol:
                        continue

                    cal_within_min_lag_tol = seq_point_position_after_or_equal_plane(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], (n * lag[dim_id]) - lag_tol[dim_id], cal_azimuth, cal_dip)
                    if not cal_within_min_lag_tol:
                        continue
                    
                    #Add point to pairs
                    pairs.append([int(point_id), int(dim_id), int(n), int(potential_pair_id)])

    return pairs