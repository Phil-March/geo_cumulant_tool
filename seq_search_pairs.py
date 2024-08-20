
from seq_search_pairs_support import (
    seq_calculate_azimuth_3d,
    seq_calculate_dip_3d,
    distance_along_horizontal_bandwidth,
    distance_along_vertical_bandwidth,
    seq_point_distance_to_shifted_plane
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
                    #print(f"Point_id : {point_id}, Potenital_id : {potential_pair_id}")
                    
                    #Ensure potential pair point is not itself
                    if point_id == potential_pair_id:
                        #print("Was skipped due to same point")
                        continue
                        
                    # Access the azimuth tolerence boundaries
                    cal_azimuth = seq_calculate_azimuth_3d(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    min_azimuth = (azm[dim_id] - azm_tol[dim_id] + 360) % 360
                    max_azimuth = (azm[dim_id] + azm_tol[dim_id] + 360) % 360
                    #print(f"cal_azimuth: {cal_azimuth}, min_azimuth: {min_azimuth}, max_azimuth: {max_azimuth}")

                    if not (min_azimuth <= cal_azimuth <= max_azimuth if min_azimuth < max_azimuth else cal_azimuth >= min_azimuth or cal_azimuth <= max_azimuth):
                        #print("Azimuth tol")
                        continue

                    # Access the dip tolerence boundaries PROBABLY NEED TO ADD 0/90 BOUNDARIES
                    cal_dip = seq_calculate_dip_3d(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_dip > dip[dim_id] + dip_tol[dim_id] or cal_dip < dip[dim_id] - dip_tol[dim_id]:
                        #print("Dip tol")
                        continue
                    
                    #Access the horizontal bandwidth boundaries
                    distance_banwh = distance_along_horizontal_bandwidth(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], azm[dim_id], dip[dim_id])
                    if  abs(distance_banwh) > bandwh[dim_id]:
                        #print("Horizontal past angle")
                        continue
                    
                    #Access the vertical bandwidth boundaries
                    distance_banwv = distance_along_vertical_bandwidth(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], azm[dim_id], dip[dim_id])
                    if abs(distance_banwv) > bandwv[dim_id]:
                        #print("Vertical past angle")
                        continue
                    
                    #Access within the lag tolerance
                    distance_max_lag_tol = seq_point_distance_to_shifted_plane(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], (n * lag[dim_id]) + lag_tol[dim_id], azm[dim_id], dip[dim_id])
                    if distance_max_lag_tol > 0:
                        #print("Max lag tol")
                        continue

                    distance_min_lag_tol = seq_point_distance_to_shifted_plane(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], (n * lag[dim_id]) - lag_tol[dim_id], azm[dim_id], dip[dim_id])
                    if distance_min_lag_tol < 0:
                        #print("Min lag tol")
                        continue
                    
                    #Add point to pairs
                    pairs.append([int(point_id), int(dim_id), int(n), int(potential_pair_id)])

    return pairs