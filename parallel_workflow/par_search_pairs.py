import cupy as cp
import numpy as np
from numba import cuda
from parallel_workflow.par_search_pairs_support import (    
    par_calculate_azimuth_3d,
    par_calculate_dip_3d,
    par_horizontal_length_difference,
    par_vertical_length_associated_with_dip,
    par_point_position_before_or_equal_plane,
    par_point_position_after_or_equal_plane)


# Constants
MAX_PAIRS = 10  # Maximum number of pairs to store for each (point_id, dim_id, n) combination

# Main parallelized function
@cuda.jit
def par_search_pairs_gen_kernel(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv, pairs, pair_counts):
    idx = cuda.grid(1)
    if idx < data_vector.shape[0]:
        p = data_vector[idx]
        point_id = p[0] + 1  # Adjust point_id to start from 1
        for dim_id in range(dim.size):
            for n in range(1, nlag[dim_id] + 1):
                for j in range(data_vector.shape[0]):
                    potential_pair = data_vector[j]
                    potential_pair_id = potential_pair[0]
                    if point_id == potential_pair_id:
                        continue
                    cal_azimuth = par_calculate_azimuth_3d(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_azimuth > azm[dim_id] + azm_tol[dim_id] or cal_azimuth < azm[dim_id] - azm_tol[dim_id]:
                        continue
                    cal_dip = par_calculate_dip_3d(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_dip > dip[dim_id] + dip_tol[dim_id] or cal_dip < dip[dim_id] - dip_tol[dim_id]:
                        continue
                    cal_hor_length_diff = par_horizontal_length_difference(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_hor_length_diff > bandwh[dim_id]:
                        continue
                    cal_ver_diff = par_vertical_length_associated_with_dip(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_ver_diff > bandwv[dim_id]:
                        continue
                    cal_within_max_lag_tol = par_point_position_before_or_equal_plane(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], (n * lag[dim_id]) + lag_tol[dim_id], cal_azimuth, cal_dip)
                    if not cal_within_max_lag_tol:
                        continue
                    cal_within_min_lag_tol = par_point_position_after_or_equal_plane(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], (n * lag[dim_id]) - lag_tol[dim_id], cal_azimuth, cal_dip)
                    if not cal_within_min_lag_tol:
                        continue
                    count = pair_counts[idx, dim_id, n]
                    if count < MAX_PAIRS:
                        pairs[idx, dim_id, n, count] = potential_pair_id
                        pair_counts[idx, dim_id, n] += 1

# Wrapper function to launch the kernel
def par_search_pairs_gen(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv):
    data_vector = cuda.to_device(data_vector)  # Ensure data is on the GPU
    dim = np.array(dim, dtype=np.int32)
    nlag = np.array(nlag, dtype=np.int32)
    lag = np.array(lag, dtype=np.float32)
    lag_tol = np.array(lag_tol, dtype=np.float32)
    azm = np.array(azm, dtype=np.float32)
    azm_tol = np.array(azm_tol, dtype=np.float32)
    bandwh = np.array(bandwh, dtype=np.float32)
    dip = np.array(dip, dtype=np.float32)
    dip_tol = np.array(dip_tol, dtype=np.float32)
    bandwv = np.array(bandwv, dtype=np.float32)
    
    dim = cuda.to_device(dim)
    nlag = cuda.to_device(nlag)
    lag = cuda.to_device(lag)
    lag_tol = cuda.to_device(lag_tol)
    azm = cuda.to_device(azm)
    azm_tol = cuda.to_device(azm_tol)
    bandwh = cuda.to_device(bandwh)
    dip = cuda.to_device(dip)
    dip_tol = cuda.to_device(dip_tol)
    bandwv = cuda.to_device(bandwv)

    pairs = cuda.device_array((data_vector.shape[0], len(dim), max(nlag) + 1, MAX_PAIRS), dtype=np.int32)
    pair_counts = cuda.device_array((data_vector.shape[0], len(dim), max(nlag) + 1), dtype=np.int32)
    threadsperblock = 256
    blockspergrid = (data_vector.shape[0] + (threadsperblock - 1)) // threadsperblock
    par_search_pairs_gen_kernel[blockspergrid, threadsperblock](data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv, pairs, pair_counts)
    pairs_host = pairs.copy_to_host()
    pair_counts_host = pair_counts.copy_to_host()

    # Extract non-zero pairs and their indices using NumPy
    non_zero_pairs = []
    indices = np.nonzero(pair_counts_host)
    for i, j, k in zip(*indices):
        count = pair_counts_host[i, j, k]
        for c in range(count):
            non_zero_pairs.append((i + 1, j, k, pairs_host[i, j, k, c]))
    
    return non_zero_pairs