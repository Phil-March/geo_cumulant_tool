import cupy as cp
import numpy as np
from numba import cuda
from par_search_pairs_support import (    
    par_calculate_azimuth_3d,
    par_calculate_dip_3d,
    par_distance_along_horizontal_bandwidth,
    par_distance_along_vertical_bandwidth,
    par_point_distance_to_shifted_plane)

# Constants
MAX_PAIRS = 100  # Maximum number of pairs to store for each (point_id, dim_id, n) combination

# Main parallelized function
@cuda.jit
def par_search_pairs_gen_kernel(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv, pairs, pair_counts):
    idx = cuda.grid(1)
    if idx < data_vector.shape[0]:
        p = data_vector[idx]
        point_id = p[0]
        for dim_id in range(dim.size):
            for n in range(1, nlag[dim_id] + 1):
                for j in range(data_vector.shape[0]):
                    potential_pair = data_vector[j]
                    potential_pair_id = potential_pair[0]
                    if point_id == potential_pair_id:
                        continue

                    # Calculate azimuth and check tolerance
                    cal_azimuth = par_calculate_azimuth_3d(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    min_azimuth = (azm[dim_id] - azm_tol[dim_id] + 360) % 360
                    max_azimuth = (azm[dim_id] + azm_tol[dim_id] + 360) % 360
                    if not (min_azimuth <= cal_azimuth <= max_azimuth if min_azimuth < max_azimuth else cal_azimuth >= min_azimuth or cal_azimuth <= max_azimuth):
                        continue

                    # Calculate dip and check tolerance
                    cal_dip = par_calculate_dip_3d(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3])
                    if cal_dip > dip[dim_id] + dip_tol[dim_id] or cal_dip < dip[dim_id] - dip_tol[dim_id]:
                        continue

                    # Calculate horizontal bandwidth and check
                    cal_hor_length_diff = par_distance_along_horizontal_bandwidth(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], azm[dim_id], dip[dim_id])
                    if abs(cal_hor_length_diff) > bandwh[dim_id]:
                        continue

                    # Calculate vertical bandwidth and check
                    cal_ver_diff = par_distance_along_vertical_bandwidth(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], azm[dim_id], dip[dim_id])
                    if abs(cal_ver_diff) > bandwv[dim_id]:
                        continue

                    # Check within maximum lag tolerance
                    distance_max_lag_tol = par_point_distance_to_shifted_plane(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], (n * lag[dim_id]) + lag_tol[dim_id], azm[dim_id], dip[dim_id])
                    if distance_max_lag_tol > 0:
                        continue

                    # Check within minimum lag tolerance
                    distance_min_lag_tol = par_point_distance_to_shifted_plane(p[1], p[2], p[3], potential_pair[1], potential_pair[2], potential_pair[3], (n * lag[dim_id]) - lag_tol[dim_id], azm[dim_id], dip[dim_id])
                    if distance_min_lag_tol < 0:
                        continue

                    # Add pair to the pairs array if within all tolerances
                    count = pair_counts[idx, dim_id, n]
                    if count < MAX_PAIRS:
                        pairs[idx, dim_id, n, count] = potential_pair_id
                        pair_counts[idx, dim_id, n] += 1

# Wrapper function to launch the kernel with chunking
def par_search_pairs_gen(data_vector, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv, num_chunks):
    dim = np.array(dim, dtype=np.int32)
    nlag = np.array(nlag, dtype=np.int32)
    lag = np.array(lag, dtype=np.float64)
    lag_tol = np.array(lag_tol, dtype=np.float64)
    azm = np.array(azm, dtype=np.float64)
    azm_tol = np.array(azm_tol, dtype=np.float64)
    bandwh = np.array(bandwh, dtype=np.float64)
    dip = np.array(dip, dtype=np.float64)
    dip_tol = np.array(dip_tol, dtype=np.float64)
    bandwv = np.array(bandwv, dtype=np.float64)

    # Transfer the arrays to GPU memory only once
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

    # Split data into chunks
    chunk_size = data_vector.shape[0] // num_chunks
    pairs_host_total = []
    pair_counts_host_total = []

    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = (chunk + 1) * chunk_size if chunk != num_chunks - 1 else data_vector.shape[0]

        data_chunk = data_vector[start_idx:end_idx, :]
        
        # Ensure data_chunk is contiguous in memory
        data_chunk = np.ascontiguousarray(data_chunk)
        data_chunk = cuda.to_device(data_chunk)

        pairs = cuda.device_array((data_chunk.shape[0], len(dim), max(nlag) + 1, MAX_PAIRS), dtype=np.int32)
        pair_counts = cuda.device_array((data_chunk.shape[0], len(dim), max(nlag) + 1), dtype=np.int32)
        
        threadsperblock = 128
        blockspergrid = (data_chunk.shape[0] + (threadsperblock - 1)) // threadsperblock
        par_search_pairs_gen_kernel[blockspergrid, threadsperblock](data_chunk, dim, nlag, lag, lag_tol, azm, azm_tol, bandwh, dip, dip_tol, bandwv, pairs, pair_counts)
        
        # Copy back to host as CuPy arrays instead of NumPy arrays
        pairs_host = cp.asarray(pairs.copy_to_host())
        pair_counts_host = cp.asarray(pair_counts.copy_to_host())

        pairs_host_total.append(pairs_host)
        pair_counts_host_total.append(pair_counts_host)

    print("trying to concat")
    # Concatenate the lists to create final results as contiguous arrays using CuPy
    pairs_host_total = cp.concatenate(pairs_host_total, axis=0)
    pair_counts_host_total = cp.concatenate(pair_counts_host_total, axis=0)
    print("done concat")
    
    # Extract non-zero pairs and their indices using CuPy
    non_zero_pairs = []
    indices = cp.nonzero(pair_counts_host_total)
    
    for i, j, k in zip(*indices):
        count = pair_counts_host_total[i, j, k].item()
        for c in range(count):
            non_zero_pairs.append((i + 1, j, k, pairs_host_total[i, j, k, c]))
    
    # Convert non-zero pairs back to NumPy if necessary for further processing
    non_zero_pairs = cp.asnumpy(non_zero_pairs)
    
    return non_zero_pairs