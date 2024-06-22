import json
import numpy as np

file_name = 'parameters.json'
ndir = 2
is_regular = 1

def load_parameters(file_name, ndir):
    # Load JSON file
    with open(file_name, 'r') as file:
        data = json.load(file)
    
    # Extract parameters from JSON based on ndir
    params = data[f'ndir{ndir}']

    # Initialize parameters
    nlag = [params.get(f'nlag{i+1}', 0) for i in range(ndir)]
    lag = [params.get(f'lag{i+1}', 0.0) for i in range(ndir)]
    ltol = [params.get(f'lagtol{i+1}', 0.0) for i in range(ndir)]
    azm = [params.get(f'az{i+1}', 0.0) for i in range(ndir)]
    atol = [params.get(f'aztol{i+1}', 0.0) for i in range(ndir)]
    bandwh = [params.get(f'bandh{i+1}', 0.0) for i in range(ndir)]
    dip = [params.get(f'dip{i+1}', 0.0) for i in range(ndir)]
    dtol = [params.get(f'dtol{i+1}', 0.0) for i in range(ndir)]
    bandwd = [params.get(f'bandv{i+1}', 0.0) for i in range(ndir)]

    return nlag, lag, ltol, azm, atol, bandwh, dip, dtol, bandwd

# Load parameters
nlag, lag, ltol, azm, atol, bandwh, dip, dtol, bandwd = load_parameters(file_name, ndir)

# Initialize grid parameters
nx_ = 1
ny_ = 1
nz_ = 1
xsize_ = 1.0
ysize_ = 1.0
zsize_ = 1.0
Ox_ = 0.0
Oy_ = 0.0
Oz_ = 0.0

# Set grid parameters based on ndir
if ndir >= 1:
    nx_ = nlag[0]
    xsize_ = lag[0]

if ndir >= 2:
    ny_ = nlag[1]
    ysize_ = lag[1]

if ndir == 3:
    nz_ = nlag[2]
    zsize_ = lag[2]

# Print the results
print(f"Lag separation in each direction: {lag[:ndir]}")
print(f"Lag number in each direction: {nlag[:ndir]}")
print("Template Shape:")
for i in range(ndir):
    print(f"Direction {i+1}: azm={azm[i]}, atol={atol[i]}, bandwh={bandwh[i]}, dip={dip[i]}, dtol={dtol[i]}, bandwd={bandwd[i]}")

print(f"Grid parameters: nx_={nx_}, ny_={ny_}, nz_={nz_}, xsize_={xsize_}, ysize_={ysize_}, zsize_={zsize_}")

def get_cum_uni(data_grid, dirs, N):
    pass  # Implement the function logic here
