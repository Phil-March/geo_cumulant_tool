# Geostatistical Cumulant Tool

Welcome to the **Geostatistical Cumulant Tool** repository! This tool is designed to compute third and fourth-order cumulants, enabling advanced geostatistical analysis in 2 and 3 directions. The tool includes a search pairing algorithm for cumulant computation and provides visualization for third and fourth-order cumulant maps.

## Features
- **Pairing Algorithm**: Efficient search pairing algorithm for accurate cumulant computation.
- **Third-Order Cumulants**: Compute third-order cumulants in two directions.
- **Fourth-Order Cumulants**: Compute fourth-order cumulants in three directions.
- **Parallel and Sequential Execution**: Both the search pairing algorithm and cumulant computation can be executed in parallel or sequentially, providing flexibility based on your computational needs.
- **Cumulant Maps Visualization**: Visualize third and fourth-order cumulant maps for insightful geostatistical analysis.

## Theory

This section provides an overview of the theoretical concepts underlying the Geostatistical Computation Cumulant Tool, specifically focusing on the Search Pair Algorithm and Cumulant Computation.

### Search Pair Algorithm

The **Search Pair Algorithm** is designed to identify and pair data points based on spatial relationships. The algorithm allows the user to input the following parameters:

- **Number of Lags**
- **Lag Distance**
- **Lag Tolerance**
- **Azimuth**
- **Azimuth Tolerance**
- **Bandwidth Horizontal**
- **Dip**
- **Dip Tolerance**
- **Bandwidth Vertical**

These parameters enable the precise control of the spatial relationships in the analysis.

Below is a visual representation of the parameters and their spatial relationships:

<p align="center">
  <img src="https://github.com/user-attachments/assets/a09ae3d3-e393-48a8-8c77-e0ec4ca5d4c3" alt="Search Pair Algorithm Parameters">
</p>

<p align="center"><strong>Figure 1</strong>: Spatial relationships and input parameters used in the Search Pair Algorithm.</p>


### Cumulant Computation

## Requirements

### Software Installation

Before running the Geostatistical Computation Cumulant Tool, ensure that the following software is installed on your system:

1. **Python 3.12.2**: This tool requires Python version 3.12.2 or higher. You can download and install it from the [official Python website](https://www.python.org/downloads/).
2. **CUDA Toolkit**: For leveraging GPU acceleration (if using parallel execution), ensure that the CUDA Toolkit is installed. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).

### Python Libraries

In addition to the software requirements, the following Python libraries need to be installed. These libraries are not included by default with Python and must be installed via `pip`. You can install all required libraries using the command:

```bash
pip install pandas numpy cupy numba
