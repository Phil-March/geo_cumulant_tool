# Geostatistical Cumulant Tool

Welcome to the **Geostatistical Cumulant Tool** repository! This tool is designed to compute third and fourth-order cumulants, enabling advanced geostatistical analysis in 2 and 3 directions. The tool includes a search pairing algorithm for cumulant computation and provides visualization for third and fourth-order cumulant maps.

## Features
- **Pairing Algorithm**: Efficient search pairing algorithm for accurate cumulant computation.
- **Third-Order Cumulants**: Compute third-order cumulants in two directions.
- **Fourth-Order Cumulants**: Compute fourth-order cumulants in three directions.
- **Parallel and Sequential Execution**: Both the search pairing algorithm and cumulant computation can be executed in parallel or sequentially, providing flexibility based on your computational needs.
- **Cumulant Maps Visualization**: Visualize third and fourth-order cumulant maps for insightful geostatistical analysis.

# Theory

This section provides an overview of the theoretical concepts underlying the Geostatistical Computation Cumulant Tool, specifically focusing on the Search Pair Algorithm and Cumulant Computation.

## Search Pair Algorithm

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


## Cumulant Computation

In this section, we discuss the computation of cumulants for the data. The data is first centered, after which the cumulants are calculated.

### General Cumulant Equation

The general formula for computing the cumulants is given by:

$$
\kappa(X_1, \dots, X_n) = \sum\limits_{\pi} (|\pi| - 1)! \cdot (-1)^{|\pi|-1} \prod\limits_{B \in \pi} \mathbb{E} \left( \prod\limits_{i \in B} X_i \right)
$$

The 3rd and 4th order cumulant equations are derived from the general cumulant equation.

### 3rd Order Cumulant

The 3rd order cumulant equation for a centered dataset is:

$$
\kappa_3 = E[Z(u) \cdot Z(u + h_1) \cdot Z(u + h_2)]
$$

### 4th Order Cumulant

The 4th order cumulant equation for a centered dataset is:

$$
\kappa_4 = E[Z(u) \cdot Z(u + h_1) \cdot Z(u + h_2) \cdot Z(u + h_3)] - E[Z(u), Z(u + h_1)] \cdot E[Z(u + h_2), Z(u + h_3)] - E[Z(u), Z(u + h_2)] \cdot E[Z(u + h_1), Z(u + h_3)] - E[Z(u + h_1), Z(u + h_2)] \cdot E[Z(u), Z(u + h_3)]
$$


# Requirements

### Software Installation

Before running the Geostatistical Computation Cumulant Tool, ensure that the following software is installed on your system:

1. **Python 3.12.2**: This tool requires Python version 3.12.2 or higher. You can download and install it from the [Python website](https://www.python.org/downloads/).
2. **CUDA Toolkit**: For leveraging GPU acceleration (if using parallel execution), ensure that the CUDA Toolkit is installed. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).

### Python Libraries

In addition to the software requirements, the following Python libraries need to be installed. These libraries are not included by default with Python and must be installed via `pip`. You can install all required libraries using the command:

```bash
pip install pandas numpy cupy numba
```

# How to Use

## Search Pairs

1. **Set Parameters:**
   - Fill in the parameters for the search pair algoritm in the `search_parameters.json` file.
     
2. **Prepare Data:**
   - Upload the CSV data file into the `input` folder. The file should follow the format: `X`, `Y`, `Z`, `GRADE` columns.

3. **Run the Script:**
   - Navigate to either the `sequential` or `parallel` workflow folder.
   - Execute the `...run.py` file using Python.

4. **Compute Pairs:**
   - When prompted, select the option to compute pairs.

5. **Choose Directions:**
   - When prompted, choose the number of directions to be used (either 2 or 3).

6. **Select Data File:**
   - When prompted, select the data file you wish to use.

7. **Save Output:**
   - The output file will be saved as a JSON file in the `output` folder.

## Compute Cumulants

1. **Run the Script:**
   - Navigate to either the `sequential` or `parallel` workflow folder.
   - Execute the `...run.py` file using Python.

2. **Select Input Data:**
   - When prompted, select the original input data file.

3. **Select Pairs File:**
   - When prompted, select the associated generated pairs file.

4. **Save Output:**
   - The output file will be saved as a CSV file in the `output` folder.

## Visualize Cumulant Map

1. **Run Visualization Script:**
   - Navigate to the `visualization` folder
   - Execute the `cumulant_map_vis.py` file.

2. **Access Visualization:**
   - Follow the displayed HTTPS link or copy and paste it into a web browser.

3. **Select File:**
   - Choose the file you want to visualize in the web app.
