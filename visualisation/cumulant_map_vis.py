import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os

# Get the parent directory of the script
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the output directory
output_dir = os.path.join(parent_dir, '..', 'output')

# List CSV or Excel files in the output directory
file_options = [{'label': f, 'value': f} for f in os.listdir(output_dir) if f.endswith(('.csv', '.xls', '.xlsx'))]

# Function to read the selected file
def read_file(filepath):
    df = pd.read_csv(filepath)
    return df

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='file-dropdown',
        options=file_options,
        placeholder='Select a file',
    ),
    html.Div(id='output-file-selected'),
    dcc.Slider(
        id='n-slider',
        min=1,
        max=16,
        step=1,
        value=1,
        marks={i: str(i) for i in range(1, 17)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Graph(id='heatmap')
])

@app.callback(
    [Output('n-slider', 'min'),
     Output('n-slider', 'max'),
     Output('n-slider', 'value'),
     Output('n-slider', 'marks'),
     Output('n-slider', 'disabled'),
     Output('output-file-selected', 'children')],
    [Input('file-dropdown', 'value')]
)
def update_slider_and_data(selected_file):
    if selected_file is not None:
        filepath = os.path.join(output_dir, selected_file)
        df = read_file(filepath)
        if df is not None:
            num_columns = df.shape[1]
            if num_columns == 3:
                # 2D Case
                n_min = df.iloc[:, 0].min()
                n_max = df.iloc[:, 0].max()
                marks = {i: str(i) for i in range(n_min, n_max + 1)}
                slider_disabled = True
                message = f"2D data detected. Slider fixed at n={n_min}"
            elif num_columns == 4:
                # 3D Case
                n_min = df.iloc[:, 2].min()
                n_max = df.iloc[:, 2].max()
                marks = {i: str(i) for i in range(n_min, n_max + 1)}
                slider_disabled = False
                message = f"3D data detected. Use the slider to change layers."
            else:
                return 1, 16, 1, {i: str(i) for i in range(1, 17)}, True, "Invalid file format."
            return n_min, n_max, n_min, marks, slider_disabled, message
    return 1, 16, 1, {i: str(i) for i in range(1, 17)}, True, "Please select a valid file."

@app.callback(
    Output('heatmap', 'figure'),
    [Input('n-slider', 'value'),
     Input('file-dropdown', 'value')]
)
def update_heatmap(selected_n, selected_file):
    if selected_file is not None:
        filepath = os.path.join(output_dir, selected_file)
        df = read_file(filepath)
        if df is not None:
            num_columns = df.shape[1]
            if num_columns == 3:
                # 2D Case
                df_dim0 = df.iloc[:, 0]
                df_dim1 = df.iloc[:, 1]
                cumulant = df.iloc[:, 2]

                # Create a grid with None for missing values
                grid_data = np.full((df_dim1.max(), df_dim0.max()), None)
                for i in range(len(cumulant)):
                    grid_data[df_dim1.iloc[i] - 1, df_dim0.iloc[i] - 1] = cumulant.iloc[i]

                cumulant_min = np.nanmin(cumulant)
                cumulant_max = np.nanmax(cumulant)

                fig = go.Figure(data=go.Heatmap(
                                   z=grid_data,
                                   x=list(range(1, df_dim0.max() + 1)),
                                   y=list(range(1, df_dim1.max() + 1)),
                                   colorscale='Viridis',
                                   zmin=cumulant_min,
                                   zmax=cumulant_max))

                fig.update_layout(
                    title='Grid Mesh of Cumulant Values (2D)',
                    xaxis_title='N for Dir 0',
                    yaxis_title='N for Dir 1',
                    autosize=False,
                    width=550,
                    height=550
                )
                return fig

            elif num_columns == 4:
                # 3D Case
                df_dim0 = df[df.iloc[:, 2] == selected_n].iloc[:, 0]
                df_dim1 = df[df.iloc[:, 2] == selected_n].iloc[:, 1]
                cumulant = df[df.iloc[:, 2] == selected_n].iloc[:, 3]

                # Create a grid with None for missing values
                grid_data = np.full((df_dim0.max(), df_dim1.max()), None)
                for i in range(len(cumulant)):
                    grid_data[df_dim0.iloc[i] - 1, df_dim1.iloc[i] - 1] = cumulant.iloc[i]

                cumulant_min = np.nanmin(cumulant)
                cumulant_max = np.nanmax(cumulant)

                fig = go.Figure(data=go.Heatmap(
                                   z=grid_data,
                                   x=list(range(1, df_dim0.max() + 1)),
                                   y=list(range(1, df_dim1.max() + 1)),
                                   colorscale='Viridis',
                                   zmin=cumulant_min,
                                   zmax=cumulant_max))

                fig.update_layout(
                    title=f'Grid Mesh of Cumulant Values (n={selected_n} for 3D)',
                    xaxis_title='N for Dir 0',
                    yaxis_title='N for Dir 1',
                    autosize=False,
                    width=550,
                    height=550
                )
                return fig
    return go.Figure()


if __name__ == '__main__':
    app.run_server(debug=True)
