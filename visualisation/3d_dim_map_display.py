import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Generate manual data
data = []

# Iterate over dim_id from 0 to 2 and n from 1 to 16
for dim_id in range(0, 3):
    for n in range(1, 17):
        cumulant_value = np.random.random()  # Random cumulant value
        data.append([dim_id, n, cumulant_value])

# Create DataFrame
df_manual = pd.DataFrame(data, columns=['dim_id', 'n', 'cumulant'])

# Filter data for dim_id
df_dim0 = df_manual[df_manual['dim_id'] == 0]
df_dim1 = df_manual[df_manual['dim_id'] == 1]
df_dim2 = df_manual[df_manual['dim_id'] == 2]

# Determine the min and max cumulant values for the fixed color scale
cumulant_min = df_dim2['cumulant'].min()
cumulant_max = df_dim2['cumulant'].max()

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Slider(
        id='n-slider',
        min=df_dim2['n'].min().item(),
        max=df_dim2['n'].max().item(),
        step=1,
        value=df_dim2['n'].min().item(),
        marks={i: str(i) for i in range(1, 17)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Graph(id='heatmap')
])

@app.callback(
    Output('heatmap', 'figure'),
    Input('n-slider', 'value')
)
def update_heatmap(selected_n):
    # Filter data for dim_id 0 and 1 for the selected n value in dim_id 2
    df_filtered_dim0 = df_manual[df_manual['dim_id'] == 0]
    df_filtered_dim1 = df_manual[df_manual['dim_id'] == 1]
    df_filtered_dim2 = df_manual[(df_manual['dim_id'] == 2) & (df_manual['n'] == selected_n)]
    
    # Create a pivot table to arrange the data into a grid
    grid_data = np.zeros((df_dim0['n'].max(), df_dim1['n'].max()))
    
    for n in range(df_dim0['n'].min(), df_dim0['n'].max() + 1):
        for m in range(df_dim1['n'].min(), df_dim1['n'].max() + 1):
            cumulant_value_n = df_filtered_dim0[df_filtered_dim0['n'] == n]['cumulant'].values[0]
            cumulant_value_m = df_filtered_dim1[df_filtered_dim1['n'] == m]['cumulant'].values[0]
            cumulant_value_l = df_filtered_dim2['cumulant'].values[0]
            grid_data[n-1, m-1] = (cumulant_value_n + cumulant_value_m + cumulant_value_l) / 3  # Averaging the values

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
                       z=grid_data,
                       x=list(range(df_dim0['n'].min(), df_dim0['n'].max() + 1)),
                       y=list(range(df_dim1['n'].min(), df_dim1['n'].max() + 1)),
                       colorscale='Viridis',
                       zmin=cumulant_min,
                       zmax=cumulant_max))

    # Update layout for better visualization and making the grid square
    fig.update_layout(
        title=f'Grid Mesh of Cumulant Values (n={selected_n} for dim_id=2)',
        xaxis_title='N for Dim 0',
        yaxis_title='N for Dim 1',
        autosize=False,
        width=550,  # Adjust the width as needed
        height=550  # Adjust the height as needed
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
