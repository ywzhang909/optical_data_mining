import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def display_3d_image(image_array):
    # Generate x, y coordinates
    x = np.arange(image_array.shape[1])
    y = np.arange(image_array.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create a 3D surface plot
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=image_array)])

    # Update layout
    fig.update_layout(
        title='3D Visualization of Far Spot Image',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Brightness'
        )
    )

    fig.show()
    
def display_image(image_array, c_x, c_y):
    fig = px.imshow(image_array, color_continuous_scale='gray')
    # Add the centroid as a red spot
    fig.add_trace(go.Scatter(
        x=[c_x],
        y=[c_y],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Centroid'
    ))
    fig.update_layout(title='Image with Centroid Marked')
    fig.show()