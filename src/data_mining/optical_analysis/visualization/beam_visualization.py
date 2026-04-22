"""
Beam visualization module for AO analysis
Handles all visualization functions using Plotly
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from datetime import datetime
import os
import json
import hashlib
from PIL import Image
import io


def plot_beam_visualization(img, title, pixel_size_um, features):
    """
    Plot beam visualization using Plotly:
    - Centroid marking
    - D4σ circle
    - XY axis intensity profiles

    Args:
        img: Raw image
        title: Title for the plot
        pixel_size_um: Pixel size in micrometers
        features: Feature dictionary with center_x, center_y, D_x, D_y, avg_diameter

    Returns:
        fig: Plotly figure
    """
    img = np.asarray(img, dtype=np.float64)

    cx, cy = features["center_x"], features["center_y"]
    # Use average sigma2 as radius
    r_pix = features["avg_diameter"] / pixel_size_um / 2  # Radius in pixels

    # Create subplots with 1 row and 3 columns
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"{title} - Centroid & D4σ",
            f"{title} - X Profile",
            f"{title} - Y Profile",
        ],
        specs=[[{"type": "heatmap"}, {"type": "scatter"}, {"type": "scatter"}]],
    )

    # 1. Heatmap with centroid and D4σ circle
    fig.add_trace(
        go.Heatmap(z=img, colorscale="Hot", name="Intensity", showscale=True),
        row=1,
        col=1,
    )

    # Add centroid marker
    fig.add_trace(
        go.Scatter(
            x=[cx],
            y=[cy],
            mode="markers",
            marker=dict(color="cyan", size=10, symbol="cross"),
            name="Centroid",
        ),
        row=1,
        col=1,
    )

    # Add D4σ circle (using average diameter)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = cx + r_pix * np.cos(theta)
    circle_y = cy + r_pix * np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="cyan", width=2, dash="dash"),
            name=f"D4σ (r={features['avg_diameter'] / 2:.1f}μm)",
        ),
        row=1,
        col=1,
    )

    # 2. X direction profile
    x_data = img[int(cy), :]
    x_pixels = np.arange(len(x_data))
    x_um = (x_pixels - cx) * pixel_size_um  # Convert to μm

    fig.add_trace(
        go.Scatter(
            x=x_um,
            y=x_data,
            mode="lines",
            line=dict(color="blue", width=2),
            name="X profile",
        ),
        row=1,
        col=2,
    )

    fig.add_hline(
        y=np.max(x_data) / np.e,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text="1/e peak",
        row=1,
        col=2,
    )

    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="gray",
        opacity=0.7,
        annotation_text="Centroid",
        row=1,
        col=2,
    )

    # 3. Y direction profile
    y_data = img[:, int(cx)]
    y_pixels = np.arange(len(y_data))
    y_um = (y_pixels - cy) * pixel_size_um  # Convert to μm

    fig.add_trace(
        go.Scatter(
            x=y_um,
            y=y_data,
            mode="lines",
            line=dict(color="green", width=2),
            name="Y profile",
        ),
        row=1,
        col=3,
    )

    fig.add_hline(
        y=np.max(y_data) / np.e,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text="1/e peak",
        row=1,
        col=3,
    )

    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="gray",
        opacity=0.7,
        annotation_text="Centroid",
        row=1,
        col=3,
    )

    fig.update_layout(title_text=title, height=500, showlegend=True)

    # Update axes labels
    fig.update_xaxes(title_text="X (pixel)", row=1, col=1)
    fig.update_yaxes(title_text="Y (pixel)", row=1, col=1)
    fig.update_xaxes(title_text="X (μm)", row=1, col=2)
    fig.update_yaxes(title_text="Intensity", row=1, col=2)
    fig.update_xaxes(title_text="Y (μm)", row=1, col=3)
    fig.update_yaxes(title_text="Intensity", row=1, col=3)

    return fig


def plot_3d_visualization(img, title, zmin=None, zmax=None):
    """
    Create 3D surface plot using Plotly

    Args:
        img: Input image array
        title: Title
        zmin: Color scale minimum (for unified scale)
        zmax: Color scale maximum (for unified scale)

    Returns:
        plotly_fig: Plotly 3D chart
    """
    img = np.asarray(img, dtype=np.float64)

    # Downsample to improve rendering speed
    step = max(1, min(img.shape[0], img.shape[1]) // 100)
    x = np.arange(0, img.shape[1], step)
    y = np.arange(0, img.shape[0], step)
    X, Y = np.meshgrid(x, y)
    Z = img[::step, ::step]

    # Unified scale
    if zmin is None:
        zmin = Z.min()
    if zmax is None:
        zmax = Z.max()

    # Compute xyz range for unified scale
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = zmax - zmin

    # Normalize Z to the same scale as XY for visual consistency
    if z_range > 0:
        target_range = (x_range + y_range) / 2
        Z_display = (Z - zmin) / z_range * target_range
    else:
        Z_display = Z - zmin

    # Create Plotly 3D surface plot
    plotly_fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z_display,
                colorscale="Hot",
                cmin=zmin,
                cmax=zmax,
                colorbar=dict(title="Intensity"),
                hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Intensity: %{z:.1f}<extra></extra>",
            )
        ]
    )

    plotly_fig.update_layout(
        title=f"{title} - 3D Surface",
        scene=dict(
            xaxis_title="X (pixel)",
            yaxis_title="Y (pixel)",
            zaxis_title="Intensity",
            aspectmode="data",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        width=800,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50),
    )

    return plotly_fig


def plot_multiple_beams_3d(beams_data, titles, zmin=None, zmax=None):
    """
    Plot multiple beams in a single 3D visualization

    Args:
        beams_data: List of beam image arrays
        titles: List of titles for each beam
        zmin: Minimum color scale value
        zmax: Maximum color scale value

    Returns:
        plotly_fig: Plotly 3D chart with multiple surfaces
    """
    if zmin is None:
        zmin = min([np.min(beam) for beam in beams_data])
    if zmax is None:
        zmax = max([np.max(beam) for beam in beams_data])

    # Create subplots for multiple 3D surfaces
    fig = make_subplots(
        rows=1,
        cols=len(beams_data),
        specs=[[{"type": "surface"} for _ in beams_data]],
        subplot_titles=titles,
    )

    for i, (beam, title) in enumerate(zip(beams_data, titles)):
        step = max(1, min(beam.shape[0], beam.shape[1]) // 100)
        x = np.arange(0, beam.shape[1], step)
        y = np.arange(0, beam.shape[0], step)
        X, Y = np.meshgrid(x, y)
        Z = beam[::step, ::step]

        # Normalize Z to the same scale as XY for visual consistency
        z_range = zmax - zmin
        if z_range > 0:
            target_range = (X.max() - X.min() + Y.max() - Y.min()) / 2
            Z_display = (Z - zmin) / z_range * target_range
        else:
            Z_display = Z - zmin

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z_display,
                colorscale="Hot",
                cmin=zmin,
                cmax=zmax,
                name=title,
                hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Intensity: %{z:.1f}<extra></extra>",
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(title="Multiple Beam Visualization", height=600, showlegend=True)

    return fig
