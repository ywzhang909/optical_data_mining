"""
Matplotlib-based rendering toolkit for AO beam analysis.

Pure visualization functions that produce matplotlib Figure objects.
No Streamlit imports — designed to be called by any UI layer.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_beam_visualization(
    img: np.ndarray,
    title: str,
    pixel_size_um: float,
    features: dict[str, Any],
    border_circles: list[dict[str, Any]] | None = None,
) -> plt.Figure:
    """
    Plot beam visualization with centroid marking, D4σ circle, and X/Y cross-section profiles.

    Parameters
    ----------
    img : np.ndarray
        Raw image array.
    title : str
        Title for the plot.
    pixel_size_um : float
        Pixel size in micrometers.
    features : dict
        Feature dictionary containing 'center_x', 'center_y', 'D_x', 'D_y', 'avg_diameter'.
    border_circles : list[dict] | None
        Optional list of extra circles to draw. Each dict should have:
        'cx', 'cy', 'radius', 'color', 'linestyle', 'label'.

    Returns
    -------
    plt.Figure
        Matplotlib figure with 2 subplots: image + profile.
    """
    img = np.asarray(img, dtype=np.float64)
    cx, cy = features["center_x"], features["center_y"]
    r_pix = features["avg_diameter"] / pixel_size_um / 2  # radius in pixels

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1. Centroid + D4σ circle ---
    ax1 = axes[0]
    im1 = ax1.imshow(img, cmap="hot", interpolation="bilinear")
    ax1.set_title(f"{title} — Centroid & D4σ", fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    ax1.plot(cx, cy, "c+", markersize=15, markeredgewidth=2, label="Centroid")
    ax1.add_patch(
        Circle(
            (cx, cy),
            radius=r_pix,
            fill=False,
            color="cyan",
            linewidth=2,
            linestyle="--",
            label=f"D4σ (r={features['avg_diameter'] / 2:.1f}μm)",
        )
    )

    # Optional extra circles (e.g. border, uniformity boundary)
    if border_circles:
        for bc in border_circles:
            ax1.add_patch(
                Circle(
                    (bc["cx"], bc["cy"]),
                    radius=bc["radius"],
                    fill=False,
                    color=bc["color"],
                    linewidth=bc.get("linewidth", 2),
                    linestyle=bc.get("linestyle", "-"),
                    label=bc.get("label", ""),
                )
            )

    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlabel("X (pixel)")
    ax1.set_ylabel("Y (pixel)")

    # --- 2. X + Y direction profiles ---
    ax2 = axes[1]
    x_data = img[int(cy), :]
    x_pixels = np.arange(len(x_data))
    x_um = (x_pixels - cx) * pixel_size_um

    y_data = img[:, int(cx)]
    y_pixels = np.arange(len(y_data))
    y_um = (y_pixels - cy) * pixel_size_um

    ax2.plot(x_um, x_data, "b-", linewidth=1.5, label="X profile")
    ax2.plot(y_um, y_data, "g-", linewidth=1.5, label="Y profile")
    ax2.axvline(x=0, color="gray", linestyle=":", alpha=0.7, label="Centroid")

    max_val = max(np.max(x_data), np.max(y_data))
    if max_val > 0:
        ax2.axhline(y=max_val / np.e, color="r", linestyle="--", alpha=0.5, label="1/e peak")
    ax2.set_title(f"{title} — Cross-section Profiles", fontsize=12)
    ax2.set_xlabel("Offset (μm)")
    ax2.set_ylabel("Intensity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_gaussian_cross_section(
    h_prof: np.ndarray,
    v_prof: np.ndarray,
    h_mu: float,
    h_sigma: float,
    h_A: float,
    h_b: float,
    v_mu: float,
    v_sigma: float,
    v_A: float,
    v_b: float,
) -> plt.Figure:
    """
    Plot horizontal and vertical cross-sections with Gaussian fits.

    Returns
    -------
    plt.Figure
        Matplotlib figure with 2 subplots (horizontal, vertical).
    """
    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(12, 4))

    # Horizontal
    x_h = np.arange(len(h_prof))
    x_h_fit = np.linspace(0, len(h_prof) - 1, 200)
    y_h_fit = _gaussian(x_h_fit, h_mu, h_sigma, h_A, h_b)
    ax_h.plot(x_h, h_prof, "b-", alpha=0.6, linewidth=1.5, label="Raw data")
    ax_h.plot(x_h_fit, y_h_fit, "r-", linewidth=2, label="Gaussian fit")
    h_peak = float(h_A) + float(h_b)
    ax_h.axvline(h_mu, color="g", linestyle="--", alpha=0.7)
    ax_h.axhline(h_peak, color="g", linestyle=":", alpha=0.4)
    ax_h.plot(h_mu, h_peak, "go", markersize=6)
    ax_h.annotate(
        f"μ={h_mu:.2f}\nI={h_peak:.1f}",
        xy=(h_mu, h_peak),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="green",
        fontweight="bold",
    )
    ax_h.set_xlabel("Pixel")
    ax_h.set_ylabel("Intensity")
    ax_h.set_title("Horizontal (X direction)")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3)

    # Vertical
    x_v = np.arange(len(v_prof))
    x_v_fit = np.linspace(0, len(v_prof) - 1, 200)
    y_v_fit = _gaussian(x_v_fit, v_mu, v_sigma, v_A, v_b)
    ax_v.plot(x_v, v_prof, "b-", alpha=0.6, linewidth=1.5, label="Raw data")
    ax_v.plot(x_v_fit, y_v_fit, "r-", linewidth=2, label="Gaussian fit")
    v_peak = float(v_A) + float(v_b)
    ax_v.axvline(v_mu, color="g", linestyle="--", alpha=0.7)
    ax_v.axhline(v_peak, color="g", linestyle=":", alpha=0.4)
    ax_v.plot(v_mu, v_peak, "go", markersize=6)
    ax_v.annotate(
        f"μ={v_mu:.2f}\nI={v_peak:.1f}",
        xy=(v_mu, v_peak),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="green",
        fontweight="bold",
    )
    ax_v.set_xlabel("Pixel")
    ax_v.set_ylabel("Intensity")
    ax_v.set_title("Vertical (Y direction)")
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ftl_polar(
    ang_theta_arr: np.ndarray,
    ang_rfl_arr: np.ndarray,
    unit_label: str = "μm",
) -> plt.Figure:
    """
    Polar plot of FTL characteristic radius R_FL(θ).

    Returns
    -------
    plt.Figure
    """
    fig_polar, ax_polar = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    cmap_val = (ang_theta_arr % (2 * np.pi)) / (2 * np.pi)
    ax_polar.scatter(ang_theta_arr, ang_rfl_arr, c=cmap_val, cmap="hsv", s=40, alpha=0.8)
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(-1)
    plt.tight_layout()
    return fig_polar


def plot_ftl_angular(
    ang_theta: list[float],
    ang_rfl_arr: np.ndarray,
    unit_label: str = "μm",
) -> plt.Figure:
    """
    Line chart of FTL R_FL(θ) vs angle with mean line.

    Returns
    -------
    plt.Figure
    """
    fig_ang, ax_ang = plt.subplots(figsize=(8, 3.5))
    ax_ang.plot(ang_theta, ang_rfl_arr, "o-", color="steelblue", markersize=4, linewidth=1.2)
    ax_ang.set_xlabel("Angle (°)")
    ax_ang.set_ylabel(f"R_FL ({unit_label})")
    ax_ang.grid(True, alpha=0.3)
    valid = ang_rfl_arr[~np.isnan(ang_rfl_arr)]
    if len(valid) > 0:
        mean_rfl = np.nanmean(valid)
        ax_ang.axhline(
            mean_rfl, color="gray", linestyle="--", alpha=0.6, label=f"Mean={mean_rfl:.2f}"
        )
    ax_ang.legend(fontsize=8)
    plt.tight_layout()
    return fig_ang


def plot_ftl_q_polar(
    ang_theta_arr: np.ndarray,
    ang_q_arr: np.ndarray,
) -> plt.Figure:
    """
    Polar plot of FTL flat-top order q(θ) with mean line.

    Returns
    -------
    plt.Figure
    """
    fig_q, ax_q = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax_q.plot(ang_theta_arr, ang_q_arr, "o-", color="darkorange", markersize=4, linewidth=1.2)
    ax_q.set_theta_zero_location("E")
    ax_q.set_theta_direction(-1)
    valid_q = ang_q_arr[~np.isnan(ang_q_arr)]
    if len(valid_q) > 0:
        mean_q = np.nanmean(ang_q_arr)
        ax_q.plot(
            ang_theta_arr,
            np.full_like(ang_q_arr, mean_q),
            color="gray",
            linestyle="--",
            alpha=0.6,
            linewidth=1,
            label=f"Mean={mean_q:.2f}",
        )
    ax_q.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig_q


def plot_zernike_barchart(
    zernike_coeffs: np.ndarray,
    zernike_noll: list[int],
    top_k: int = 10,
) -> plt.Figure:
    """
    Horizontal bar chart of top Zernike coefficients (sorted by |value|).

    Parameters
    ----------
    zernike_coeffs : np.ndarray
        Full array of Zernike coefficients.
    zernike_noll : list[int]
        Corresponding Noll index for each coefficient.
    top_k : int
        Number of top coefficients to show (default 10).

    Returns
    -------
    plt.Figure
    """
    sorted_indices = np.argsort(np.abs(zernike_coeffs))[::-1]
    top_k = min(top_k, len(sorted_indices))
    top_idx = sorted_indices[:top_k]
    top_coeffs = zernike_coeffs[top_idx]
    top_noll = [zernike_noll[i] for i in top_idx]

    labels = [f"Noll {n}" for n in top_noll]
    colors = ["steelblue" if c >= 0 else "crimson" for c in top_coeffs]

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    ax_bar.barh(y_pos, top_coeffs, color=colors)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Coefficient value")
    ax_bar.set_title(f"Top {top_k} Zernike coefficients")
    ax_bar.axvline(0, color="gray", linewidth=0.8)
    ax_bar.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig_bar


def plot_energy_pie(
    cum_norm: np.ndarray,
    sorted_R: np.ndarray,
    r50: float,
    r80: float,
    r95: float,
) -> plt.Figure:
    """
    Plot energy concentration η(r) as a radial ring pie chart.

    Parameters
    ----------
    cum_norm : np.ndarray
        Normalised cumulative energy (0→1) sorted by radius.
    sorted_R : np.ndarray
        Radius values corresponding to each cum_norm entry.
    r50, r80, r95 : float
        Radii containing 50 %, 80 %, 95 % of total energy.

    Returns
    -------
    plt.Figure
        Matplotlib pie chart figure.
    """
    import math

    if np.isnan(r50) or np.isnan(r80) or np.isnan(r95):
        # Fallback: single grey slice
        fig_pie, ax_pie = plt.subplots(figsize=(4.5, 3.5))
        ax_pie.pie([1.0], labels=["N/A"], colors=["#bdc3c7"],
                   startangle=90, textprops={"fontsize": 8},
                   wedgeprops={"linewidth": 1, "edgecolor": "white"})
        ax_pie.set_title("Energy concentration η(r) — radial ring distribution", fontsize=10)
        return fig_pie

    idx50 = np.searchsorted(cum_norm, 0.50)
    idx80 = np.searchsorted(cum_norm, 0.80)
    idx95 = np.searchsorted(cum_norm, 0.95)

    e50 = cum_norm[min(idx50, len(cum_norm) - 1)]
    e80 = cum_norm[min(idx80, len(cum_norm) - 1)] - e50 if idx80 < len(cum_norm) else 0.30
    e95 = cum_norm[min(idx95, len(cum_norm) - 1)] - cum_norm[min(idx80, len(cum_norm) - 1)] if idx95 < len(cum_norm) else 0.15
    erem = max(0.0, 1.0 - e50 - e80 - e95)

    sizes = [e50, e80, e95, erem]
    labels = [
        f"Core\n(r≤{r50:.0f}px)\n{e50 * 100:.0f}%",
        f"Inner\n({r50:.0f}<r≤{r80:.0f}px)\n{e80 * 100:.0f}%",
        f"Outer\n({r80:.0f}<r≤{r95:.0f}px)\n{e95 * 100:.0f}%",
        f"Rim\n(r>{r95:.0f}px)\n{erem * 100:.0f}%",
    ]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#95a5a6"]

    fig_pie, ax_pie = plt.subplots(figsize=(4.5, 3.5))
    ax_pie.pie(
        sizes, labels=labels, colors=colors,
        startangle=90, textprops={"fontsize": 8},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax_pie.set_title("Energy concentration η(r) — radial ring distribution", fontsize=10)
    plt.tight_layout()
    return fig_pie


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian(x: np.ndarray, mu: float, sigma: float, A: float, b: float) -> np.ndarray:
    """Gaussian function used for cross-section fitting display."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b
