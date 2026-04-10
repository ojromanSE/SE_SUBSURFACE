"""
Multi-well management, log correlation, and formation detection.

Provides:
  - Well collection management (add/remove wells, store metadata)
  - Well-to-well log correlation display
  - Automatic formation boundary detection from GR inflections
  - Multi-well map view
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


# ---------------------------------------------------------------------------
# Formation boundary detection (GR-based)
# ---------------------------------------------------------------------------

def detect_formation_boundaries(
    df: pd.DataFrame,
    gr_col: str,
    depth_col: str = "DEPTH",
    window: int = 15,
    min_delta_gr: float = 20.0,
) -> pd.DataFrame:
    """
    Detect formation boundaries from GR log inflections.

    Uses smoothed GR gradient to find significant transitions
    (shale-to-sand and sand-to-shale).

    Parameters
    ----------
    df : DataFrame with GR and DEPTH columns.
    gr_col : Column name for GR curve.
    depth_col : Column name for depth.
    window : Smoothing window size (number of samples).
    min_delta_gr : Minimum GR change across boundary to qualify.

    Returns
    -------
    DataFrame with columns: depth, gr_above, gr_below, delta_gr, boundary_type
    """
    if gr_col not in df.columns or depth_col not in df.columns:
        return pd.DataFrame()

    depth = df[depth_col].values
    gr = df[gr_col].values

    # Smooth GR
    gr_smooth = pd.Series(gr).rolling(window, center=True, min_periods=1).mean().values

    # Compute gradient
    gr_grad = np.gradient(gr_smooth, depth)

    # Find inflection points (local extrema of gradient)
    maxima = argrelextrema(np.abs(gr_grad), np.greater, order=window)[0]

    boundaries = []
    for idx in maxima:
        if idx < window or idx >= len(gr) - window:
            continue
        gr_above = np.nanmean(gr_smooth[max(0, idx - window):idx])
        gr_below = np.nanmean(gr_smooth[idx:min(len(gr), idx + window)])
        delta = gr_below - gr_above

        if abs(delta) < min_delta_gr:
            continue

        btype = "sand-to-shale" if delta > 0 else "shale-to-sand"
        boundaries.append({
            "depth": depth[idx],
            "gr_above": gr_above,
            "gr_below": gr_below,
            "delta_gr": delta,
            "boundary_type": btype,
        })

    return pd.DataFrame(boundaries)


# ---------------------------------------------------------------------------
# Log correlation plot (multi-well side by side)
# ---------------------------------------------------------------------------

def plot_well_correlation(
    wells: list[dict],
    curve: str = "GR",
    depth_col: str = "DEPTH",
    boundaries: dict = None,
) -> plt.Figure:
    """
    Plot multiple wells side by side for visual correlation.

    Parameters
    ----------
    wells : list of dicts, each with:
        - 'name': str (well name)
        - 'df': DataFrame (interpreted log data)
        - 'detected': dict (curve mapping)
        - 'lat': float or None
        - 'lon': float or None
    curve : Which curve type to correlate ('GR', 'RESISTIVITY_DEEP', etc.)
    depth_col : Depth column name.
    boundaries : dict mapping well_name -> DataFrame of formation boundaries.

    Returns
    -------
    matplotlib Figure
    """
    n = len(wells)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No wells to display", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 12), sharey=True)
    if n == 1:
        axes = [axes]

    # Find global depth range
    all_depths = []
    for w in wells:
        if depth_col in w["df"].columns:
            all_depths.extend([w["df"][depth_col].min(), w["df"][depth_col].max()])

    if all_depths:
        global_min = min(all_depths)
        global_max = max(all_depths)
    else:
        global_min, global_max = 0, 1000

    # Color map for curve types
    curve_colors = {
        "GR": "green",
        "RESISTIVITY_DEEP": "red",
        "RESISTIVITY_SHALLOW": "blue",
        "DENSITY": "red",
        "NEUTRON": "blue",
        "SONIC": "purple",
    }

    for i, (w, ax) in enumerate(zip(wells, axes)):
        df = w["df"]
        detected = w["detected"]
        name = w.get("name", f"Well {i+1}")

        if depth_col not in df.columns:
            ax.set_title(name, fontsize=9, fontweight="bold")
            ax.text(0.5, 0.5, "No depth", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        depth = df[depth_col]
        ax.set_ylim(global_max, global_min)  # Inverted
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_ylabel("Depth", fontsize=10)

        if curve in detected:
            col = detected[curve]
            vals = df[col]
            color = curve_colors.get(curve, "black")
            ax.plot(vals, depth, color=color, linewidth=0.7)
            ax.set_xlabel(col, fontsize=8)

            if curve == "GR":
                ax.set_xlim(0, 150)
                ax.fill_betweenx(depth, vals, 0, alpha=0.15, color="green")
                ax.fill_betweenx(depth, vals, 150, alpha=0.15, color="yellow")
            elif curve in ("RESISTIVITY_DEEP", "RESISTIVITY_SHALLOW"):
                ax.set_xscale("log")
                ax.set_xlim(0.2, 2000)
        else:
            ax.text(0.5, 0.5, f"No {curve}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")

        # Draw formation boundaries if available
        if boundaries and name in boundaries:
            fb = boundaries[name]
            for _, row in fb.iterrows():
                ax.axhline(y=row["depth"], color="red", linewidth=1.0,
                           linestyle="--", alpha=0.7)

    fig.suptitle(f"Well Correlation — {curve}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Multi-well map
# ---------------------------------------------------------------------------

def plot_well_map(wells: list[dict]) -> plt.Figure:
    """
    Simple scatter plot of well locations (lat/lon).

    Parameters
    ----------
    wells : list of dicts with 'name', 'lat', 'lon' keys.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    has_locations = False
    for w in wells:
        lat = w.get("lat")
        lon = w.get("lon")
        if lat is not None and lon is not None:
            ax.plot(lon, lat, "o", markersize=10, color="#6B1D2A")
            ax.annotate(
                w.get("name", ""),
                (lon, lat),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                fontweight="bold",
            )
            has_locations = True

    if not has_locations:
        ax.text(0.5, 0.5, "No well locations available",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)

    ax.set_title("Well Location Map", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig
