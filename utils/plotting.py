"""
Well log plotting utilities using matplotlib.

Includes:
  - Triple-combo vertical log display
  - Horizontal well lateral visualization
  - Vertical well cross-section (TVD stick plot)
  - Wellbore trajectory (MD vs TVD)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def plot_triple_combo(
    df: pd.DataFrame,
    detected: dict,
    vshale_col: str = None,
    porosity_col: str = None,
    sw_col: str = None,
    net_pay_col: str = None,
    depth_col: str = "DEPTH",
) -> plt.Figure:
    """
    Create a classic triple-combo well log display:
      Track 1: GR + Caliper
      Track 2: Resistivity (deep & shallow)
      Track 3: Density + Neutron porosity
      Track 4: Computed curves (Vshale, Porosity, Sw)
      Track 5: Net Pay flag
    """
    num_tracks = 5
    fig, axes = plt.subplots(1, num_tracks, figsize=(16, 12), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    if depth_col not in df.columns:
        depth_col = "DEPTH" if "DEPTH" in df.columns else df.columns[0]
    depth = df[depth_col]

    # ---- Track 1: GR ----
    ax = axes[0]
    ax.set_ylabel("Depth", fontsize=10)
    ax.invert_yaxis()

    if "GR" in detected:
        gr = df[detected["GR"]]
        ax.plot(gr, depth, color="green", linewidth=0.7, label="GR")
        ax.set_xlim(0, 150)
        ax.set_xlabel("GR (API)", fontsize=8)
        ax.fill_betweenx(depth, gr, 0, alpha=0.15, color="green")
        ax.fill_betweenx(depth, gr, 150, alpha=0.15, color="yellow")
    if "CALIPER" in detected:
        ax_cal = ax.twiny()
        cal = df[detected["CALIPER"]]
        ax_cal.plot(cal, depth, color="black", linewidth=0.5, linestyle="--", label="CALI")
        ax_cal.set_xlim(6, 16)
        ax_cal.set_xlabel("Caliper (in)", fontsize=7, color="black")
    ax.set_title("GR / Caliper", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ---- Track 2: Resistivity ----
    ax = axes[1]
    plotted_res = False
    if "RESISTIVITY_DEEP" in detected:
        rt = df[detected["RESISTIVITY_DEEP"]]
        ax.plot(rt, depth, color="red", linewidth=0.7, label="Rt (deep)")
        plotted_res = True
    if "RESISTIVITY_SHALLOW" in detected:
        rs = df[detected["RESISTIVITY_SHALLOW"]]
        ax.plot(rs, depth, color="blue", linewidth=0.7, label="Rs (shallow)")
        plotted_res = True
    if plotted_res:
        ax.set_xscale("log")
        ax.set_xlim(0.2, 2000)
        ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel("Resistivity (ohm-m)", fontsize=8)
    ax.set_title("Resistivity", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    # ---- Track 3: Density / Neutron ----
    ax = axes[2]
    if "DENSITY" in detected:
        rhob = df[detected["DENSITY"]]
        ax.plot(rhob, depth, color="red", linewidth=0.7, label="RHOB")
        ax.set_xlim(1.95, 2.95)
        ax.set_xlabel("RHOB (g/cc)", fontsize=8, color="red")
    if "NEUTRON" in detected:
        ax_n = ax.twiny()
        nphi = df[detected["NEUTRON"]]
        ax_n.plot(nphi, depth, color="blue", linewidth=0.7, label="NPHI")
        ax_n.set_xlim(0.45, -0.15)  # Reversed scale
        ax_n.set_xlabel("NPHI (v/v)", fontsize=7, color="blue")
    # Gas crossover shading
    if "DENSITY" in detected and "NEUTRON" in detected:
        phid = (2.65 - rhob) / (2.65 - 1.0)
        ax.fill_betweenx(
            depth, rhob, 2.65,
            where=(nphi < phid),
            alpha=0.2, color="red",
            label="Gas effect"
        )
    ax.set_title("Density / Neutron", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ---- Track 4: Computed Curves ----
    ax = axes[3]
    if vshale_col and vshale_col in df.columns:
        ax.plot(df[vshale_col], depth, color="brown", linewidth=0.7, label="Vshale")
        ax.fill_betweenx(depth, df[vshale_col], 0, alpha=0.15, color="brown")
    if porosity_col and porosity_col in df.columns:
        ax.plot(df[porosity_col], depth, color="blue", linewidth=0.7, label="PHIE")
    if sw_col and sw_col in df.columns:
        ax.plot(df[sw_col], depth, color="cyan", linewidth=0.7, label="Sw")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction", fontsize=8)
    ax.set_title("Vsh / PHI / Sw", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- Track 5: Net Pay ----
    ax = axes[4]
    if net_pay_col and net_pay_col in df.columns:
        net_res = df.get("NET_RESERVOIR", pd.Series(dtype=float))
        net_pay = df.get("NET_PAY", pd.Series(dtype=float))

        if not net_res.empty:
            ax.fill_betweenx(
                depth, 0, net_res * 0.5,
                alpha=0.3, color="gold", label="Net Reservoir"
            )
        if not net_pay.empty:
            ax.fill_betweenx(
                depth, 0, net_pay,
                alpha=0.5, color="green", label="Net Pay"
            )
        ax.set_xlim(0, 1.1)
        ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel("Flag", fontsize=8)
    ax.set_title("Net Pay", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Well Log Interpretation – Triple Combo", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


# ---------------------------------------------------------------------------
# Wellbore trajectory plot (MD vs TVD)
# ---------------------------------------------------------------------------

def plot_wellbore_trajectory(
    df: pd.DataFrame,
    detected: dict,
    depth_col: str = "DEPTH",
    well_name: str = "",
    color_by: str = "GR",
) -> plt.Figure:
    """
    Plot the wellbore trajectory as MD vs TVD, optionally colored by a log curve.

    If no TVD column exists, synthesizes a simple vertical trajectory for
    reference.  Works for vertical, deviated, and horizontal wells.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    if depth_col not in df.columns:
        depth_col = "DEPTH" if "DEPTH" in df.columns else df.columns[0]
    md = df[depth_col].values

    # Get TVD
    if "TVD" in detected and detected["TVD"] in df.columns:
        tvd = df[detected["TVD"]].values
    else:
        # No TVD — assume vertical (TVD = MD)
        tvd = md.copy()

    # Color by a log curve if available
    has_color = False
    if color_by in detected and detected[color_by] in df.columns:
        cvals = df[detected[color_by]].values
        has_color = True

    if has_color:
        # Create a colored line
        valid = ~(np.isnan(md) | np.isnan(tvd) | np.isnan(cvals))
        md_v, tvd_v, cv = md[valid], tvd[valid], cvals[valid]
        points = np.array([md_v, tvd_v]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis_r", linewidth=3)
        lc.set_array(cv)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, label=color_by, shrink=0.7)
        ax.set_xlim(md_v.min() - 50, md_v.max() + 50)
        ax.set_ylim(tvd_v.max() + 50, tvd_v.min() - 50)
    else:
        valid = ~(np.isnan(md) | np.isnan(tvd))
        ax.plot(md[valid], tvd[valid], color="#6B1D2A", linewidth=2.5)
        ax.invert_yaxis()

    ax.set_xlabel("Measured Depth (MD)", fontsize=11)
    ax.set_ylabel("True Vertical Depth (TVD)", fontsize=11)
    ax.grid(True, alpha=0.3)

    title = "Wellbore Trajectory"
    if well_name:
        title = f"{well_name} — Wellbore Trajectory"
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate KOP and landing point if detectable
    if "TVD" in detected and detected["TVD"] in df.columns:
        md_clean = md[valid] if has_color else md[~np.isnan(md) & ~np.isnan(tvd)]
        tvd_clean = tvd[valid] if has_color else tvd[~np.isnan(md) & ~np.isnan(tvd)]
        if len(md_clean) > 20:
            # Detect kickoff point: where TVD starts diverging from MD
            diff = np.abs(np.diff(tvd_clean) / np.diff(md_clean))
            diff_smooth = pd.Series(diff).rolling(10, min_periods=1, center=True).mean().values
            kop_candidates = np.where(diff_smooth < 0.85)[0]
            if len(kop_candidates) > 0:
                kop_idx = kop_candidates[0]
                ax.plot(md_clean[kop_idx], tvd_clean[kop_idx], "o",
                        color="orange", markersize=10, zorder=5)
                ax.annotate("KOP", (md_clean[kop_idx], tvd_clean[kop_idx]),
                            textcoords="offset points", xytext=(12, -12),
                            fontsize=9, fontweight="bold", color="orange")

            # Detect landing point: where inclination goes > ~85 degrees
            # Approximate: where TVD change per MD step becomes near-zero
            landing_candidates = np.where(diff_smooth < 0.1)[0]
            if len(landing_candidates) > 0:
                land_idx = landing_candidates[0]
                ax.plot(md_clean[land_idx], tvd_clean[land_idx], "s",
                        color="red", markersize=10, zorder=5)
                ax.annotate("Landing", (md_clean[land_idx], tvd_clean[land_idx]),
                            textcoords="offset points", xytext=(12, 12),
                            fontsize=9, fontweight="bold", color="red")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Horizontal well lateral view
# ---------------------------------------------------------------------------

def plot_hz_lateral(
    df: pd.DataFrame,
    detected: dict,
    depth_col: str = "DEPTH",
    well_name: str = "",
    vshale_col: str = "VSHALE",
    porosity_col: str = "PHIE",
    sw_col: str = "SW",
    net_pay_col: str = "NET_PAY",
) -> plt.Figure:
    """
    Horizontal well lateral visualization.

    Plots log curves along the measured-depth axis (horizontal) to show
    how reservoir quality varies along the lateral. This is the standard
    way to view HZ well data for completion and landing-zone decisions.

    Layout (4 rows, shared horizontal MD axis):
      Row 1: GR with shale/sand fill
      Row 2: Resistivity (deep + shallow)
      Row 3: Vshale, Porosity, Sw (computed curves)
      Row 4: Net Pay flag with completion-quality color coding
    """
    if depth_col not in df.columns:
        depth_col = "DEPTH" if "DEPTH" in df.columns else df.columns[0]

    md = df[depth_col]
    num_rows = 4
    fig, axes = plt.subplots(num_rows, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # ---- Row 1: GR (horizontal axis = MD, vertical = GR value) ----
    ax = axes[0]
    if "GR" in detected and detected["GR"] in df.columns:
        gr = df[detected["GR"]]
        ax.plot(md, gr, color="green", linewidth=0.7)
        ax.fill_between(md, gr, 0, alpha=0.2, color="green", label="Sand")
        ax.fill_between(md, gr, 150, alpha=0.15, color="yellow", label="Shale")
        ax.set_ylim(0, 150)
        ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("GR (API)", fontsize=9)
    ax.set_title("Gamma Ray", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ---- Row 2: Resistivity ----
    ax = axes[1]
    plotted = False
    if "RESISTIVITY_DEEP" in detected and detected["RESISTIVITY_DEEP"] in df.columns:
        ax.plot(md, df[detected["RESISTIVITY_DEEP"]], color="red", linewidth=0.7, label="Deep Res")
        plotted = True
    if "RESISTIVITY_SHALLOW" in detected and detected["RESISTIVITY_SHALLOW"] in df.columns:
        ax.plot(md, df[detected["RESISTIVITY_SHALLOW"]], color="blue", linewidth=0.7, label="Shallow Res")
        plotted = True
    if plotted:
        ax.set_yscale("log")
        ax.set_ylim(0.2, 2000)
        ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("Res (ohm-m)", fontsize=9)
    ax.set_title("Resistivity", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    # ---- Row 3: Computed curves (Vsh, PHI, Sw) ----
    ax = axes[2]
    if vshale_col in df.columns:
        ax.plot(md, df[vshale_col], color="brown", linewidth=0.7, label="Vshale")
        ax.fill_between(md, df[vshale_col], 0, alpha=0.12, color="brown")
    if porosity_col in df.columns:
        ax.plot(md, df[porosity_col], color="blue", linewidth=0.7, label="PHIE")
    if sw_col in df.columns:
        ax.plot(md, df[sw_col], color="cyan", linewidth=0.7, label="Sw")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction", fontsize=9)
    ax.set_title("Vshale / Porosity / Sw", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- Row 4: Net Pay along lateral ----
    ax = axes[3]
    if net_pay_col in df.columns:
        net_pay = df[net_pay_col]
        net_res = df.get("NET_RESERVOIR", pd.Series(dtype=float))

        # Color-code the lateral: green = pay, gold = reservoir (no pay), gray = non-reservoir
        colors = []
        for _, row in df.iterrows():
            if net_pay_col in row and row[net_pay_col] == 1:
                colors.append("#2ca02c")  # green — net pay
            elif "NET_RESERVOIR" in row and row.get("NET_RESERVOIR", 0) == 1:
                colors.append("#d4a017")  # gold — reservoir
            else:
                colors.append("#cccccc")  # gray — non-reservoir
        ax.bar(md, 1, width=md.diff().median() if len(md) > 1 else 1.0,
               color=colors, edgecolor="none", alpha=0.8)

        # Legend patches
        patches = [
            mpatches.Patch(color="#2ca02c", label="Net Pay"),
            mpatches.Patch(color="#d4a017", label="Net Reservoir"),
            mpatches.Patch(color="#cccccc", label="Non-Reservoir"),
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_ylabel("Quality", fontsize=9)
    ax.set_title("Completion Quality Along Lateral", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    axes[-1].set_xlabel("Measured Depth (MD)", fontsize=11)

    title = "Horizontal Well — Lateral Log View"
    if well_name:
        title = f"{well_name} — Lateral Log View"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Vertical well cross-section (TVD stick plot)
# ---------------------------------------------------------------------------

def plot_vt_cross_section(
    df: pd.DataFrame,
    detected: dict,
    depth_col: str = "DEPTH",
    well_name: str = "",
    vshale_col: str = "VSHALE",
    porosity_col: str = "PHIE",
    sw_col: str = "SW",
    net_pay_col: str = "NET_PAY",
) -> plt.Figure:
    """
    Vertical well cross-section visualization.

    Shows the well as a vertical stick with lithology fill (from GR/Vshale),
    porosity and saturation side panels, and net pay flag. Designed to give
    a quick spatial sense of the reservoir column — useful for vertical
    and deviated wells.

    Layout:
      Left panel:  Lithology column (sand/shale fill from Vshale)
      Center:      Porosity + Sw profile
      Right panel: Net pay flag + formation summary
    """
    if depth_col not in df.columns:
        depth_col = "DEPTH" if "DEPTH" in df.columns else df.columns[0]

    depth = df[depth_col]
    fig, axes = plt.subplots(1, 4, figsize=(14, 12), sharey=True,
                             gridspec_kw={"width_ratios": [1.5, 2, 2, 1]})
    fig.subplots_adjust(wspace=0.05)

    # Invert Y axis (depth increasing downward)
    axes[0].invert_yaxis()

    # ---- Panel 1: Lithology column ----
    ax = axes[0]
    if vshale_col in df.columns:
        vsh = df[vshale_col]
        # Fill between: yellow (sand) on left, brown (shale) on right
        ax.fill_betweenx(depth, 0, vsh, color="saddlebrown", alpha=0.6, label="Shale")
        ax.fill_betweenx(depth, vsh, 1, color="#F5DEB3", alpha=0.6, label="Sand")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Vshale", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
    elif "GR" in detected and detected["GR"] in df.columns:
        gr = df[detected["GR"]]
        gr_norm = (gr - gr.min()) / (gr.max() - gr.min() + 1e-9)
        ax.fill_betweenx(depth, 0, gr_norm, color="saddlebrown", alpha=0.6)
        ax.fill_betweenx(depth, gr_norm, 1, color="#F5DEB3", alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("GR (norm)", fontsize=8)
    ax.set_ylabel("Depth", fontsize=11)
    ax.set_title("Lithology", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # ---- Panel 2: Porosity profile ----
    ax = axes[1]
    if porosity_col in df.columns:
        phi = df[porosity_col]
        ax.plot(phi, depth, color="blue", linewidth=1.0, label="PHIE")
        ax.fill_betweenx(depth, phi, 0, alpha=0.2, color="blue")
        ax.set_xlim(0, max(0.35, phi.max() * 1.1))
    if "DENSITY" in detected and detected["DENSITY"] in df.columns:
        rhob = df[detected["DENSITY"]]
        ax_rho = ax.twiny()
        ax_rho.plot(rhob, depth, color="red", linewidth=0.6, alpha=0.6, label="RHOB")
        ax_rho.set_xlim(1.95, 2.95)
        ax_rho.set_xlabel("RHOB (g/cc)", fontsize=7, color="red")
    ax.set_xlabel("Porosity (v/v)", fontsize=9, color="blue")
    ax.set_title("Porosity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ---- Panel 3: Saturation profile ----
    ax = axes[2]
    if sw_col in df.columns:
        sw = df[sw_col]
        sh = 1.0 - sw  # hydrocarbon saturation
        ax.fill_betweenx(depth, 0, sh, color="green", alpha=0.4, label="Hydrocarbon (Sh)")
        ax.fill_betweenx(depth, sh, 1, color="lightblue", alpha=0.5, label="Water (Sw)")
        ax.plot(sw, depth, color="cyan", linewidth=0.8, label="Sw")
        ax.set_xlim(0, 1)
    ax.set_xlabel("Saturation", fontsize=9)
    ax.set_title("Fluid Saturation", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ---- Panel 4: Net Pay flag ----
    ax = axes[3]
    if net_pay_col in df.columns:
        net_pay = df[net_pay_col]
        net_res = df.get("NET_RESERVOIR", pd.Series(dtype=float))

        if not net_res.empty:
            ax.fill_betweenx(depth, 0, net_res * 0.5, color="gold", alpha=0.4, label="Net Res")
        ax.fill_betweenx(depth, 0, net_pay, color="green", alpha=0.6, label="Net Pay")
        ax.set_xlim(0, 1.1)
        ax.legend(fontsize=6, loc="upper right")

    ax.set_xlabel("Flag", fontsize=8)
    ax.set_title("Net Pay", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2)

    title = "Vertical Well — Reservoir Cross-Section"
    if well_name:
        title = f"{well_name} — Reservoir Cross-Section"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
