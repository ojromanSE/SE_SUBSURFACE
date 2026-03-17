"""
Well log plotting utilities using matplotlib for a classic triple-combo display.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
