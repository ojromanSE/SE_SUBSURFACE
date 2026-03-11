"""
Petrophysical interpretation engine following SPE/AAPG/SPWLA standards.

References:
- SPE-PRMS (2018): Petroleum Resources Management System
- Archie, G.E. (1942): "The Electrical Resistivity Log as an Aid in
  Determining Some Reservoir Characteristics", Trans. AIME 146, 54-62.
- Worthington, P.F. (2010): SPE 131529 - "Net Pay: What Is It? What Does
  It Do? How Do We Quantify It? How Do We Use It?"
- Asquith & Krygowski (2004): "Basic Well Log Analysis", AAPG Methods in
  Exploration No. 16, 2nd Ed.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shale Volume (Vshale) Calculations
# ---------------------------------------------------------------------------

def vshale_linear(gr: pd.Series, gr_clean: float, gr_shale: float) -> pd.Series:
    """
    Linear Vshale from Gamma Ray (IGR).
    Ref: Asquith & Krygowski (2004), Ch. 2.
    """
    ish = (gr - gr_clean) / (gr_shale - gr_clean)
    return ish.clip(0.0, 1.0)


def vshale_larionov_tertiary(igr: pd.Series) -> pd.Series:
    """
    Larionov (1969) non-linear Vshale correction for Tertiary (young) rocks.
    Vsh = 0.083 * (2^(3.7 * IGR) - 1)
    """
    vsh = 0.083 * (np.power(2, 3.7 * igr) - 1)
    return vsh.clip(0.0, 1.0)


def vshale_larionov_older(igr: pd.Series) -> pd.Series:
    """
    Larionov (1969) non-linear Vshale correction for older (pre-Tertiary) rocks.
    Vsh = 0.33 * (2^(2 * IGR) - 1)
    """
    vsh = 0.33 * (np.power(2, 2.0 * igr) - 1)
    return vsh.clip(0.0, 1.0)


def vshale_steiber(igr: pd.Series) -> pd.Series:
    """
    Steiber (1970) non-linear Vshale correction.
    Vsh = IGR / (3 - 2 * IGR)
    """
    vsh = igr / (3.0 - 2.0 * igr)
    return vsh.clip(0.0, 1.0)


def vshale_clavier(igr: pd.Series) -> pd.Series:
    """
    Clavier (1971) non-linear Vshale correction.
    Vsh = 1.7 - sqrt(3.38 - (IGR + 0.7)^2)
    """
    inner = 3.38 - np.power(igr + 0.7, 2)
    inner = inner.clip(lower=0.0)
    vsh = 1.7 - np.sqrt(inner)
    return vsh.clip(0.0, 1.0)


VSHALE_METHODS = {
    "Linear (IGR)": lambda igr: igr.clip(0.0, 1.0),
    "Larionov – Tertiary": vshale_larionov_tertiary,
    "Larionov – Pre-Tertiary": vshale_larionov_older,
    "Steiber (1970)": vshale_steiber,
    "Clavier (1971)": vshale_clavier,
}


# ---------------------------------------------------------------------------
# Porosity Calculations
# ---------------------------------------------------------------------------

def porosity_density(
    rhob: pd.Series,
    rho_matrix: float = 2.65,
    rho_fluid: float = 1.0,
) -> pd.Series:
    """
    Density porosity.
    Ref: Asquith & Krygowski (2004), Ch. 5.

    PHID = (rho_matrix - rhob) / (rho_matrix - rho_fluid)

    Typical matrix densities:
        Sandstone: 2.65 g/cc
        Limestone: 2.71 g/cc
        Dolomite:  2.87 g/cc
    """
    phi = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    return phi.clip(0.0, 0.45)


def porosity_sonic(
    dt: pd.Series,
    dt_matrix: float = 55.5,
    dt_fluid: float = 189.0,
) -> pd.Series:
    """
    Wyllie time-average sonic porosity.
    Ref: Wyllie et al. (1956); Asquith & Krygowski (2004), Ch. 5.

    PHIS = (DT - DT_matrix) / (DT_fluid - DT_matrix)

    Typical matrix travel times (us/ft):
        Sandstone: 55.5
        Limestone: 47.5
        Dolomite:  43.5
    Fluid: 189 us/ft (freshwater)
    """
    phi = (dt - dt_matrix) / (dt_fluid - dt_matrix)
    return phi.clip(0.0, 0.45)


def porosity_neutron_density(
    phin: pd.Series,
    phid: pd.Series,
) -> pd.Series:
    """
    Neutron-density crossplot porosity.
    Ref: Asquith & Krygowski (2004), Ch. 6.

    PHI_ND = sqrt((PHIN^2 + PHID^2) / 2)
    """
    phi = np.sqrt((phin**2 + phid**2) / 2.0)
    return phi.clip(0.0, 0.45)


def effective_porosity(phit: pd.Series, vshale: pd.Series) -> pd.Series:
    """
    Effective porosity corrected for shale volume.
    PHIE = PHIT * (1 - Vsh)
    """
    phie = phit * (1.0 - vshale)
    return phie.clip(0.0, 0.45)


# ---------------------------------------------------------------------------
# Water Saturation – Archie Family
# ---------------------------------------------------------------------------

def sw_archie(
    rt: pd.Series,
    porosity: pd.Series,
    rw: float = 0.05,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> pd.Series:
    """
    Archie's equation for water saturation.
    Ref: Archie (1942), Trans. AIME 146, 54-62.

    Sw^n = (a * Rw) / (PHI^m * Rt)
    Sw = ((a * Rw) / (PHI^m * Rt))^(1/n)

    Parameters:
        a: tortuosity factor (1.0 for carbonates, 0.62–0.81 for sandstones)
        m: cementation exponent (2.0 typical)
        n: saturation exponent (2.0 typical)
        Rw: formation water resistivity (ohm-m)
    """
    # Avoid division by zero
    phi_safe = porosity.clip(lower=0.001)
    rt_safe = rt.clip(lower=0.001)

    sw = np.power((a * rw) / (np.power(phi_safe, m) * rt_safe), 1.0 / n)
    return sw.clip(0.0, 1.0)


def sw_simandoux(
    rt: pd.Series,
    porosity: pd.Series,
    vshale: pd.Series,
    rw: float = 0.05,
    rsh: float = 5.0,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> pd.Series:
    """
    Simandoux (1963) equation for shaly sands.
    Accounts for additional conductivity from shale.
    Ref: Simandoux, P. (1963), Dielectric measurements on porous media.

    1/Rt = (PHI^m * Sw^n) / (a * Rw) + (Vsh * Sw) / Rsh
    Solved as quadratic in Sw.
    """
    phi_safe = porosity.clip(lower=0.001)
    rt_safe = rt.clip(lower=0.001)
    vsh_safe = vshale.clip(lower=0.0)

    c = 1.0 / rt_safe
    b_coef = vsh_safe / rsh
    a_coef = np.power(phi_safe, m) / (a * rw)

    # Quadratic: a_coef * Sw^2 + b_coef * Sw - c = 0
    discriminant = b_coef**2 + 4.0 * a_coef * c
    discriminant = discriminant.clip(lower=0.0)
    sw = (-b_coef + np.sqrt(discriminant)) / (2.0 * a_coef)
    return sw.clip(0.0, 1.0)


def sw_indonesia(
    rt: pd.Series,
    porosity: pd.Series,
    vshale: pd.Series,
    rw: float = 0.05,
    rsh: float = 5.0,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> pd.Series:
    """
    Poupon-Leveaux (Indonesia) equation for shaly sands.
    Commonly used in SE Asia / Indonesia formations.
    Ref: Poupon & Leveaux (1971), SPWLA 12th Annual Symposium.

    1/sqrt(Rt) = (Vsh^(1-Vsh/2)) / sqrt(Rsh) + (PHI^(m/2)) / sqrt(a*Rw) * Sw^(n/2)
    """
    phi_safe = porosity.clip(lower=0.001)
    rt_safe = rt.clip(lower=0.001)
    vsh_safe = vshale.clip(lower=0.0)

    term_shale = np.power(vsh_safe, 1.0 - vsh_safe / 2.0) / np.sqrt(rsh)
    term_clean = np.power(phi_safe, m / 2.0) / np.sqrt(a * rw)

    # Sw^(n/2) = (1/sqrt(Rt) - term_shale) / term_clean
    lhs = 1.0 / np.sqrt(rt_safe) - term_shale
    lhs = lhs.clip(lower=0.0)
    term_clean_safe = term_clean.clip(lower=1e-10)
    sw_half = lhs / term_clean_safe
    sw = np.power(sw_half, 2.0 / n)
    return sw.clip(0.0, 1.0)


SW_METHODS = {
    "Archie (Clean Sands)": "archie",
    "Simandoux (Shaly Sands)": "simandoux",
    "Indonesia / Poupon-Leveaux": "indonesia",
}


# ---------------------------------------------------------------------------
# Net Pay Determination
# ---------------------------------------------------------------------------

def compute_net_pay(
    df: pd.DataFrame,
    vshale_col: str,
    porosity_col: str,
    sw_col: str,
    vsh_cutoff: float = 0.50,
    phi_cutoff: float = 0.06,
    sw_cutoff: float = 0.60,
    depth_col: str = "DEPTH",
) -> pd.DataFrame:
    """
    Determine net reservoir and net pay using cutoffs.

    Ref: Worthington (2010), SPE 131529.

    Net Reservoir = intervals passing Vsh AND porosity cutoffs.
    Net Pay = Net Reservoir intervals also passing Sw cutoff.

    Typical cutoffs (vary by formation):
        Vshale:   0.30 - 0.50
        Porosity: 0.05 - 0.10
        Sw:       0.50 - 0.70
    """
    result = df.copy()

    result["NET_RESERVOIR"] = (
        (result[vshale_col] <= vsh_cutoff) &
        (result[porosity_col] >= phi_cutoff)
    ).astype(int)

    result["NET_PAY"] = (
        (result["NET_RESERVOIR"] == 1) &
        (result[sw_col] <= sw_cutoff)
    ).astype(int)

    return result


def compute_net_pay_summary(
    df: pd.DataFrame,
    depth_col: str = "DEPTH",
) -> dict:
    """Compute net pay thickness and net-to-gross ratio."""
    if len(df) < 2:
        return {"gross_thickness": 0, "net_reservoir": 0, "net_pay": 0, "ntg": 0}

    # Estimate depth step
    depth_step = df[depth_col].diff().median()
    if pd.isna(depth_step) or depth_step <= 0:
        depth_step = 0.5  # default half-foot

    gross = df[depth_col].max() - df[depth_col].min()
    net_res = df["NET_RESERVOIR"].sum() * abs(depth_step)
    net_pay = df["NET_PAY"].sum() * abs(depth_step)
    ntg = net_pay / gross if gross > 0 else 0

    return {
        "gross_thickness": round(gross, 1),
        "net_reservoir": round(net_res, 1),
        "net_pay": round(net_pay, 1),
        "ntg_ratio": round(ntg, 3),
    }


# ---------------------------------------------------------------------------
# Volumetric Reserves Estimation (SPE-PRMS Deterministic)
# ---------------------------------------------------------------------------

def ooip_volumetric(
    area_acres: float,
    net_pay_ft: float,
    porosity: float,
    sw: float,
    boi: float = 1.2,
) -> float:
    """
    Original Oil In Place – volumetric method.
    Ref: SPE-PRMS (2018); Craft & Hawkins.

    OOIP (STB) = 7758 * A * h * PHI * (1 - Sw) / Boi

    Where:
        7758 = bbl per acre-ft
        A = drainage area (acres)
        h = net pay thickness (ft)
        PHI = average porosity (fraction)
        Sw = average water saturation (fraction)
        Boi = initial oil formation volume factor (rb/stb)
    """
    return 7758.0 * area_acres * net_pay_ft * porosity * (1.0 - sw) / boi


def ogip_volumetric(
    area_acres: float,
    net_pay_ft: float,
    porosity: float,
    sw: float,
    bgi: float = 0.005,
) -> float:
    """
    Original Gas In Place – volumetric method.
    Ref: SPE-PRMS (2018).

    OGIP (SCF) = 43560 * A * h * PHI * (1 - Sw) / Bgi

    Where:
        43560 = cubic feet per acre-ft
        A = drainage area (acres)
        h = net pay thickness (ft)
        Bgi = initial gas formation volume factor (rcf/scf)
    """
    return 43560.0 * area_acres * net_pay_ft * porosity * (1.0 - sw) / bgi


# ---------------------------------------------------------------------------
# Interpretation Summary / Report
# ---------------------------------------------------------------------------

def generate_interpretation_summary(
    df: pd.DataFrame,
    net_pay_stats: dict,
    detected_curves: dict,
    vshale_col: str = None,
    porosity_col: str = None,
    sw_col: str = None,
) -> str:
    """Generate a text summary of the petrophysical interpretation."""
    lines = []
    lines.append("=" * 60)
    lines.append("PETROPHYSICAL INTERPRETATION SUMMARY")
    lines.append("Following SPE-PRMS / AAPG / SPWLA Standards")
    lines.append("=" * 60)

    # Available curves
    lines.append("\n--- Available Log Curves ---")
    for curve_type, col_name in detected_curves.items():
        lines.append(f"  {curve_type:25s} -> {col_name}")

    # Depth range
    if "DEPTH" in df.columns:
        lines.append(f"\n--- Depth Range ---")
        lines.append(f"  Top:    {df['DEPTH'].min():.1f}")
        lines.append(f"  Bottom: {df['DEPTH'].max():.1f}")

    # Vshale statistics
    if vshale_col and vshale_col in df.columns:
        vsh = df[vshale_col]
        lines.append(f"\n--- Shale Volume (Vshale) ---")
        lines.append(f"  Mean:   {vsh.mean():.3f}")
        lines.append(f"  Median: {vsh.median():.3f}")
        lines.append(f"  P10:    {vsh.quantile(0.10):.3f}")
        lines.append(f"  P90:    {vsh.quantile(0.90):.3f}")

    # Porosity statistics
    if porosity_col and porosity_col in df.columns:
        phi = df[porosity_col]
        lines.append(f"\n--- Porosity ---")
        lines.append(f"  Mean:   {phi.mean():.3f}")
        lines.append(f"  Median: {phi.median():.3f}")
        lines.append(f"  P10:    {phi.quantile(0.10):.3f}")
        lines.append(f"  P90:    {phi.quantile(0.90):.3f}")

    # Sw statistics
    if sw_col and sw_col in df.columns:
        sw = df[sw_col]
        lines.append(f"\n--- Water Saturation (Sw) ---")
        lines.append(f"  Mean:   {sw.mean():.3f}")
        lines.append(f"  Median: {sw.median():.3f}")
        lines.append(f"  P10:    {sw.quantile(0.10):.3f}")
        lines.append(f"  P90:    {sw.quantile(0.90):.3f}")
        avg_sh = 1.0 - sw.mean()
        lines.append(f"  Avg Hydrocarbon Sat (1-Sw): {avg_sh:.3f}")

    # Net pay
    lines.append(f"\n--- Net Pay Analysis (SPE 131529) ---")
    lines.append(f"  Gross Thickness: {net_pay_stats['gross_thickness']:.1f}")
    lines.append(f"  Net Reservoir:   {net_pay_stats['net_reservoir']:.1f}")
    lines.append(f"  Net Pay:         {net_pay_stats['net_pay']:.1f}")
    lines.append(f"  Net-to-Gross:    {net_pay_stats['ntg_ratio']:.3f}")

    # Qualitative assessment
    lines.append(f"\n--- Reservoir Quality Assessment ---")
    avg_phi = df[porosity_col].mean() if porosity_col and porosity_col in df.columns else 0
    avg_sw = df[sw_col].mean() if sw_col and sw_col in df.columns else 1
    ntg = net_pay_stats["ntg_ratio"]

    quality = _assess_reservoir_quality(avg_phi, avg_sw, ntg)
    for item in quality:
        lines.append(f"  {item}")

    lines.append("\n" + "=" * 60)
    lines.append("NOTE: This is an automated screening interpretation.")
    lines.append("Final reserves booking must incorporate geological,")
    lines.append("engineering, and economic analyses per SPE-PRMS (2018).")
    lines.append("=" * 60)

    return "\n".join(lines)


def _assess_reservoir_quality(avg_phi: float, avg_sw: float, ntg: float) -> list:
    """Provide qualitative reservoir quality indicators."""
    items = []

    # Porosity quality
    if avg_phi >= 0.20:
        items.append("Porosity: EXCELLENT (>20%)")
    elif avg_phi >= 0.15:
        items.append("Porosity: GOOD (15-20%)")
    elif avg_phi >= 0.10:
        items.append("Porosity: MODERATE (10-15%)")
    elif avg_phi >= 0.05:
        items.append("Porosity: LOW (5-10%) – tight reservoir")
    else:
        items.append("Porosity: VERY LOW (<5%) – may not be commercial")

    # Hydrocarbon saturation
    sh = 1.0 - avg_sw
    if sh >= 0.70:
        items.append(f"Hydrocarbon Saturation: HIGH ({sh:.0%})")
    elif sh >= 0.50:
        items.append(f"Hydrocarbon Saturation: MODERATE ({sh:.0%})")
    elif sh >= 0.30:
        items.append(f"Hydrocarbon Saturation: LOW ({sh:.0%})")
    else:
        items.append(f"Hydrocarbon Saturation: VERY LOW ({sh:.0%}) – likely water-bearing")

    # Net-to-gross
    if ntg >= 0.70:
        items.append(f"Net-to-Gross: EXCELLENT ({ntg:.1%})")
    elif ntg >= 0.50:
        items.append(f"Net-to-Gross: GOOD ({ntg:.1%})")
    elif ntg >= 0.30:
        items.append(f"Net-to-Gross: MODERATE ({ntg:.1%})")
    else:
        items.append(f"Net-to-Gross: LOW ({ntg:.1%}) – heterogeneous reservoir")

    return items
