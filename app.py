"""
Subsurface Log Interpreter – Streamlit Application

A tool for reservoir engineers to upload, interpret, and analyze
well logs following SPE-PRMS, AAPG, and SPWLA standards.

Usage:
    streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
import numpy as np

from utils.parsers import parse_las, parse_csv_excel, parse_pdf, detect_log_curves
from utils.petrophysics import (
    vshale_linear,
    VSHALE_METHODS,
    porosity_density,
    porosity_sonic,
    porosity_neutron_density,
    effective_porosity,
    sw_archie,
    sw_simandoux,
    sw_indonesia,
    SW_METHODS,
    compute_net_pay,
    compute_net_pay_summary,
    ooip_volumetric,
    ogip_volumetric,
    generate_interpretation_summary,
)
from utils.plotting import plot_triple_combo

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Subsurface Log Interpreter",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Subsurface Log Interpreter")
st.markdown(
    "Upload well logs (LAS, CSV, Excel, or PDF) for petrophysical interpretation "
    "following **SPE-PRMS**, **AAPG**, and **SPWLA** standards."
)

# ---------------------------------------------------------------------------
# Sidebar – File Upload
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload Log File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a well log file",
    type=["las", "csv", "xlsx", "xls", "pdf"],
    help="Supported formats: LAS, CSV, Excel (.xlsx/.xls), PDF (text-based tables)",
)

if uploaded_file is None:
    st.info(
        "**Getting started:** Upload a well log file using the sidebar.\n\n"
        "Supported formats:\n"
        "- **LAS** – Log ASCII Standard (preferred)\n"
        "- **CSV / Excel** – Tabular log data with a depth column\n"
        "- **PDF** – Text-based tables extracted from log reports\n\n"
        "The tool will automatically detect log curves (GR, resistivity, density, "
        "neutron, sonic, etc.) and run a full petrophysical interpretation."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Parse the uploaded file
# ---------------------------------------------------------------------------
file_bytes = uploaded_file.read()
filename = uploaded_file.name.lower()

try:
    if filename.endswith(".las"):
        df = parse_las(file_bytes, filename)
    elif filename.endswith((".csv", ".xlsx", ".xls")):
        df = parse_csv_excel(file_bytes, filename)
    elif filename.endswith(".pdf"):
        df = parse_pdf(file_bytes, filename)
    else:
        st.error(f"Unsupported file format: {filename}")
        st.stop()
except Exception as e:
    st.error(f"Error parsing file: {e}")
    st.stop()

# Detect available curves
detected = detect_log_curves(df)

st.success(f"Loaded **{uploaded_file.name}** — {len(df)} data points, {len(df.columns)} curves")

# ---------------------------------------------------------------------------
# Sidebar – Show detected curves
# ---------------------------------------------------------------------------
st.sidebar.header("2. Detected Curves")
if detected:
    for curve_type, col_name in detected.items():
        st.sidebar.write(f"**{curve_type}**: {col_name}")
else:
    st.sidebar.warning("No standard curves auto-detected. You can manually map columns below.")

# Allow manual column mapping if auto-detection misses something
st.sidebar.header("3. Manual Column Mapping")
st.sidebar.caption("Override auto-detected mappings if needed.")
all_cols = ["(none)"] + list(df.columns)

manual_gr = st.sidebar.selectbox(
    "Gamma Ray (GR)", all_cols,
    index=all_cols.index(detected["GR"]) if "GR" in detected else 0,
)
manual_rt = st.sidebar.selectbox(
    "Deep Resistivity (Rt)", all_cols,
    index=all_cols.index(detected["RESISTIVITY_DEEP"]) if "RESISTIVITY_DEEP" in detected else 0,
)
manual_rhob = st.sidebar.selectbox(
    "Bulk Density (RHOB)", all_cols,
    index=all_cols.index(detected["DENSITY"]) if "DENSITY" in detected else 0,
)
manual_nphi = st.sidebar.selectbox(
    "Neutron Porosity (NPHI)", all_cols,
    index=all_cols.index(detected["NEUTRON"]) if "NEUTRON" in detected else 0,
)
manual_dt = st.sidebar.selectbox(
    "Sonic (DT)", all_cols,
    index=all_cols.index(detected["SONIC"]) if "SONIC" in detected else 0,
)

# Update detected dict with manual overrides
if manual_gr != "(none)":
    detected["GR"] = manual_gr
if manual_rt != "(none)":
    detected["RESISTIVITY_DEEP"] = manual_rt
if manual_rhob != "(none)":
    detected["DENSITY"] = manual_rhob
if manual_nphi != "(none)":
    detected["NEUTRON"] = manual_nphi
if manual_dt != "(none)":
    detected["SONIC"] = manual_dt

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_data, tab_interp, tab_plot, tab_report, tab_volumetrics = st.tabs([
    "Raw Data",
    "Interpretation",
    "Log Plot",
    "Report",
    "Volumetrics",
])

# ---------------------------------------------------------------------------
# Tab 1: Raw Data
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Raw Log Data")
    st.dataframe(df, use_container_width=True, height=500)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
    with col2:
        st.write("**Statistics:**")
        st.dataframe(df.describe(), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Interpretation Parameters & Computation
# ---------------------------------------------------------------------------
with tab_interp:
    st.subheader("Petrophysical Interpretation Parameters")
    st.markdown(
        "Configure parameters below following **Asquith & Krygowski (2004)** "
        "and **Archie (1942)** conventions."
    )

    col_vsh, col_phi, col_sw = st.columns(3)

    # --- Vshale Parameters ---
    with col_vsh:
        st.markdown("#### Shale Volume")
        can_compute_vsh = "GR" in detected

        if can_compute_vsh:
            gr_data = df[detected["GR"]].dropna()
            gr_min_default = float(gr_data.quantile(0.05))
            gr_max_default = float(gr_data.quantile(0.95))

            gr_clean = st.number_input(
                "GR clean (API)", value=gr_min_default, step=1.0,
                help="Gamma ray reading in clean sand (low GR)"
            )
            gr_shale = st.number_input(
                "GR shale (API)", value=gr_max_default, step=1.0,
                help="Gamma ray reading in pure shale (high GR)"
            )
            vsh_method = st.selectbox(
                "Vshale method", list(VSHALE_METHODS.keys()),
                help="Larionov Tertiary is common for younger formations"
            )
        else:
            st.warning("No GR curve detected – cannot compute Vshale.")

    # --- Porosity Parameters ---
    with col_phi:
        st.markdown("#### Porosity")

        porosity_method = st.selectbox(
            "Porosity method",
            ["Density", "Sonic (Wyllie)", "Neutron-Density Crossplot"],
        )
        lithology = st.selectbox(
            "Lithology", ["Sandstone", "Limestone", "Dolomite"],
        )
        lith_params = {
            "Sandstone": {"rho_ma": 2.65, "dt_ma": 55.5},
            "Limestone": {"rho_ma": 2.71, "dt_ma": 47.5},
            "Dolomite": {"rho_ma": 2.87, "dt_ma": 43.5},
        }
        rho_matrix = st.number_input(
            "Matrix density (g/cc)", value=lith_params[lithology]["rho_ma"],
            step=0.01, format="%.2f",
        )
        rho_fluid = st.number_input(
            "Fluid density (g/cc)", value=1.0, step=0.01, format="%.2f",
        )
        dt_matrix = st.number_input(
            "Matrix DT (us/ft)", value=lith_params[lithology]["dt_ma"],
            step=0.5, format="%.1f",
        )
        dt_fluid = st.number_input(
            "Fluid DT (us/ft)", value=189.0, step=1.0, format="%.1f",
        )

    # --- Sw Parameters ---
    with col_sw:
        st.markdown("#### Water Saturation")
        sw_method_name = st.selectbox(
            "Sw method", list(SW_METHODS.keys()),
            help="Use Archie for clean sands, Simandoux/Indonesia for shaly sands"
        )
        rw = st.number_input(
            "Rw – formation water resistivity (ohm-m)",
            value=0.05, step=0.01, format="%.3f",
            help="Obtain from water analysis or SP log"
        )
        a_tort = st.number_input(
            "a – tortuosity factor",
            value=1.0, step=0.1, format="%.2f",
            help="1.0 (carbonates), 0.62 (Humble), 0.81 (sandstones)"
        )
        m_cem = st.number_input(
            "m – cementation exponent",
            value=2.0, step=0.1, format="%.2f",
        )
        n_sat = st.number_input(
            "n – saturation exponent",
            value=2.0, step=0.1, format="%.2f",
        )
        rsh = st.number_input(
            "Rsh – shale resistivity (ohm-m)",
            value=5.0, step=0.5, format="%.1f",
            help="Used for Simandoux and Indonesia methods"
        )

    # --- Cutoffs ---
    st.markdown("---")
    st.markdown("#### Net Pay Cutoffs (SPE 131529 – Worthington, 2010)")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        vsh_cutoff = st.slider("Vshale cutoff", 0.0, 1.0, 0.50, 0.05)
    with cc2:
        phi_cutoff = st.slider("Porosity cutoff", 0.0, 0.30, 0.06, 0.01)
    with cc3:
        sw_cutoff = st.slider("Sw cutoff", 0.0, 1.0, 0.60, 0.05)

    # ---- Run Interpretation ----
    st.markdown("---")
    run_btn = st.button("Run Interpretation", type="primary", use_container_width=True)

    if run_btn:
        result = df.copy()

        # --- Compute Vshale ---
        if can_compute_vsh:
            igr = vshale_linear(result[detected["GR"]], gr_clean, gr_shale)
            vsh_func = VSHALE_METHODS[vsh_method]
            result["VSHALE"] = vsh_func(igr)
        else:
            result["VSHALE"] = 0.5  # default if no GR

        # --- Compute Porosity ---
        can_compute_phi = False
        if porosity_method == "Density" and "DENSITY" in detected:
            result["PHIT"] = porosity_density(
                result[detected["DENSITY"]], rho_matrix, rho_fluid,
            )
            can_compute_phi = True
        elif porosity_method == "Sonic (Wyllie)" and "SONIC" in detected:
            result["PHIT"] = porosity_sonic(
                result[detected["SONIC"]], dt_matrix, dt_fluid,
            )
            can_compute_phi = True
        elif porosity_method == "Neutron-Density Crossplot" and "NEUTRON" in detected and "DENSITY" in detected:
            phid = porosity_density(result[detected["DENSITY"]], rho_matrix, rho_fluid)
            phin = result[detected["NEUTRON"]]
            result["PHIT"] = porosity_neutron_density(phin, phid)
            can_compute_phi = True
        else:
            st.warning(
                f"Cannot compute {porosity_method} porosity – required curve not found. "
                "Using a default porosity of 0.10."
            )
            result["PHIT"] = 0.10

        # Effective porosity
        result["PHIE"] = effective_porosity(result["PHIT"], result["VSHALE"])

        # --- Compute Sw ---
        can_compute_sw = "RESISTIVITY_DEEP" in detected and can_compute_phi
        if can_compute_sw:
            rt = result[detected["RESISTIVITY_DEEP"]]
            phi_for_sw = result["PHIE"]
            sw_method_key = SW_METHODS[sw_method_name]

            if sw_method_key == "archie":
                result["SW"] = sw_archie(rt, phi_for_sw, rw, a_tort, m_cem, n_sat)
            elif sw_method_key == "simandoux":
                result["SW"] = sw_simandoux(
                    rt, phi_for_sw, result["VSHALE"], rw, rsh, a_tort, m_cem, n_sat,
                )
            elif sw_method_key == "indonesia":
                result["SW"] = sw_indonesia(
                    rt, phi_for_sw, result["VSHALE"], rw, rsh, a_tort, m_cem, n_sat,
                )
        else:
            st.warning("Cannot compute Sw – missing deep resistivity or porosity. Using Sw=0.50.")
            result["SW"] = 0.50

        # --- Net Pay ---
        result = compute_net_pay(
            result, "VSHALE", "PHIE", "SW",
            vsh_cutoff, phi_cutoff, sw_cutoff,
        )
        net_stats = compute_net_pay_summary(result)

        # Store results in session state
        st.session_state["result_df"] = result
        st.session_state["net_stats"] = net_stats
        st.session_state["detected"] = detected

        st.success("Interpretation complete!")

        # Show summary table of computed curves
        st.markdown("**Computed Curves Summary:**")
        summary_cols = ["VSHALE", "PHIT", "PHIE", "SW"]
        summary_data = result[summary_cols].describe().round(4)
        st.dataframe(summary_data, use_container_width=True)

        # Net pay summary
        st.markdown("**Net Pay Summary:**")
        npc1, npc2, npc3, npc4 = st.columns(4)
        npc1.metric("Gross Thickness", f"{net_stats['gross_thickness']:.1f}")
        npc2.metric("Net Reservoir", f"{net_stats['net_reservoir']:.1f}")
        npc3.metric("Net Pay", f"{net_stats['net_pay']:.1f}")
        npc4.metric("Net-to-Gross", f"{net_stats['ntg_ratio']:.3f}")

# ---------------------------------------------------------------------------
# Tab 3: Log Plot
# ---------------------------------------------------------------------------
with tab_plot:
    st.subheader("Well Log Display – Triple Combo")

    if "result_df" not in st.session_state:
        st.info("Run the interpretation first (Interpretation tab) to generate plots.")
    else:
        result = st.session_state["result_df"]
        det = st.session_state["detected"]

        fig = plot_triple_combo(
            result, det,
            vshale_col="VSHALE",
            porosity_col="PHIE",
            sw_col="SW",
            net_pay_col="NET_PAY",
        )
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tab 4: Report
# ---------------------------------------------------------------------------
with tab_report:
    st.subheader("Interpretation Report")

    if "result_df" not in st.session_state:
        st.info("Run the interpretation first (Interpretation tab) to generate the report.")
    else:
        result = st.session_state["result_df"]
        net_stats = st.session_state["net_stats"]
        det = st.session_state["detected"]

        report = generate_interpretation_summary(
            result, net_stats, det,
            vshale_col="VSHALE",
            porosity_col="PHIE",
            sw_col="SW",
        )
        st.code(report, language=None)

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download Report (TXT)",
                report,
                file_name="interpretation_report.txt",
                mime="text/plain",
            )
        with col_dl2:
            csv_buf = io.StringIO()
            result.to_csv(csv_buf, index=False)
            st.download_button(
                "Download Interpreted Data (CSV)",
                csv_buf.getvalue(),
                file_name="interpreted_log_data.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Tab 5: Volumetrics
# ---------------------------------------------------------------------------
with tab_volumetrics:
    st.subheader("Volumetric Reserves Estimation (SPE-PRMS)")
    st.markdown(
        "Deterministic volumetric calculation using average petrophysical "
        "properties from the interpreted interval."
    )

    if "result_df" not in st.session_state:
        st.info("Run the interpretation first (Interpretation tab).")
    else:
        result = st.session_state["result_df"]
        net_stats = st.session_state["net_stats"]

        # Only use net pay intervals for averages
        pay_mask = result["NET_PAY"] == 1
        if pay_mask.sum() > 0:
            avg_phi = result.loc[pay_mask, "PHIE"].mean()
            avg_sw = result.loc[pay_mask, "SW"].mean()
        else:
            avg_phi = result["PHIE"].mean()
            avg_sw = result["SW"].mean()

        fluid_type = st.radio("Fluid Type", ["Oil", "Gas"], horizontal=True)

        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**From Log Interpretation (editable):**")
            net_pay_ft = st.number_input(
                "Net Pay (ft)", value=net_stats["net_pay"], step=1.0, format="%.1f",
            )
            avg_porosity = st.number_input(
                "Avg Porosity (fraction)", value=round(avg_phi, 4),
                step=0.01, format="%.4f",
            )
            avg_water_sat = st.number_input(
                "Avg Sw (fraction)", value=round(avg_sw, 4),
                step=0.01, format="%.4f",
            )

        with vc2:
            st.markdown("**Reservoir / Economic Parameters:**")
            area = st.number_input(
                "Drainage Area (acres)", value=160.0, step=10.0,
                help="Typical spacing: 40-640 acres depending on regulations"
            )
            if fluid_type == "Oil":
                boi = st.number_input(
                    "Boi – Oil FVF (rb/stb)", value=1.20, step=0.05, format="%.2f",
                )
                rf = st.number_input(
                    "Recovery Factor (%)", value=20.0, step=5.0,
                    help="Primary: 10-25%, Secondary: 25-40%, Tertiary: up to 60%"
                )
            else:
                bgi = st.number_input(
                    "Bgi – Gas FVF (rcf/scf)", value=0.005, step=0.001, format="%.4f",
                )
                rf = st.number_input(
                    "Recovery Factor (%)", value=70.0, step=5.0,
                    help="Gas: typically 50-85%"
                )

        if st.button("Calculate Volumetrics", type="primary", use_container_width=True):
            if fluid_type == "Oil":
                ooip = ooip_volumetric(area, net_pay_ft, avg_porosity, avg_water_sat, boi)
                reserves = ooip * (rf / 100.0)

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("OOIP", f"{ooip:,.0f} STB")
                rc2.metric("Reserves (EUR)", f"{reserves:,.0f} STB")
                rc3.metric("Reserves", f"{reserves / 1000:,.1f} MSTB")

                st.markdown("---")
                st.markdown("**Calculation Details:**")
                st.latex(r"OOIP = \frac{7758 \times A \times h \times \phi \times (1 - S_w)}{B_{oi}}")
                st.markdown(
                    f"= 7758 × {area:.0f} × {net_pay_ft:.1f} × {avg_porosity:.4f} "
                    f"× (1 - {avg_water_sat:.4f}) / {boi:.2f} = **{ooip:,.0f} STB**"
                )
            else:
                ogip = ogip_volumetric(area, net_pay_ft, avg_porosity, avg_water_sat, bgi)
                reserves = ogip * (rf / 100.0)

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("OGIP", f"{ogip / 1e6:,.1f} MMSCF")
                rc2.metric("Reserves (EUR)", f"{reserves / 1e6:,.1f} MMSCF")
                rc3.metric("Reserves", f"{reserves / 1e9:,.3f} BSCF")

                st.markdown("---")
                st.markdown("**Calculation Details:**")
                st.latex(r"OGIP = \frac{43560 \times A \times h \times \phi \times (1 - S_w)}{B_{gi}}")
                st.markdown(
                    f"= 43560 × {area:.0f} × {net_pay_ft:.1f} × {avg_porosity:.4f} "
                    f"× (1 - {avg_water_sat:.4f}) / {bgi:.4f} = **{ogip / 1e6:,.1f} MMSCF**"
                )

        st.markdown("---")
        st.caption(
            "**Disclaimer:** Volumetric estimates are for screening purposes only. "
            "Final reserves booking requires full subsurface integration, economic analysis, "
            "and classification per SPE-PRMS (2018) guidelines. Reserves categories (1P/2P/3P) "
            "should be assigned based on uncertainty analysis and project maturity."
        )
