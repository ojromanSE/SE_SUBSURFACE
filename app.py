"""
Subsurface Log Interpreter – Streamlit Application

A tool for anyone to upload well logs and get a plain-English
interpretation. No petrophysics knowledge required.

Usage:
    streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
import numpy as np

from utils.parsers import (
    parse_las,
    parse_csv_excel,
    parse_pdf,
    extract_pdf_images,
    detect_log_curves,
)
from utils.petrophysics import (
    auto_interpret,
    compute_net_pay_summary,
    generate_verbal_interpretation,
    generate_interpretation_summary,
    ooip_volumetric,
    ogip_volumetric,
    VSHALE_METHODS,
    SW_METHODS,
    vshale_linear,
    porosity_density,
    porosity_sonic,
    porosity_neutron_density,
    effective_porosity,
    sw_archie,
    sw_simandoux,
    sw_indonesia,
    compute_net_pay,
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
    "Upload a well log file and get an **instant plain-English interpretation**. "
    "No petrophysics knowledge required."
)

# ---------------------------------------------------------------------------
# Sidebar – File Upload
# ---------------------------------------------------------------------------
st.sidebar.header("Upload Log File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a well log file",
    type=["las", "csv", "xlsx", "xls", "pdf"],
    help="Supported: LAS, CSV, Excel, PDF (text tables or scanned images)",
)

if uploaded_file is None:
    st.info(
        "**Getting started:** Upload a well log file using the sidebar.\n\n"
        "Supported formats:\n"
        "- **LAS** – Log ASCII Standard (preferred)\n"
        "- **CSV / Excel** – Tabular log data with a depth column\n"
        "- **PDF** – Text-based tables *or* scanned raster images\n\n"
        "The tool will **automatically interpret** the data and give you a "
        "plain-English summary — no adjustments needed."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Parse the uploaded file
# ---------------------------------------------------------------------------
file_bytes = uploaded_file.read()
filename = uploaded_file.name.lower()

df = pd.DataFrame()
pdf_images = []
is_raster_pdf = False

try:
    if filename.endswith(".las"):
        df = parse_las(file_bytes, filename)
    elif filename.endswith((".csv", ".xlsx", ".xls")):
        df = parse_csv_excel(file_bytes, filename)
    elif filename.endswith(".pdf"):
        df = parse_pdf(file_bytes, filename)
        if df.empty:
            # Raster / scanned PDF – extract images instead
            pdf_images = extract_pdf_images(file_bytes)
            is_raster_pdf = True
    else:
        st.error(f"Unsupported file format: {filename}")
        st.stop()
except Exception as e:
    st.error(f"Error parsing file: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Handle raster PDFs (images only)
# ---------------------------------------------------------------------------
if is_raster_pdf:
    st.success(f"Loaded **{uploaded_file.name}** — scanned/raster PDF with {len(pdf_images)} page(s)")

    tab_images, tab_raw_data = st.tabs(["Log Images", "Upload Digitized Data"])

    with tab_images:
        st.subheader("Scanned Well Log Images")
        st.markdown(
            "This PDF contains **scanned images** rather than digital data. "
            "The log images are displayed below for visual inspection."
        )
        for i, img in enumerate(pdf_images):
            st.image(img, caption=f"Page {i + 1}", use_container_width=True)

        st.markdown("---")
        st.markdown(
            "**To get a full interpretation**, you can:\n"
            "1. Digitize these logs using a tool like Neuralog or DigitizeIt\n"
            "2. Export the data as CSV or LAS\n"
            "3. Re-upload the digitized file here for automatic interpretation"
        )

    with tab_raw_data:
        st.subheader("Upload Digitized Data")
        st.markdown(
            "If you have a digitized version of this log (CSV, LAS, or Excel), "
            "upload it here for interpretation."
        )
        digitized_file = st.file_uploader(
            "Upload digitized data",
            type=["las", "csv", "xlsx", "xls"],
            key="digitized_upload",
        )
        if digitized_file:
            dig_bytes = digitized_file.read()
            dig_name = digitized_file.name.lower()
            try:
                if dig_name.endswith(".las"):
                    df = parse_las(dig_bytes, dig_name)
                else:
                    df = parse_csv_excel(dig_bytes, dig_name)
                is_raster_pdf = False  # Now we have data
                st.success(f"Loaded digitized data: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error parsing digitized file: {e}")

    if is_raster_pdf:
        st.stop()  # No numeric data to interpret

# ---------------------------------------------------------------------------
# We have numeric data – proceed with interpretation
# ---------------------------------------------------------------------------
detected = detect_log_curves(df)

st.success(
    f"Loaded **{uploaded_file.name}** — {len(df)} data points, {len(df.columns)} curves detected"
)

# Auto-run interpretation
result = auto_interpret(df, detected)
net_stats = compute_net_pay_summary(result)

# Store in session state
st.session_state["result_df"] = result
st.session_state["net_stats"] = net_stats
st.session_state["detected"] = detected

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_verbal, tab_plot, tab_data, tab_volumetrics, tab_advanced = st.tabs([
    "Interpretation",
    "Log Plot",
    "Raw Data",
    "Quick Volumetrics",
    "Advanced Settings",
])

# ---------------------------------------------------------------------------
# Tab 1: Plain-English Verbal Interpretation (PRIMARY)
# ---------------------------------------------------------------------------
with tab_verbal:
    st.subheader("What does this well log tell us?")

    verbal = generate_verbal_interpretation(result, net_stats, detected)
    st.text(verbal)

    # Key metrics at a glance
    st.markdown("---")
    st.markdown("### Key Numbers at a Glance")
    mc1, mc2, mc3, mc4 = st.columns(4)

    avg_phi = result["PHIE"].mean() if "PHIE" in result.columns else 0
    avg_sw = result["SW"].mean() if "SW" in result.columns else 1
    avg_vsh = result["VSHALE"].mean() if "VSHALE" in result.columns else 0.5

    mc1.metric("Avg Porosity", f"{avg_phi:.1%}")
    mc2.metric("Avg Water Saturation", f"{avg_sw:.1%}")
    mc3.metric("Net Pay", f"{net_stats['net_pay']:.0f}")
    mc4.metric("Net-to-Gross", f"{net_stats['ntg_ratio']:.1%}")

    # Download
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download Interpretation (TXT)",
            verbal,
            file_name="well_interpretation.txt",
            mime="text/plain",
        )
    with col_dl2:
        technical_report = generate_interpretation_summary(
            result, net_stats, detected,
            vshale_col="VSHALE", porosity_col="PHIE", sw_col="SW",
        )
        st.download_button(
            "Download Technical Report (TXT)",
            technical_report,
            file_name="technical_report.txt",
            mime="text/plain",
        )

# ---------------------------------------------------------------------------
# Tab 2: Log Plot
# ---------------------------------------------------------------------------
with tab_plot:
    st.subheader("Well Log Display")

    fig = plot_triple_combo(
        result, detected,
        vshale_col="VSHALE",
        porosity_col="PHIE",
        sw_col="SW",
        net_pay_col="NET_PAY",
    )
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tab 3: Raw Data
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Raw & Interpreted Data")
    st.dataframe(result, use_container_width=True, height=500)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Columns:**", list(result.columns))
    with col2:
        st.write("**Statistics:**")
        st.dataframe(result.describe().round(4), use_container_width=True)

    csv_buf = io.StringIO()
    result.to_csv(csv_buf, index=False)
    st.download_button(
        "Download Full Data (CSV)",
        csv_buf.getvalue(),
        file_name="interpreted_log_data.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Tab 4: Quick Volumetrics
# ---------------------------------------------------------------------------
with tab_volumetrics:
    st.subheader("Quick Reserves Estimate")
    st.markdown(
        "A rough estimate of how much oil or gas might be in this reservoir. "
        "Only the **drainage area** is needed from you — everything else comes from the log."
    )

    pay_mask = result["NET_PAY"] == 1
    if pay_mask.sum() > 0:
        avg_phi_pay = result.loc[pay_mask, "PHIE"].mean()
        avg_sw_pay = result.loc[pay_mask, "SW"].mean()
    else:
        avg_phi_pay = result["PHIE"].mean()
        avg_sw_pay = result["SW"].mean()

    fluid_type = st.radio("What fluid are you looking for?", ["Oil", "Gas"], horizontal=True)

    area = st.number_input(
        "Drainage area (acres) — how large is the field?",
        value=160.0, step=10.0,
        help="A typical well spacing unit is 40-640 acres. If unsure, 160 is a common default."
    )

    net_pay_ft = net_stats["net_pay"]

    if fluid_type == "Oil":
        ooip = ooip_volumetric(area, net_pay_ft, avg_phi_pay, avg_sw_pay, 1.2)
        reserves = ooip * 0.20  # 20% RF default

        rc1, rc2 = st.columns(2)
        rc1.metric("Oil In Place (OOIP)", f"{ooip:,.0f} STB")
        rc2.metric("Estimated Recoverable (20% RF)", f"{reserves:,.0f} STB")

        if reserves > 0:
            st.markdown(
                f"**In plain terms:** There could be roughly **{ooip / 1000:,.0f} thousand barrels** "
                f"of oil in place. With typical primary recovery (~20%), you might recover about "
                f"**{reserves / 1000:,.0f} thousand barrels**."
            )
    else:
        ogip = ogip_volumetric(area, net_pay_ft, avg_phi_pay, avg_sw_pay, 0.005)
        reserves = ogip * 0.70  # 70% RF default

        rc1, rc2 = st.columns(2)
        rc1.metric("Gas In Place (OGIP)", f"{ogip / 1e6:,.1f} MMSCF")
        rc2.metric("Estimated Recoverable (70% RF)", f"{reserves / 1e6:,.1f} MMSCF")

        if reserves > 0:
            st.markdown(
                f"**In plain terms:** There could be roughly **{ogip / 1e6:,.1f} million cubic feet** "
                f"of gas in place. With typical gas recovery (~70%), you might recover about "
                f"**{reserves / 1e6:,.1f} million cubic feet**."
            )

    st.markdown("---")
    st.caption(
        "**Disclaimer:** These are rough screening estimates only. "
        "Actual reserves require full engineering analysis, economic evaluation, "
        "and classification per SPE-PRMS (2018) guidelines."
    )

# ---------------------------------------------------------------------------
# Tab 5: Advanced Settings (for experts who want to tweak)
# ---------------------------------------------------------------------------
with tab_advanced:
    st.subheader("Advanced Settings")
    st.markdown(
        "The interpretation above used **automatic defaults**. "
        "If you're a petrophysicist and want to adjust parameters, you can re-run below."
    )

    with st.expander("Petrophysical Parameters", expanded=False):
        col_vsh, col_phi, col_sw = st.columns(3)

        with col_vsh:
            st.markdown("**Shale Volume**")
            if "GR" in detected:
                gr_data = df[detected["GR"]].dropna()
                gr_clean = st.number_input(
                    "GR clean (API)", value=float(gr_data.quantile(0.05)), step=1.0,
                )
                gr_shale = st.number_input(
                    "GR shale (API)", value=float(gr_data.quantile(0.95)), step=1.0,
                )
                vsh_method = st.selectbox("Vshale method", list(VSHALE_METHODS.keys()))
            else:
                st.warning("No GR curve detected.")
                gr_clean, gr_shale, vsh_method = 0, 150, "Linear (IGR)"

        with col_phi:
            st.markdown("**Porosity**")
            porosity_method = st.selectbox(
                "Porosity method",
                ["Density", "Sonic (Wyllie)", "Neutron-Density Crossplot"],
            )
            rho_matrix = st.number_input("Matrix density (g/cc)", value=2.65, step=0.01, format="%.2f")
            rho_fluid = st.number_input("Fluid density (g/cc)", value=1.0, step=0.01, format="%.2f")
            dt_matrix = st.number_input("Matrix DT (us/ft)", value=55.5, step=0.5, format="%.1f")
            dt_fluid = st.number_input("Fluid DT (us/ft)", value=189.0, step=1.0, format="%.1f")

        with col_sw:
            st.markdown("**Water Saturation**")
            sw_method_name = st.selectbox("Sw method", list(SW_METHODS.keys()))
            rw = st.number_input("Rw (ohm-m)", value=0.05, step=0.01, format="%.3f")
            a_tort = st.number_input("a (tortuosity)", value=1.0, step=0.1, format="%.2f")
            m_cem = st.number_input("m (cementation)", value=2.0, step=0.1, format="%.2f")
            n_sat = st.number_input("n (saturation)", value=2.0, step=0.1, format="%.2f")
            rsh = st.number_input("Rsh (ohm-m)", value=5.0, step=0.5, format="%.1f")

    with st.expander("Net Pay Cutoffs", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            vsh_cutoff = st.slider("Vshale cutoff", 0.0, 1.0, 0.40, 0.05)
        with cc2:
            phi_cutoff = st.slider("Porosity cutoff", 0.0, 0.30, 0.08, 0.01)
        with cc3:
            sw_cutoff = st.slider("Sw cutoff", 0.0, 1.0, 0.60, 0.05)

    if st.button("Re-run with Custom Parameters", type="primary", use_container_width=True):
        custom = df.copy()

        # Vshale
        if "GR" in detected:
            igr = vshale_linear(custom[detected["GR"]], gr_clean, gr_shale)
            vsh_func = VSHALE_METHODS[vsh_method]
            custom["VSHALE"] = vsh_func(igr)
        else:
            custom["VSHALE"] = 0.5

        # Porosity
        if porosity_method == "Density" and "DENSITY" in detected:
            custom["PHIT"] = porosity_density(custom[detected["DENSITY"]], rho_matrix, rho_fluid)
        elif porosity_method == "Sonic (Wyllie)" and "SONIC" in detected:
            custom["PHIT"] = porosity_sonic(custom[detected["SONIC"]], dt_matrix, dt_fluid)
        elif porosity_method == "Neutron-Density Crossplot" and "NEUTRON" in detected and "DENSITY" in detected:
            phid = porosity_density(custom[detected["DENSITY"]], rho_matrix, rho_fluid)
            phin = custom[detected["NEUTRON"]]
            custom["PHIT"] = porosity_neutron_density(phin, phid)
        else:
            custom["PHIT"] = 0.10

        custom["PHIE"] = effective_porosity(custom["PHIT"], custom["VSHALE"])

        # Sw
        if "RESISTIVITY_DEEP" in detected:
            rt = custom[detected["RESISTIVITY_DEEP"]]
            sw_key = SW_METHODS[sw_method_name]
            if sw_key == "archie":
                custom["SW"] = sw_archie(rt, custom["PHIE"], rw, a_tort, m_cem, n_sat)
            elif sw_key == "simandoux":
                custom["SW"] = sw_simandoux(rt, custom["PHIE"], custom["VSHALE"], rw, rsh, a_tort, m_cem, n_sat)
            elif sw_key == "indonesia":
                custom["SW"] = sw_indonesia(rt, custom["PHIE"], custom["VSHALE"], rw, rsh, a_tort, m_cem, n_sat)
        else:
            custom["SW"] = 0.50

        custom = compute_net_pay(custom, "VSHALE", "PHIE", "SW", vsh_cutoff, phi_cutoff, sw_cutoff)
        custom_stats = compute_net_pay_summary(custom)

        st.session_state["result_df"] = custom
        st.session_state["net_stats"] = custom_stats

        st.success("Custom interpretation complete! Switch to other tabs to see updated results.")

    st.markdown("---")
    st.markdown("**Column Mapping** (auto-detected)")
    if detected:
        for curve_type, col_name in detected.items():
            st.write(f"**{curve_type}**: {col_name}")
