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
from utils.digitizer import (
    suggest_tracks,
    detect_log_area,
    digitize_log_image,
    annotate_image_with_tracks,
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
        try:
            df = parse_pdf(file_bytes, filename)
        except Exception:
            df = pd.DataFrame()
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
# Handle raster PDFs – built-in digitizer
# ---------------------------------------------------------------------------
if is_raster_pdf:
    st.success(f"Loaded **{uploaded_file.name}** — scanned/raster PDF with {len(pdf_images)} page(s)")

    tab_digitizer, tab_images, tab_raw_data = st.tabs([
        "Digitize Log",
        "View Pages",
        "Upload Digitized Data",
    ])

    # -- Tab: View raw pages --
    with tab_images:
        st.subheader("Scanned Well Log Pages")
        for i, img in enumerate(pdf_images):
            st.image(img, caption=f"Page {i + 1}", width="stretch")

    # -- Tab: Upload pre-digitized file --
    with tab_raw_data:
        st.subheader("Upload Digitized Data")
        st.markdown(
            "If you already have a digitized version (CSV, LAS, or Excel), upload it here."
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
                is_raster_pdf = False
                st.success(f"Loaded digitized data: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error parsing digitized file: {e}")

    # -- Tab: Built-in Digitizer (PRIMARY) --
    with tab_digitizer:
        st.subheader("Log Image Digitizer")
        st.markdown(
            "Extract numerical curve data directly from the scanned image. "
            "The tool will auto-detect the log tracks (columns). You just need to "
            "tell it the **depth range** and **what each track measures**."
        )

        if not pdf_images:
            st.warning("No images could be extracted from this PDF.")
        else:
            # --- Step 1: Select page ---
            if len(pdf_images) > 1:
                page_idx = st.selectbox(
                    "Select page to digitize",
                    range(len(pdf_images)),
                    format_func=lambda i: f"Page {i + 1}",
                )
            else:
                page_idx = 0

            work_img = pdf_images[page_idx]
            img_w, img_h = work_img.size

            # --- Step 2: Depth range ---
            st.markdown("### Step 1: Set the depth range")
            st.markdown(
                "Look at the depth scale on the log image. "
                "Enter the depth at the **top** and **bottom** of the log."
            )
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                depth_top = st.number_input("Depth at top of log", value=0.0, step=10.0, format="%.1f")
            with dc2:
                depth_bottom = st.number_input("Depth at bottom of log", value=1000.0, step=10.0, format="%.1f")
            with dc3:
                sample_interval = st.number_input(
                    "Sample interval", value=0.5, step=0.1, format="%.1f",
                    help="How often to sample (in depth units). 0.5 is typical.",
                )

            if depth_bottom <= depth_top:
                st.error("Bottom depth must be greater than top depth.")
                st.stop()

            # --- Step 3: Auto-detect tracks ---
            st.markdown("### Step 2: Define the log tracks")
            st.markdown(
                "The tool detected the column boundaries automatically. "
                "For each track, select **what curve it contains** and set the **scale range** "
                "(the min/max values printed at the top of each track on the log)."
            )

            auto_tracks = suggest_tracks(work_img)

            # Show annotated image with detected boundaries
            if auto_tracks:
                annotated = annotate_image_with_tracks(work_img, auto_tracks)
                st.image(annotated, caption="Detected track boundaries (colored lines)", width="stretch")
            else:
                st.image(work_img, caption="Original image", width="stretch")
                st.warning("Could not auto-detect tracks. Define them manually below.")

            # --- Step 4: Configure each track ---
            num_tracks = st.number_input(
                "Number of tracks to digitize",
                min_value=1,
                max_value=10,
                value=min(len(auto_tracks), 5) if auto_tracks else 3,
            )

            COMMON_CURVES = {
                "GR (Gamma Ray)": {"name": "GR", "min": 0.0, "max": 150.0},
                "ILD / RT (Deep Resistivity)": {"name": "ILD", "min": 0.2, "max": 2000.0},
                "ILS / RS (Shallow Resistivity)": {"name": "ILS", "min": 0.2, "max": 2000.0},
                "RHOB (Bulk Density)": {"name": "RHOB", "min": 1.95, "max": 2.95},
                "NPHI (Neutron Porosity)": {"name": "NPHI", "min": 0.45, "max": -0.15},
                "DT (Sonic)": {"name": "DT", "min": 140.0, "max": 40.0},
                "SP (Spontaneous Potential)": {"name": "SP", "min": -160.0, "max": 40.0},
                "CALI (Caliper)": {"name": "CALI", "min": 6.0, "max": 16.0},
                "Custom": {"name": "CUSTOM", "min": 0.0, "max": 100.0},
            }
            curve_options = list(COMMON_CURVES.keys())

            COLOR_OPTIONS = {
                "Darkest line (auto)": "dark",
                "Red curve": "red",
                "Blue curve": "blue",
                "Green curve": "green",
            }

            track_configs = []
            for i in range(num_tracks):
                with st.expander(f"Track {i + 1}", expanded=(i < 3)):
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        # Curve type selection
                        curve_choice = st.selectbox(
                            "Curve type",
                            curve_options,
                            key=f"curve_type_{i}",
                            index=min(i, len(curve_options) - 1),
                        )
                        defaults = COMMON_CURVES[curve_choice]

                        curve_name = defaults["name"]
                        if curve_choice == "Custom":
                            curve_name = st.text_input("Curve mnemonic", value="CUSTOM", key=f"custom_name_{i}")

                        color_choice = st.selectbox(
                            "Trace which color?",
                            list(COLOR_OPTIONS.keys()),
                            key=f"color_{i}",
                            help="Pick the color of the curve you want to trace in this track",
                        )

                    with tc2:
                        # Pixel boundaries
                        auto_left = auto_tracks[i]["left"] if i < len(auto_tracks) else int(img_w * i / num_tracks)
                        auto_right = auto_tracks[i]["right"] if i < len(auto_tracks) else int(img_w * (i + 1) / num_tracks)

                        left_px = st.number_input(
                            "Left boundary (pixels)", value=auto_left,
                            min_value=0, max_value=img_w, key=f"left_{i}",
                        )
                        right_px = st.number_input(
                            "Right boundary (pixels)", value=auto_right,
                            min_value=0, max_value=img_w, key=f"right_{i}",
                        )

                        val_min = st.number_input(
                            "Scale: left edge value", value=defaults["min"],
                            format="%.2f", key=f"vmin_{i}",
                            help="Value printed at the LEFT edge of this track",
                        )
                        val_max = st.number_input(
                            "Scale: right edge value", value=defaults["max"],
                            format="%.2f", key=f"vmax_{i}",
                            help="Value printed at the RIGHT edge of this track",
                        )

                    track_configs.append({
                        "left": int(left_px),
                        "right": int(right_px),
                        "curve_name": curve_name,
                        "min_value": val_min,
                        "max_value": val_max,
                        "color_channel": COLOR_OPTIONS[color_choice],
                    })

            # --- Step 5: Digitize ---
            st.markdown("### Step 3: Digitize")
            if st.button("Digitize Log Image", type="primary", width="stretch"):
                with st.spinner("Tracing curves from image..."):
                    digitized_df = digitize_log_image(
                        work_img,
                        track_configs,
                        depth_top,
                        depth_bottom,
                        sample_interval,
                    )

                if digitized_df.empty or len(digitized_df.columns) <= 1:
                    st.error("Could not extract any curves. Try adjusting the track boundaries.")
                else:
                    st.success(
                        f"Digitized **{len(digitized_df.columns) - 1} curves** "
                        f"over **{len(digitized_df)} depth points**"
                    )

                    # Preview the digitized data
                    st.markdown("**Preview of digitized data:**")
                    st.dataframe(digitized_df.head(20), width="stretch")

                    # Quick plot of digitized curves
                    st.markdown("**Digitized curves:**")
                    import matplotlib.pyplot as plt
                    n_curves = len(digitized_df.columns) - 1
                    if n_curves > 0:
                        fig, axes = plt.subplots(1, n_curves, figsize=(4 * n_curves, 10), sharey=True)
                        if n_curves == 1:
                            axes = [axes]
                        for idx, col in enumerate(digitized_df.columns[1:]):
                            ax = axes[idx]
                            ax.plot(digitized_df[col], digitized_df["DEPTH"], linewidth=0.8)
                            ax.set_title(col, fontsize=10, fontweight="bold")
                            ax.set_xlabel(col)
                            ax.invert_yaxis()
                            ax.grid(True, alpha=0.3)
                            if idx == 0:
                                ax.set_ylabel("Depth")
                        plt.tight_layout()
                        st.pyplot(fig)

                    # Store and continue to interpretation
                    st.session_state["digitized_df"] = digitized_df
                    df = digitized_df
                    is_raster_pdf = False

                    st.markdown("---")
                    st.markdown(
                        "The digitized data is ready. **Scroll down** or switch tabs to see "
                        "the full automatic interpretation."
                    )

                    # Download digitized data
                    csv_buf = io.StringIO()
                    digitized_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "Download Digitized Data (CSV)",
                        csv_buf.getvalue(),
                        file_name="digitized_log.csv",
                        mime="text/csv",
                    )

    # If still raster (user hasn't digitized yet), check session state
    if is_raster_pdf:
        if "digitized_df" in st.session_state:
            df = st.session_state["digitized_df"]
            is_raster_pdf = False
        else:
            st.stop()

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
    st.dataframe(result, width="stretch", height=500)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Columns:**", list(result.columns))
    with col2:
        st.write("**Statistics:**")
        st.dataframe(result.describe().round(4), width="stretch")

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

    if st.button("Re-run with Custom Parameters", type="primary", width="stretch"):
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
