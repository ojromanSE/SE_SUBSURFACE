"""
PDF report generation for well log interpretations.

Uses fpdf2 to build a formal, multi-section PDF that combines:
  - Verbal (plain-English) interpretation
  - Key metrics summary
  - Technical petrophysical report
  - Log plot figure (embedded as image)
  - AI-enhanced interpretation (if available)
"""

import io
import tempfile
from datetime import datetime

import pandas as pd
from fpdf import FPDF


def _sanitize(text: str) -> str:
    """Replace Unicode characters unsupported by built-in PDF fonts."""
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "--",  # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "*",   # bullet
        "\u2026": "...", # ellipsis
        "\u00b0": "deg", # degree
        "\u00b1": "+/-", # plus-minus
        "\u2265": ">=",  # greater-equal
        "\u2264": "<=",  # less-equal
        "\u03c6": "phi", # phi
        "\u03a6": "Phi", # Phi
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Fallback: encode to latin-1, replacing anything still unsupported
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _ReportPDF(FPDF):
    """Custom PDF with header/footer branding."""

    _title_text: str = "Well Log Interpretation Report"

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, self._title_text, align="L")
        self.cell(0, 8, datetime.now().strftime("%Y-%m-%d"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # -- helpers -------------------------------------------------------------

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0, 70, 140)
        self.ln(4)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 70, 140)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(3)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, _sanitize(text))
        self.ln(2)

    def metric_row(self, label: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(70, 6, label)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")


def generate_report_pdf(
    verbal: str,
    technical_report: str,
    net_stats: dict,
    result_df: pd.DataFrame,
    fig=None,
    ai_interpretation: str | None = None,
) -> bytes:
    """Build a formal PDF report and return its bytes.

    Parameters
    ----------
    verbal : str
        Plain-English interpretation text.
    technical_report : str
        Detailed petrophysical summary text.
    net_stats : dict
        Net-pay statistics dict (from compute_net_pay_summary).
    result_df : pd.DataFrame
        Interpreted log DataFrame (used for key metrics).
    fig : matplotlib.figure.Figure, optional
        Log plot figure to embed.
    ai_interpretation : str, optional
        AI-generated interpretation text to include.

    Returns
    -------
    bytes
        PDF file content.
    """
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()

    # -- Cover / title page --------------------------------------------------
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 14, "Well Log Interpretation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Data points: {len(result_df)}  |  Curves: {len(result_df.columns)}", align="C", new_x="LMARGIN", new_y="NEXT")

    # -- Key Metrics ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title("Key Metrics")

    avg_phi = result_df["PHIE"].mean() if "PHIE" in result_df.columns else 0
    avg_sw = result_df["SW"].mean() if "SW" in result_df.columns else 1
    avg_vsh = result_df["VSHALE"].mean() if "VSHALE" in result_df.columns else 0.5

    pdf.metric_row("Average Porosity:", f"{avg_phi:.1%}")
    pdf.metric_row("Average Water Saturation:", f"{avg_sw:.1%}")
    pdf.metric_row("Average Shale Volume:", f"{avg_vsh:.1%}")
    pdf.metric_row("Net Pay:", f"{net_stats['net_pay']:.1f}")
    pdf.metric_row("Net-to-Gross Ratio:", f"{net_stats['ntg_ratio']:.1%}")
    pdf.ln(4)

    # -- Verbal Interpretation -----------------------------------------------
    pdf.section_title("Plain-English Interpretation")
    pdf.body_text(verbal)

    # -- Log Plot ------------------------------------------------------------
    if fig is not None:
        pdf.add_page()
        pdf.section_title("Well Log Display")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
            img_w = pdf.w - 20  # 10 mm margin each side
            pdf.image(tmp.name, x=10, w=img_w)

    # -- AI Interpretation ---------------------------------------------------
    if ai_interpretation:
        pdf.add_page()
        pdf.section_title("AI-Enhanced Interpretation")
        pdf.body_text(ai_interpretation)

    # -- Technical Report ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("Technical Petrophysical Report")
    pdf.set_font("Courier", "", 8)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 4, _sanitize(technical_report))

    # -- Output --------------------------------------------------------------
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
