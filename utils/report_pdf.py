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
from pathlib import Path

import pandas as pd
from fpdf import FPDF


# Common paths where DejaVu fonts live on Linux
_DEJAVU_PATHS = [
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/dejavu"),
    Path("/usr/share/fonts/truetype"),
]


def _find_font(name: str) -> str | None:
    """Locate a DejaVu TTF font file on the system."""
    for d in _DEJAVU_PATHS:
        p = d / name
        if p.exists():
            return str(p)
    return None


class _ReportPDF(FPDF):
    """Custom PDF with header/footer branding and Unicode font support."""

    _title_text: str = "Well Log Interpretation Report"
    _fonts_loaded: bool = False
    _use_unicode: bool = False

    def _load_fonts(self):
        if self._fonts_loaded:
            return
        self._fonts_loaded = True
        regular = _find_font("DejaVuSans.ttf")
        bold = _find_font("DejaVuSans-Bold.ttf")
        oblique = _find_font("DejaVuSans-Oblique.ttf")
        mono = _find_font("DejaVuSansMono.ttf")
        if regular and bold:
            self.add_font("DejaVu", "", regular)
            self.add_font("DejaVu", "B", bold)
            if oblique:
                self.add_font("DejaVu", "I", oblique)
            else:
                self.add_font("DejaVu", "I", regular)
            if mono:
                self.add_font("DejaVuMono", "", mono)
            self._use_unicode = True

    @property
    def _body_font(self):
        return "DejaVu" if self._use_unicode else "Helvetica"

    @property
    def _mono_font(self):
        return "DejaVuMono" if self._use_unicode else "Courier"

    def _safe(self, text: str) -> str:
        """Sanitize text if using non-Unicode fonts."""
        if self._use_unicode:
            return text
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def header(self):
        self._load_fonts()
        self.set_font(self._body_font, "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, self._title_text, align="L")
        self.cell(0, 8, datetime.now().strftime("%Y-%m-%d"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(self._body_font, "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # -- helpers -------------------------------------------------------------

    def section_title(self, title: str):
        self.set_font(self._body_font, "B", 13)
        self.set_text_color(0, 70, 140)
        self.ln(4)
        self.cell(0, 9, self._safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 70, 140)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(3)

    def body_text(self, text: str):
        self.set_font(self._body_font, "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, self._safe(text))
        self.ln(2)

    def metric_row(self, label: str, value: str):
        self.set_font(self._body_font, "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(70, 6, label)
        self.set_font(self._body_font, "", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    def _available_height(self) -> float:
        """Return remaining vertical space (mm) on the current page."""
        return self.h - self.get_y() - 20  # 20 mm bottom margin

    def body_text_fit_page(self, text: str, font_family: str | None = None,
                           style: str = "", max_size: float = 10,
                           min_size: float = 6):
        """Render *text* so it fits entirely on the current page.

        Starts at *max_size* pt and shrinks by 0.5 pt until the text
        fits the available vertical space (down to *min_size*).  If it
        still overflows at the minimum size the text is truncated.
        """
        font_family = font_family or self._body_font
        avail = self._available_height()

        # Try decreasing font sizes until the text fits
        for sz_x10 in range(int(max_size * 10), int(min_size * 10) - 1, -5):
            sz = sz_x10 / 10
            lh = max(sz * 0.45, 3.0)  # line height in mm
            self.set_font(font_family, style, sz)
            # Estimate height via a dry-run multi_cell
            lines = self.multi_cell(
                0, lh, self._safe(text), dry_run=True, output="LINES"
            )
            needed = len(lines) * lh
            if needed <= avail:
                self.set_text_color(30, 30, 30)
                self.multi_cell(0, lh, self._safe(text))
                self.ln(2)
                return

        # At minimum size – truncate line-by-line until it fits
        sz = min_size
        lh = max(sz * 0.45, 3.0)
        self.set_font(font_family, style, sz)
        max_lines = int(avail / lh) - 1  # reserve one line for note
        lines = self.multi_cell(
            0, lh, self._safe(text), dry_run=True, output="LINES"
        )
        truncated = lines[:max_lines]
        self.set_text_color(30, 30, 30)
        for ln_text in truncated:
            self.cell(0, lh, ln_text, new_x="LMARGIN", new_y="NEXT")
        self.set_font(font_family, "I", sz)
        self.set_text_color(128, 128, 128)
        self.cell(0, lh, "[... truncated to fit page]",
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


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
    font = pdf._body_font
    pdf.set_font(font, "B", 26)
    pdf.set_text_color(0, 70, 140)
    pdf.cell(0, 14, "Well Log Interpretation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font(font, "", 12)
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
        pdf.body_text_fit_page(ai_interpretation, max_size=10, min_size=6)

    # -- Technical Report ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("Technical Petrophysical Report")
    pdf.body_text_fit_page(technical_report, font_family=pdf._mono_font,
                           max_size=8, min_size=5)

    # -- Output --------------------------------------------------------------
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
