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
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting symbols from text.

    Converts markdown headers, bold, italic, and bullet markers to
    plain text so the PDF renders cleanly.
    """
    # Remove header markers (### Header -> Header)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers (**text** or *text* or ***text***)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Remove underline-style bold (__text__)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    return text


# Brand colours (maroon / burgundy)
_MAROON = (100, 20, 40)
_MAROON_LIGHT = (140, 40, 60)
_GREY_DARK = (50, 50, 50)
_GREY_TEXT = (30, 30, 30)
_GREY_SUB = (80, 80, 80)
_GREY_MUTED = (128, 128, 128)

# Logo path (relative to this file)
_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "se_logo.png"

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
        # Logo in header (small)
        if _LOGO_PATH.exists():
            self.image(str(_LOGO_PATH), x=10, y=8, h=10)
            text_x = 22
        else:
            text_x = 10
        self.set_x(text_x)
        self.set_font(self._body_font, "B", 10)
        self.set_text_color(*_MAROON)
        self.cell(0, 8, self._title_text, align="L")
        self.cell(0, 8, datetime.now().strftime("%Y-%m-%d"),
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_MAROON)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(self._body_font, "I", 8)
        self.set_text_color(*_GREY_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # -- helpers -------------------------------------------------------------

    def section_title(self, title: str):
        self.set_font(self._body_font, "B", 13)
        self.set_text_color(*_MAROON)
        self.ln(4)
        self.cell(0, 9, self._safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_MAROON)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(3)

    def body_text(self, text: str):
        self.set_font(self._body_font, "", 10)
        self.set_text_color(*_GREY_TEXT)
        self.multi_cell(0, 5, self._safe(text))
        self.ln(2)

    def metric_row(self, label: str, value: str):
        self.set_font(self._body_font, "B", 10)
        self.set_text_color(*_GREY_DARK)
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
                self.set_text_color(*_GREY_TEXT)
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
        self.set_text_color(*_GREY_TEXT)
        for ln_text in truncated:
            self.cell(0, lh, ln_text, new_x="LMARGIN", new_y="NEXT")
        self.set_font(font_family, "I", sz)
        self.set_text_color(*_GREY_MUTED)
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
    well_name: str = "",
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
    well_name : str, optional
        Well name to display on the report.

    Returns
    -------
    bytes
        PDF file content.
    """
    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    if well_name:
        pdf._title_text = f"{well_name} — Interpretation Report"
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()

    # -- Cover / title page --------------------------------------------------
    pdf.add_page()
    font = pdf._body_font

    # Logo centred on cover
    if _LOGO_PATH.exists():
        logo_h = 50
        logo_x = (pdf.w - logo_h) / 2  # roughly square
        pdf.ln(20)
        pdf.image(str(_LOGO_PATH), x=logo_x, h=logo_h)
        pdf.ln(10)
    else:
        pdf.ln(30)

    pdf.set_font(font, "B", 26)
    pdf.set_text_color(*_MAROON)
    pdf.cell(0, 14, "Well Log Interpretation", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    if well_name:
        pdf.ln(6)
        pdf.set_font(font, "B", 18)
        pdf.cell(0, 12, pdf._safe(well_name), align="C",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font(font, "", 12)
    pdf.set_text_color(*_GREY_SUB)
    pdf.cell(0, 8,
             f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8,
             f"Data points: {len(result_df)}  |  "
             f"Curves: {len(result_df.columns)}",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # -- Key Metrics ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title("Key Metrics")

    avg_phi = result_df["PHIE"].mean() if "PHIE" in result_df.columns else 0
    avg_sw = result_df["SW"].mean() if "SW" in result_df.columns else 1
    avg_vsh = (result_df["VSHALE"].mean()
               if "VSHALE" in result_df.columns else 0.5)

    pdf.metric_row("Average Porosity:", f"{avg_phi:.1%}")
    pdf.metric_row("Average Water Saturation:", f"{avg_sw:.1%}")
    pdf.metric_row("Average Shale Volume:", f"{avg_vsh:.1%}")
    pdf.metric_row("Net Pay:", f"{net_stats['net_pay']:.1f}")
    pdf.metric_row("Net-to-Gross Ratio:", f"{net_stats['ntg_ratio']:.1%}")
    pdf.ln(4)

    # -- Verbal Interpretation -----------------------------------------------
    pdf.add_page()
    pdf.section_title("Plain-English Interpretation")
    pdf.body_text_fit_page(verbal, max_size=10, min_size=6)

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
        clean_ai = _strip_markdown(ai_interpretation)
        pdf.body_text_fit_page(clean_ai, max_size=10, min_size=6)

    # -- Technical Report ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("Technical Petrophysical Report")
    pdf.body_text_fit_page(technical_report, font_family=pdf._mono_font,
                           max_size=8, min_size=5)

    # -- Methodology & References --------------------------------------------
    pdf.add_page()
    pdf.section_title("Methodology & References")
    font = pdf._body_font

    methodology_text = (
        "This report was generated using industry-standard deterministic "
        "petrophysical methods. All calculations are reproducible and "
        "follow published SPE and PRMS guidelines.\n\n"
        "PETROPHYSICAL METHODS\n\n"
        "Shale Volume (Vshale)\n"
        "Gamma-ray index (IGR) computed from GR log using auto-picked "
        "clean (P5) and shale (P95) endpoints. Non-linear correction "
        "applied via Larionov (1969) Tertiary equation:\n"
        "  Vsh = 0.083 * (2^(3.7 * IGR) - 1)\n"
        "Alternative methods available: Linear IGR, Larionov Pre-Tertiary, "
        "Steiber (1970), Clavier (1971).\n\n"
        "Porosity\n"
        "Density porosity (Wyllie equation):\n"
        "  PHID = (rho_ma - rho_b) / (rho_ma - rho_fl)\n"
        "Neutron-density crossplot (when both curves available):\n"
        "  PHIT = sqrt((PHID^2 + PHIN^2) / 2)\n"
        "Effective porosity corrected for shale content:\n"
        "  PHIE = PHIT * (1 - Vsh)\n\n"
        "Water Saturation (Sw)\n"
        "Three models available:\n"
        "  Archie (1942): Sw = ((a * Rw) / (PHI^m * Rt))^(1/n)\n"
        "    Standard for clean formations.\n"
        "  Simandoux (1963): Accounts for shale conductivity.\n"
        "    Recommended for unconventional / shaly-sand reservoirs.\n"
        "  Indonesia / Poupon-Leveaux (1971): Empirical shaly-sand model.\n\n"
        "Net Pay (SPE 131529)\n"
        "Net reservoir = intervals where Vsh <= cutoff AND Phi >= cutoff.\n"
        "Net pay = net reservoir intervals where Sw <= cutoff.\n"
        "Cutoffs are adjusted based on resource type (conventional vs "
        "unconventional) per SPE 170830 guidelines.\n\n"
        "RESOURCE TYPE PRESETS\n\n"
        "Conventional: rho_ma=2.65, a=1.0, m=2.0, n=2.0, Rw=0.05,\n"
        "  Vsh<0.40, Phi>0.08, Sw<0.60\n\n"
        "Unconventional: rho_ma=2.55 (kerogen correction), a=1.0, m=1.7,\n"
        "  n=1.8, Rw=0.03, Vsh<0.55, Phi>0.04, Sw<0.45\n"
        "  Lower m reflects micro-/nano-porosity and natural fractures.\n"
        "  Matrix density reduced for kerogen effect per Passey et al. (1990).\n\n"
        "VOLUMETRIC RESERVES (SPE-PRMS 2018)\n\n"
        "OOIP = 7758 * A * h * PHI * (1 - Sw) / Boi  [STB]\n"
        "OGIP = 43560 * A * h * PHI * (1 - Sw) / Bgi  [SCF]\n"
        "Classification follows SPE/WPC/AAPG/SPEE/SEG Petroleum Resources\n"
        "Management System (PRMS) 2018 guidelines.\n\n"
        "REFERENCES\n\n"
        "Archie, G.E. (1942). The Electrical Resistivity Log as an Aid in\n"
        "  Determining Some Reservoir Characteristics. Trans. AIME 146,\n"
        "  54-62.\n\n"
        "Simandoux, P. (1963). Dielectric measurements on porous media,\n"
        "  application to the measurement of water saturations: study of\n"
        "  the behavior of argillaceous formations. Revue de l'Institut\n"
        "  Francais du Petrole 18, Supplementary Issue, 193-215.\n\n"
        "Poupon, A. & Leveaux, J. (1971). Evaluation of Water Saturation\n"
        "  in Shaly Formations. SPWLA 12th Annual Logging Symposium.\n\n"
        "Larionov, V.V. (1969). Borehole Radiometry. Moscow, Nedra.\n\n"
        "Steiber, R.G. (1970). Pulsed Neutron Interpretation in Shaly\n"
        "  Sands. SPE 2456.\n\n"
        "Clavier, C. et al. (1971). Theoretical and Experimental Bases\n"
        "  for the Dual-Water Model. SPE J. 21(2), 153-168.\n\n"
        "Passey, Q.R. et al. (1990). A Practical Model for Organic\n"
        "  Richness from Porosity and Resistivity Logs. AAPG Bulletin\n"
        "  74(12), 1777-1794.\n\n"
        "Aguilera, R. (1976). Analysis of Naturally Fractured Reservoirs\n"
        "  from Conventional Well Logs. JPT 28(7), 764-772.\n\n"
        "Glover, P. (2000). Petrophysics MSc Course Notes. University\n"
        "  of Leeds.\n\n"
        "Worthington, P.F. (2010). Net Pay: What Is It? What Does It\n"
        "  Do? How Do We Quantify It? How Do We Use It? SPE 131529.\n\n"
        "SPE/WPC/AAPG/SPEE/SEG (2018). Petroleum Resources Management\n"
        "  System (PRMS). SPE, Richardson, TX.\n\n"
        "Asquith, G. & Krygowski, D. (2004). Basic Well Log Analysis,\n"
        "  2nd ed. AAPG Methods in Exploration No. 16.\n\n"
        "SPE 170830 (2014). Petrophysical Evaluation of Unconventional\n"
        "  Reservoirs — Net Pay and Cutoff Considerations.\n\n"
        "DISCLAIMER\n\n"
        "This report is generated for screening purposes only. Volumetric\n"
        "estimates are deterministic and do not represent certified reserves.\n"
        "Actual reserves require full engineering analysis, economic\n"
        "evaluation, and classification per SPE-PRMS (2018) guidelines by\n"
        "a qualified reserves evaluator."
    )

    pdf.body_text_fit_page(methodology_text, font_family=font,
                           max_size=9, min_size=5)

    # -- Output --------------------------------------------------------------
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
