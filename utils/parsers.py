"""
Parsers for well log files: LAS, CSV/Excel, and PDF extraction.
Supports both text-based and raster (scanned image) PDFs.
"""

import io
import re
import lasio
import pandas as pd
import pdfplumber
from PIL import Image


def parse_las(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse a LAS file and return a DataFrame with depth as the index."""
    text = file_bytes.decode("utf-8", errors="replace")
    las = lasio.read(io.StringIO(text), engine="normal")
    df = las.df().reset_index()
    # Normalize the depth column name
    depth_col = df.columns[0]
    df = df.rename(columns={depth_col: "DEPTH"})
    # Deduplicate column names to prevent downstream errors
    if df.columns.duplicated().any():
        cols = list(df.columns)
        seen = {}
        for i, c in enumerate(cols):
            if c in seen:
                seen[c] += 1
                cols[i] = f"{c}_{seen[c]}"
            else:
                seen[c] = 0
        df.columns = cols
    return df


def parse_csv_excel(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse a CSV or Excel file containing log data."""
    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    # Try to identify and rename the depth column
    for col in df.columns:
        if col.upper() in ("DEPTH", "DEPT", "MD", "TVD", "MEASURED_DEPTH"):
            df = df.rename(columns={col: "DEPTH"})
            break

    return df


def parse_pdf(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Extract tabular data from a PDF well log.
    Attempts to find tables with numeric log data.
    Falls back to text extraction for structured data.
    """
    all_rows = []
    header = None

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                except Exception:
                    continue
                for table in tables:
                    if not table:
                        continue
                    for i, row in enumerate(table):
                        if not isinstance(row, (list, tuple)):
                            continue
                        # Clean cells
                        cleaned = []
                        for cell in row:
                            if cell is None:
                                cleaned.append("")
                            else:
                                cleaned.append(str(cell).strip())

                        # Identify header row: mostly non-numeric text
                        if header is None and _looks_like_header(cleaned):
                            header = cleaned
                            continue

                        # Only keep rows with mostly numeric data
                        if _has_numeric_data(cleaned):
                            all_rows.append(cleaned)

            # If no tables found, try text-based extraction
            if not all_rows:
                header, all_rows = _extract_from_text(pdf)
    except Exception:
        return pd.DataFrame()

    if not all_rows:
        # No tabular data found – this is likely a raster/scanned PDF.
        # Return empty DataFrame; caller should use extract_pdf_images() instead.
        return pd.DataFrame()

    if header:
        # Ensure header and rows have same length
        max_len = max(len(header), max(len(r) for r in all_rows))
        header = header + [""] * (max_len - len(header))
        all_rows = [r + [""] * (max_len - len(r)) for r in all_rows]
        # Deduplicate header names to avoid DataFrame returning a DF for a column
        seen: dict[str, int] = {}
        for i, h in enumerate(header):
            if h in seen:
                seen[h] += 1
                header[i] = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
        df = pd.DataFrame(all_rows, columns=header)
    else:
        df = pd.DataFrame(all_rows)

    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop fully empty columns/rows
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # Try to identify depth column
    for col in df.columns:
        if str(col).upper() in ("DEPTH", "DEPT", "MD", "TVD", "MEASURED_DEPTH"):
            df = df.rename(columns={col: "DEPTH"})
            break

    return df


def _looks_like_header(row: list) -> bool:
    """Check if a row looks like a table header (mostly non-numeric)."""
    if not row or all(c == "" for c in row):
        return False
    non_empty = [c for c in row if c]
    if not non_empty:
        return False
    numeric_count = sum(1 for c in non_empty if _is_numeric(c))
    return numeric_count / len(non_empty) < 0.5


def _has_numeric_data(row: list) -> bool:
    """Check if a row contains mostly numeric data."""
    non_empty = [c for c in row if c]
    if not non_empty:
        return False
    numeric_count = sum(1 for c in non_empty if _is_numeric(c))
    return numeric_count / len(non_empty) > 0.4


def _is_numeric(s: str) -> bool:
    """Check if a string is numeric."""
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, AttributeError):
        return False


def _extract_from_text(pdf) -> tuple:
    """Try to extract structured data from PDF text when tables fail."""
    all_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    lines = all_text.strip().split("\n")
    header = None
    rows = []

    for line in lines:
        # Split on whitespace (common in log printouts)
        parts = re.split(r"\s{2,}|\t", line.strip())
        if len(parts) < 2:
            continue

        if header is None and _looks_like_header(parts):
            header = parts
            continue

        if _has_numeric_data(parts):
            rows.append(parts)

    return header, rows


def extract_pdf_images(file_bytes: bytes) -> list:
    """
    Extract raster images from a PDF file.
    Returns a list of PIL Image objects, one per page.
    """
    images = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    page_img = page.to_image(resolution=150)
                    pil_img = page_img.original
                    if isinstance(pil_img, Image.Image):
                        images.append(pil_img.copy())
                    else:
                        # Fallback: convert via annotated image
                        annotated = page_img.annotated
                        if isinstance(annotated, Image.Image):
                            images.append(annotated.copy())
                except Exception:
                    continue
    except Exception:
        pass
    return images


def detect_log_curves(df: pd.DataFrame) -> dict:
    """
    Detect which standard log curves are present in the DataFrame
    by matching column names to known mnemonics.
    Returns a dict mapping curve type to column name.
    """
    curve_patterns = {
        "GR": r"^(GR|GAMMA|GAMMA_RAY|SGR|CGR)$",
        "DEPTH": r"^(DEPTH|DEPT|MD|TVD|MEASURED_DEPTH)$",
        "RESISTIVITY_DEEP": r"^(ILD|RT|LLD|RLLD|RDEP|RD|AT90|RDEEP|RES_DEEP)$",
        "RESISTIVITY_SHALLOW": r"^(ILS|RS|LLS|RLLS|RSHA|RSHAL|AT10|RES_SHALLOW)$",
        "DENSITY": r"^(RHOB|RHOZ|DEN|DENSITY|ZDEN)$",
        "NEUTRON": r"^(NPHI|TNPH|NEU|NEUTRON|PHIN|NPOR)$",
        "SONIC": r"^(DT|DTC|DTCO|SONIC|AC)$",
        "CALIPER": r"^(CALI|CAL|CALIPER|HCAL)$",
        "SP": r"^(SP|SPONTANEOUS)$",
        "PE": r"^(PE|PEF|PEFZ)$",
        "DENSITY_CORRECTION": r"^(DRHO|DCOR)$",
    }

    detected = {}
    for col in df.columns:
        col_upper = str(col).upper().strip()
        for curve_type, pattern in curve_patterns.items():
            if re.match(pattern, col_upper):
                detected[curve_type] = col
                break

    return detected
