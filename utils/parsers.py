"""
Parsers for well log files: LAS, CSV/Excel, and PDF extraction.
Supports both text-based and raster (scanned image) PDFs.

Also extracts well header metadata (lat/long, well name, field, etc.)
from LAS file headers and provides unit detection/conversion utilities.
"""

import io
import re
import lasio
import pandas as pd
import numpy as np
import pdfplumber
from PIL import Image


# ---------------------------------------------------------------------------
# Unit detection & conversion
# ---------------------------------------------------------------------------

# Depth unit patterns
_DEPTH_UNIT_MAP = {
    "F": "ft", "FT": "ft", "FEET": "ft", "FOOT": "ft", "FT.": "ft",
    "M": "m", "MT": "m", "METER": "m", "METERS": "m", "METRE": "m",
    "METRES": "m", "M.": "m",
}

# Curve unit hints – maps raw unit strings to canonical form
_UNIT_CANONICAL = {
    # Porosity
    "V/V": "v/v", "FRAC": "v/v", "FRACTION": "v/v", "DEC": "v/v",
    "%": "pct", "PU": "pct", "P.U.": "pct", "PCT": "pct", "PERCENT": "pct",
    # Resistivity
    "OHMM": "ohmm", "OHM.M": "ohmm", "OHM-M": "ohmm", "OHMM2/M": "ohmm",
    "OHM.M2/M": "ohmm", "OHMS": "ohmm",
    # Density
    "G/C3": "g/cc", "G/CC": "g/cc", "GM/CC": "g/cc", "KG/M3": "kg/m3",
    "G/CM3": "g/cc",
    # Sonic
    "US/F": "us/ft", "US/FT": "us/ft", "USEC/FT": "us/ft",
    "US/M": "us/m", "USEC/M": "us/m",
    # GR
    "GAPI": "gapi", "API": "gapi",
    # Caliper
    "IN": "in", "INCH": "in", "INCHES": "in",
    "CM": "cm", "MM": "mm",
}


def detect_depth_unit(las_or_unit_str) -> str:
    """Return 'ft' or 'm' from a lasio LASFile or raw unit string."""
    if hasattr(las_or_unit_str, "well"):
        # lasio object – check STRT/STOP/STEP units
        for item in las_or_unit_str.well:
            if item.mnemonic.upper() in ("STRT", "STOP", "STEP"):
                u = str(item.unit).strip().upper()
                if u in _DEPTH_UNIT_MAP:
                    return _DEPTH_UNIT_MAP[u]
        return "ft"  # default
    u = str(las_or_unit_str).strip().upper()
    return _DEPTH_UNIT_MAP.get(u, "ft")


def detect_curve_units(las) -> dict:
    """
    Return a dict mapping curve mnemonic -> canonical unit string
    from a lasio LASFile object.
    """
    units = {}
    for curve in las.curves:
        raw = str(curve.unit).strip().upper()
        units[curve.mnemonic] = _UNIT_CANONICAL.get(raw, raw.lower())
    return units


def convert_depth(df: pd.DataFrame, from_unit: str, to_unit: str,
                  depth_col: str = "DEPTH") -> pd.DataFrame:
    """Convert the DEPTH column between ft and m in-place."""
    if from_unit == to_unit or depth_col not in df.columns:
        return df
    if from_unit == "m" and to_unit == "ft":
        df[depth_col] = df[depth_col] * 3.28084
    elif from_unit == "ft" and to_unit == "m":
        df[depth_col] = df[depth_col] / 3.28084
    return df


def normalize_porosity_units(df: pd.DataFrame, curve_units: dict) -> pd.DataFrame:
    """Convert porosity columns from % to v/v if detected as percentage."""
    porosity_mnemonics = {"NPHI", "TNPH", "DPHI", "PHIN", "NPOR", "PHI", "PHIT", "PHIE"}
    for col in df.columns:
        if col.upper() in porosity_mnemonics:
            unit = curve_units.get(col, "")
            if unit == "pct":
                # Values > 1 strongly suggest percentage
                if df[col].dropna().max() > 1.0:
                    df[col] = df[col] / 100.0
    return df


def normalize_density_units(df: pd.DataFrame, curve_units: dict) -> pd.DataFrame:
    """Convert density from kg/m3 to g/cc if needed."""
    density_mnemonics = {"RHOB", "RHOZ", "DEN", "DENSITY", "ZDEN"}
    for col in df.columns:
        if col.upper() in density_mnemonics:
            unit = curve_units.get(col, "")
            if unit == "kg/m3" or (df[col].dropna().median() > 100):
                df[col] = df[col] / 1000.0
    return df


# ---------------------------------------------------------------------------
# Lat/Long extraction from LAS headers
# ---------------------------------------------------------------------------

def _parse_dms_to_decimal(dms_str: str) -> float | None:
    """
    Parse a lat/long string in various formats to decimal degrees.

    Handles:
      - Decimal: "29.123456", "-95.456"
      - DMS: "29 07 23.45 N", "29d 07' 23.45\" N"
      - Compact DMS: "29:07:23.45N"
      - Degrees + decimal minutes: "29 07.391N"
    """
    if not dms_str or not dms_str.strip():
        return None
    s = dms_str.strip()

    # Determine hemisphere sign
    sign = 1
    if s[-1].upper() in ("S", "W"):
        sign = -1
        s = s[:-1].strip()
    elif s[-1].upper() in ("N", "E"):
        s = s[:-1].strip()
    elif s[0] == "-":
        sign = -1
        s = s[1:].strip()

    # Try plain decimal first
    try:
        return sign * float(s)
    except ValueError:
        pass

    # Split on delimiters: space, :, d, °, ', "
    parts = re.split(r"[:\s°d'\"]+", s)
    parts = [p for p in parts if p]

    try:
        if len(parts) == 1:
            return sign * float(parts[0])
        elif len(parts) == 2:
            # Degrees + decimal minutes
            deg = float(parts[0])
            mins = float(parts[1])
            return sign * (deg + mins / 60.0)
        elif len(parts) >= 3:
            deg = float(parts[0])
            mins = float(parts[1])
            secs = float(parts[2])
            return sign * (deg + mins / 60.0 + secs / 3600.0)
    except (ValueError, IndexError):
        pass

    return None


def extract_las_metadata(file_bytes: bytes) -> dict:
    """
    Extract well header metadata from a LAS file.

    Returns a dict with keys:
      - well_name: str
      - field: str
      - company: str
      - county: str
      - state: str
      - country: str
      - latitude: float or None
      - longitude: float or None
      - lat_raw: str (original string)
      - lon_raw: str (original string)
      - api_number: str
      - depth_unit: str ('ft' or 'm')
      - curve_units: dict mapping mnemonic -> unit
      - date: str
      - service_company: str
      - uwi: str
    """
    text = file_bytes.decode("utf-8", errors="replace")
    las = lasio.read(io.StringIO(text), engine="normal")

    meta = {
        "well_name": "", "field": "", "company": "", "county": "",
        "state": "", "country": "", "latitude": None, "longitude": None,
        "lat_raw": "", "lon_raw": "", "api_number": "", "depth_unit": "ft",
        "curve_units": {}, "date": "", "service_company": "", "uwi": "",
        "location": "", "well_type": "",
    }

    # Map of LAS well-section mnemonics to our metadata keys
    _WELL_FIELD_MAP = {
        "WELL": "well_name", "COMP": "company", "FLD": "field",
        "CNTY": "county", "STAT": "state", "CTRY": "country",
        "LOC": "location", "SRVC": "service_company",
        "DATE": "date", "API": "api_number", "UWI": "uwi",
    }
    _LAT_MNEMONICS = {"LATI", "LAT", "SLAT", "LATITUDE", "YLOC", "SURF_LAT"}
    _LON_MNEMONICS = {"LONG", "LON", "SLON", "LONGITUDE", "XLON", "XLOC", "SURF_LONG"}
    _WELL_TYPE_MNEMONICS = {"WTYP", "WELL_TYPE", "WELLTYPE", "TYPE"}

    for item in las.well:
        mnem = item.mnemonic.strip().upper()
        val = str(item.value).strip() if item.value else ""

        if mnem in _WELL_FIELD_MAP and val:
            meta[_WELL_FIELD_MAP[mnem]] = val
        elif mnem in _LAT_MNEMONICS and val:
            meta["lat_raw"] = val
            meta["latitude"] = _parse_dms_to_decimal(val)
        elif mnem in _LON_MNEMONICS and val:
            meta["lon_raw"] = val
            meta["longitude"] = _parse_dms_to_decimal(val)
        elif mnem in _WELL_TYPE_MNEMONICS and val:
            meta["well_type"] = val

    meta["depth_unit"] = detect_depth_unit(las)
    meta["curve_units"] = detect_curve_units(las)

    return meta


def merge_log_dataframes(dfs: list, depth_col: str = "DEPTH") -> pd.DataFrame:
    """
    Merge multiple well-log DataFrames on a common depth column.

    Handles:
      - Different depth ranges (e.g. shallow run + deep run)
      - Overlapping depths with different curves
      - Duplicate curve names across files (suffixed automatically)

    Strategy:
      1. Round depth to a common precision so nearby samples align.
      2. Outer-join on depth so no data is lost.
      3. For overlapping curves at the same depth, keep the first non-NaN value.
    """
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0]

    # Determine depth precision from the finest step across all files
    precisions = []
    for df in dfs:
        if depth_col in df.columns and len(df) > 1:
            step = df[depth_col].dropna().diff().dropna().abs()
            if len(step) > 0:
                med = float(np.nanmedian(step.values))
                if med > 0:
                    # number of decimals needed
                    precisions.append(max(0, int(-np.floor(np.log10(med))) + 1))
    precision = max(precisions) if precisions else 2

    merged = None
    for i, df in enumerate(dfs):
        if depth_col not in df.columns:
            continue
        tmp = df.copy()
        tmp[depth_col] = tmp[depth_col].round(precision)

        if merged is None:
            merged = tmp
        else:
            # Suffix overlapping columns (except DEPTH)
            overlap = set(merged.columns) & set(tmp.columns) - {depth_col}
            suffix_map = {c: f"{c}_run{i + 1}" for c in overlap}
            tmp = tmp.rename(columns=suffix_map)

            merged = pd.merge(merged, tmp, on=depth_col, how="outer")

    if merged is None:
        return dfs[0]

    merged = merged.sort_values(depth_col).reset_index(drop=True)

    # Consolidate duplicate curves: e.g. GR and GR_run2 -> fill NaNs in GR from GR_run2
    base_cols = [c for c in merged.columns if not re.search(r"_run\d+$", c) and c != depth_col]
    for base in base_cols:
        run_cols = [c for c in merged.columns if re.match(rf"^{re.escape(base)}_run\d+$", c)]
        if run_cols:
            for rc in run_cols:
                merged[base] = merged[base].fillna(merged[rc])
            merged = merged.drop(columns=run_cols)

    return merged


def parse_las(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse a LAS file and return (DataFrame, metadata_dict).

    The DataFrame has DEPTH as first column.
    The metadata dict contains well header info (lat/long, name, units, etc.).
    """
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

    # Extract metadata (lat/long, well name, units, etc.)
    meta = extract_las_metadata(file_bytes)

    # Apply unit normalizations
    curve_units = meta.get("curve_units", {})
    df = normalize_porosity_units(df, curve_units)
    df = normalize_density_units(df, curve_units)

    return df, meta


def parse_csv_excel(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """Parse a CSV or Excel file containing log data. Returns (DataFrame, metadata).

    Handles DrillingInfo/Enverus export formats with verbose column names
    like "Gamma Ray (API)" by stripping units in parentheses and mapping
    to standard mnemonics.
    """
    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    # Clean column names: strip whitespace, normalize
    df.columns = [str(c).strip() for c in df.columns]

    # Map verbose DI/Enverus-style column names to standard mnemonics
    _DI_COLUMN_MAP = {
        "GAMMA RAY": "GR", "GAMMA_RAY": "GR",
        "DEEP RESISTIVITY": "ILD", "DEEP RES": "ILD", "RES DEEP": "ILD",
        "SHALLOW RESISTIVITY": "ILS", "SHALLOW RES": "ILS", "RES SHALLOW": "ILS",
        "BULK DENSITY": "RHOB", "DENSITY": "RHOB",
        "NEUTRON POROSITY": "NPHI", "NEUTRON": "NPHI",
        "SONIC": "DT", "COMPRESSIONAL SONIC": "DT",
        "CALIPER": "CALI",
        "MEASURED DEPTH": "DEPTH",
        "TRUE VERTICAL DEPTH": "TVD",
    }

    renamed = {}
    for col in df.columns:
        # Strip units in parentheses: "Gamma Ray (API)" -> "Gamma Ray"
        clean = re.sub(r"\s*\([^)]*\)\s*$", "", col).strip().upper()
        clean = re.sub(r"[_\s]+", " ", clean)
        if clean in _DI_COLUMN_MAP:
            renamed[col] = _DI_COLUMN_MAP[clean]
    if renamed:
        df = df.rename(columns=renamed)

    # Try to identify and rename the depth column
    for col in df.columns:
        if str(col).upper() in ("DEPTH", "DEPT", "MD", "TVD", "MEASURED_DEPTH"):
            df = df.rename(columns={col: "DEPTH"})
            break

    return df, {}


def parse_pdf(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """
    Extract tabular data from a PDF well log. Returns (DataFrame, metadata).
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
        return pd.DataFrame(), {}

    if not all_rows:
        # No tabular data found – this is likely a raster/scanned PDF.
        # Return empty DataFrame; caller should use extract_pdf_images() instead.
        return pd.DataFrame(), {}

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
    depth_found = False
    for col in df.columns:
        if str(col).upper() in ("DEPTH", "DEPT", "MD", "TVD", "MEASURED_DEPTH"):
            df = df.rename(columns={col: "DEPTH"})
            depth_found = True
            break

    # If no depth column found, check if first column is monotonically increasing (likely depth)
    if not depth_found and len(df.columns) > 0:
        first_col = df.iloc[:, 0].dropna()
        if len(first_col) > 1 and first_col.is_monotonic_increasing:
            df = df.rename(columns={df.columns[0]: "DEPTH"})
            depth_found = True

    # Last resort: create a synthetic depth column from row index
    if not depth_found:
        df.insert(0, "DEPTH", range(len(df)))

    return df, {}


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


def extract_pdf_header_metadata(file_bytes: bytes) -> dict:
    """
    Try to extract well metadata (lat/long, well name, API) from text
    on the first page of a PDF log. Works for PDFs that have a text layer
    in their header block (common for digitally-generated log prints).

    Returns a dict similar to extract_las_metadata().
    """
    meta = {
        "well_name": "", "field": "", "company": "", "county": "",
        "state": "", "latitude": None, "longitude": None,
        "lat_raw": "", "lon_raw": "", "api_number": "", "location": "",
    }

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return meta
            # Only look at the first page header area (top 25%)
            page = pdf.pages[0]
            header_bbox = (0, 0, page.width, page.height * 0.25)
            text = page.within_bbox(header_bbox).extract_text() or ""
            if not text.strip():
                # Try full first page
                text = page.extract_text() or ""
    except Exception:
        return meta

    if not text.strip():
        return meta

    lines = text.strip().split("\n")

    # Patterns to look for in header text
    _patterns = {
        "well_name": [
            r"(?:WELL|WELL\s*NAME)\s*[:=\-]\s*(.+)",
        ],
        "company": [
            r"(?:COMP(?:ANY)?|OPERATOR)\s*[:=\-]\s*(.+)",
        ],
        "field": [
            r"(?:FIELD|FLD)\s*[:=\-]\s*(.+)",
        ],
        "county": [
            r"(?:COUNTY|CNTY)\s*[:=\-]\s*(.+)",
        ],
        "state": [
            r"(?:STATE|STAT)\s*[:=\-]\s*(.+)",
        ],
        "api_number": [
            r"(?:API\s*(?:NO|NUMBER|#)?)\s*[:=\-]?\s*([\d\-]+)",
        ],
    }

    # Lat/long patterns
    _lat_patterns = [
        r"(?:LAT(?:ITUDE)?)\s*[:=\-]\s*([\d°\'\"\.\s\-NSEW:dms]+)",
        r"(?:SURF(?:ACE)?\s*LAT)\s*[:=\-]\s*([\d°\'\"\.\s\-NSEW:dms]+)",
    ]
    _lon_patterns = [
        r"(?:LON(?:G(?:ITUDE)?)?)\s*[:=\-]\s*([\d°\'\"\.\s\-NSEW:dms]+)",
        r"(?:SURF(?:ACE)?\s*LON(?:G)?)\s*[:=\-]\s*([\d°\'\"\.\s\-NSEW:dms]+)",
    ]

    full_text = "\n".join(lines)

    for key, patterns in _patterns.items():
        for pat in patterns:
            m = re.search(pat, full_text, re.IGNORECASE)
            if m and m.group(1).strip():
                meta[key] = m.group(1).strip()
                break

    for pat in _lat_patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m and m.group(1).strip():
            meta["lat_raw"] = m.group(1).strip()
            meta["latitude"] = _parse_dms_to_decimal(meta["lat_raw"])
            break

    for pat in _lon_patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m and m.group(1).strip():
            meta["lon_raw"] = m.group(1).strip()
            meta["longitude"] = _parse_dms_to_decimal(meta["lon_raw"])
            break

    return meta


def detect_log_curves(df: pd.DataFrame) -> dict:
    """
    Detect which standard log curves are present in the DataFrame
    by matching column names to known mnemonics.
    Returns a dict mapping curve type to column name.

    Supports standard LAS mnemonics, DrillingInfo/Enverus export names,
    and common vendor variations.
    """
    curve_patterns = {
        "GR": r"^(GR|GAMMA|GAMMA_RAY|GAMMA\.RAY|SGR|CGR|GR_CORR|GRD|GRS)$",
        "DEPTH": r"^(DEPTH|DEPT|MD|MEASURED_DEPTH|MEASURED\.DEPTH|TDEP)$",
        "TVD": r"^(TVD|TVDSS|TVD_SS|TRUE_VERT|TVDKB)$",
        "INCLINATION": r"^(INCL|INC|DEVI|DEVIATION|INCLIN|HDEV|AZIA)$",
        "AZIMUTH": r"^(AZIM|AZI|AZIMUTH|AZ)$",
        "RESISTIVITY_DEEP": r"^(ILD|RT|LLD|RLLD|RDEP|RD|AT90|RDEEP|RES_DEEP|RESD|R_DEEP|HDRS|M2RX)$",
        "RESISTIVITY_SHALLOW": r"^(ILS|RS|LLS|RLLS|RSHA|RSHAL|AT10|RES_SHALLOW|RESS|R_SHALL|HMRS|M2R1)$",
        "DENSITY": r"^(RHOB|RHOZ|DEN|DENSITY|ZDEN|BULK_DENSITY|BULK\.DEN)$",
        "NEUTRON": r"^(NPHI|TNPH|NEU|NEUTRON|PHIN|NPOR|NEUT|NEUTRON_POR)$",
        "SONIC": r"^(DT|DTC|DTCO|SONIC|AC|DT_COMP|DTLN)$",
        "CALIPER": r"^(CALI|CAL|CALIPER|HCAL|BS|BIT_SIZE)$",
        "SP": r"^(SP|SPONTANEOUS|SPONT)$",
        "PE": r"^(PE|PEF|PEFZ|PHOTO)$",
        "DENSITY_CORRECTION": r"^(DRHO|DCOR|DENS_CORR)$",
    }

    detected = {}
    for col in df.columns:
        col_upper = str(col).upper().strip()
        for curve_type, pattern in curve_patterns.items():
            if re.match(pattern, col_upper):
                detected[curve_type] = col
                break

    return detected


def infer_well_type(
    df: pd.DataFrame,
    detected: dict,
    metadata: dict,
) -> str:
    """
    Infer well type as 'vertical', 'deviated', or 'horizontal'.

    Uses these signals in priority order:
      1. Explicit well_type from LAS header metadata
      2. Well name heuristics (#H, HZ, HORIZ in name)
      3. Inclination data (if INCL column present)
      4. MD vs TVD divergence (if both columns exist)

    Returns one of: 'vertical', 'deviated', 'horizontal'
    """
    # 1. LAS header well type field
    wt = metadata.get("well_type", "").upper()
    if wt:
        if any(k in wt for k in ("HORIZ", "HZ", "LATERAL")):
            return "horizontal"
        if any(k in wt for k in ("VERT", "VT", "STRAIGHT")):
            return "vertical"
        if any(k in wt for k in ("DEV", "DIRECT", "SLANT")):
            return "deviated"

    # 2. Well name heuristics
    name = metadata.get("well_name", "").upper()
    if re.search(r"\d+\s*H\b|HZ\b|HORIZ|LATERAL", name):
        return "horizontal"

    # 3. Inclination data
    if "INCLINATION" in detected and detected["INCLINATION"] in df.columns:
        incl = df[detected["INCLINATION"]].dropna()
        if len(incl) > 10:
            max_incl = incl.max()
            # Sustained high inclination = horizontal
            high_pct = (incl > 80).mean()
            if max_incl > 85 and high_pct > 0.3:
                return "horizontal"
            if max_incl > 20:
                return "deviated"

    # 4. MD vs TVD divergence
    if "TVD" in detected and detected["TVD"] in df.columns and "DEPTH" in df.columns:
        md = df["DEPTH"].dropna()
        tvd = df[detected["TVD"]].dropna()
        if len(md) > 10 and len(tvd) > 10:
            md_range = md.max() - md.min()
            tvd_range = tvd.max() - tvd.min()
            if md_range > 0:
                ratio = tvd_range / md_range
                if ratio < 0.3:
                    return "horizontal"
                if ratio < 0.85:
                    return "deviated"

    return "vertical"
