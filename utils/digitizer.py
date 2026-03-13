"""
Well log raster image digitizer.

Extracts numerical curve data from scanned well log images by:
1. Auto-detecting track (column) boundaries from vertical separators
2. Tracing dark curves within each track
3. Mapping pixel coordinates to depth and measurement values

Designed to work with standard well log prints (triple-combo, dual-combo, etc.)
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Track boundary detection
# ---------------------------------------------------------------------------

def detect_track_boundaries(img: Image.Image, min_gap: int = 40) -> list[int]:
    """
    Detect vertical track separator lines in a well log image.

    Scans column-wise for consistently dark vertical lines.
    Returns a sorted list of x-pixel positions where separators are found.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Use the middle 80% of the image height to avoid headers/footers
    y_start = int(h * 0.10)
    y_end = int(h * 0.90)
    crop = gray[y_start:y_end, :]

    # For each column, compute mean darkness (lower = darker)
    col_means = crop.mean(axis=0)

    # Dark columns (potential separators) have low mean values
    # Use adaptive threshold: columns darker than 70% of the median
    threshold = np.median(col_means) * 0.70

    # Find runs of dark columns
    is_dark = col_means < threshold
    boundaries = []
    in_run = False
    run_start = 0

    for x in range(w):
        if is_dark[x] and not in_run:
            in_run = True
            run_start = x
        elif not is_dark[x] and in_run:
            in_run = False
            # Take the center of the dark run
            center = (run_start + x) // 2
            boundaries.append(center)

    if in_run:
        boundaries.append((run_start + w) // 2)

    # Filter out boundaries that are too close together
    if boundaries:
        filtered = [boundaries[0]]
        for b in boundaries[1:]:
            if b - filtered[-1] >= min_gap:
                filtered.append(b)
        boundaries = filtered

    # Always include image edges
    if not boundaries or boundaries[0] > min_gap:
        boundaries.insert(0, 0)
    if not boundaries or boundaries[-1] < w - min_gap:
        boundaries.append(w - 1)

    return boundaries


def suggest_tracks(img: Image.Image) -> list[dict]:
    """
    Auto-detect tracks and return a list of track definitions.

    Each track is a dict with:
        - left: left pixel boundary
        - right: right pixel boundary
        - name: suggested name (Track 1, Track 2, ...)
    """
    bounds = detect_track_boundaries(img)

    tracks = []
    for i in range(len(bounds) - 1):
        left = bounds[i]
        right = bounds[i + 1]
        width = right - left
        if width < 20:
            continue
        tracks.append({
            "left": left,
            "right": right,
            "name": f"Track {i + 1}",
            "curve_name": f"CURVE_{i + 1}",
            "min_value": 0.0,
            "max_value": 150.0,
        })

    return tracks


# ---------------------------------------------------------------------------
# Depth zone detection
# ---------------------------------------------------------------------------

def detect_log_area(img: Image.Image) -> dict:
    """
    Estimate the pixel region containing actual log data
    (excluding headers, footers, and margins).

    Returns dict with top_y, bottom_y, left_x, right_x pixel coords.
    """
    gray = np.array(img.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # Row-wise variance — log data rows have high variance, blank rows have low
    row_var = gray.var(axis=1)
    threshold = np.median(row_var) * 0.3

    active_rows = np.where(row_var > threshold)[0]

    if len(active_rows) == 0:
        return {"top_y": 0, "bottom_y": h - 1, "left_x": 0, "right_x": w - 1}

    return {
        "top_y": int(active_rows[0]),
        "bottom_y": int(active_rows[-1]),
        "left_x": 0,
        "right_x": w - 1,
    }


# ---------------------------------------------------------------------------
# Curve tracing
# ---------------------------------------------------------------------------

def trace_curve(
    img: Image.Image,
    left_x: int,
    right_x: int,
    top_y: int,
    bottom_y: int,
    color_channel: str = "dark",
) -> np.ndarray:
    """
    Trace a curve within a rectangular region of the image.

    For each row of pixels (depth level), finds the horizontal position
    of the most prominent curve pixel.

    Args:
        img: PIL Image
        left_x, right_x: horizontal pixel bounds of the track
        top_y, bottom_y: vertical pixel bounds of the log
        color_channel: "dark" (trace darkest pixels), "red", "blue", "green"

    Returns:
        Array of shape (num_rows, 2):
            column 0 = row index (pixel y)
            column 1 = normalized x position within track (0.0 to 1.0)
    """
    arr = np.array(img)

    # Ensure bounds are valid
    h, w = arr.shape[:2]
    left_x = max(0, min(left_x, w - 1))
    right_x = max(left_x + 1, min(right_x, w))
    top_y = max(0, min(top_y, h - 1))
    bottom_y = max(top_y + 1, min(bottom_y, h))

    # Extract the track region
    track = arr[top_y:bottom_y, left_x:right_x]
    track_h, track_w = track.shape[:2]

    if track_w < 3 or track_h < 3:
        return np.array([])

    # Build a 1D signal for each row: where is the curve?
    if len(track.shape) == 3:
        if color_channel == "dark":
            # Invert grayscale — darkest pixels get highest weight
            signal = 255.0 - track.mean(axis=2)
        elif color_channel == "red":
            signal = track[:, :, 0].astype(float)
            # Suppress where green/blue are equally high (gray/white)
            signal = signal - track[:, :, 1].astype(float) * 0.5 - track[:, :, 2].astype(float) * 0.5
            signal = np.clip(signal, 0, 255)
        elif color_channel == "blue":
            signal = track[:, :, 2].astype(float)
            signal = signal - track[:, :, 0].astype(float) * 0.5 - track[:, :, 1].astype(float) * 0.5
            signal = np.clip(signal, 0, 255)
        elif color_channel == "green":
            signal = track[:, :, 1].astype(float)
            signal = signal - track[:, :, 0].astype(float) * 0.5 - track[:, :, 2].astype(float) * 0.5
            signal = np.clip(signal, 0, 255)
        else:
            signal = 255.0 - track.mean(axis=2)
    else:
        signal = 255.0 - track.astype(float)

    # For each row, find the peak position using weighted average
    results = []
    for row_idx in range(track_h):
        row_signal = signal[row_idx, :]

        # Apply threshold: only consider pixels above background
        bg_level = np.percentile(row_signal, 30)
        peak_threshold = bg_level + (row_signal.max() - bg_level) * 0.4

        mask = row_signal >= peak_threshold
        if mask.sum() < 1:
            results.append((top_y + row_idx, np.nan))
            continue

        # Weighted centroid of above-threshold pixels
        weights = row_signal * mask
        positions = np.arange(track_w)
        if weights.sum() > 0:
            centroid = np.average(positions, weights=weights)
            norm_x = centroid / (track_w - 1) if track_w > 1 else 0.5
            results.append((top_y + row_idx, norm_x))
        else:
            results.append((top_y + row_idx, np.nan))

    return np.array(results)


def smooth_curve(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a moving median filter to smooth traced curve data."""
    if len(data) == 0:
        return data

    result = data.copy()
    values = result[:, 1]

    # Interpolate NaN gaps first
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return result

    indices = np.arange(len(values))
    values[~valid] = np.interp(indices[~valid], indices[valid], values[valid])

    # Moving median
    half = window // 2
    smoothed = values.copy()
    for i in range(half, len(values) - half):
        smoothed[i] = np.median(values[i - half:i + half + 1])

    result[:, 1] = smoothed
    return result


# ---------------------------------------------------------------------------
# Digitization pipeline
# ---------------------------------------------------------------------------

def digitize_track(
    img: Image.Image,
    left_x: int,
    right_x: int,
    top_y: int,
    bottom_y: int,
    depth_top: float,
    depth_bottom: float,
    value_min: float,
    value_max: float,
    color_channel: str = "dark",
    smooth_window: int = 7,
    sample_interval: float = 0.5,
) -> pd.Series:
    """
    Digitize a single log track from an image.

    Args:
        img: PIL Image of the log
        left_x, right_x: pixel boundaries of the track
        top_y, bottom_y: pixel boundaries of the log area
        depth_top, depth_bottom: depth values at top/bottom of image
        value_min, value_max: scale range for this track (left edge = min, right = max)
        color_channel: which curve color to trace
        smooth_window: median filter window size
        sample_interval: depth sampling interval for output

    Returns:
        pandas Series indexed by depth with digitized values
    """
    # Trace the curve
    raw = trace_curve(img, left_x, right_x, top_y, bottom_y, color_channel)

    if len(raw) == 0:
        return pd.Series(dtype=float)

    # Smooth
    smoothed = smooth_curve(raw, window=smooth_window)

    # Convert pixel positions to depth and values
    pixel_depths = smoothed[:, 0]
    norm_positions = smoothed[:, 1]

    total_pixels = bottom_y - top_y
    if total_pixels <= 0:
        return pd.Series(dtype=float)

    depths = depth_top + (pixel_depths - top_y) / total_pixels * (depth_bottom - depth_top)
    values = value_min + norm_positions * (value_max - value_min)

    # Resample to regular depth intervals
    depth_range = np.arange(depth_top, depth_bottom, sample_interval)
    resampled = np.interp(depth_range, depths, values)

    return pd.Series(resampled, index=depth_range, name="value")


def digitize_log_image(
    img: Image.Image,
    track_configs: list[dict],
    depth_top: float,
    depth_bottom: float,
    sample_interval: float = 0.5,
) -> pd.DataFrame:
    """
    Digitize multiple tracks from a well log image.

    Args:
        img: PIL Image
        track_configs: list of dicts, each with:
            - left: left pixel x
            - right: right pixel x
            - curve_name: column name for output (e.g., "GR", "RHOB")
            - min_value: value at left edge of track
            - max_value: value at right edge of track
            - color_channel: "dark", "red", "blue", "green" (default: "dark")
        depth_top: depth at top of image
        depth_bottom: depth at bottom of image
        sample_interval: output sampling interval in depth units

    Returns:
        DataFrame with DEPTH column and one column per digitized track
    """
    log_area = detect_log_area(img)
    top_y = log_area["top_y"]
    bottom_y = log_area["bottom_y"]

    depth_range = np.arange(depth_top, depth_bottom, sample_interval)
    result = pd.DataFrame({"DEPTH": depth_range})

    for track in track_configs:
        color = track.get("color_channel", "dark")
        series = digitize_track(
            img,
            left_x=track["left"],
            right_x=track["right"],
            top_y=top_y,
            bottom_y=bottom_y,
            depth_top=depth_top,
            depth_bottom=depth_bottom,
            value_min=track["min_value"],
            value_max=track["max_value"],
            color_channel=color,
            sample_interval=sample_interval,
        )

        if not series.empty:
            result[track["curve_name"]] = series.values[:len(result)]

    return result


def annotate_image_with_tracks(
    img: Image.Image,
    tracks: list[dict],
) -> Image.Image:
    """
    Draw track boundary lines on the image for visual confirmation.
    Returns a new annotated image.
    """
    from PIL import ImageDraw, ImageFont

    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    h = annotated.height

    colors = ["#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#FFFF00", "#00FFFF",
              "#FF8800", "#8800FF", "#00FF88", "#FF0088"]

    for i, track in enumerate(tracks):
        color = colors[i % len(colors)]
        left = track["left"]
        right = track["right"]

        # Draw vertical boundary lines
        draw.line([(left, 0), (left, h)], fill=color, width=2)
        draw.line([(right, 0), (right, h)], fill=color, width=2)

        # Label
        label = track.get("curve_name", track.get("name", f"Track {i+1}"))
        mid_x = (left + right) // 2
        try:
            draw.text((mid_x - 20, 5), label, fill=color)
        except Exception:
            pass

    return annotated
