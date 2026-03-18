"""
AI-enhanced well log interpretation using Claude or Google Gemini.

Takes the deterministic petrophysical results (Vshale, porosity, Sw, net pay)
and sends them to an LLM for a richer, context-aware narrative interpretation
that reads like an expert petrophysicist's report.

Supported providers:
  - Google Gemini (free tier available)
  - Anthropic Claude (paid)
"""

import numpy as np
import pandas as pd

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# ---------------------------------------------------------------------------
# Metrics builder (shared by all providers)
# ---------------------------------------------------------------------------

def _build_metrics_summary(
    df: pd.DataFrame,
    net_stats: dict,
    detected: dict,
) -> str:
    """Build a concise text summary of all computed petrophysical metrics."""
    lines = []

    # Depth range
    if "DEPTH" in df.columns:
        lines.append(f"Depth range: {df['DEPTH'].min():.1f} – {df['DEPTH'].max():.1f}")

    # Available curves
    curve_labels = {
        "GR": "Gamma Ray",
        "RESISTIVITY_DEEP": "Deep Resistivity",
        "RESISTIVITY_SHALLOW": "Shallow Resistivity",
        "DENSITY": "Bulk Density",
        "NEUTRON": "Neutron Porosity",
        "SONIC": "Sonic",
        "CALIPER": "Caliper",
        "SP": "Spontaneous Potential",
    }
    avail = [curve_labels.get(k, k) for k in detected if k != "DEPTH"]
    lines.append(f"Available curves: {', '.join(avail)}")

    # Vshale
    if "VSHALE" in df.columns:
        vsh = df["VSHALE"]
        pct_clean = (vsh < 0.30).mean() * 100
        pct_shaly = (vsh >= 0.50).mean() * 100
        lines.append(
            f"Vshale: mean={vsh.mean():.3f}, median={vsh.median():.3f}, "
            f"P10={vsh.quantile(0.10):.3f}, P90={vsh.quantile(0.90):.3f}, "
            f"clean_pct={pct_clean:.1f}%, shaly_pct={pct_shaly:.1f}%"
        )

    # Porosity
    if "PHIE" in df.columns:
        phi = df["PHIE"]
        lines.append(
            f"Effective porosity (PHIE): mean={phi.mean():.3f}, "
            f"median={phi.median():.3f}, max={phi.max():.3f}, "
            f"P10={phi.quantile(0.10):.3f}, P90={phi.quantile(0.90):.3f}"
        )
    if "PHIT" in df.columns:
        phit = df["PHIT"]
        lines.append(f"Total porosity (PHIT): mean={phit.mean():.3f}")

    # Water saturation
    if "SW" in df.columns:
        sw = df["SW"]
        sh = 1.0 - sw
        lines.append(
            f"Water saturation (Sw): mean={sw.mean():.3f}, "
            f"median={sw.median():.3f}, P10={sw.quantile(0.10):.3f}, "
            f"P90={sw.quantile(0.90):.3f}"
        )
        lines.append(f"Hydrocarbon saturation (1-Sw): mean={sh.mean():.3f}")

    # Gas indicators
    if "DENSITY" in detected and "NEUTRON" in detected:
        try:
            phid = (2.65 - df[detected["DENSITY"]]) / (2.65 - 1.0)
            phin = df[detected["NEUTRON"]]
            gas_crossover = (phin < phid).mean() * 100
            lines.append(f"Neutron-density crossover (gas indicator): {gas_crossover:.1f}% of interval")
        except Exception:
            pass

    # Net pay
    lines.append(
        f"Gross thickness: {net_stats['gross_thickness']:.1f}, "
        f"Net reservoir: {net_stats['net_reservoir']:.1f}, "
        f"Net pay: {net_stats['net_pay']:.1f}, "
        f"NTG ratio: {net_stats['ntg_ratio']:.3f}"
    )

    # Zonal breakdown — top/middle/bottom thirds
    if "DEPTH" in df.columns and len(df) > 30:
        third = len(df) // 3
        zones = [
            ("Upper third", df.iloc[:third]),
            ("Middle third", df.iloc[third:2*third]),
            ("Lower third", df.iloc[2*third:]),
        ]
        zone_lines = []
        for name, zone in zones:
            parts = [f"{name}:"]
            if "VSHALE" in zone.columns:
                parts.append(f"Vsh={zone['VSHALE'].mean():.2f}")
            if "PHIE" in zone.columns:
                parts.append(f"Phi={zone['PHIE'].mean():.3f}")
            if "SW" in zone.columns:
                parts.append(f"Sw={zone['SW'].mean():.2f}")
            if "NET_PAY" in zone.columns:
                parts.append(f"PayPts={zone['NET_PAY'].sum():.0f}")
            zone_lines.append(", ".join(parts))
        lines.append("Zonal breakdown: " + " | ".join(zone_lines))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior petrophysicist with 25+ years of experience in formation \
evaluation. You interpret well log data for a living and write clear, \
insightful reports.

You will receive computed petrophysical metrics from a well log \
(Vshale, porosity, water saturation, net pay statistics, and available \
curve information). These were computed using standard industry methods:
- Vshale: Larionov Tertiary correction from Gamma Ray
- Porosity: Density porosity (sandstone matrix 2.65 g/cc) or \
neutron-density crossplot
- Water saturation: Archie equation (a=1, m=2, n=2, Rw=0.05 ohm-m)
- Net pay cutoffs: Vsh<0.40, Phi>0.08, Sw<0.60

Your job is to provide an expert interpretation that goes BEYOND what \
simple threshold-based rules can offer. Specifically:

1. PATTERN RECOGNITION: Look at how metrics vary across the interval \
(zonal breakdown). Identify trends, sweet spots, and problem zones.

2. CROSS-CURVE INSIGHTS: Note relationships between curves that tell a \
deeper story (e.g., porosity-Sw relationship suggesting transition zone, \
Vshale-porosity suggesting laminated vs. dispersed shale).

3. RISK FACTORS: Identify potential concerns — are the Archie defaults \
appropriate? Could Rw be different? Is the matrix density assumption valid? \
What would change the interpretation?

4. RECOMMENDATIONS: What should the user investigate next? Core data? \
Pressure tests? Different Sw model? Analog comparison?

5. PLAIN LANGUAGE: Write for someone who is NOT a petrophysicist. Use \
technical terms only when necessary, and always explain them.

Keep the interpretation concise but insightful — aim for 300-500 words. \
Use clear section headers. Do NOT repeat the raw numbers excessively; \
focus on what they MEAN.

If the user provides geological context (basin, formation, fluid type), \
incorporate that knowledge into your interpretation.\
"""


# ---------------------------------------------------------------------------
# Provider: Google Gemini (free tier available)
# ---------------------------------------------------------------------------

def generate_gemini_interpretation(
    df: pd.DataFrame,
    net_stats: dict,
    detected: dict,
    api_key: str,
    geological_context: str = "",
) -> str:
    """Generate AI interpretation using Google Gemini."""
    if not HAS_GEMINI:
        return (
            "The 'google-genai' package is not installed. "
            "Run: pip install google-genai"
        )

    metrics = _build_metrics_summary(df, net_stats, detected)

    user_message = f"Here are the computed petrophysical metrics for this well:\n\n{metrics}"
    if geological_context.strip():
        user_message += (
            f"\n\nGeological context provided by the user:\n{geological_context}"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_message,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1500,
            temperature=0.7,
        ),
    )

    return response.text


# ---------------------------------------------------------------------------
# Provider: Anthropic Claude (paid)
# ---------------------------------------------------------------------------

def generate_claude_interpretation(
    df: pd.DataFrame,
    net_stats: dict,
    detected: dict,
    api_key: str,
    geological_context: str = "",
) -> str:
    """Generate AI interpretation using Claude."""
    if not HAS_ANTHROPIC:
        return (
            "The 'anthropic' package is not installed. "
            "Run: pip install anthropic"
        )

    metrics = _build_metrics_summary(df, net_stats, detected)

    user_message = f"Here are the computed petrophysical metrics for this well:\n\n{metrics}"
    if geological_context.strip():
        user_message += (
            f"\n\nGeological context provided by the user:\n{geological_context}"
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return message.content[0].text


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

PROVIDERS = {
    "Gemini (Free)": {
        "func": generate_gemini_interpretation,
        "available": lambda: HAS_GEMINI,
        "key_help": "Get a free API key from aistudio.google.com",
        "key_prefix": "",
    },
    "Claude (Paid)": {
        "func": generate_claude_interpretation,
        "available": lambda: HAS_ANTHROPIC,
        "key_help": "Get an API key from console.anthropic.com (requires credits)",
        "key_prefix": "sk-ant-",
    },
}


def generate_ai_interpretation(
    df: pd.DataFrame,
    net_stats: dict,
    detected: dict,
    api_key: str,
    provider: str = "Gemini (Free)",
    geological_context: str = "",
) -> str:
    """
    Generate an AI-enhanced interpretation using the selected provider.

    Parameters
    ----------
    df : pd.DataFrame
        Interpreted log data with VSHALE, PHIE, SW, NET_PAY columns.
    net_stats : dict
        Net pay summary from compute_net_pay_summary().
    detected : dict
        Detected curve mapping from detect_log_curves().
    api_key : str
        API key for the selected provider.
    provider : str
        One of the keys in PROVIDERS dict.
    geological_context : str, optional
        User-provided context (basin, formation, fluid type, etc.).

    Returns
    -------
    str
        AI-generated interpretation text.
    """
    prov = PROVIDERS.get(provider)
    if prov is None:
        return f"Unknown provider: {provider}"

    if not prov["available"]():
        pkg = "google-genai" if "Gemini" in provider else "anthropic"
        return f"Required package not installed. Run: pip install {pkg}"

    return prov["func"](df, net_stats, detected, api_key, geological_context)


def get_available_providers() -> list:
    """Return list of provider names whose packages are installed."""
    return [name for name, p in PROVIDERS.items() if p["available"]()]


def is_available() -> bool:
    """Check if at least one AI provider is installed."""
    return HAS_GEMINI or HAS_ANTHROPIC
