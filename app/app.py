"""Azure Capacity Advisor — Streamlit GUI.

Provides a web interface for uploading rightsizing datasets, validating
VM SKU availability against Azure regions, and exporting the results.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import inspect
import io
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path so modules can be imported cleanly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import (
    AppConfig,
    DEFAULT_REGIONS,
    MAX_DATASET_ROWS,
    REGION_DISPLAY_NAMES,
)
from azure_client.auth import AzureAuthError, AuthMethod, reset_credential, test_connection
from azure_client.deployment_validator import DeploymentValidator, DeploymentValidationError
from azure_client.pricing_service import PricingService, SkuPricing
from azure_client.sku_service import SkuService, SkuServiceError
from engine.alternatives import AlternativeEngine
from engine.analyzer import Analyzer
from engine.capacity_validator import CapacityValidator
from engine.disk_analyzer import DiskAnalyzer, DiskAnalysisResult, DiskStatus
from models.result import AnalysisSummary, SkuStatus
from models.disk import Disk
from parsers.dataset_parser import DatasetParseError, ExcelParseResult, parse_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Azure Capacity Advisor",
    page_icon="\u2601\ufe0f",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Adam System — Design System CSS
# ---------------------------------------------------------------------------
def _inject_custom_css() -> None:
    """Inject the Adam System design tokens and component styles."""
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        /* ===========================================
           Adam System — Design Tokens & Components
           Calm · Structured · Signal > Decoration
           =========================================== */

        /* ---- Base Typography ---- */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"],
        .stMarkdown, .stText {
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        }

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0F172A; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

        /* ---- Top Header Bar ---- */
        [data-testid="stHeader"] {
            background: rgba(15, 23, 42, 0.88) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid #374151;
        }

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            border-right: 1px solid #374151 !important;
        }

        /* ---- Branded Header ---- */
        .adam-header {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 8px 0 24px;
            border-bottom: 1px solid #374151;
            margin-bottom: 4px;
        }

        .adam-header-icon {
            background: #2563EB;
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            flex-shrink: 0;
        }

        .adam-header h1 {
            margin: 0 !important;
            font-size: 32px !important;
            font-weight: 700 !important;
            color: #F9FAFB !important;
            letter-spacing: -0.02em !important;
            line-height: 1.2 !important;
            padding: 0 !important;
        }

        .adam-header p {
            margin: 4px 0 0 !important;
            color: #9CA3AF !important;
            font-size: 16px !important;
            line-height: 1.5 !important;
            max-width: 65ch;
        }

        /* ---- Sidebar Brand ---- */
        .adam-sidebar-brand {
            padding: 0 0 16px;
            border-bottom: 1px solid #374151;
            margin-bottom: 16px;
        }

        .adam-sidebar-brand h2 {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #F9FAFB !important;
            margin: 0 !important;
            padding: 0 !important;
            letter-spacing: -0.01em !important;
        }

        .adam-sidebar-brand p {
            font-size: 12px !important;
            color: #9CA3AF !important;
            margin: 2px 0 0 !important;
            letter-spacing: 0.04em !important;
            text-transform: uppercase !important;
        }

        /* ---- Section Headers ---- */
        .adam-section {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 32px 0 16px !important;
            padding-bottom: 12px;
            border-bottom: 1px solid #374151;
        }

        .adam-section .step {
            background: #2563EB;
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
            flex-shrink: 0;
        }

        .adam-section h3 {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 20px !important;
            font-weight: 600 !important;
            color: #F9FAFB !important;
        }

        /* ---- Metrics ---- */
        [data-testid="stMetric"] {
            background: #111827 !important;
            border: 1px solid #374151 !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }

        [data-testid="stMetricValue"] {
            font-weight: 700 !important;
            font-size: 32px !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 12px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            color: #9CA3AF !important;
        }

        /* ---- Buttons ---- */
        .stButton > button {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            min-height: 40px !important;
            padding-left: 16px !important;
            padding-right: 16px !important;
            transition: all 0.15s ease !important;
        }

        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: #2563EB !important;
            border-color: #2563EB !important;
        }

        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover {
            background: #1d4ed8 !important;
            border-color: #1d4ed8 !important;
        }

        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="baseButton-secondary"] {
            background: #1F2937 !important;
            border: 1px solid #374151 !important;
            color: #D1D5DB !important;
        }

        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid="baseButton-secondary"]:hover {
            border-color: #2563EB !important;
            color: #F9FAFB !important;
        }

        /* ---- Inputs ---- */
        .stTextInput input,
        .stNumberInput input {
            background: #1F2937 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
            color: #F9FAFB !important;
            font-family: 'Inter', sans-serif !important;
        }

        .stTextInput input:focus,
        .stNumberInput input:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
        }

        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stFileUploader label {
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #D1D5DB !important;
        }

        /* ---- Selectbox / Multiselect (baseweb) ---- */
        [data-baseweb="select"] > div {
            background: #1F2937 !important;
            border-color: #374151 !important;
            border-radius: 8px !important;
        }

        [data-baseweb="select"] > div:focus-within {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
        }

        /* Dropdown menu */
        [data-baseweb="popover"] {
            background: #1F2937 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
        }

        [data-baseweb="menu"] {
            background: #1F2937 !important;
        }

        [role="option"] {
            color: #D1D5DB !important;
        }

        [role="option"]:hover,
        [aria-selected="true"] {
            background: #2563EB !important;
            color: #F9FAFB !important;
        }

        /* ---- File Uploader ---- */
        [data-testid="stFileUploader"] section {
            border: 1px dashed #374151 !important;
            border-radius: 12px !important;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: #2563EB !important;
        }

        /* ---- Expanders ---- */
        [data-testid="stExpander"] {
            background: #111827 !important;
            border: 1px solid #374151 !important;
            border-radius: 12px !important;
        }

        [data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: #D1D5DB !important;
        }

        /* ---- Download Button ---- */
        [data-testid="stDownloadButton"] button {
            background: #2563EB !important;
            border-color: #2563EB !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            min-height: 40px !important;
        }

        [data-testid="stDownloadButton"] button:hover {
            background: #1d4ed8 !important;
            border-color: #1d4ed8 !important;
        }

        /* ---- Dividers ---- */
        hr {
            border-color: #374151 !important;
            opacity: 1 !important;
        }

        /* ---- Dataframe ---- */
        [data-testid="stDataFrame"] {
            border: 1px solid #374151 !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }

        /* ---- Alert Messages ---- */
        [data-testid="stAlert"] {
            border-radius: 8px !important;
        }

        /* ---- Sidebar Footer ---- */
        .adam-version {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #9CA3AF;
            letter-spacing: 0.04em;
            padding-top: 8px;
            border-top: 1px solid #374151;
        }

        /* ---- Tabs (if used) ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            font-weight: 600;
        }

        /* ---- Family Group Header ---- */
        .fam-hdr {
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 10px;
        }
        .fam-hdr .fam-title { font-size: 16px; font-weight: 700; color: #F9FAFB; }
        .fam-hdr .fam-sub { color: #9CA3AF; font-size: 12px; margin-top: 2px; }
        .fam-hdr .fam-pills { display: flex; gap: 6px; flex-wrap: wrap; }
        .fam-hdr .pill {
            padding: 2px 10px; border-radius: 12px;
            font-size: 12px; font-weight: 600; line-height: 1.5;
        }
        /* ---- Server Row ---- */
        .srv {
            display: grid;
            grid-template-columns: minmax(0,1fr) auto;
            align-items: start;
            gap: 16px;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 4px;
        }
        .srv .srv-left { min-width: 0; }
        .srv .srv-name {
            font-size: 14px; font-weight: 600; color: #F9FAFB;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .srv .srv-meta { color: #9CA3AF; font-size: 12px; margin-top: 1px; }
        .srv .srv-right { text-align: right; white-space: nowrap; }
        .srv .srv-verdict {
            font-size: 13px; font-weight: 700;
            padding: 2px 10px; border-radius: 4px;
            display: inline-block;
        }
        .srv .srv-action {
            font-size: 12px; margin-top: 3px; color: #D1D5DB;
        }
        .srv .srv-action .alt-name { font-weight: 600; }
        .srv .srv-action .verified { color: #22c55e; }
        .srv .srv-action .unchecked { color: #9CA3AF; }
        .srv .srv-action .failed { color: #ef4444; }
        .srv .srv-action .cost-save { color: #22c55e; font-weight: 600; font-size: 11px; }
        .srv .srv-action .cost-more { color: #f59e0b; font-weight: 600; font-size: 11px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_header(number: int, title: str) -> None:
    """Render a numbered section header in Adam System style."""
    st.markdown(
        f'<div class="adam-section"><span class="step">{number}</span>'
        f"<h3>{title}</h3></div>",
        unsafe_allow_html=True,
    )


def _build_region_options() -> list[str]:
    """Build display labels for the region dropdown."""
    return [REGION_DISPLAY_NAMES.get(r, r) for r in DEFAULT_REGIONS]


def _display_label_to_region(label: str) -> str:
    """Convert a display label back to an Azure region id."""
    for region_id, display in REGION_DISPLAY_NAMES.items():
        if display == label:
            return region_id
    return label.lower().replace(" ", "")


def _status_badge(status: SkuStatus) -> str:
    """Return a coloured status label for display."""
    colors = {
        SkuStatus.OK: "green",
        SkuStatus.RISK: "orange",
        SkuStatus.BLOCKED: "red",
        SkuStatus.UNKNOWN: "gray",
    }
    color = colors.get(status, "gray")
    return f":{color}[**{status.value}**]"


def _format_alternatives(result) -> str:
    """Format alternatives with specs for display."""
    if not result.alternatives:
        return "\u2014"
    if result.alternatives_detail:
        parts = []
        for d in result.alternatives_detail:
            parts.append(
                f"{d['name']} ({d['vcpu']}vCPU, {d['memory_gb']}GB, "
                f"{d['max_disks']} disks)"
            )
        return " | ".join(parts)
    return ", ".join(result.alternatives)


def _build_readiness_label(result) -> str:
    """Build a short readiness label from status + alternatives."""
    if result.status == SkuStatus.OK:
        if result.capacity_verified is True:
            return "Ready (Verified)"
        return "Ready"
    if result.status == SkuStatus.RISK:
        return "Zone Limited"
    if result.status == SkuStatus.BLOCKED:
        n = len(result.alternatives)
        if n > 0:
            return f"Alt Available ({n})"
        return "Blocked"
    return "Unknown"


def _best_alternative_label(result) -> str:
    """Return the best alternative SKU name with its capacity status."""
    if not result.alternatives_detail:
        return "\u2014"
    best = result.alternatives_detail[0]  # Already sorted verified-first
    name = best["name"]
    cap = best.get("capacity", "Not Checked")
    if cap == "Verified":
        return f"{name} (Verified)"
    if cap == "Failed":
        return f"{name} (No Capacity)"
    return name


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


def _short_reason(result) -> str:
    """Extract a short, human-readable reason from the full analysis reason.

    Returns a brief label suitable for inline display in server rows.
    """
    if result.status == SkuStatus.OK:
        if result.capacity_verified is True:
            return "Capacity confirmed"
        if result.capacity_verified is False:
            return "Catalog OK but capacity exhausted"
        return ""
    if result.status == SkuStatus.RISK:
        return "Not in all availability zones"

    reason = result.reason or ""
    if "NOT offered in" in reason:
        return "Not available in this region"
    if "RESTRICTED" in reason:
        if "NotAvailableForSubscription" in reason:
            return "Subscription lacks quota for this SKU"
        return "Restricted on this subscription"
    if "does not exist" in reason:
        return "SKU not found in Azure catalog"
    if result.capacity_error_code:
        return f"Capacity: {result.capacity_error_code}"
    return ""


def _format_device_code_expiry(expires_on: int | None) -> str:
    """Format device code expiry timestamp for display."""
    if not expires_on:
        return ""
    try:
        expires_utc = datetime.fromtimestamp(int(expires_on), tz=timezone.utc)
        return expires_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError, OSError):
        return ""


def _render_device_code_prompt(
    container,
    verification_uri: str,
    user_code: str,
    expires_on: int | None,
) -> None:
    """Render in-app instructions for Azure device code login."""
    expiry_text = _format_device_code_expiry(expires_on)
    lines = [
        "Azure device login required.",
        "",
        f"1. Open `{verification_uri}`",
        f"2. Enter code: `{user_code}`",
        "3. Complete sign-in in that browser tab and return here.",
    ]
    if expiry_text:
        lines.append(f"Code expires at: `{expiry_text}`")

    container.info("\n".join(lines))
    logger.info(
        "Device code prompt shown in UI (url=%s, code=%s, expires_on=%s)",
        verification_uri,
        user_code,
        expires_on,
    )


def _friendly_auth_error(auth_method: AuthMethod, exc: AzureAuthError) -> str:
    """Convert noisy Azure SDK auth errors into user-actionable guidance."""
    msg = str(exc)
    if "unexpected keyword argument 'device_code_callback'" in msg:
        return (
            "Authentication component versions are out of sync in this deployment. "
            "Redeploy with the latest code so app and auth modules match."
        )
    if auth_method == AuthMethod.INTERACTIVE_BROWSER and "Failed to open a browser" in msg:
        return (
            "Interactive Browser auth is not supported in hosted Streamlit environments. "
            "Use Device Code (Browser Login) or Service Principal."
        )
    if auth_method == AuthMethod.DEFAULT and "DefaultAzureCredential failed to retrieve a token" in msg:
        return (
            "Default auth could not find usable credentials in this environment "
            "(no Azure CLI/managed identity/env credentials). Use Device Code "
            "(Browser Login) or Service Principal."
        )
    return f"Auth failed: {msg}"


def _prefer_device_code_default() -> bool:
    """Return True when auth dropdown should default to Device Code.

    Override via env var:
    - `AZURE_AUTH_DEFAULT=device_code` to force Device Code
    - `AZURE_AUTH_DEFAULT=default` to force Default chain
    """
    override = os.environ.get("AZURE_AUTH_DEFAULT", "").strip().lower()
    if override in {"device_code", "device", "dc"}:
        return True
    if override in {"default", "cli"}:
        return False

    # Streamlit Community Cloud commonly sets this flag. Keep heuristic narrow
    # to avoid changing behavior for local development.
    return bool(os.environ.get("STREAMLIT_SHARING_MODE"))


def _supports_kwarg(target, kwarg: str) -> bool:
    """Return True if a callable signature accepts a specific keyword arg."""
    try:
        return kwarg in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


def _short_verdict(result) -> str:
    """One-line verdict for expander labels."""
    if result.status == SkuStatus.OK and result.capacity_verified is True:
        return "Deploy"
    if result.status == SkuStatus.OK and result.capacity_verified is None:
        return "Catalogue OK"
    # Needs alternative
    if result.alternatives_detail:
        top = result.alternatives_detail[0]
        cap = top.get("capacity", "")
        tag = "verified" if cap == "Verified" else "not checked" if cap != "Failed" else "no capacity"
        return f"Use {top['name']} ({tag})"
    if result.status == SkuStatus.RISK:
        return "Zone limited"
    if result.status == SkuStatus.BLOCKED:
        return "Blocked — no alternative"
    return "Unknown"


def _render_server_row(result) -> str:
    """Build HTML for a single server row inside a family group.

    Layout:
        Left:  server name + SKU / specs meta line
        Right: verdict badge + action line (alternative if needed)
    """
    # --- Determine the verdict ---
    if result.status == SkuStatus.OK and result.capacity_verified is True:
        verdict_text = "DEPLOY"
        verdict_bg = "#22c55e"
        verdict_fg = "#fff"
        row_border = "#22c55e"
    elif result.status == SkuStatus.OK and result.capacity_verified is False:
        verdict_text = "NO CAPACITY"
        verdict_bg = "#ef4444"
        verdict_fg = "#fff"
        row_border = "#ef4444"
    elif result.status == SkuStatus.OK:
        verdict_text = "CATALOGUE OK"
        verdict_bg = "rgba(22,163,74,0.15)"
        verdict_fg = "#22c55e"
        row_border = "#22c55e"
    elif result.status == SkuStatus.RISK:
        verdict_text = "ZONE LIMITED"
        verdict_bg = "rgba(245,158,11,0.15)"
        verdict_fg = "#f59e0b"
        row_border = "#f59e0b"
    elif result.status == SkuStatus.BLOCKED:
        verdict_text = "BLOCKED"
        verdict_bg = "rgba(239,68,68,0.15)"
        verdict_fg = "#ef4444"
        row_border = "#ef4444"
    else:
        verdict_text = "UNKNOWN"
        verdict_bg = "rgba(107,114,128,0.15)"
        verdict_fg = "#9CA3AF"
        row_border = "#6b7280"

    # --- Action line: best alternative + cost delta ---
    action_html = ""
    if result.status in (SkuStatus.BLOCKED, SkuStatus.RISK) or (
        result.status == SkuStatus.OK and result.capacity_verified is False
    ):
        if result.alternatives_detail:
            top = result.alternatives_detail[0]
            alt_name = top["name"]
            cap = top.get("capacity", "Not Checked")

            if cap == "Verified":
                cap_cls = "verified"
                cap_label = "capacity verified"
            elif cap == "Failed":
                cap_cls = "failed"
                cap_label = "no capacity"
            else:
                cap_cls = "unchecked"
                cap_label = "not checked"

            # Cost delta
            cost_bit = ""
            delta_raw = top.get("delta_payg", "")
            if delta_raw:
                dv = float(delta_raw)
                if dv < 0:
                    cost_bit = f' <span class="cost-save">&darr; ${abs(dv):,.0f}/mo</span>'
                elif dv > 0:
                    cost_bit = f' <span class="cost-more">&uarr; ${dv:,.0f}/mo</span>'

            action_html = (
                f'<div class="srv-action">'
                f'Use <span class="alt-name">{alt_name}</span> '
                f'<span class="{cap_cls}">({cap_label})</span>'
                f'{cost_bit}'
                f'</div>'
            )
        else:
            short = _short_reason(result)
            reason_line = f" &mdash; {short}" if short else ""
            action_html = (
                f'<div class="srv-action" style="color:#ef4444;">'
                f'No alternative available{reason_line}'
                f'</div>'
            )

    row_bg = f"rgba({int(row_border[1:3],16)},{int(row_border[3:5],16)},{int(row_border[5:7],16)},0.04)"

    return (
        f'<div class="srv" style="background:{row_bg}; border:1px solid {row_border}22;">'
        f'<div class="srv-left">'
        f'<div class="srv-name">{result.machine_name}</div>'
        f'<div class="srv-meta">'
        f'{result.requested_sku}'
        f' &middot; {result.vcpu or "?"} vCPU'
        f' &middot; {result.memory_gb or "?"} GB'
        f' &middot; {result.disk_count} disk(s)'
        f' &middot; {result.display_region}'
        f'</div>'
        f'</div>'
        f'<div class="srv-right">'
        f'<span class="srv-verdict" style="background:{verdict_bg}; color:{verdict_fg};">'
        f'{verdict_text}</span>'
        f'{action_html}'
        f'</div>'
        f'</div>'
    )


def _render_grouped_view(
    results: list,
    filtered_df: pd.DataFrame,
    disk_results,
) -> None:
    """Render machines grouped by VM Family with drill-down to individual servers.

    Family header answers: "Can this family deploy?"
    Server rows answer:    "Can THIS server deploy? If not, use what instead?"
    """
    machine_names_set = set(filtered_df["Machine"].tolist())
    if not machine_names_set:
        st.info("No machines match the current filters.")
        return

    result_map = {r.machine_name: r for r in results}
    filtered_results = [result_map[n] for n in machine_names_set if n in result_map]

    # Group by VM Family
    families: dict[str, list] = defaultdict(list)
    for r in filtered_results:
        fam = r.vm_family or "Unknown"
        families[fam].append(r)

    # Sort families: most action-needed first
    def _family_sort_key(item):
        _fam, members = item
        blocked = sum(1 for m in members if m.status == SkuStatus.BLOCKED)
        cap_fail = sum(
            1 for m in members
            if m.status == SkuStatus.OK and m.capacity_verified is False
        )
        risk = sum(1 for m in members if m.status == SkuStatus.RISK)
        return (-(blocked + cap_fail), -risk, _fam)

    sorted_families = sorted(families.items(), key=_family_sort_key)

    st.caption(
        f"{len(filtered_results)} servers across {len(sorted_families)} VM families"
    )

    for fam_name, members in sorted_families:
        count = len(members)

        # Classify members into outcome groups
        can_deploy = sum(
            1 for m in members
            if m.status == SkuStatus.OK and m.capacity_verified is True
        )
        catalogue_ok = sum(
            1 for m in members
            if m.status == SkuStatus.OK and m.capacity_verified is None
        )
        need_alt = sum(
            1 for m in members
            if m.status == SkuStatus.BLOCKED
            or (m.status == SkuStatus.OK and m.capacity_verified is False)
        )
        risk_n = sum(1 for m in members if m.status == SkuStatus.RISK)
        has_verified_alt = sum(
            1 for m in members
            if (m.status == SkuStatus.BLOCKED
                or (m.status == SkuStatus.OK and m.capacity_verified is False))
            and m.alternatives_detail
            and m.alternatives_detail[0].get("capacity") == "Verified"
        )

        # Family colour based on worst outcome
        if need_alt > 0 and has_verified_alt < need_alt:
            border = "#ef4444"
        elif need_alt > 0:
            border = "#f59e0b"  # all blocked have verified alts — actionable
        elif risk_n > 0:
            border = "#f59e0b"
        else:
            border = "#22c55e"
        border_bg = f"rgba({int(border[1:3],16)},{int(border[3:5],16)},{int(border[5:7],16)},0.05)"

        # Pills — outcome-focused
        pills = []
        if can_deploy:
            pills.append(
                f'<span class="pill" style="background:#22c55e22; '
                f'color:#22c55e;">{can_deploy} deploy</span>'
            )
        if catalogue_ok:
            pills.append(
                f'<span class="pill" style="background:#22c55e15; '
                f'color:#86efac;">{catalogue_ok} catalogue&nbsp;OK</span>'
            )
        if risk_n:
            pills.append(
                f'<span class="pill" style="background:#f59e0b22; '
                f'color:#f59e0b;">{risk_n} zone&nbsp;limited</span>'
            )
        if need_alt:
            alt_note = f" ({has_verified_alt} have alt)" if has_verified_alt else ""
            pills.append(
                f'<span class="pill" style="background:#ef444422; '
                f'color:#ef4444;">{need_alt} need&nbsp;alternative{alt_note}</span>'
            )

        # Unique SKUs
        skus = sorted(set(m.requested_sku for m in members))
        sku_label = skus[0] if len(skus) == 1 else f"{len(skus)} SKU variants"

        header_html = (
            f'<div class="fam-hdr" style="background:{border_bg}; '
            f'border:1px solid {border}33; border-left:3px solid {border};">'
            f'<div style="display:flex; justify-content:space-between; '
            f'align-items:flex-start; flex-wrap:wrap; gap:8px;">'
            f'<div>'
            f'<span class="fam-title">{fam_name}</span>'
            f'<div class="fam-sub">'
            f'{count} server{"s" if count != 1 else ""}'
            f' &middot; {sku_label}'
            f'</div>'
            f'</div>'
            f'<div class="fam-pills">{"".join(pills)}</div>'
            f'</div>'
            f'</div>'
        )

        # Sort: needs-action first (blocked, cap-failed), then risk, then OK
        def _member_order(m):
            if m.status == SkuStatus.BLOCKED:
                return (0, m.machine_name)
            if m.status == SkuStatus.OK and m.capacity_verified is False:
                return (1, m.machine_name)
            if m.status == SkuStatus.RISK:
                return (2, m.machine_name)
            if m.status == SkuStatus.UNKNOWN:
                return (3, m.machine_name)
            return (4, m.machine_name)

        members_sorted = sorted(members, key=_member_order)

        # Expander label — lead with outcome
        if need_alt == 0 and risk_n == 0:
            exp_icon = "\u2705"  # green tick
        elif need_alt > 0 and has_verified_alt >= need_alt:
            exp_icon = "\U0001f504"  # swap arrows
        elif need_alt > 0:
            exp_icon = "\u26a0\ufe0f"  # warning
        else:
            exp_icon = "\u26a0\ufe0f"

        with st.expander(
            f"{exp_icon} {fam_name} — {count} servers",
            expanded=(need_alt > 0),
        ):
            st.markdown(header_html, unsafe_allow_html=True)

            # Each server is its own expander — click to see detail
            for m in members_sorted:
                row_html = _render_server_row(m)
                # Build a short label for the expander toggle
                verdict = _short_verdict(m)
                with st.expander(f"{m.machine_name}  —  {verdict}", expanded=False):
                    st.markdown(row_html, unsafe_allow_html=True)
                    _render_machine_detail(m, disk_results, show_header=False)


def _results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert analysis results to a Pandas DataFrame for display."""
    rows = []
    for r in results:
        if r.capacity_verified is True:
            capacity = "Verified"
        elif r.capacity_verified is False:
            capacity = "Failed"
        else:
            capacity = "Not Checked"

        # PAYG cost delta for best alternative
        cost_delta_str = "\u2014"
        if r.alternatives_detail:
            delta_raw = r.alternatives_detail[0].get("delta_payg", "")
            if delta_raw:
                dv = float(delta_raw)
                if dv < 0:
                    cost_delta_str = f"\u2193 ${abs(dv):,.0f}/mo"
                elif dv > 0:
                    cost_delta_str = f"\u2191 ${dv:,.0f}/mo"
                else:
                    cost_delta_str = "="

        rows.append({
            "Machine": r.machine_name,
            "Region": r.display_region,
            "Requested SKU": r.requested_sku,
            "vCPU": int(r.vcpu) if r.vcpu else 0,
            "Memory (GB)": float(r.memory_gb) if r.memory_gb else 0.0,
            "VM Family": r.vm_family if r.vm_family else "",
            "Disks": int(r.disk_count) if r.disk_count else 0,
            "Status": r.status.value,
            "Live Capacity": capacity,
            "Readiness": _build_readiness_label(r),
            "Best Alternative": _best_alternative_label(r),
            "Alt Cost \u0394": cost_delta_str,
        })
    return pd.DataFrame(rows)


def _results_to_export_dataframe(results: list) -> pd.DataFrame:
    """Convert analysis results to a full DataFrame for CSV export (includes Reason & Alternatives)."""
    machine_costs = st.session_state.get("machine_costs", {})
    pricing_data = st.session_state.get("pricing_data", {})
    rows = []
    for r in results:
        if r.capacity_verified is True:
            capacity = "Verified"
        elif r.capacity_verified is False:
            capacity = "Failed"
        else:
            capacity = "Not Checked"

        # Current costs (Excel → API fallback)
        mc = machine_costs.get(r.machine_name, {})
        api_p = pricing_data.get((r.requested_sku, r.region))
        cur_payg = mc.get("payg") or (api_p.payg if api_p else None)
        cur_ri1 = mc.get("ri_1yr") or (api_p.ri_1yr if api_p else None)
        cur_ri3 = mc.get("ri_3yr") or (api_p.ri_3yr if api_p else None)

        # Best alternative pricing
        alt_payg = alt_ri1 = alt_ri3 = ""
        delta_payg = delta_ri1 = delta_ri3 = ""
        if r.alternatives_detail:
            top = r.alternatives_detail[0]
            alt_payg = top.get("price_payg", "")
            alt_ri1 = top.get("price_ri_1yr", "")
            alt_ri3 = top.get("price_ri_3yr", "")
            delta_payg = top.get("delta_payg", "")
            delta_ri1 = top.get("delta_ri_1yr", "")
            delta_ri3 = top.get("delta_ri_3yr", "")

        rows.append({
            "Machine": r.machine_name,
            "Region": r.display_region,
            "Requested SKU": r.requested_sku,
            "vCPU": int(r.vcpu) if r.vcpu else 0,
            "Memory (GB)": float(r.memory_gb) if r.memory_gb else 0.0,
            "VM Family": r.vm_family if r.vm_family else "",
            "Disks": int(r.disk_count) if r.disk_count else 0,
            "Status": r.status.value,
            "Live Capacity": capacity,
            "Readiness": _build_readiness_label(r),
            "Best Alternative": _best_alternative_label(r),
            "Current PAYG $/mo": f"{cur_payg:.2f}" if cur_payg else "",
            "Current 1yr RI $/mo": f"{cur_ri1:.2f}" if cur_ri1 else "",
            "Current 3yr RI $/mo": f"{cur_ri3:.2f}" if cur_ri3 else "",
            "Alt PAYG $/mo": alt_payg,
            "Alt 1yr RI $/mo": alt_ri1,
            "Alt 3yr RI $/mo": alt_ri3,
            "Delta PAYG $/mo": delta_payg,
            "Delta 1yr RI $/mo": delta_ri1,
            "Delta 3yr RI $/mo": delta_ri3,
            "Reason": r.reason,
            "Alternatives": _format_alternatives(r),
        })
    return pd.DataFrame(rows)


def _build_updated_export(results: list) -> bytes:
    """Build an Excel file that mirrors the original input format.

    Re-reads the original uploaded Excel, updates the "Chosen SKU" column
    to the recommended alternative where the original is blocked or has no
    capacity, and appends advisory columns so the output can be re-ingested
    by the same tool or Dr Migrate.

    Falls back to a flat CSV-style Excel if the original file wasn't Excel.
    """
    raw_bytes = st.session_state.get("raw_bytes")
    uploaded_name = st.session_state.get("uploaded_name", "")

    # Build lookup: machine_name -> result
    result_map = {r.machine_name: r for r in results}

    is_excel = uploaded_name.lower().endswith((".xlsx", ".xls"))

    if not is_excel or not raw_bytes:
        # Fallback: flat export with the key columns
        return _build_flat_export(results)

    output = io.BytesIO()

    try:
        # Re-read the original Excel preserving all sheets
        original = pd.ExcelFile(io.BytesIO(raw_bytes), engine="openpyxl")

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_name in original.sheet_names:
                if sheet_name.lower() == "servers":
                    _write_updated_servers_sheet(
                        writer, io.BytesIO(raw_bytes), sheet_name, result_map,
                    )
                else:
                    # Copy other sheets as-is (e.g. Disks, Summary)
                    df = pd.read_excel(
                        io.BytesIO(raw_bytes),
                        sheet_name=sheet_name,
                        header=None,
                        engine="openpyxl",
                    )
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

    except Exception as exc:
        logger.warning("Failed to build mirrored Excel export: %s", exc)
        return _build_flat_export(results)

    return output.getvalue()


def _write_updated_servers_sheet(
    writer: pd.ExcelWriter,
    raw_source: io.BytesIO,
    sheet_name: str,
    result_map: dict,
) -> None:
    """Write the Servers sheet with updated SKUs and advisory columns.

    Preserves the original header rows (1-5) and structure. Headers are on
    row 6. Data starts at row 7.
    """
    # Read the full sheet with no header (so rows 1-5 are kept as data)
    raw_df = pd.read_excel(raw_source, sheet_name=sheet_name, header=None, engine="openpyxl")

    # Row 6 (0-indexed row 5) is the header row
    headers = raw_df.iloc[5].tolist()
    header_lower = [str(h).strip().lower() for h in headers]

    # Find key column indices
    server_idx = None
    sku_idx = None
    for i, h in enumerate(header_lower):
        if h == "server":
            server_idx = i
        if h == "chosen sku":
            sku_idx = i

    if server_idx is None or sku_idx is None:
        # Can't map — write as-is
        raw_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        return

    # Determine advisory column positions (append after last existing column)
    n_cols = len(headers)
    col_orig_sku = n_cols
    col_status = n_cols + 1
    col_verdict = n_cols + 2
    col_reason = n_cols + 3

    # Add header labels for new columns in the header row
    raw_df.iloc[5, col_orig_sku] = "Original SKU"
    raw_df.iloc[5, col_status] = "Capacity Status"
    raw_df.iloc[5, col_verdict] = "Verdict"
    raw_df.iloc[5, col_reason] = "Reason"

    # Update data rows (row 7 onwards = 0-indexed row 6+)
    for row_idx in range(6, len(raw_df)):
        server_name = str(raw_df.iloc[row_idx, server_idx]).strip()
        r = result_map.get(server_name)
        if not r:
            continue

        original_sku = str(raw_df.iloc[row_idx, sku_idx]).strip()

        # Decide whether to swap the SKU
        needs_swap = (
            r.status == SkuStatus.BLOCKED
            or (r.status == SkuStatus.OK and r.capacity_verified is False)
        )

        best_alt = None
        if needs_swap and r.alternatives_detail:
            # Prefer verified alternatives
            for alt in r.alternatives_detail:
                if alt.get("capacity") == "Verified":
                    best_alt = alt["name"]
                    break
            if not best_alt:
                best_alt = r.alternatives_detail[0]["name"]

        if best_alt:
            raw_df.iloc[row_idx, sku_idx] = best_alt
            raw_df.iloc[row_idx, col_orig_sku] = original_sku
        else:
            raw_df.iloc[row_idx, col_orig_sku] = ""

        # Status
        if r.status == SkuStatus.OK and r.capacity_verified is True:
            raw_df.iloc[row_idx, col_status] = "Deploy"
        elif r.status == SkuStatus.OK and r.capacity_verified is None:
            raw_df.iloc[row_idx, col_status] = "Catalogue OK"
        elif r.status == SkuStatus.OK and r.capacity_verified is False:
            raw_df.iloc[row_idx, col_status] = "No Capacity"
        elif r.status == SkuStatus.RISK:
            raw_df.iloc[row_idx, col_status] = "Zone Limited"
        elif r.status == SkuStatus.BLOCKED:
            raw_df.iloc[row_idx, col_status] = "Blocked"
        else:
            raw_df.iloc[row_idx, col_status] = "Unknown"

        # Verdict
        if best_alt:
            cap = next(
                (a.get("capacity", "") for a in r.alternatives_detail if a["name"] == best_alt),
                "",
            )
            if cap == "Verified":
                cap_note = " (Capacity Verified)"
            elif cap == "Failed":
                cap_note = " (No Capacity - check manually)"
            else:
                cap_note = " (Not Capacity Checked)"
            raw_df.iloc[row_idx, col_verdict] = f"Use {best_alt}{cap_note}"
        elif r.status == SkuStatus.OK:
            raw_df.iloc[row_idx, col_verdict] = "Keep"
        elif r.status == SkuStatus.RISK:
            raw_df.iloc[row_idx, col_verdict] = "Review zones"
        elif r.status == SkuStatus.BLOCKED:
            raw_df.iloc[row_idx, col_verdict] = "No alternative available — manual review required"
        else:
            raw_df.iloc[row_idx, col_verdict] = "No alternative available"

        # Reason
        raw_df.iloc[row_idx, col_reason] = _short_reason(r) or ""

    raw_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)


def _build_flat_export(results: list) -> bytes:
    """Fallback flat Excel export when the original wasn't an Excel file."""
    rows = []
    for r in results:
        best_alt = ""
        alt_cap = ""
        if r.alternatives_detail:
            best_alt = r.alternatives_detail[0]["name"]
            alt_cap = r.alternatives_detail[0].get("capacity", "Not Checked")

        needs_swap = (
            r.status == SkuStatus.BLOCKED
            or (r.status == SkuStatus.OK and r.capacity_verified is False)
        )

        if needs_swap and best_alt:
            cap_note = {"Verified": " (Capacity Verified)", "Failed": " (No Capacity)"}.get(alt_cap, " (Not Capacity Checked)")
            verdict = f"Use {best_alt}{cap_note}"
        elif needs_swap:
            verdict = "No alternative available — manual review required"
        elif r.status == SkuStatus.RISK:
            verdict = "Review zones"
        else:
            verdict = "Keep"

        rows.append({
            "Server": r.machine_name,
            "Target Azure Region": r.display_region,
            "Chosen SKU": best_alt if (needs_swap and best_alt) else r.requested_sku,
            "Original SKU": r.requested_sku if (needs_swap and best_alt) else "",
            "Capacity Status": _build_readiness_label(r),
            "Verdict": verdict,
            "Reason": _short_reason(r) or "",
        })

    output = io.BytesIO()
    pd.DataFrame(rows).to_excel(output, index=False, engine="openpyxl")
    return output.getvalue()



def _render_executive_summary(results: list, summary: AnalysisSummary) -> None:
    """Render the executive summary dashboard with donut chart, metrics, and heatmap."""
    capacity_ran = st.session_state.get("capacity_validation_ran", False)

    # Categorise machines into deployment readiness groups
    ready = sum(
        1 for r in results
        if r.status == SkuStatus.OK
        or (r.status == SkuStatus.OK and r.capacity_verified is True)
    )
    risk = sum(1 for r in results if r.status == SkuStatus.RISK)
    needs_alt = sum(
        1 for r in results
        if r.status == SkuStatus.BLOCKED and len(r.alternatives) > 0
    )
    blocked_no_path = sum(
        1 for r in results
        if (r.status == SkuStatus.BLOCKED and len(r.alternatives) == 0)
        or r.status == SkuStatus.UNKNOWN
    )
    # Include capacity-failed OK machines as needing action
    capacity_failed = sum(
        1 for r in results
        if r.capacity_verified is False and r.status == SkuStatus.OK
    )
    if capacity_ran and capacity_failed:
        ready -= capacity_failed
        needs_alt += capacity_failed

    # --- Row 1: Donut chart + Metrics ---
    chart_col, metrics_col = st.columns([1, 2])

    with chart_col:
        labels = ["Ready to Deploy", "Needs Action", "Blocked — No Path"]
        values = [ready, risk + needs_alt, blocked_no_path]
        colors = ["#22c55e", "#f59e0b", "#ef4444"]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors),
            textinfo="value+percent",
            textfont=dict(size=13, color="#F9FAFB"),
            hovertemplate="<b>%{label}</b><br>%{value} machines<br>%{percent}<extra></extra>",
        )])
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                font=dict(color="#D1D5DB", size=12),
            ),
            margin=dict(t=10, b=40, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text=f"<b>{summary.total}</b><br><span style='font-size:12px;color:#9CA3AF'>machines</span>",
                x=0.5, y=0.5, font=dict(size=28, color="#F9FAFB"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with metrics_col:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ready to Deploy", ready)
        m2.metric("Needs Alternative", needs_alt)
        m3.metric("Blocked — No Path", blocked_no_path)
        m4.metric("Risk — Zone Limited", risk)

        # Actionable summary text
        parts = [f"**{ready}** of **{summary.total}** machines ready to deploy."]
        if needs_alt:
            # Count how many alternatives across blocked machines are capacity-verified
            alts_verified = sum(
                1 for r in results
                if r.status == SkuStatus.BLOCKED and r.alternatives_detail
                for a in r.alternatives_detail if a.get("capacity") == "Verified"
            )
            if alts_verified:
                parts.append(
                    f"**{needs_alt}** need alternatives — **{alts_verified}** alternative(s) capacity-verified."
                )
            else:
                parts.append(f"**{needs_alt}** need alternative SKUs (alternatives found).")
        if blocked_no_path:
            parts.append(f"**{blocked_no_path}** blocked with no path forward.")
        if risk:
            parts.append(f"**{risk}** available but zone-limited.")
        if capacity_ran and capacity_failed:
            parts.append(
                f"**{capacity_failed}** passed catalog check but failed live capacity."
            )
        st.markdown(" ".join(parts))

    # --- Row 2: Region Heatmap ---
    regions_in_data = sorted(set(r.display_region for r in results))
    if len(regions_in_data) > 1:
        with st.expander("Region Breakdown", expanded=True):
            status_cols = ["OK", "RISK", "BLOCKED", "UNKNOWN"]
            heatmap_data: dict[str, list[int]] = {s: [] for s in status_cols}
            for region in regions_in_data:
                region_results = [r for r in results if r.display_region == region]
                for s in status_cols:
                    heatmap_data[s].append(
                        sum(1 for r in region_results if r.status.value == s)
                    )

            fig_hm = go.Figure(data=go.Heatmap(
                z=[heatmap_data[s] for s in status_cols],
                x=regions_in_data,
                y=status_cols,
                colorscale=[
                    [0.0, "#1F2937"],
                    [0.25, "#374151"],
                    [0.5, "#f59e0b"],
                    [0.75, "#ef4444"],
                    [1.0, "#dc2626"],
                ],
                text=[[v for v in heatmap_data[s]] for s in status_cols],
                texttemplate="%{text}",
                textfont=dict(size=14, color="#F9FAFB"),
                hovertemplate="<b>%{y}</b> in %{x}<br>Count: %{z}<extra></extra>",
                showscale=False,
            ))
            fig_hm.update_layout(
                height=max(180, 45 * len(status_cols)),
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    tickfont=dict(color="#D1D5DB", size=11),
                    tickangle=-45 if len(regions_in_data) > 6 else 0,
                ),
                yaxis=dict(tickfont=dict(color="#D1D5DB", size=12)),
            )
            st.plotly_chart(fig_hm, width="stretch", config={"displayModeBar": False})


def _render_machine_detail(
    result,
    disk_results: list[DiskAnalysisResult] | None,
    show_header: bool = True,
) -> None:
    """Render a detail card for a single selected machine.

    Args:
        show_header: If False, skips the header card (used when rendered
                     inside a card's expander where the header is redundant).
    """
    # Status color mapping
    status_colors = {
        "OK": ("#22c55e", "rgba(22, 163, 74, 0.15)"),
        "RISK": ("#f59e0b", "rgba(245, 158, 11, 0.15)"),
        "BLOCKED": ("#ef4444", "rgba(239, 68, 68, 0.15)"),
        "UNKNOWN": ("#9CA3AF", "rgba(156, 163, 175, 0.10)"),
    }
    text_color, bg_color = status_colors.get(
        result.status.value, ("#9CA3AF", "rgba(156,163,175,0.10)")
    )

    if show_header:
        # Header card
        st.markdown(
            f"""
            <div style="background:{bg_color}; border:1px solid {text_color}33;
                        border-radius:12px; padding:20px; margin:8px 0 16px;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                    <span style="background:{text_color}; color:#fff; padding:4px 12px;
                                 border-radius:6px; font-weight:700; font-size:13px;
                                 letter-spacing:0.04em;">
                        {result.status.value}
                    </span>
                    <span style="font-size:20px; font-weight:700; color:#F9FAFB;">
                        {result.machine_name}
                    </span>
                </div>
                <div style="color:#9CA3AF; font-size:14px;">
                    {result.display_region} &nbsp;|&nbsp;
                    {result.requested_sku} &nbsp;|&nbsp;
                    {result.vcpu or '?'} vCPU &nbsp;|&nbsp;
                    {result.memory_gb or '?'} GB RAM &nbsp;|&nbsp;
                    {result.disk_count} disk(s)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Reason
    st.markdown("**Assessment**")
    st.markdown(
        f'<div style="background:#111827; border:1px solid #374151; border-radius:8px; '
        f'padding:16px; color:#D1D5DB; line-height:1.6; font-size:14px;">'
        f"{result.reason}</div>",
        unsafe_allow_html=True,
    )

    # Alternatives comparison table
    if result.alternatives_detail:
        verified_count = sum(1 for a in result.alternatives_detail if a.get("capacity") == "Verified")
        total_alts = len(result.alternatives_detail)
        if verified_count:
            st.markdown(f"**Alternative SKUs** ({verified_count} of {total_alts} capacity-verified)")
        else:
            st.markdown(f"**Alternative SKUs** ({total_alts})")

        # Show current SKU cost as reference (Excel data preferred, API fallback)
        machine_costs = st.session_state.get("machine_costs", {})
        pricing_data = st.session_state.get("pricing_data", {})
        mc = machine_costs.get(result.machine_name, {})
        api_p = pricing_data.get((result.requested_sku, result.region))

        cur_payg = mc.get("payg") or (api_p.payg if api_p else None)
        cur_ri1 = mc.get("ri_1yr") or (api_p.ri_1yr if api_p else None)
        cur_ri3 = mc.get("ri_3yr") or (api_p.ri_3yr if api_p else None)

        cost_parts = []
        if cur_payg:
            cost_parts.append(f"PAYG ${cur_payg:,.2f}")
        if cur_ri1:
            cost_parts.append(f"1yr RI ${cur_ri1:,.2f}")
        if cur_ri3:
            cost_parts.append(f"3yr RI ${cur_ri3:,.2f}")
        if cost_parts:
            source = "assessment" if mc.get("payg") else "Azure retail"
            st.caption(
                f"Current SKU ({result.requested_sku}): "
                f"**{' / '.join(cost_parts)}** /mo — source: {source}"
            )

        alt_rows = []
        for d in result.alternatives_detail:
            # Compute delta vs original
            try:
                alt_vcpu = int(d["vcpu"])
                orig_vcpu = result.vcpu or 0
                vcpu_delta = alt_vcpu - orig_vcpu
                vcpu_str = f"+{vcpu_delta}" if vcpu_delta > 0 else ("=" if vcpu_delta == 0 else str(vcpu_delta))
            except (ValueError, TypeError):
                vcpu_str = "?"

            try:
                alt_mem = float(d["memory_gb"])
                orig_mem = result.memory_gb or 0
                mem_delta = alt_mem - orig_mem
                mem_str = f"+{mem_delta:g}" if mem_delta > 0 else ("=" if mem_delta == 0 else f"{mem_delta:g}")
            except (ValueError, TypeError):
                mem_str = "?"

            row_data = {
                "SKU": d["name"],
                "vCPU": d["vcpu"],
                "Memory (GB)": d["memory_gb"],
                "Max Disks": d["max_disks"],
                "Capacity": d.get("capacity", "Not Checked"),
                "\u0394 vCPU": vcpu_str,
                "\u0394 Mem": mem_str,
            }

            # Multi-tier pricing columns
            for tier_key, tier_label, cur_val in [
                ("price_payg", "PAYG $/mo", cur_payg),
                ("price_ri_1yr", "1yr RI $/mo", cur_ri1),
                ("price_ri_3yr", "3yr RI $/mo", cur_ri3),
            ]:
                raw = d.get(tier_key, "")
                if raw:
                    row_data[tier_label] = f"${float(raw):,.2f}"
                else:
                    row_data[tier_label] = "\u2014"

            # Primary delta (PAYG) for quick scanning
            delta_raw = d.get("delta_payg", "")
            if delta_raw:
                dv = float(delta_raw)
                if dv < 0:
                    row_data["\u0394 PAYG"] = f"\u2193 ${abs(dv):,.0f}"
                elif dv > 0:
                    row_data["\u0394 PAYG"] = f"\u2191 ${dv:,.0f}"
                else:
                    row_data["\u0394 PAYG"] = "="
            else:
                row_data["\u0394 PAYG"] = "\u2014"

            alt_rows.append(row_data)

        alt_df = pd.DataFrame(alt_rows)

        def _highlight_capacity(row: pd.Series) -> list[str]:
            cap = row.get("Capacity", "")
            if cap == "Verified":
                s = "background-color: rgba(22, 163, 74, 0.18); color: #22c55e"
            elif cap == "Failed":
                s = "background-color: rgba(220, 38, 38, 0.12); color: #f87171"
            else:
                s = ""
            return [s] * len(row)

        st.dataframe(
            alt_df.style.apply(_highlight_capacity, axis=1),
            hide_index=True,
            width="stretch",
        )
    elif result.status == SkuStatus.BLOCKED:
        st.warning("No alternative SKUs found for this machine.")

    # Attached disks
    if disk_results:
        machine_disks = [
            dr for dr in disk_results
            if dr.server_name == result.machine_name
        ]
        if machine_disks:
            st.markdown(f"**Attached Disks** ({len(machine_disks)})")
            disk_rows = []
            for dr in machine_disks:
                disk_rows.append({
                    "Disk": dr.disk_name,
                    "Size (GB)": dr.disk_size_gb or "",
                    "SKU": dr.chosen_sku,
                    "Tier": dr.sku_tier_label,
                    "Status": dr.status.value,
                })
            disk_df = pd.DataFrame(disk_rows)
            styled_disks = disk_df.style.apply(_highlight_status, axis=1)
            st.dataframe(styled_disks, hide_index=True, width="stretch")


def _highlight_status(row: pd.Series) -> list[str]:
    """Apply row-level styling based on status, capacity, and alternatives.

    BLOCKED machines with a capacity-verified alternative get amber instead
    of red, because a viable migration path exists.
    """
    status = row.get("Status", "")
    capacity = row.get("Live Capacity", "Not Checked")
    best_alt = row.get("Best Alternative", "\u2014")

    # Capacity-verified OK gets stronger green
    if status == "OK" and capacity == "Verified":
        style = "background-color: rgba(22, 163, 74, 0.22); color: #22c55e"
    elif status == "OK" and capacity == "Failed":
        style = "background-color: rgba(220, 38, 38, 0.12); color: #f87171"
    elif status == "BLOCKED" and isinstance(best_alt, str) and "(Verified)" in best_alt:
        # Blocked but has a verified alternative — amber (actionable, not stuck)
        style = "background-color: rgba(217, 119, 6, 0.15); color: #fbbf24"
    else:
        styles = {
            "OK": "background-color: rgba(22, 163, 74, 0.12); color: #4ade80",
            "RISK": "background-color: rgba(217, 119, 6, 0.12); color: #fbbf24",
            "BLOCKED": "background-color: rgba(220, 38, 38, 0.12); color: #f87171",
            "UNKNOWN": "background-color: rgba(156, 163, 175, 0.10); color: #9CA3AF",
        }
        style = styles.get(status, "")
    return [style] * len(row)


def _disks_to_dataframe(disks: list[Disk]) -> pd.DataFrame:
    """Convert disk objects to a DataFrame for display and export."""
    rows = []
    for d in disks:
        rows.append(
            {
                "Server": d.server_name,
                "Disk": d.disk_name,
                "Size (GB)": d.disk_size_gb if d.disk_size_gb else "",
                "Chosen SKU": d.chosen_disk_sku or "",
                "Recommended SKU": d.recommended_disk_sku or "",
                "Tier": d.sku_tier,
                "Region": d.region or "",
                "Storage Target": d.storage_target or "",
                "Read MBPS": d.disk_read_mbps if d.disk_read_mbps else "",
                "Write MBPS": d.disk_write_mbps if d.disk_write_mbps else "",
                "Read IOPS": d.disk_read_iops if d.disk_read_iops else "",
                "Write IOPS": d.disk_write_iops if d.disk_write_iops else "",
                "Scope": d.scope or "",
            }
        )
    return pd.DataFrame(rows)


def _disk_results_to_dataframe(results: list[DiskAnalysisResult]) -> pd.DataFrame:
    """Convert disk analysis results to a DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "Server": r.server_name,
                "Disk": r.disk_name,
                "Region": r.region,
                "Chosen SKU": r.chosen_sku,
                "Azure Tier": r.azure_tier,
                "Tier Label": r.sku_tier_label,
                "Size (GB)": r.disk_size_gb if r.disk_size_gb else "",
                "Storage Target": r.storage_target or "",
                "Status": r.status.value,
                "Reason": r.reason,
            }
        )
    return pd.DataFrame(rows)


def _render_disk_analysis(disk_results: list[DiskAnalysisResult], disks: list[Disk]) -> None:
    """Render disk analysis with capacity check results."""
    if not disk_results:
        return

    st.info(
        "**How we verified disks:** Each disk SKU (e.g. S10, E6, P30) maps to an "
        "Azure managed disk tier (Standard HDD, Standard SSD, Premium SSD, etc.). "
        "We queried the Azure Resource SKUs API for disk resources and checked "
        "whether each tier is offered in your target region with no restrictions."
    )

    disk_results_df = _disk_results_to_dataframe(disk_results)

    # Summary metrics
    total = len(disk_results)
    ok_count = sum(1 for r in disk_results if r.status == DiskStatus.OK)
    risk_count = sum(1 for r in disk_results if r.status == DiskStatus.RISK)
    blocked_count = sum(1 for r in disk_results if r.status == DiskStatus.BLOCKED)
    unknown_count = sum(1 for r in disk_results if r.status == DiskStatus.UNKNOWN)
    total_storage_gb = sum(r.disk_size_gb or 0 for r in disk_results)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Disks", total)
    col2.metric("OK", ok_count)
    col3.metric("Risk", risk_count)
    col4.metric("Blocked", blocked_count + unknown_count)
    col5.metric("Total Storage", f"{total_storage_gb:,.0f} GB")

    st.divider()

    # Filter controls
    d_filter_col1, d_filter_col2, d_filter_col3 = st.columns(3)
    with d_filter_col1:
        disk_status_filter = st.multiselect(
            "Filter by status",
            options=["OK", "RISK", "BLOCKED", "UNKNOWN"],
            default=["OK", "RISK", "BLOCKED", "UNKNOWN"],
            key="disk_status_filter",
        )
    with d_filter_col2:
        disk_search = st.text_input(
            "Search disks",
            placeholder="Server name, disk name, SKU...",
            key="disk_search",
        )
    with d_filter_col3:
        tier_options = sorted(
            disk_results_df["Tier Label"].dropna().unique().tolist()
        )
        tier_options = [t for t in tier_options if t]
        disk_tier_filter = st.multiselect(
            "Filter by disk tier",
            options=tier_options,
            default=[],
            help="Leave empty to show all tiers.",
            key="disk_tier_filter",
        )

    # Apply filters
    disk_filtered = disk_results_df.copy()
    if disk_status_filter:
        disk_filtered = disk_filtered[
            disk_filtered["Status"].isin(disk_status_filter)
        ]
    if disk_search:
        mask = disk_filtered.apply(
            lambda row: disk_search.lower()
            in row.astype(str).str.lower().str.cat(sep=" "),
            axis=1,
        )
        disk_filtered = disk_filtered[mask]
    if disk_tier_filter:
        disk_filtered = disk_filtered[
            disk_filtered["Tier Label"].isin(disk_tier_filter)
        ]

    # Results table with status highlighting
    styled = disk_filtered.style.apply(_highlight_status, axis=1)
    st.dataframe(
        styled,
        width="stretch",
        height=min(400 + len(disk_filtered) * 5, 800),
    )

    st.caption(f"Showing {len(disk_filtered)} of {len(disk_results_df)} disk results.")

    # Tier breakdown
    with st.expander("Disk Tier Breakdown", expanded=False):
        tier_col1, tier_col2 = st.columns(2)
        with tier_col1:
            st.markdown("**Disk Count by Tier**")
            tier_series = disk_filtered["Tier Label"].value_counts()
            st.bar_chart(tier_series)
        with tier_col2:
            st.markdown("**Storage by Tier (GB)**")
            storage_by_tier: dict[str, float] = {}
            for r in disk_results:
                label = r.sku_tier_label or "Unknown"
                storage_by_tier[label] = storage_by_tier.get(label, 0) + (r.disk_size_gb or 0)
            st.bar_chart(pd.Series(storage_by_tier))

    # Per-server disk summary
    with st.expander("Per-Server Disk Summary", expanded=False):
        servers_seen: dict[str, dict] = {}
        for r in disk_results:
            if r.server_name not in servers_seen:
                servers_seen[r.server_name] = {
                    "Server": r.server_name,
                    "Region": r.region,
                    "Disk Count": 0,
                    "Total Storage (GB)": 0.0,
                    "Disk SKUs": set(),
                    "Statuses": set(),
                }
            entry = servers_seen[r.server_name]
            entry["Disk Count"] += 1
            entry["Total Storage (GB)"] += r.disk_size_gb or 0
            if r.chosen_sku:
                entry["Disk SKUs"].add(r.chosen_sku)
            entry["Statuses"].add(r.status.value)

        server_rows = []
        for entry in servers_seen.values():
            server_rows.append(
                {
                    "Server": entry["Server"],
                    "Region": entry["Region"],
                    "Disk Count": entry["Disk Count"],
                    "Total Storage (GB)": round(entry["Total Storage (GB)"], 1),
                    "Disk SKUs": ", ".join(sorted(entry["Disk SKUs"])),
                    "Statuses": ", ".join(sorted(entry["Statuses"])),
                }
            )
        st.dataframe(
            pd.DataFrame(server_rows),
            width="stretch",
            height=min(400 + len(server_rows) * 5, 600),
        )

    # Export
    st.divider()
    disk_csv = disk_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download disk results as CSV",
        data=disk_csv,
        file_name="capacity_advisor_disks.csv",
        mime="text/csv",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit application."""
    _inject_custom_css()

    # Branded header
    st.markdown(
        """
        <div class="adam-header">
            <div class="adam-header-icon">\u2601</div>
            <div>
                <h1>Azure Capacity Advisor</h1>
                <p>Validate VM SKU availability for Azure migrations and rightsizing exercises.
                Upload your dataset, run the analysis, and export the results.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    device_code_callback = None

    # --- Sidebar: Azure Login ---
    with st.sidebar:
        st.markdown(
            """
            <div class="adam-sidebar-brand">
                <h2>Capacity Advisor</h2>
                <p>Azure Tools</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.header("Azure Login")

        auth_options = [m.value for m in AuthMethod]
        default_auth = (
            AuthMethod.DEVICE_CODE if _prefer_device_code_default() else AuthMethod.DEFAULT
        )

        auth_method_label = st.selectbox(
            "Authentication method",
            options=auth_options,
            index=auth_options.index(default_auth.value),
            help=(
                "**Default** — uses Azure CLI (`az login`), managed identity, "
                "or env vars automatically.\n\n"
                "**Service Principal** — enter client credentials directly.\n\n"
                "**Device Code** — log in via browser with a one-time code "
                "(recommended for hosted deployments).\n\n"
                "**Interactive Browser** — opens a browser window on the app host "
                "(local development only)."
            ),
        )
        auth_method = AuthMethod(auth_method_label)

        if default_auth == AuthMethod.DEVICE_CODE:
            st.caption("Hosted runtime detected. Defaulting auth to Device Code.")

        subscription_id = st.text_input(
            "Subscription ID",
            value=AppConfig.from_env().subscription_id,
            type="password",
            help="Your Azure subscription ID. Can also be set via AZURE_SUBSCRIPTION_ID env var.",
        )

        # Conditional fields based on auth method
        tenant_id = ""
        client_id = ""
        client_secret = ""

        if auth_method == AuthMethod.SERVICE_PRINCIPAL:
            tenant_id = st.text_input(
                "Tenant ID",
                type="password",
                help="Azure AD tenant ID (Directory ID).",
            )
            client_id = st.text_input(
                "Client ID",
                type="password",
                help="App registration client ID (Application ID).",
            )
            client_secret = st.text_input(
                "Client Secret",
                type="password",
                help="App registration client secret.",
            )

        if auth_method in (AuthMethod.DEVICE_CODE, AuthMethod.INTERACTIVE_BROWSER):
            tenant_id = st.text_input(
                "Tenant ID (optional)",
                help="Azure AD tenant ID. Leave blank for multi-tenant.",
            )

        device_code_prompt = st.empty()
        if auth_method == AuthMethod.DEVICE_CODE:
            st.caption(
                "Recommended for hosted deployments. You will get a login URL "
                "and one-time code after clicking Test Connection."
            )
            def _device_code_callback(
                verification_uri: str,
                user_code: str,
                expires_on: int,
            ) -> None:
                _render_device_code_prompt(
                    device_code_prompt,
                    verification_uri=verification_uri,
                    user_code=user_code,
                    expires_on=expires_on,
                )

            device_code_callback = _device_code_callback
        elif auth_method == AuthMethod.INTERACTIVE_BROWSER:
            st.warning(
                "Interactive Browser auth requires opening a browser on the app host "
                "and usually fails in hosted Streamlit. Use Device Code instead."
            )

        # Test connection button
        col_test, col_clear = st.columns(2)
        with col_test:
            if st.button("Test Connection", width="stretch"):
                if not subscription_id:
                    st.error("Enter a Subscription ID first.")
                else:
                    try:
                        with st.spinner("Authenticating..."):
                            test_kwargs = dict(
                                subscription_id=subscription_id,
                                method=auth_method,
                                tenant_id=tenant_id,
                                client_id=client_id,
                                client_secret=client_secret,
                            )
                            if _supports_kwarg(test_connection, "device_code_callback"):
                                test_kwargs["device_code_callback"] = device_code_callback
                            test_connection(**test_kwargs)
                        st.session_state["auth_verified"] = True
                        st.session_state["auth_method"] = auth_method
                        device_code_prompt.empty()
                        st.success("Connected!")
                    except AzureAuthError as exc:
                        st.session_state["auth_verified"] = False
                        st.error(_friendly_auth_error(auth_method, exc))
        with col_clear:
            if st.button("Reset Auth", width="stretch"):
                reset_credential()
                st.session_state.pop("auth_verified", None)
                st.session_state.pop("auth_method", None)
                device_code_prompt.empty()
                st.info("Credentials cleared.")

        # Show auth status
        if st.session_state.get("auth_verified"):
            current = st.session_state.get("auth_method", auth_method)
            st.success(f"Authenticated via {current}")

        st.divider()
        st.header("Options")
        region_override_label = st.selectbox(
            "Region override (optional)",
            options=["(use dataset regions)"] + _build_region_options(),
            index=0,
            help="Override the region for all machines. Leave as default to use each row's region.",
        )
        region_override: str | None = None
        if region_override_label != "(use dataset regions)":
            region_override = _display_label_to_region(region_override_label)

        st.divider()
        st.header("Live Capacity Check")
        resource_group = st.text_input(
            "Resource Group",
            value=os.environ.get("AZURE_RESOURCE_GROUP", "rg-capacity-advisor"),
            help=(
                "Resource group for ARM deployment validation. Will be auto-created "
                "if it doesn't exist. No VMs or resources are created — it is only "
                "used as a container for the validation API calls. "
                "If left blank, only catalog availability is checked."
            ),
        )

        st.divider()
        st.markdown(
            '<div class="adam-version">Azure Capacity Advisor v1.0</div>',
            unsafe_allow_html=True,
        )

    # --- File upload ---
    _section_header(1, "Upload Dataset")
    uploaded = st.file_uploader(
        "Upload a rightsizing dataset (CSV, JSON, or Excel)",
        type=["csv", "json", "xlsx", "xls"],
        help=(
            "**CSV/JSON**: Expects columns: MachineName, Region, RecommendedSKU "
            "(and optionally vCPU, MemoryGB, VMFamily).\n\n"
            "**Excel (.xlsx)**: Supports Dr Migrate / Azure Migrate exports with "
            "Servers and Disks sheets (headers on row 6)."
        ),
    )

    if uploaded is None:
        st.info("Upload a file to get started.")
        return

    # Parse uploaded file
    disks: list[Disk] = []
    try:
        raw_bytes = uploaded.read()
        result = parse_file(io.BytesIO(raw_bytes), uploaded.name)
        if isinstance(result, ExcelParseResult):
            machines = result.machines
            disks = result.disks
        else:
            machines = result
    except DatasetParseError as exc:
        st.error(f"Failed to parse dataset: {exc}")
        return

    if len(machines) > MAX_DATASET_ROWS:
        st.error(
            f"Dataset has {len(machines)} rows, which exceeds the maximum of {MAX_DATASET_ROWS}."
        )
        return

    loaded_msg = f"Loaded **{len(machines)}** servers from `{uploaded.name}`."
    if disks:
        loaded_msg += f" Also loaded **{len(disks)}** disk entries."
    st.success(loaded_msg)

    # Store parsed data in session state for later display and export
    st.session_state["machines"] = machines
    st.session_state["raw_bytes"] = raw_bytes
    st.session_state["uploaded_name"] = uploaded.name
    if disks:
        st.session_state["disks"] = disks
    else:
        st.session_state.pop("disks", None)

    # Preview
    with st.expander("Preview uploaded data", expanded=False):
        if disks:
            server_tab, disk_tab = st.tabs(["Servers", "Disks"])
        else:
            server_tab = st.container()
            disk_tab = None

        with server_tab:
            preview_rows = [
                {
                    "Machine": m.name,
                    "Region": m.display_region,
                    "SKU": m.recommended_sku,
                    "vCPU": m.vcpu or "",
                    "Memory (GB)": m.memory_gb or "",
                    "Family": m.vm_family or "",
                }
                for m in machines[:50]
            ]
            st.dataframe(pd.DataFrame(preview_rows), width="stretch")
            if len(machines) > 50:
                st.caption(f"Showing first 50 of {len(machines)} rows.")

        if disk_tab is not None:
            with disk_tab:
                disk_preview = [
                    {
                        "Server": d.server_name,
                        "Disk": d.disk_name,
                        "Size (GB)": d.disk_size_gb or "",
                        "Chosen SKU": d.chosen_disk_sku or "",
                        "Recommended SKU": d.recommended_disk_sku or "",
                        "Tier": d.sku_tier,
                        "Storage Target": d.storage_target or "",
                    }
                    for d in disks[:50]
                ]
                st.dataframe(pd.DataFrame(disk_preview), width="stretch")
                if len(disks) > 50:
                    st.caption(f"Showing first 50 of {len(disks)} disk entries.")

    # --- Run analysis ---
    _section_header(2, "Run Analysis")

    if not subscription_id:
        st.warning(
            "Enter your Azure Subscription ID in the sidebar or set the "
            "`AZURE_SUBSCRIPTION_ID` environment variable."
        )
        return

    if st.button("Run Analysis", type="primary", width="stretch"):
        try:
            with st.spinner("Authenticating with Azure..."):
                sku_kwargs = dict(
                    subscription_id=subscription_id,
                    auth_method=auth_method,
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret,
                )
                if _supports_kwarg(SkuService.__init__, "device_code_callback"):
                    sku_kwargs["device_code_callback"] = device_code_callback
                sku_service = SkuService(**sku_kwargs)

            with st.spinner("Fetching Azure VM SKU catalog (this may take a moment on first run)..."):
                sku_service.fetch_skus()

            alt_engine = AlternativeEngine(sku_service=sku_service)

            # Build disk map: machine name -> list of its disks
            disk_map: dict[str, list[Disk]] = {}
            for d in disks:
                disk_map.setdefault(d.server_name, []).append(d)

            analyzer = Analyzer(
                sku_service=sku_service,
                alternative_engine=alt_engine,
                region_override=region_override,
                disk_map=disk_map,
            )

            with st.spinner("Analyzing VM SKU availability (catalog check)..."):
                results = analyzer.analyze(machines)

            # Live capacity validation (Method 2) — runs automatically if RG provided
            capacity_ran = False
            if resource_group:
                unique_pairs = {
                    (r.requested_sku, r.region)
                    for r in results
                    if r.status in (SkuStatus.OK, SkuStatus.RISK)
                    and r.requested_sku and r.region
                }
                if unique_pairs:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def _update_progress(done: int, total: int) -> None:
                        progress_bar.progress(done / total if total else 1.0)
                        status_text.text(
                            f"Phase 1 — Checking primary SKUs: {done}/{total} "
                            f"unique SKU+region combinations..."
                        )

                    def _update_alt_progress(done: int, total: int) -> None:
                        progress_bar.progress(done / total if total else 1.0)
                        status_text.text(
                            f"Phase 2 — Checking alternative SKUs: {done}/{total} "
                            f"unique alternatives..."
                        )

                    deployment_kwargs = dict(
                        subscription_id=subscription_id,
                        resource_group=resource_group,
                        auth_method=auth_method,
                        tenant_id=tenant_id,
                        client_id=client_id,
                        client_secret=client_secret,
                    )
                    if _supports_kwarg(DeploymentValidator.__init__, "device_code_callback"):
                        deployment_kwargs["device_code_callback"] = device_code_callback
                    deployment_validator = DeploymentValidator(**deployment_kwargs)

                    # Auto-create the resource group if it doesn't exist
                    rg_location = next(iter(
                        r.region for r in results if r.region
                    ), "uksouth")
                    with st.spinner(
                        f"Ensuring resource group '{resource_group}' exists..."
                    ):
                        deployment_validator.ensure_resource_group(
                            location=rg_location
                        )

                    with st.spinner(
                        f"Validating live capacity for {len(unique_pairs)} "
                        f"unique SKU+region combinations..."
                    ):
                        cap_validator = CapacityValidator(
                            deployment_validator,
                            alternative_engine=alt_engine,
                            disk_map=disk_map,
                        )
                        results = cap_validator.validate_results(
                            results,
                            progress_callback=_update_progress,
                            alt_progress_callback=_update_alt_progress,
                        )

                    progress_bar.empty()
                    status_text.empty()
                    capacity_ran = True
            else:
                st.warning(
                    "No resource group provided — skipping live capacity check. "
                    "Only catalog availability was verified. Add a resource group "
                    "in the sidebar to enable real-time capacity testing."
                )

            # Fetch pricing data for cost delta indicators
            with st.spinner("Fetching pricing data from Azure Retail Prices API..."):
                pricing_service = PricingService()

                # Build machine lookup for Excel cost data
                machine_map = {m.name: m for m in machines}

                # We only need API pricing for alternative SKUs (current costs
                # come from the Excel where available). Also fetch current SKUs
                # as fallback when Excel data is missing.
                price_pairs: set[tuple[str, str]] = set()
                for r in results:
                    if r.requested_sku and r.region:
                        price_pairs.add((r.requested_sku, r.region))
                    for alt in r.alternatives_detail:
                        alt_name = alt.get("name", "")
                        if alt_name and r.region:
                            price_pairs.add((alt_name, r.region))

                pricing_data: dict[tuple[str, str], SkuPricing] = {}
                if price_pairs:
                    pricing_data = pricing_service.fetch_prices_batch(price_pairs)

                # Enrich alternatives_detail with multi-tier pricing
                for r in results:
                    # Current cost: prefer Excel data, fall back to API
                    m = machine_map.get(r.machine_name)
                    current_payg = _safe_float(m.extra.get("compute_cost_monthly")) if m else None
                    current_ri1 = _safe_float(m.extra.get("ri_1yr_cost_monthly")) if m else None
                    current_ri3 = _safe_float(m.extra.get("ri_3yr_cost_monthly")) if m else None

                    api_current = pricing_data.get((r.requested_sku, r.region))
                    if api_current:
                        current_payg = current_payg or api_current.payg
                        current_ri1 = current_ri1 or api_current.ri_1yr
                        current_ri3 = current_ri3 or api_current.ri_3yr

                    for alt in r.alternatives_detail:
                        alt_name = alt.get("name", "")
                        alt_pricing = pricing_data.get((alt_name, r.region))
                        if not alt_pricing:
                            continue

                        # Store all three tiers
                        if alt_pricing.payg is not None:
                            alt["price_payg"] = f"{alt_pricing.payg:.2f}"
                        if alt_pricing.ri_1yr is not None:
                            alt["price_ri_1yr"] = f"{alt_pricing.ri_1yr:.2f}"
                        if alt_pricing.ri_3yr is not None:
                            alt["price_ri_3yr"] = f"{alt_pricing.ri_3yr:.2f}"

                        # Deltas against current cost (PAYG as primary)
                        if alt_pricing.payg is not None and current_payg:
                            delta = alt_pricing.payg - current_payg
                            alt["delta_payg"] = f"{delta:+.2f}"
                        if alt_pricing.ri_1yr is not None and current_ri1:
                            delta = alt_pricing.ri_1yr - current_ri1
                            alt["delta_ri_1yr"] = f"{delta:+.2f}"
                        if alt_pricing.ri_3yr is not None and current_ri3:
                            delta = alt_pricing.ri_3yr - current_ri3
                            alt["delta_ri_3yr"] = f"{delta:+.2f}"

                # Store for use in rendering
                st.session_state["pricing_data"] = pricing_data
                st.session_state["machine_costs"] = {
                    m.name: {
                        "payg": _safe_float(m.extra.get("compute_cost_monthly")),
                        "ri_1yr": _safe_float(m.extra.get("ri_1yr_cost_monthly")),
                        "ri_3yr": _safe_float(m.extra.get("ri_3yr_cost_monthly")),
                    }
                    for m in machines
                }

            # Store results in session state
            st.session_state["results"] = results
            st.session_state["results_df"] = _results_to_dataframe(results)
            st.session_state["summary"] = AnalysisSummary.from_results(results)
            st.session_state["capacity_validation_ran"] = capacity_ran

            # Disk analysis (if disks are loaded)
            if disks:
                with st.spinner("Fetching Azure disk SKU catalog..."):
                    sku_service.fetch_disk_skus()

                with st.spinner("Analyzing disk SKU availability..."):
                    disk_analyzer = DiskAnalyzer(sku_service=sku_service)
                    disk_results = disk_analyzer.analyze(disks)

                st.session_state["disk_results"] = disk_results

        except AzureAuthError as exc:
            st.error(_friendly_auth_error(auth_method, exc))
            return
        except SkuServiceError as exc:
            st.error(f"Azure SKU service error: {exc}")
            return
        except DeploymentValidationError as exc:
            st.error(f"Live capacity validation error: {exc}")
            return
        except Exception as exc:
            logger.exception("Unexpected error during analysis")
            st.error(f"An unexpected error occurred: {exc}")
            return

    # --- Display results ---
    if "results_df" not in st.session_state:
        return

    results = st.session_state["results"]
    results_df: pd.DataFrame = st.session_state["results_df"]
    summary: AnalysisSummary = st.session_state["summary"]
    has_disks = bool(st.session_state.get("disk_results"))

    _section_header(3, "Results")

    # How we checked — methodology explanation
    capacity_ran = st.session_state.get("capacity_validation_ran", False)
    if capacity_ran:
        st.info(
            "**Two-level verification:**\n\n"
            "**Step 1 — Live Capacity Check:** For each SKU that passed the catalog "
            "check, we submitted an ARM deployment validation request to the Azure "
            "Resource Manager. This tests whether physical hardware is actually "
            "available right now, catching capacity exhaustion and quota limits. "
            "No VMs were created.\n\n"
            "**Step 2 — Catalog Check:** We queried the Azure Resource SKUs API "
            "(Microsoft.Compute/skus) for your subscription to verify which VM "
            "sizes are listed, restricted, and zone-supported in each region."
        )
    else:
        st.info(
            "**How we verified this:** We queried the Azure Resource SKUs API "
            "(Microsoft.Compute/skus) for your subscription. This returns the "
            "complete catalog of every VM size and disk tier Microsoft offers, "
            "including which regions each is available in, any restrictions on "
            "your subscription, and which availability zones are supported. "
            "Each row below was checked against this live data.\n\n"
            "*Add a Resource Group in the sidebar to also test live capacity.*"
        )

    # Status legend
    with st.expander("What do the statuses mean?", expanded=False):
        if capacity_ran:
            cols = st.columns(5)
            with cols[0]:
                st.markdown(
                    ':green[**CAPACITY VERIFIED**]\n\n'
                    'Passed both the catalog check and ARM deployment '
                    'validation. Physical hardware is confirmed available. '
                    'Highest confidence for deployment.'
                )
            with cols[1]:
                st.markdown(
                    ':orange[**RISK — Available with caveats**]\n\n'
                    'Available in your region but not in all availability '
                    'zones. Fine for most deployments, but check if you need '
                    'zone-redundant HA.'
                )
            with cols[2]:
                st.markdown(
                    ':red[**BLOCKED — Cannot deploy**]\n\n'
                    'Not offered in your region, restricted on your '
                    'subscription, or physical capacity is exhausted. '
                    'Check the Reason column for details.'
                )
            with cols[3]:
                st.markdown(
                    ':gray[**UNKNOWN — Not found**]\n\n'
                    'Not found in the Azure catalog at all. '
                    'Check for typos or retired SKU names.'
                )
            with cols[4]:
                st.markdown(
                    ':red[**CAPACITY EXHAUSTED**]\n\n'
                    'Passed catalog check but ARM validation reports '
                    'no physical hardware available. The SKU exists but '
                    'the region is saturated.'
                )
        else:
            legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
            with legend_col1:
                st.markdown(
                    ':green[**OK — Ready to deploy**]\n\n'
                    'Available in your target region with no '
                    'restrictions. You can proceed with your migration.'
                )
            with legend_col2:
                st.markdown(
                    ':orange[**RISK — Available with caveats**]\n\n'
                    'Available in your region but not in all availability '
                    'zones. Fine for most deployments, but check if you need '
                    'zone-redundant HA.'
                )
            with legend_col3:
                st.markdown(
                    ':red[**BLOCKED — Cannot deploy**]\n\n'
                    'Not offered in your region, or '
                    'your subscription is restricted from using it. '
                    'You must use an alternative or contact Azure support.'
                )
            with legend_col4:
                st.markdown(
                    ':gray[**UNKNOWN — Not found**]\n\n'
                    'Not found in the Azure catalog at all. '
                    'Check for typos or retired SKU names.'
                )

    # --- Tabbed interface: Servers | Disks ---
    if has_disks:
        server_tab, disk_tab = st.tabs(["Servers", "Disks"])
    else:
        server_tab = st.container()
        disk_tab = None

    # ==================== SERVERS TAB ====================
    with server_tab:
        # Layer 1: Executive Summary Dashboard
        _render_executive_summary(results, summary)

        st.divider()

        # View toggle + Filter controls
        toggle_col, spacer_col = st.columns([1, 3])
        with toggle_col:
            view_mode = st.radio(
                "View",
                options=["Grouped", "Table"],
                horizontal=True,
                key="server_view_mode",
                label_visibility="collapsed",
            )

        if capacity_ran:
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        else:
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            filter_col4 = None
        with filter_col1:
            status_filter = st.multiselect(
                "Filter by status",
                options=["OK", "RISK", "BLOCKED", "UNKNOWN"],
                default=["OK", "RISK", "BLOCKED", "UNKNOWN"],
                key="server_status_filter",
            )
        with filter_col2:
            search_term = st.text_input(
                "Search machines",
                placeholder="Type to search...",
                key="server_search",
            )
        with filter_col3:
            family_options = sorted(results_df["VM Family"].dropna().unique().tolist())
            family_options = [f for f in family_options if f]
            family_filter = st.multiselect(
                "Filter by VM family",
                options=family_options,
                default=[],
                help="Leave empty to show all families.",
                key="server_family_filter",
            )
        capacity_filter: list[str] = []
        if filter_col4 is not None:
            with filter_col4:
                capacity_filter = st.multiselect(
                    "Filter by capacity",
                    options=["Verified", "Failed", "Not Checked"],
                    default=["Verified", "Failed", "Not Checked"],
                    key="server_capacity_filter",
                )

        # Apply filters
        filtered = results_df.copy()
        if status_filter:
            filtered = filtered[filtered["Status"].isin(status_filter)]
        if search_term:
            mask = filtered.apply(
                lambda row: search_term.lower()
                in row.astype(str).str.lower().str.cat(sep=" "),
                axis=1,
            )
            filtered = filtered[mask]
        if family_filter:
            filtered = filtered[filtered["VM Family"].isin(family_filter)]
        if capacity_filter:
            filtered = filtered[filtered["Live Capacity"].isin(capacity_filter)]

        disk_results_for_detail = st.session_state.get("disk_results")

        # ---- Grouped View (primary) ----
        if view_mode == "Grouped":
            _render_grouped_view(results, filtered, disk_results_for_detail)

        # ---- Table View (secondary) ----
        else:
            styled = filtered.style.apply(_highlight_status, axis=1)
            selection = st.dataframe(
                styled,
                width="stretch",
                height=min(400 + len(filtered) * 5, 800),
                selection_mode="single-row",
                on_select="rerun",
                key="server_table_selection",
            )

            st.caption(
                f"Showing {len(filtered)} of {len(results_df)} results. "
                "Click a row for details."
            )

            # Machine Detail Panel driven by table selection
            selected_idx = None
            if selection and selection.selection and selection.selection.rows:
                selected_idx = selection.selection.rows[0]

            machine_names = filtered["Machine"].tolist()
            dropdown_default = 0

            if selected_idx is not None and selected_idx < len(machine_names):
                dropdown_default = selected_idx + 1

            selected_machine_name = st.selectbox(
                "Machine details",
                options=["(select a machine)"] + machine_names,
                index=dropdown_default,
                key="machine_detail_select",
                help="Click a table row above or pick from this dropdown.",
            )

            if (
                selected_machine_name
                and selected_machine_name != "(select a machine)"
            ):
                selected_result = next(
                    (r for r in results if r.machine_name == selected_machine_name),
                    None,
                )
                if selected_result:
                    _render_machine_detail(selected_result, disk_results_for_detail)

        # Charts in expander
        with st.expander("Summary Charts", expanded=False):
            dash_col1, dash_col2 = st.columns(2)
            with dash_col1:
                st.markdown("**Status Distribution**")
                status_counts = filtered["Status"].value_counts()
                st.bar_chart(status_counts)
            with dash_col2:
                st.markdown("**Top Requested SKUs**")
                sku_counts = filtered["Requested SKU"].value_counts().head(10)
                st.bar_chart(sku_counts)

        # Export (full data including Reason & Alternatives)
        st.divider()
        export_df = _results_to_export_dataframe(results)
        # Apply same filters to export
        if status_filter:
            export_df = export_df[export_df["Status"].isin(status_filter)]
        if search_term:
            mask = export_df.apply(
                lambda row: search_term.lower()
                in row.astype(str).str.lower().str.cat(sep=" "),
                axis=1,
            )
            export_df = export_df[mask]
        if family_filter:
            export_df = export_df[export_df["VM Family"].isin(family_filter)]
        if capacity_filter:
            export_df = export_df[export_df["Live Capacity"].isin(capacity_filter)]
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            excel_bytes = _build_updated_export(results)
            st.download_button(
                label="Download updated rightsizing export (.xlsx)",
                data=excel_bytes,
                file_name="capacity_advisor_updated.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with dl_col2:
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download full analysis as CSV",
                data=csv_data,
                file_name="capacity_advisor_servers.csv",
                mime="text/csv",
                width="stretch",
            )

    # ==================== DISKS TAB ====================
    if disk_tab is not None:
        with disk_tab:
            _render_disk_analysis(
                st.session_state["disk_results"],
                st.session_state.get("disks", []),
            )


if __name__ == "__main__":
    main()
