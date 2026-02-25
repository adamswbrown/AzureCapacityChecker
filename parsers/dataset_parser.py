"""Parser for rightsizing datasets in CSV, JSON, and Excel formats.

Handles column name normalization so that slight variations in column naming
(e.g. 'Machine Name' vs 'MachineName' vs 'machine_name') are mapped to a
canonical set of fields.

Excel (.xlsx) files from Dr Migrate / Azure Migrate exports are supported:
  - "Servers" sheet (headers on row 6) is parsed into Machine objects.
  - "Disks" sheet (headers on row 6) is parsed into Disk objects.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from models.disk import Disk
from models.machine import Machine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical column names used internally (for CSV / JSON flat files)
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS: dict[str, str] = {
    "name": "name",
    "machinename": "name",
    "machine_name": "name",
    "machine name": "name",
    "hostname": "name",
    "host_name": "name",
    "host name": "name",
    "vm_name": "name",
    "vmname": "name",
    "vm name": "name",
    "server": "name",
    "servername": "name",
    "server_name": "name",
    "server name": "name",
    "region": "region",
    "location": "region",
    "azure_region": "region",
    "azureregion": "region",
    "azure region": "region",
    "target_region": "region",
    "targetregion": "region",
    "target region": "region",
    "recommendedsku": "recommended_sku",
    "recommended_sku": "recommended_sku",
    "recommended sku": "recommended_sku",
    "sku": "recommended_sku",
    "vm_sku": "recommended_sku",
    "vmsku": "recommended_sku",
    "vm sku": "recommended_sku",
    "target_sku": "recommended_sku",
    "targetsku": "recommended_sku",
    "target sku": "recommended_sku",
    "recommended_vm_size": "recommended_sku",
    "vmsize": "recommended_sku",
    "vm_size": "recommended_sku",
    "vm size": "recommended_sku",
    "vcpu": "vcpu",
    "vcpus": "vcpu",
    "cpu": "vcpu",
    "cpus": "vcpu",
    "cores": "vcpu",
    "core_count": "vcpu",
    "corecount": "vcpu",
    "core count": "vcpu",
    "num_cpus": "vcpu",
    "memorygb": "memory_gb",
    "memory_gb": "memory_gb",
    "memory gb": "memory_gb",
    "memory": "memory_gb",
    "ram": "memory_gb",
    "ramgb": "memory_gb",
    "ram_gb": "memory_gb",
    "ram gb": "memory_gb",
    "memory_size_gb": "memory_gb",
    "vmfamily": "vm_family",
    "vm_family": "vm_family",
    "vm family": "vm_family",
    "family": "vm_family",
    "sku_family": "vm_family",
    "skufamily": "vm_family",
    "sku family": "vm_family",
    "series": "vm_family",
    "vm_series": "vm_family",
    "vmseries": "vm_family",
    "vm series": "vm_family",
}

REQUIRED_COLUMNS: set[str] = {"name", "region", "recommended_sku"}

# ---------------------------------------------------------------------------
# Excel-specific column mappings (Dr Migrate / Azure Migrate export format)
# Headers are on row 6; these map the exact header text to internal fields.
# ---------------------------------------------------------------------------
EXCEL_SERVER_COLUMNS: dict[str, str] = {
    "server": "name",
    "target azure region": "region",
    "chosen sku": "recommended_sku",
    "current cores": "vcpu",
    "current ram (mb)": "memory_mb",  # note: MB — converted to GB later
    "current cpu usage (%)": "cpu_usage_pct",
    "current memory usage (%)": "memory_usage_pct",
    "storage (gb)": "storage_gb",
    "server category": "server_category",
    "scope": "scope",
    "associated apps": "associated_apps",
    "associated envs": "associated_envs",
    "migration strategy": "migration_strategy",
    "chosen payment model": "payment_model",
    "hybrid benefit setting": "hybrid_benefit",
    "sql hybrid benefit setting": "sql_hybrid_benefit",
    "dev/test setting": "dev_test",
    "hours powered on (daily)": "hours_powered_on",
    "backup setting": "backup",
    "disaster recovery setting": "disaster_recovery",
    "chosen compute cost monthly": "compute_cost_monthly",
    "chosen 1 year ri cost monthly": "ri_1yr_cost_monthly",
    "chosen 3 year ri cost monthly": "ri_3yr_cost_monthly",
    "chosen dev/test cost monthly": "devtest_cost_monthly",
    "chosen windows cost monthly": "windows_cost_monthly",
    "sql license cost monthly": "sql_license_cost_monthly",
    "azure backup cost": "backup_cost",
    "azure site recovery cost": "site_recovery_cost",
    "dr migrate recommended sku": "dr_migrate_recommended_sku",
}

EXCEL_DISK_COLUMNS: dict[str, str] = {
    "server": "server_name",
    "server category": "server_category",
    "scope": "scope",
    "associated apps": "associated_apps",
    "associated envs": "associated_envs",
    "migration strategy": "migration_strategy",
    "disk name": "disk_name",
    "disk size (gb)": "disk_size_gb",
    "disk read (mbps)": "disk_read_mbps",
    "disk write (mbps)": "disk_write_mbps",
    "disk read (iops)": "disk_read_iops",
    "disk write (iops)": "disk_write_iops",
    "chosen disk sku": "chosen_disk_sku",
    "disk size gb (ultra & premium ssdv2 only)": "ultra_disk_size_gb",
    "provisioned iops (ultra & premium ssdv2 only)": "ultra_provisioned_iops",
    "provisioned throughput (mbps) (ultra & premium ssdv2 only)": "ultra_provisioned_throughput",
    "storage target": "storage_target",
    "payment model (blob storage only)": "blob_payment_model",
    "storage redundancy (blob storage only)": "blob_redundancy",
    "chosen disk cost monthly": "disk_cost_monthly",
    "chosen 1 year ri cost monthly": "disk_ri_1yr_cost_monthly",
    "chosen 3 year ri cost monthly": "disk_ri_3yr_cost_monthly",
    "azure migrate recommended sku": "recommended_disk_sku",
    "azure migrate recommmended sku": "recommended_disk_sku",  # typo in source
    "guid": "guid",
}


@dataclass
class ExcelParseResult:
    """Container for parsed Excel data (servers + disks)."""

    machines: list[Machine] = field(default_factory=list)
    disks: list[Disk] = field(default_factory=list)


class DatasetParseError(Exception):
    """Raised when the dataset cannot be parsed."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw column names to canonical names.

    Unrecognized columns are kept under their original lowered name.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        lookup = col.strip().lower().replace("-", "_")
        canonical = CANONICAL_COLUMNS.get(lookup)
        if canonical:
            rename_map[col] = canonical
        else:
            rename_map[col] = lookup
    df = df.rename(columns=rename_map)
    return df


def _validate_required(df: pd.DataFrame) -> None:
    """Ensure the dataframe has all required columns after normalization."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DatasetParseError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}. "
            f"Found columns: {', '.join(sorted(df.columns))}"
        )


def _to_machines(df: pd.DataFrame) -> list[Machine]:
    """Convert dataframe rows to Machine objects."""
    machines: list[Machine] = []
    known_fields = {"name", "region", "recommended_sku", "vcpu", "memory_gb", "vm_family"}
    extra_cols = [c for c in df.columns if c not in known_fields]

    for _, row in df.iterrows():
        vcpu: Optional[int] = None
        if "vcpu" in df.columns and pd.notna(row.get("vcpu")):
            try:
                vcpu = int(float(row["vcpu"]))
            except (ValueError, TypeError):
                pass

        memory_gb: Optional[float] = None
        if "memory_gb" in df.columns and pd.notna(row.get("memory_gb")):
            try:
                memory_gb = float(row["memory_gb"])
            except (ValueError, TypeError):
                pass

        vm_family: Optional[str] = None
        if "vm_family" in df.columns and pd.notna(row.get("vm_family")):
            vm_family = str(row["vm_family"])

        extra = {}
        for c in extra_cols:
            if pd.notna(row.get(c)):
                extra[c] = str(row[c])

        machines.append(
            Machine(
                name=str(row["name"]),
                region=str(row["region"]),
                recommended_sku=str(row["recommended_sku"]),
                vcpu=vcpu,
                memory_gb=memory_gb,
                vm_family=vm_family,
                extra=extra,
            )
        )
    return machines


# ---------------------------------------------------------------------------
# Excel parsing helpers
# ---------------------------------------------------------------------------

def _read_excel_sheet(
    source: Union[str, Path, io.BytesIO],
    sheet_name: str,
    header_row: int,
) -> pd.DataFrame:
    """Read a specific sheet from an Excel file with headers at a given row.

    Args:
        source: File path or BytesIO stream.
        sheet_name: The sheet/tab name to read.
        header_row: 1-based row number where headers are located.

    Returns:
        DataFrame with headers from the specified row and data from the rows below.
    """
    # pandas header param is 0-based
    df = pd.read_excel(source, sheet_name=sheet_name, header=header_row - 1, engine="openpyxl")
    # Drop rows that are completely empty
    df = df.dropna(how="all")
    return df


def _map_excel_columns(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Rename Excel columns using a mapping from lowercase header → internal name.

    Handles duplicate target column names (e.g. two source columns that both map
    to the same canonical name) by keeping the first non-null value and dropping
    the duplicate.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        lookup = str(col).strip().lower()
        mapped = column_map.get(lookup)
        if mapped:
            rename_map[col] = mapped
        else:
            rename_map[col] = lookup
    df = df.rename(columns=rename_map)

    # Deduplicate columns: if multiple columns share the same name, coalesce
    # them (take the first non-null value) and keep only one.
    if df.columns.duplicated().any():
        seen: dict[str, int] = {}  # col_name → first positional index
        keep_mask = [True] * len(df.columns)
        for i, col_name in enumerate(df.columns):
            if col_name in seen:
                # Coalesce: fill NaN in the first occurrence with values from this column
                first_idx = seen[col_name]
                df.iloc[:, first_idx] = df.iloc[:, first_idx].fillna(df.iloc[:, i])
                keep_mask[i] = False
            else:
                seen[col_name] = i
        df = df.iloc[:, [i for i, keep in enumerate(keep_mask) if keep]]

    return df


def _parse_currency(value: str) -> Optional[float]:
    """Parse a currency string like '£52.92' to a float."""
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip().replace("£", "").replace("$", "").replace(",", "").replace("€", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _excel_servers_to_machines(df: pd.DataFrame) -> list[Machine]:
    """Convert an Excel Servers-tab DataFrame into Machine objects."""
    machines: list[Machine] = []
    known_machine_fields = {"name", "region", "recommended_sku", "vcpu", "memory_gb"}

    for _, row in df.iterrows():
        name = row.get("name")
        region = row.get("region")
        sku = row.get("recommended_sku")

        # Skip rows with missing required fields
        if pd.isna(name) or pd.isna(region) or pd.isna(sku):
            continue

        # vCPU
        vcpu: Optional[int] = None
        raw_vcpu = row.get("vcpu")
        if pd.notna(raw_vcpu):
            try:
                vcpu = int(float(raw_vcpu))
            except (ValueError, TypeError):
                pass

        # Memory: source is in MB, convert to GB
        memory_gb: Optional[float] = None
        raw_mem = row.get("memory_mb")
        if pd.notna(raw_mem):
            try:
                memory_gb = round(float(raw_mem) / 1024, 2)
            except (ValueError, TypeError):
                pass

        # Collect extra fields
        extra: dict[str, str] = {}
        for col in df.columns:
            if col not in known_machine_fields and col != "memory_mb" and pd.notna(row.get(col)):
                val = str(row[col])
                if val:
                    extra[col] = val

        machines.append(
            Machine(
                name=str(name),
                region=str(region),
                recommended_sku=str(sku),
                vcpu=vcpu,
                memory_gb=memory_gb,
                vm_family=None,  # Not present in Dr Migrate export
                extra=extra,
            )
        )
    return machines


def _excel_disks_to_disk_objects(
    df: pd.DataFrame,
    server_region_map: dict[str, str],
) -> list[Disk]:
    """Convert an Excel Disks-tab DataFrame into Disk objects.

    Args:
        df: Disks DataFrame after column mapping.
        server_region_map: Mapping of server name → region from the Servers tab.
    """
    disks: list[Disk] = []

    for _, row in df.iterrows():
        server_name = row.get("server_name")
        if pd.isna(server_name):
            continue

        server_name = str(server_name).strip()
        disk_name = str(row.get("disk_name", "")) if pd.notna(row.get("disk_name")) else ""

        # Region comes from the server
        region = server_region_map.get(server_name)

        # Numeric fields
        def _float_or_none(key: str) -> Optional[float]:
            val = row.get(key)
            if pd.notna(val):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            return None

        chosen_sku = str(row.get("chosen_disk_sku", "")) if pd.notna(row.get("chosen_disk_sku")) else None
        recommended_sku = str(row.get("recommended_disk_sku", "")) if pd.notna(row.get("recommended_disk_sku")) else None
        storage_target = str(row.get("storage_target", "")) if pd.notna(row.get("storage_target")) else None
        server_category = str(row.get("server_category", "")) if pd.notna(row.get("server_category")) else None
        scope = str(row.get("scope", "")) if pd.notna(row.get("scope")) else None

        extra: dict[str, str] = {}
        skip_keys = {
            "server_name", "disk_name", "disk_size_gb", "chosen_disk_sku",
            "recommended_disk_sku", "server_category", "scope",
            "disk_read_mbps", "disk_write_mbps", "disk_read_iops",
            "disk_write_iops", "storage_target",
        }
        for col in df.columns:
            if col not in skip_keys and pd.notna(row.get(col)):
                val = str(row[col])
                if val:
                    extra[col] = val

        disks.append(
            Disk(
                server_name=server_name,
                disk_name=disk_name,
                disk_size_gb=_float_or_none("disk_size_gb"),
                chosen_disk_sku=chosen_sku,
                recommended_disk_sku=recommended_sku,
                region=region,
                server_category=server_category,
                scope=scope,
                disk_read_mbps=_float_or_none("disk_read_mbps"),
                disk_write_mbps=_float_or_none("disk_write_mbps"),
                disk_read_iops=_float_or_none("disk_read_iops"),
                disk_write_iops=_float_or_none("disk_write_iops"),
                storage_target=storage_target,
                extra=extra,
            )
        )
    return disks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_csv(source: Union[str, Path, io.StringIO, io.BytesIO]) -> list[Machine]:
    """Parse a CSV file or stream into a list of Machine objects.

    Args:
        source: File path, StringIO, or BytesIO containing CSV data.

    Returns:
        List of Machine objects.

    Raises:
        DatasetParseError: If the CSV cannot be parsed or required columns are missing.
    """
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise DatasetParseError(f"Failed to read CSV: {exc}") from exc

    df = _normalize_columns(df)
    _validate_required(df)
    logger.info("Parsed CSV with %d rows and columns: %s", len(df), list(df.columns))
    return _to_machines(df)


def parse_json(source: Union[str, Path, io.StringIO, io.BytesIO]) -> list[Machine]:
    """Parse a JSON file or stream into a list of Machine objects.

    Expects either a JSON array of objects or a JSON object with a top-level key
    containing the array (e.g. {"machines": [...]}).

    Args:
        source: File path, StringIO, or BytesIO containing JSON data.

    Returns:
        List of Machine objects.

    Raises:
        DatasetParseError: If the JSON cannot be parsed or required columns are missing.
    """
    try:
        if isinstance(source, (str, Path)):
            with open(source, "r", encoding="utf-8") as f:
                raw = json.load(f)
        elif isinstance(source, io.BytesIO):
            raw = json.loads(source.read().decode("utf-8"))
        else:
            raw = json.loads(source.read())
    except Exception as exc:
        raise DatasetParseError(f"Failed to read JSON: {exc}") from exc

    # If raw is a dict, look for the first key whose value is a list
    if isinstance(raw, dict):
        for key, val in raw.items():
            if isinstance(val, list):
                raw = val
                break
        else:
            raise DatasetParseError(
                "JSON object does not contain an array of records."
            )

    if not isinstance(raw, list):
        raise DatasetParseError("Expected a JSON array of machine records.")

    try:
        df = pd.DataFrame(raw)
    except Exception as exc:
        raise DatasetParseError(f"Failed to convert JSON to DataFrame: {exc}") from exc

    df = _normalize_columns(df)
    _validate_required(df)
    logger.info("Parsed JSON with %d rows and columns: %s", len(df), list(df.columns))
    return _to_machines(df)


def parse_excel(source: Union[str, Path, io.BytesIO]) -> ExcelParseResult:
    """Parse an Excel (.xlsx) rightsizing export into machines and disks.

    Expects Dr Migrate / Azure Migrate export format with:
      - "Servers" sheet with headers on row 6
      - "Disks" sheet with headers on row 6 (optional)

    Args:
        source: File path or BytesIO stream.

    Returns:
        ExcelParseResult containing machines and disks.

    Raises:
        DatasetParseError: If the file cannot be parsed or required sheets/columns
            are missing.
    """
    try:
        xls = pd.ExcelFile(source, engine="openpyxl")
    except Exception as exc:
        raise DatasetParseError(f"Failed to open Excel file: {exc}") from exc

    sheet_names_lower = {s.lower(): s for s in xls.sheet_names}

    # --- Servers sheet ---
    servers_sheet = sheet_names_lower.get("servers")
    if not servers_sheet:
        raise DatasetParseError(
            f"Excel file has no 'Servers' sheet. Found sheets: {xls.sheet_names}"
        )

    try:
        servers_df = _read_excel_sheet(source, servers_sheet, header_row=6)
    except Exception as exc:
        raise DatasetParseError(f"Failed to read Servers sheet: {exc}") from exc

    servers_df = _map_excel_columns(servers_df, EXCEL_SERVER_COLUMNS)
    machines = _excel_servers_to_machines(servers_df)
    logger.info("Parsed %d servers from Excel Servers sheet", len(machines))

    if not machines:
        raise DatasetParseError(
            "No valid server rows found in the Servers sheet. "
            "Ensure the sheet has Server, Target Azure Region, and Chosen SKU columns."
        )

    # Build server → region map for disk parsing
    server_region_map: dict[str, str] = {}
    for m in machines:
        server_region_map[m.name] = m.region

    # --- Disks sheet (optional) ---
    disks: list[Disk] = []
    disks_sheet = sheet_names_lower.get("disks") or sheet_names_lower.get("discs")
    if disks_sheet:
        try:
            disks_df = _read_excel_sheet(source, disks_sheet, header_row=6)
            disks_df = _map_excel_columns(disks_df, EXCEL_DISK_COLUMNS)
            disks = _excel_disks_to_disk_objects(disks_df, server_region_map)
            logger.info("Parsed %d disks from Excel Disks sheet", len(disks))
        except Exception as exc:
            logger.warning("Could not parse Disks sheet: %s", exc)
    else:
        logger.info("No Disks sheet found in Excel file — skipping disk parsing")

    return ExcelParseResult(machines=machines, disks=disks)


def parse_file(
    source: Union[str, Path, io.BytesIO],
    filename: str,
) -> Union[list[Machine], ExcelParseResult]:
    """Auto-detect format from filename and parse accordingly.

    Args:
        source: File path or BytesIO stream.
        filename: Original filename used for format detection.

    Returns:
        For CSV/JSON: List of Machine objects.
        For Excel: ExcelParseResult with machines and disks.

    Raises:
        DatasetParseError: If format is unsupported or parsing fails.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return parse_csv(source)
    elif ext == ".json":
        return parse_json(source)
    elif ext in (".xlsx", ".xls"):
        return parse_excel(source)
    else:
        raise DatasetParseError(
            f"Unsupported file format '{ext}'. Use .csv, .json, or .xlsx files."
        )
