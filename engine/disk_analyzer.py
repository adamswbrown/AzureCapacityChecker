"""Disk capacity analyzer engine.

Evaluates each disk's chosen SKU against the Azure disk tier catalog to
determine availability in the target region.

Disk SKU names (S4, S10, E6, P30, etc.) are mapped to their underlying
Azure storage tier (Standard_LRS, StandardSSD_LRS, Premium_LRS, etc.)
and checked for regional availability and restrictions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from azure_client.sku_service import SkuInfo, SkuService
from models.disk import Disk

logger = logging.getLogger(__name__)


class DiskStatus(str, Enum):
    """Status of a disk SKU in a given region."""

    OK = "OK"
    RISK = "RISK"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        return self.value


@dataclass
class DiskAnalysisResult:
    """Result of analysing a single disk's SKU availability."""

    server_name: str
    disk_name: str
    region: str
    chosen_sku: str
    azure_tier: str
    disk_size_gb: Optional[float]
    storage_target: Optional[str]
    status: DiskStatus
    reason: str = ""
    sku_tier_label: str = ""


# ---------------------------------------------------------------------------
# Mapping from disk size SKU names to Azure storage tiers
# ---------------------------------------------------------------------------

# Standard HDD sizes (S-series) → Standard_LRS
_STANDARD_HDD_RE = re.compile(r"^S\d+$", re.IGNORECASE)

# Standard SSD sizes (E-series or StandardSSD_ prefix) → StandardSSD_LRS
_STANDARD_SSD_RE = re.compile(r"^(StandardSSD_)?E\d+$", re.IGNORECASE)

# Premium SSD sizes (P-series or Premium_ prefix) → Premium_LRS
_PREMIUM_SSD_RE = re.compile(r"^(Premium_)?P\d+$", re.IGNORECASE)

# Ultra Disk
_ULTRA_RE = re.compile(r"^Ultra", re.IGNORECASE)

# Premium SSD v2
_PREMIUMV2_RE = re.compile(r"^PremiumV2", re.IGNORECASE)


def _map_disk_sku_to_tier(chosen_sku: str) -> tuple[Optional[str], str]:
    """Map a disk size SKU name to its Azure storage tier.

    Args:
        chosen_sku: The disk SKU from the export (e.g. S10, E6, P30, Ultra).

    Returns:
        Tuple of (azure_tier_name, human_label).
        azure_tier_name is None if the SKU cannot be mapped.
    """
    sku = chosen_sku.strip()

    if _STANDARD_HDD_RE.match(sku):
        return "Standard_LRS", "Standard HDD"
    if _STANDARD_SSD_RE.match(sku):
        return "StandardSSD_LRS", "Standard SSD"
    if _PREMIUM_SSD_RE.match(sku):
        return "Premium_LRS", "Premium SSD"
    if _ULTRA_RE.match(sku):
        return "UltraSSD_LRS", "Ultra Disk"
    if _PREMIUMV2_RE.match(sku):
        return "PremiumV2_LRS", "Premium SSD v2"

    # Try matching full tier names directly
    tier_map = {
        "standard_lrs": ("Standard_LRS", "Standard HDD"),
        "standardssd_lrs": ("StandardSSD_LRS", "Standard SSD"),
        "premium_lrs": ("Premium_LRS", "Premium SSD"),
        "ultrassd_lrs": ("UltraSSD_LRS", "Ultra Disk"),
        "premiumv2_lrs": ("PremiumV2_LRS", "Premium SSD v2"),
    }
    result = tier_map.get(sku.lower())
    if result:
        return result

    return None, "Unknown"


class DiskAnalyzer:
    """Evaluates disk SKU availability for a list of disks."""

    def __init__(self, sku_service: SkuService) -> None:
        self._sku_service = sku_service

    def analyze(self, disks: list[Disk]) -> list[DiskAnalysisResult]:
        """Analyze a list of disks and return availability results."""
        results: list[DiskAnalysisResult] = []
        for disk in disks:
            result = self._evaluate(disk)
            results.append(result)
        logger.info("Analyzed %d disks", len(results))
        return results

    def _evaluate(self, disk: Disk) -> DiskAnalysisResult:
        """Evaluate a single disk's SKU against its region."""
        chosen_sku = disk.effective_sku
        region = disk.region or ""

        base = dict(
            server_name=disk.server_name,
            disk_name=disk.disk_name,
            region=region,
            chosen_sku=chosen_sku,
            disk_size_gb=disk.disk_size_gb,
            storage_target=disk.storage_target,
        )

        size_str = f" ({disk.disk_size_gb:g} GB)" if disk.disk_size_gb else ""

        if not chosen_sku:
            return DiskAnalysisResult(
                **base,
                azure_tier="",
                status=DiskStatus.UNKNOWN,
                reason=(
                    f"No disk SKU was specified for this disk{size_str}. "
                    f"The export file did not include a recommended or chosen "
                    f"disk type, so we cannot verify availability."
                ),
                sku_tier_label="",
            )

        if not region:
            return DiskAnalysisResult(
                **base,
                azure_tier="",
                status=DiskStatus.UNKNOWN,
                reason=(
                    f"Cannot check disk '{chosen_sku}'{size_str} — no target "
                    f"region is set. This disk's server was not found in the "
                    f"Servers tab, so we don't know which region to check against."
                ),
                sku_tier_label="",
            )

        # Map disk SKU to Azure storage tier
        azure_tier, tier_label = _map_disk_sku_to_tier(chosen_sku)

        if azure_tier is None:
            return DiskAnalysisResult(
                **base,
                azure_tier="",
                status=DiskStatus.UNKNOWN,
                reason=(
                    f"Disk SKU '{chosen_sku}'{size_str} could not be mapped to an "
                    f"Azure storage tier. Azure managed disks use tiers like "
                    f"S4/S10 (Standard HDD), E6/E10 (Standard SSD), P10/P30 "
                    f"(Premium SSD), or Ultra. Check if the SKU name is correct."
                ),
                sku_tier_label=tier_label,
            )

        # Look up the tier in the disk cache
        tier_info: Optional[SkuInfo] = self._sku_service.get_disk_tier_info(azure_tier)

        if tier_info is None:
            return DiskAnalysisResult(
                **base,
                azure_tier=azure_tier,
                status=DiskStatus.UNKNOWN,
                reason=(
                    f"Storage tier '{azure_tier}' ({tier_label}) not found in the "
                    f"Azure disk catalog. We queried the Microsoft Compute SKU "
                    f"API for disk resources and this tier was not returned."
                ),
                sku_tier_label=tier_label,
            )

        total_regions = len(tier_info.locations)

        # Build a display region name for disk (reuse server's region logic)
        from models.machine import Machine
        display_region = Machine(
            name="", region=region, recommended_sku=""
        ).display_region

        # Check region availability
        if not tier_info.is_available_in_region(region):
            sample_regions = sorted(tier_info.locations)[:5]
            sample_str = ", ".join(sample_regions)
            more = f" and {total_regions - 5} more" if total_regions > 5 else ""
            return DiskAnalysisResult(
                **base,
                azure_tier=azure_tier,
                status=DiskStatus.BLOCKED,
                reason=(
                    f"CANNOT USE: '{chosen_sku}'{size_str} maps to {tier_label} "
                    f"({azure_tier}), which is NOT available in {display_region}. "
                    f"Microsoft does not offer this disk tier in this region. "
                    f"It is available in {total_regions} other region(s): "
                    f"{sample_str}{more}. "
                    f"You need a different disk tier or a different region."
                ),
                sku_tier_label=tier_label,
            )

        # Check location restrictions
        if tier_info.is_restricted_in_region(region):
            reason_codes = [
                r.reason_code
                for r in tier_info.restrictions
                if r.type == "Location"
                and region.lower() in [v.lower() for v in r.values]
            ]
            reason_explanations: list[str] = []
            for code in reason_codes:
                if code == "NotAvailableForSubscription":
                    reason_explanations.append(
                        "your subscription does not have access to this disk tier"
                    )
                else:
                    reason_explanations.append(code)
            reason_str = (
                "; ".join(reason_explanations) if reason_explanations
                else "restricted by Microsoft"
            )
            return DiskAnalysisResult(
                **base,
                azure_tier=azure_tier,
                status=DiskStatus.BLOCKED,
                reason=(
                    f"CANNOT USE: '{chosen_sku}'{size_str} maps to {tier_label} "
                    f"({azure_tier}), which exists in {display_region} but is "
                    f"RESTRICTED — {reason_str}. "
                    f"The disk tier is in this region but blocked for your "
                    f"subscription. Contact Azure support to request access."
                ),
                sku_tier_label=tier_label,
            )

        # Check zone restrictions
        if tier_info.is_zone_limited_in_region(region):
            zones = sorted(tier_info.zones) if tier_info.zones else []
            zone_str = ", ".join(zones) if zones else "unknown"
            return DiskAnalysisResult(
                **base,
                azure_tier=azure_tier,
                status=DiskStatus.RISK,
                reason=(
                    f"USABLE WITH CAVEATS: '{chosen_sku}'{size_str} maps to "
                    f"{tier_label} ({azure_tier}), which is in {display_region} "
                    f"but not in all availability zones (zones: {zone_str}). "
                    f"Fine for single-zone deployments but may affect "
                    f"zone-redundant high availability."
                ),
                sku_tier_label=tier_label,
            )

        # Fully available
        zones = sorted(tier_info.zones) if tier_info.zones else []
        zone_str = ", ".join(zones) if zones else "all"
        return DiskAnalysisResult(
            **base,
            azure_tier=azure_tier,
            status=DiskStatus.OK,
            reason=(
                f"READY: '{chosen_sku}'{size_str} maps to {tier_label} "
                f"({azure_tier}), confirmed available in {display_region} "
                f"with no restrictions. Zones: {zone_str}. "
                f"Available in {total_regions} region(s) total. No action needed."
            ),
            sku_tier_label=tier_label,
        )
