"""Disk model representing a single disk entry from a rightsizing dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Disk:
    """Represents a single disk row from the input dataset.

    Attributes:
        server_name: The parent server/host name.
        disk_name: The disk identifier (e.g. SCSI0:0).
        disk_size_gb: Disk capacity in gigabytes.
        chosen_disk_sku: The chosen disk SKU (e.g. S10, P30, E6).
        recommended_disk_sku: The Azure Migrate recommended disk SKU.
        region: Azure region (inherited from server or dataset).
        server_category: OS category (Windows/Linux).
        scope: Migration scope (In Scope/Out of Scope).
        disk_read_mbps: Disk read throughput in MBPS.
        disk_write_mbps: Disk write throughput in MBPS.
        disk_read_iops: Disk read IOPS.
        disk_write_iops: Disk write IOPS.
        storage_target: Storage target type (Managed Disk, Blob, etc.).
        extra: Additional columns not mapped to known fields.
    """

    server_name: str
    disk_name: str
    disk_size_gb: Optional[float] = None
    chosen_disk_sku: Optional[str] = None
    recommended_disk_sku: Optional[str] = None
    region: Optional[str] = None
    server_category: Optional[str] = None
    scope: Optional[str] = None
    disk_read_mbps: Optional[float] = None
    disk_write_mbps: Optional[float] = None
    disk_read_iops: Optional[float] = None
    disk_write_iops: Optional[float] = None
    storage_target: Optional[str] = None
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.server_name = self.server_name.strip()
        self.disk_name = self.disk_name.strip() if self.disk_name else ""
        if self.chosen_disk_sku:
            self.chosen_disk_sku = self.chosen_disk_sku.strip()
        if self.recommended_disk_sku:
            self.recommended_disk_sku = self.recommended_disk_sku.strip()
        if self.region:
            self.region = self.region.strip().lower().replace(" ", "")

    @property
    def effective_sku(self) -> str:
        """Return the chosen SKU, falling back to recommended."""
        return self.chosen_disk_sku or self.recommended_disk_sku or ""

    @property
    def sku_tier(self) -> str:
        """Classify disk SKU tier from the SKU name prefix."""
        sku = self.effective_sku.lower().replace("standard_", "").replace("standardssd_", "").replace("premium_", "")
        raw = self.effective_sku.lower()
        if raw.startswith("premium") or sku.startswith("p"):
            return "Premium SSD"
        elif raw.startswith("standardssd") or sku.startswith("e"):
            return "Standard SSD"
        elif raw.startswith("standard") or sku.startswith("s"):
            return "Standard HDD"
        elif "ultra" in raw:
            return "Ultra"
        elif "premiumv2" in raw:
            return "Premium SSD v2"
        return "Unknown"
