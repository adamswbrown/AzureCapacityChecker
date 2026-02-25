"""Machine model representing a single VM entry from a rightsizing dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Regex to parse Azure VM SKU names like Standard_D2as_v5, Standard_E192is_v6
# Groups: (1) family prefix, (2) size digits, (3) feature suffix, (4) generation
_SKU_FAMILY_RE = re.compile(
    r"Standard_([A-Za-z]+?)(\d+)([a-z]*)_v(\d+)$", re.IGNORECASE
)


def _derive_vm_family(sku_name: str) -> Optional[str]:
    """Derive the VM family from a SKU name.

    Examples:
        Standard_D2as_v5   → Dasv5
        Standard_E4bs_v5   → Ebsv5
        Standard_F8als_v6  → Falsv6
        Standard_E192is_v6 → Eisv6
        Standard_B2s_v2    → Bsv2
    """
    match = _SKU_FAMILY_RE.match(sku_name)
    if match:
        prefix = match.group(1)   # D, E, F, B, etc.
        suffix = match.group(3)   # as, bs, als, s, ds, etc.
        gen = match.group(4)      # 5, 6, 2, etc.
        return f"{prefix}{suffix}v{gen}"
    return None


@dataclass
class Machine:
    """Represents a single machine row from the input dataset.

    Attributes:
        name: The machine/host name.
        region: Azure region where the VM should be deployed.
        recommended_sku: The SKU recommended by the rightsizing tool.
        vcpu: Number of virtual CPUs.
        memory_gb: Memory in gigabytes.
        vm_family: The Azure VM family (e.g. Dasv5, Ebsv5).
    """

    name: str
    region: str
    recommended_sku: str
    vcpu: Optional[int] = None
    memory_gb: Optional[float] = None
    vm_family: Optional[str] = None
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.region = self.region.strip().lower().replace(" ", "")
        self.recommended_sku = self.recommended_sku.strip()
        if self.vm_family:
            self.vm_family = self.vm_family.strip()
        elif self.recommended_sku:
            self.vm_family = _derive_vm_family(self.recommended_sku)

    @property
    def display_region(self) -> str:
        """Return a human-readable region name."""
        region_map: dict[str, str] = {
            "uksouth": "UK South",
            "ukwest": "UK West",
            "eastus": "East US",
            "eastus2": "East US 2",
            "westus": "West US",
            "westus2": "West US 2",
            "westus3": "West US 3",
            "centralus": "Central US",
            "northeurope": "North Europe",
            "westeurope": "West Europe",
            "southeastasia": "Southeast Asia",
            "eastasia": "East Asia",
            "australiaeast": "Australia East",
            "australiasoutheast": "Australia Southeast",
            "japaneast": "Japan East",
            "japanwest": "Japan West",
            "canadacentral": "Canada Central",
            "canadaeast": "Canada East",
            "brazilsouth": "Brazil South",
            "southcentralus": "South Central US",
            "northcentralus": "North Central US",
            "westcentralus": "West Central US",
            "germanywestcentral": "Germany West Central",
            "switzerlandnorth": "Switzerland North",
            "norwayeast": "Norway East",
            "swedencentral": "Sweden Central",
            "francecentral": "France Central",
            "francesouth": "France South",
            "koreacentral": "Korea Central",
            "koreasouth": "Korea South",
            "southafricanorth": "South Africa North",
            "uaenorth": "UAE North",
            "centralindia": "Central India",
            "southindia": "South India",
            "westindia": "West India",
        }
        return region_map.get(self.region, self.region)
