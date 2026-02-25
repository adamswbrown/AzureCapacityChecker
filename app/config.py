"""Application configuration for Azure Capacity Advisor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Azure Resource Manager API version for Compute SKU listing
AZURE_COMPUTE_SKU_API_VERSION: str = "2023-07-01"

# Default Azure regions presented in the UI
DEFAULT_REGIONS: list[str] = [
    "eastus",
    "eastus2",
    "westus",
    "westus2",
    "westus3",
    "centralus",
    "southcentralus",
    "northcentralus",
    "westcentralus",
    "uksouth",
    "ukwest",
    "northeurope",
    "westeurope",
    "francecentral",
    "germanywestcentral",
    "switzerlandnorth",
    "norwayeast",
    "swedencentral",
    "southeastasia",
    "eastasia",
    "australiaeast",
    "australiasoutheast",
    "japaneast",
    "japanwest",
    "canadacentral",
    "canadaeast",
    "brazilsouth",
    "koreacentral",
    "centralindia",
    "southafricanorth",
    "uaenorth",
]

# Human-readable region labels for the UI dropdown
REGION_DISPLAY_NAMES: dict[str, str] = {
    "eastus": "East US",
    "eastus2": "East US 2",
    "westus": "West US",
    "westus2": "West US 2",
    "westus3": "West US 3",
    "centralus": "Central US",
    "southcentralus": "South Central US",
    "northcentralus": "North Central US",
    "westcentralus": "West Central US",
    "uksouth": "UK South",
    "ukwest": "UK West",
    "northeurope": "North Europe",
    "westeurope": "West Europe",
    "francecentral": "France Central",
    "germanywestcentral": "Germany West Central",
    "switzerlandnorth": "Switzerland North",
    "norwayeast": "Norway East",
    "swedencentral": "Sweden Central",
    "southeastasia": "Southeast Asia",
    "eastasia": "East Asia",
    "australiaeast": "Australia East",
    "australiasoutheast": "Australia Southeast",
    "japaneast": "Japan East",
    "japanwest": "Japan West",
    "canadacentral": "Canada Central",
    "canadaeast": "Canada East",
    "brazilsouth": "Brazil South",
    "koreacentral": "Korea Central",
    "centralindia": "Central India",
    "southafricanorth": "South Africa North",
    "uaenorth": "UAE North",
}

# Number of alternative SKUs to recommend
MAX_ALTERNATIVES: int = 5

# SKU cache TTL in seconds (1 hour)
SKU_CACHE_TTL_SECONDS: int = 3600

# Maximum dataset rows supported
MAX_DATASET_ROWS: int = 10_000

# Azure API retry settings
AZURE_MAX_RETRIES: int = 3
AZURE_RETRY_WAIT_SECONDS: int = 2
AZURE_RETRY_MAX_WAIT_SECONDS: int = 30

# ARM Deployment Validation (Method 2 — live capacity check) settings
ARM_DEPLOYMENT_API_VERSION: str = "2024-11-01"
ARM_VALIDATION_MAX_WORKERS: int = 5
ARM_VALIDATION_RATE_LIMIT_PER_MINUTE: int = 150
ARM_VALIDATION_POLL_INTERVAL_SECONDS: int = 5
ARM_VALIDATION_POLL_TIMEOUT_SECONDS: int = 60
ARM_VALIDATION_REQUEST_TIMEOUT_SECONDS: int = 30


@dataclass
class AppConfig:
    """Runtime configuration, resolved from environment variables and defaults."""

    subscription_id: str = ""
    cache_ttl: int = SKU_CACHE_TTL_SECONDS
    max_alternatives: int = MAX_ALTERNATIVES
    max_retries: int = AZURE_MAX_RETRIES
    default_regions: list[str] = field(default_factory=lambda: list(DEFAULT_REGIONS))

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build configuration from environment variables."""
        return cls(
            subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID", ""),
            cache_ttl=int(
                os.environ.get("SKU_CACHE_TTL", str(SKU_CACHE_TTL_SECONDS))
            ),
            max_alternatives=int(
                os.environ.get("MAX_ALTERNATIVES", str(MAX_ALTERNATIVES))
            ),
            max_retries=int(
                os.environ.get("AZURE_MAX_RETRIES", str(AZURE_MAX_RETRIES))
            ),
        )
