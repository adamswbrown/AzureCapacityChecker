"""Azure SKU Service — fetches and caches VM SKU data from the Azure Resource Manager API.

Uses the Compute Resource SKUs endpoint to retrieve the full list of available
VM sizes across regions, including restriction and capability metadata.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from azure_client.auth import AuthMethod, get_access_token
from app.config import (
    AZURE_COMPUTE_SKU_API_VERSION,
    AZURE_MAX_RETRIES,
    AZURE_RETRY_WAIT_SECONDS,
    AZURE_RETRY_MAX_WAIT_SECONDS,
    SKU_CACHE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)


class SkuServiceError(Exception):
    """Raised when the SKU service encounters an error."""


@dataclass
class SkuCapability:
    """A single capability of a VM SKU (e.g. vCPUs, MemoryGB)."""

    name: str
    value: str


@dataclass
class SkuRestriction:
    """A restriction on a VM SKU in a region."""

    type: str
    values: list[str] = field(default_factory=list)
    reason_code: str = ""


@dataclass
class SkuInfo:
    """Parsed information about a single VM SKU."""

    name: str
    tier: str
    size: str
    family: str
    locations: list[str] = field(default_factory=list)
    capabilities: dict[str, str] = field(default_factory=dict)
    restrictions: list[SkuRestriction] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)

    @property
    def vcpus(self) -> Optional[int]:
        """Extract vCPU count from capabilities."""
        val = self.capabilities.get("vCPUs") or self.capabilities.get("vCPUsAvailable")
        if val:
            try:
                return int(val)
            except ValueError:
                return None
        return None

    @property
    def memory_gb(self) -> Optional[float]:
        """Extract memory in GB from capabilities."""
        val = self.capabilities.get("MemoryGB")
        if val:
            try:
                return float(val)
            except ValueError:
                return None
        return None

    def is_restricted_in_region(self, region: str) -> bool:
        """Check if this SKU has a Location-level restriction in the given region.

        Only checks Location restrictions (full block). Zone restrictions are
        handled separately by is_zone_limited_in_region().
        """
        region_lower = region.lower()
        for restriction in self.restrictions:
            if restriction.type == "Location":
                if region_lower in [v.lower() for v in restriction.values]:
                    return True
        return False

    def is_available_in_region(self, region: str) -> bool:
        """Check if this SKU lists the region in its locations."""
        return region.lower() in [loc.lower() for loc in self.locations]

    def is_zone_limited_in_region(self, region: str) -> bool:
        """Check if the SKU has zone-level restrictions in this region."""
        for restriction in self.restrictions:
            if restriction.type == "Zone":
                # Zone restrictions list regions where some zones are unavailable
                restricted_regions = [v.lower() for v in restriction.values]
                if region.lower() in restricted_regions:
                    return True
        return False


def _parse_sku(raw: dict[str, Any]) -> Optional[SkuInfo]:
    """Parse a raw SKU dict from the API into a SkuInfo, filtering to virtualMachines."""
    if raw.get("resourceType") != "virtualMachines":
        return None

    capabilities: dict[str, str] = {}
    for cap in raw.get("capabilities", []):
        capabilities[cap["name"]] = cap["value"]

    restrictions: list[SkuRestriction] = []
    for rest in raw.get("restrictions", []):
        restriction_type = rest.get("type", "")
        values: list[str] = []
        reason_code = rest.get("reasonCode", "")
        restriction_info = rest.get("restrictionInfo", {})
        if restriction_type == "Location":
            values = restriction_info.get("locations", [])
        elif restriction_type == "Zone":
            values = restriction_info.get("locations", [])
        restrictions.append(
            SkuRestriction(type=restriction_type, values=values, reason_code=reason_code)
        )

    locations = raw.get("locations", [])
    location_info = raw.get("locationInfo", [])
    zones: list[str] = []
    if location_info:
        zones = location_info[0].get("zones", [])

    return SkuInfo(
        name=raw.get("name", ""),
        tier=raw.get("tier", ""),
        size=raw.get("size", ""),
        family=raw.get("family", ""),
        locations=locations,
        capabilities=capabilities,
        restrictions=restrictions,
        zones=zones,
    )


# Managed disk storage tier names in the Azure API
_DISK_TIER_NAMES = {
    "standard_lrs", "standardssd_lrs", "standardssd_zrs",
    "premium_lrs", "premium_zrs", "premiumv2_lrs", "ultrassd_lrs",
}


def _parse_disk_sku(raw: dict[str, Any]) -> Optional[SkuInfo]:
    """Parse a raw SKU dict from the API into a SkuInfo for disk tiers."""
    if raw.get("resourceType") != "disks":
        return None

    name = raw.get("name", "")
    # Only keep actual managed disk storage tiers, not VM host types
    if name.lower() not in _DISK_TIER_NAMES:
        return None

    capabilities: dict[str, str] = {}
    for cap in raw.get("capabilities", []):
        capabilities[cap["name"]] = cap["value"]

    restrictions: list[SkuRestriction] = []
    for rest in raw.get("restrictions", []):
        restriction_type = rest.get("type", "")
        values: list[str] = []
        reason_code = rest.get("reasonCode", "")
        restriction_info = rest.get("restrictionInfo", {})
        if restriction_type == "Location":
            values = restriction_info.get("locations", [])
        elif restriction_type == "Zone":
            values = restriction_info.get("locations", [])
        restrictions.append(
            SkuRestriction(type=restriction_type, values=values, reason_code=reason_code)
        )

    locations = raw.get("locations", [])
    location_info = raw.get("locationInfo", [])
    zones: list[str] = []
    if location_info:
        zones = location_info[0].get("zones", [])

    return SkuInfo(
        name=name,
        tier=raw.get("tier", ""),
        size=raw.get("size", ""),
        family=raw.get("family", ""),
        locations=locations,
        capabilities=capabilities,
        restrictions=restrictions,
        zones=zones,
    )


class SkuCache:
    """In-memory cache for SKU data with TTL."""

    def __init__(self, ttl: int = SKU_CACHE_TTL_SECONDS) -> None:
        self._data: dict[str, SkuInfo] = {}
        self._region_index: dict[str, list[str]] = {}
        self._timestamp: float = 0.0
        self._ttl = ttl

    @property
    def is_valid(self) -> bool:
        """Check if the cache is still within its TTL."""
        return bool(self._data) and (time.time() - self._timestamp) < self._ttl

    def store(self, skus: list[SkuInfo]) -> None:
        """Store a list of SKUs, indexing by name and region.

        The Azure API returns one entry per SKU per region. This method merges
        duplicate entries so each SKU name maps to a single SkuInfo with all
        locations, restrictions, and zone data combined.
        """
        self._data.clear()
        self._region_index.clear()
        for sku in skus:
            key = sku.name.lower()
            if key in self._data:
                existing = self._data[key]
                # Merge locations (avoid duplicates)
                for loc in sku.locations:
                    if loc.lower() not in [l.lower() for l in existing.locations]:
                        existing.locations.append(loc)
                # Merge restrictions
                existing.restrictions.extend(sku.restrictions)
                # Merge zones (avoid duplicates)
                for z in sku.zones:
                    if z not in existing.zones:
                        existing.zones.append(z)
            else:
                self._data[key] = sku
            for loc in sku.locations:
                loc_lower = loc.lower()
                if loc_lower not in self._region_index:
                    self._region_index[loc_lower] = []
                if key not in self._region_index[loc_lower]:
                    self._region_index[loc_lower].append(key)
        self._timestamp = time.time()
        logger.info(
            "Cached %d unique VM SKUs across %d regions (from %d API entries)",
            len(self._data), len(self._region_index), len(skus),
        )

    def get_sku(self, name: str) -> Optional[SkuInfo]:
        """Look up a SKU by name (case-insensitive)."""
        return self._data.get(name.lower())

    def get_skus_in_region(self, region: str) -> list[SkuInfo]:
        """Return all SKUs available in a given region."""
        sku_names = self._region_index.get(region.lower(), [])
        return [self._data[n] for n in sku_names if n in self._data]

    def all_skus(self) -> list[SkuInfo]:
        """Return all cached SKUs."""
        return list(self._data.values())


class SkuService:
    """Service for fetching and querying Azure VM SKU availability."""

    def __init__(
        self,
        subscription_id: str,
        cache_ttl: int = SKU_CACHE_TTL_SECONDS,
        auth_method: AuthMethod = AuthMethod.DEFAULT,
        tenant_id: str = "",
        client_id: str = "",
        client_secret: str = "",
    ) -> None:
        """Initialize the SKU service.

        Args:
            subscription_id: Azure subscription ID.
            cache_ttl: Cache time-to-live in seconds.
            auth_method: Authentication method to use.
            tenant_id: For Service Principal / Device Code.
            client_id: For Service Principal.
            client_secret: For Service Principal.
        """
        if not subscription_id:
            raise SkuServiceError(
                "Azure subscription ID is required. "
                "Set the AZURE_SUBSCRIPTION_ID environment variable."
            )
        self._subscription_id = subscription_id
        self._auth_method = auth_method
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._cache = SkuCache(ttl=cache_ttl)
        self._disk_cache = SkuCache(ttl=cache_ttl)
        self._base_url = (
            f"https://management.azure.com/subscriptions/{subscription_id}"
            f"/providers/Microsoft.Compute/skus"
        )

    @retry(
        stop=stop_after_attempt(AZURE_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=AZURE_RETRY_WAIT_SECONDS,
            max=AZURE_RETRY_MAX_WAIT_SECONDS,
        ),
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch_page(self, url: str, token: str) -> dict[str, Any]:
        """Fetch a single page from the SKU API with retry logic.

        Args:
            url: The full URL to request.
            token: Bearer token for authentication.

        Returns:
            The parsed JSON response.
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_skus(self, force_refresh: bool = False) -> list[SkuInfo]:
        """Fetch all VM SKUs from Azure, using cache when possible.

        Handles pagination via the nextLink field in the API response.

        Args:
            force_refresh: If True, ignore the cache and re-fetch.

        Returns:
            List of all SkuInfo objects for virtualMachines.

        Raises:
            SkuServiceError: If the API call fails.
        """
        if self._cache.is_valid and not force_refresh:
            logger.info("Returning %d SKUs from cache", len(self._cache.all_skus()))
            return self._cache.all_skus()

        token = get_access_token(
            method=self._auth_method,
            tenant_id=self._tenant_id,
            client_id=self._client_id,
            client_secret=self._client_secret,
        )
        url = f"{self._base_url}?api-version={AZURE_COMPUTE_SKU_API_VERSION}&$filter=resourceType eq 'virtualMachines'"

        all_skus: list[SkuInfo] = []
        page_count = 0

        try:
            while url:
                page_count += 1
                logger.info("Fetching SKU page %d ...", page_count)
                data = self._fetch_page(url, token)

                for raw_sku in data.get("value", []):
                    parsed = _parse_sku(raw_sku)
                    if parsed:
                        all_skus.append(parsed)

                url = data.get("nextLink")

        except requests.HTTPError as exc:
            raise SkuServiceError(
                f"Azure API returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except requests.RequestException as exc:
            raise SkuServiceError(f"Network error fetching SKUs: {exc}") from exc

        self._cache.store(all_skus)
        logger.info("Fetched %d VM SKUs in %d pages", len(all_skus), page_count)
        return all_skus

    def get_sku_info(self, sku_name: str) -> Optional[SkuInfo]:
        """Look up a single SKU by name.

        Requires that fetch_skus() has been called at least once.

        Args:
            sku_name: The SKU name (e.g. 'Standard_D4s_v5').

        Returns:
            SkuInfo if found, else None.
        """
        return self._cache.get_sku(sku_name)

    def get_skus_in_region(self, region: str) -> list[SkuInfo]:
        """Return all VM SKUs available in a specific region.

        Requires that fetch_skus() has been called at least once.

        Args:
            region: Azure region name (e.g. 'uksouth').

        Returns:
            List of SkuInfo objects available in the region.
        """
        return self._cache.get_skus_in_region(region)

    def fetch_disk_skus(self, force_refresh: bool = False) -> list[SkuInfo]:
        """Fetch managed disk tier SKUs from Azure.

        Returns SkuInfo entries for disk storage tiers (Standard_LRS,
        Premium_LRS, StandardSSD_LRS, UltraSSD_LRS, PremiumV2_LRS, etc.).

        Args:
            force_refresh: If True, ignore the cache and re-fetch.

        Returns:
            List of merged SkuInfo objects for disk tiers.
        """
        if self._disk_cache.is_valid and not force_refresh:
            logger.info("Returning %d disk SKUs from cache", len(self._disk_cache.all_skus()))
            return self._disk_cache.all_skus()

        token = get_access_token(
            method=self._auth_method,
            tenant_id=self._tenant_id,
            client_id=self._client_id,
            client_secret=self._client_secret,
        )
        url = (
            f"{self._base_url}?api-version={AZURE_COMPUTE_SKU_API_VERSION}"
            f"&$filter=resourceType eq 'disks'"
        )

        all_disk_skus: list[SkuInfo] = []
        page_count = 0

        try:
            while url:
                page_count += 1
                logger.info("Fetching disk SKU page %d ...", page_count)
                data = self._fetch_page(url, token)

                for raw_sku in data.get("value", []):
                    parsed = _parse_disk_sku(raw_sku)
                    if parsed:
                        all_disk_skus.append(parsed)

                url = data.get("nextLink")

        except requests.HTTPError as exc:
            raise SkuServiceError(
                f"Azure API returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except requests.RequestException as exc:
            raise SkuServiceError(f"Network error fetching disk SKUs: {exc}") from exc

        self._disk_cache.store(all_disk_skus)
        logger.info("Fetched %d disk SKU entries in %d pages", len(all_disk_skus), page_count)
        return all_disk_skus

    def get_disk_tier_info(self, tier_name: str) -> Optional[SkuInfo]:
        """Look up a disk storage tier by name (e.g. 'Standard_LRS').

        Requires that fetch_disk_skus() has been called at least once.
        """
        return self._disk_cache.get_sku(tier_name)
