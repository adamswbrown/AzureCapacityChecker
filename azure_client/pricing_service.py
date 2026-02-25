"""Azure Retail Prices API client.

Fetches VM pricing from the public Azure Retail Prices API.
No authentication required — this is a free, public Microsoft API.

Returns three pricing tiers per SKU:
    - PAYG (Pay-As-You-Go / Consumption): hourly rate × 730 hours/month
    - 1-Year RI (Reserved Instance): term cost ÷ 12
    - 3-Year RI (Reserved Instance): term cost ÷ 36

API docs: https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://prices.azure.com/api/retail/prices"
_HOURS_PER_MONTH = 730  # Azure standard: 365 days / 12 months * 24 hours


@dataclass
class SkuPricing:
    """Monthly pricing for a single SKU across tiers."""

    payg: Optional[float] = None
    ri_1yr: Optional[float] = None
    ri_3yr: Optional[float] = None

    @property
    def cheapest(self) -> Optional[float]:
        """Return the lowest available monthly price across all tiers."""
        prices = [p for p in (self.payg, self.ri_1yr, self.ri_3yr) if p is not None]
        return min(prices) if prices else None

    def as_dict(self) -> dict[str, Optional[float]]:
        return {"payg": self.payg, "ri_1yr": self.ri_1yr, "ri_3yr": self.ri_3yr}


class PricingService:
    """Fetch VM retail pricing (PAYG + RI) from the Azure Retail Prices API."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], SkuPricing] = {}

    def get_pricing(self, sku_name: str, region: str) -> SkuPricing:
        """Get pricing for a single SKU in a region.

        Returns a SkuPricing with payg, ri_1yr, ri_3yr (any may be None).
        """
        key = (sku_name.lower(), region.lower())
        if key in self._cache:
            return self._cache[key]

        result = self._fetch_single(sku_name, region)
        self._cache[key] = result
        return result

    def fetch_prices_batch(
        self,
        sku_region_pairs: set[tuple[str, str]],
        progress_callback=None,
    ) -> dict[tuple[str, str], SkuPricing]:
        """Fetch pricing for multiple SKU+region pairs efficiently.

        Groups by region to minimise API calls. Each request fetches
        PAYG + 1yr RI + 3yr RI in one go.

        Returns:
            Dict mapping (sku_name, region) -> SkuPricing.
        """
        by_region: dict[str, list[str]] = {}
        for sku, region in sku_region_pairs:
            by_region.setdefault(region.lower(), []).append(sku)

        results: dict[tuple[str, str], SkuPricing] = {}
        total_regions = len(by_region)
        done = 0

        for region, skus in by_region.items():
            # Batch up to 15 SKUs per request (OData filter grows with RI terms)
            for i in range(0, len(skus), 15):
                batch = skus[i : i + 15]
                batch_results = self._fetch_batch(batch, region)
                results.update(batch_results)

            done += 1
            if progress_callback:
                progress_callback(done, total_regions)

        # Cache all results
        for key, pricing in results.items():
            self._cache[(key[0].lower(), key[1].lower())] = pricing

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_single(self, sku_name: str, region: str) -> SkuPricing:
        """Fetch all tiers for a single SKU from the API."""
        odata_filter = (
            f"serviceName eq 'Virtual Machines' "
            f"and armRegionName eq '{region}' "
            f"and armSkuName eq '{sku_name}' "
            f"and (priceType eq 'Consumption' or priceType eq 'Reservation') "
            f"and contains(meterName, 'Spot') eq false "
            f"and contains(meterName, 'Low Priority') eq false "
            f"and contains(productName, 'Windows') eq false"
        )

        try:
            resp = requests.get(
                _API_URL,
                params={"$filter": odata_filter, "api-version": "2023-01-01-preview"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Pricing API error for %s in %s: %s", sku_name, region, exc)
            return SkuPricing()

        return self._items_to_pricing(data.get("Items", []), sku_name, region)

    def _fetch_batch(
        self, sku_names: list[str], region: str
    ) -> dict[tuple[str, str], SkuPricing]:
        """Fetch pricing for a batch of SKUs in one region."""
        sku_clauses = " or ".join(f"armSkuName eq '{s}'" for s in sku_names)
        odata_filter = (
            f"serviceName eq 'Virtual Machines' "
            f"and armRegionName eq '{region}' "
            f"and ({sku_clauses}) "
            f"and (priceType eq 'Consumption' or priceType eq 'Reservation') "
            f"and contains(meterName, 'Spot') eq false "
            f"and contains(meterName, 'Low Priority') eq false "
            f"and contains(productName, 'Windows') eq false"
        )

        # Collect all items across pages
        all_items: list[dict] = []
        next_url: Optional[str] = _API_URL

        try:
            while next_url:
                if next_url == _API_URL:
                    resp = requests.get(
                        next_url,
                        params={
                            "$filter": odata_filter,
                            "api-version": "2023-01-01-preview",
                        },
                        timeout=15,
                    )
                else:
                    resp = requests.get(next_url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                all_items.extend(data.get("Items", []))
                next_url = data.get("NextPageLink")

        except Exception as exc:
            logger.warning("Pricing batch API error for %s: %s", region, exc)

        # Group items by SKU, then build SkuPricing per SKU
        by_sku: dict[str, list[dict]] = {}
        for item in all_items:
            sku = item.get("armSkuName", "")
            if sku:
                by_sku.setdefault(sku, []).append(item)

        results: dict[tuple[str, str], SkuPricing] = {}
        for sku, items in by_sku.items():
            results[(sku, region)] = self._items_to_pricing(items, sku, region)

        return results

    @staticmethod
    def _items_to_pricing(
        items: list[dict], sku_name: str, region: str
    ) -> SkuPricing:
        """Convert API items into a SkuPricing object.

        Uses the same cost model as azure_sku_compute:
            PAYG:   hourly × 730
            1yr RI: term cost ÷ 12
            3yr RI: term cost ÷ 36
        """
        payg_candidates: list[float] = []
        ri_1yr_candidates: list[float] = []
        ri_3yr_candidates: list[float] = []

        for item in items:
            price = item.get("retailPrice", 0)
            if price <= 0:
                continue

            price_type = item.get("type", item.get("priceType", ""))
            reservation_term = item.get("reservationTerm", "")

            if price_type == "Consumption":
                payg_candidates.append(round(price * _HOURS_PER_MONTH, 2))
            elif price_type == "Reservation":
                if reservation_term == "1 Year":
                    ri_1yr_candidates.append(round(price / 12, 2))
                elif reservation_term == "3 Years":
                    ri_3yr_candidates.append(round(price / 36, 2))

        return SkuPricing(
            payg=min(payg_candidates) if payg_candidates else None,
            ri_1yr=min(ri_1yr_candidates) if ri_1yr_candidates else None,
            ri_3yr=min(ri_3yr_candidates) if ri_3yr_candidates else None,
        )
