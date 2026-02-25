"""Capacity and risk analyzer engine.

Evaluates each machine's recommended SKU against the Azure SKU catalog
to determine availability status in the target region.

Statuses:
    OK       - SKU is available and unrestricted.
    RISK     - SKU is available but has zone restrictions or is in a constrained family.
    BLOCKED  - SKU is not available or explicitly restricted in the region.
    UNKNOWN  - SKU was not found in the Azure catalog at all.
"""

from __future__ import annotations

import logging
from typing import Optional

from azure_client.sku_service import SkuInfo, SkuService
from engine.alternatives import AlternativeEngine, DiskRequirements
from models.disk import Disk
from models.machine import Machine
from models.result import AnalysisResult, SkuStatus

logger = logging.getLogger(__name__)


class Analyzer:
    """Evaluates SKU availability and risk for a list of machines."""

    def __init__(
        self,
        sku_service: SkuService,
        alternative_engine: AlternativeEngine,
        region_override: Optional[str] = None,
        disk_map: Optional[dict[str, list[Disk]]] = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            sku_service: The Azure SKU service (must already have data fetched).
            alternative_engine: Engine for recommending alternative SKUs.
            region_override: If set, evaluate all machines against this region
                             instead of their individual region fields.
            disk_map: Mapping of machine name to its disks for disk-aware alternatives.
        """
        self._sku_service = sku_service
        self._alt_engine = alternative_engine
        self._region_override = region_override
        self._disk_map = disk_map or {}

    def analyze(self, machines: list[Machine]) -> list[AnalysisResult]:
        """Analyze a list of machines and return results.

        Args:
            machines: Parsed machine entries from the dataset.

        Returns:
            A list of AnalysisResult objects, one per machine.
        """
        results: list[AnalysisResult] = []
        for machine in machines:
            region = self._region_override or machine.region
            result = self._evaluate(machine, region)
            results.append(result)
        logger.info("Analyzed %d machines", len(results))
        return results

    def _evaluate(self, machine: Machine, region: str) -> AnalysisResult:
        """Evaluate a single machine's SKU against the region.

        Args:
            machine: The machine to evaluate.
            region: The target Azure region.

        Returns:
            An AnalysisResult with status, reason, and alternatives.
        """
        sku_name = machine.recommended_sku
        sku_info: Optional[SkuInfo] = self._sku_service.get_sku_info(sku_name)
        display = machine.display_region

        # Build disk requirements for this machine
        machine_disks = self._disk_map.get(machine.name, [])
        disk_reqs = DiskRequirements.from_disks(machine_disks) if machine_disks else None

        # Construct base result
        base = dict(
            machine_name=machine.name,
            region=region,
            display_region=display,
            requested_sku=sku_name,
            vcpu=machine.vcpu,
            memory_gb=machine.memory_gb,
            vm_family=machine.vm_family,
            disk_count=len(machine_disks),
        )

        # SKU not found in any region — it doesn't exist in Azure at all
        if sku_info is None:
            alternatives = self._alt_engine.find_alternatives(
                sku_name=sku_name,
                region=region,
                vcpu=machine.vcpu,
                memory_gb=machine.memory_gb,
                vm_family=machine.vm_family,
                disk_requirements=disk_reqs,
            )
            return AnalysisResult(
                **base,
                status=SkuStatus.UNKNOWN,
                reason=(
                    f"'{sku_name}' does not exist in the Azure VM catalog. "
                    f"We searched the full Microsoft Compute SKU list for your "
                    f"subscription and this name was not found in any region. "
                    f"This could be a typo, a retired SKU, or a name that doesn't "
                    f"match Azure's naming format (e.g. Standard_D2as_v5)."
                ),
                alternatives=alternatives,
                alternatives_detail=self._alt_engine.get_alternative_details(alternatives),
            )

        # Build specs string from what Azure reports for this SKU
        specs_parts: list[str] = []
        if sku_info.vcpus:
            specs_parts.append(f"{sku_info.vcpus} vCPUs")
        if sku_info.memory_gb:
            specs_parts.append(f"{sku_info.memory_gb:g} GB RAM")
        if sku_info.family:
            specs_parts.append(f"family: {sku_info.family}")
        specs_str = f" ({', '.join(specs_parts)})" if specs_parts else ""

        total_regions = len(sku_info.locations)

        # SKU exists but is not listed in the target region
        if not sku_info.is_available_in_region(region):
            # Show a sample of regions where it IS available
            sample_regions = sorted(sku_info.locations)[:5]
            sample_str = ", ".join(sample_regions)
            more = f" and {total_regions - 5} more" if total_regions > 5 else ""

            alternatives = self._alt_engine.find_alternatives(
                sku_name=sku_name,
                region=region,
                vcpu=sku_info.vcpus or machine.vcpu,
                memory_gb=sku_info.memory_gb or machine.memory_gb,
                vm_family=machine.vm_family or sku_info.family,
                disk_requirements=disk_reqs,
            )
            return AnalysisResult(
                **base,
                status=SkuStatus.BLOCKED,
                reason=(
                    f"CANNOT DEPLOY: '{sku_name}'{specs_str} exists in Azure but "
                    f"is NOT offered in {display}. Microsoft does not list this "
                    f"VM size for this region. It is available in {total_regions} "
                    f"other region(s): {sample_str}{more}. "
                    f"You need to either pick a different region or use one of "
                    f"the alternative SKUs listed."
                ),
                alternatives=alternatives,
                alternatives_detail=self._alt_engine.get_alternative_details(alternatives),
            )

        # SKU is in the region but has a location-level restriction
        if sku_info.is_restricted_in_region(region):
            reason_codes = [
                r.reason_code
                for r in sku_info.restrictions
                if r.type == "Location"
                and region.lower() in [v.lower() for v in r.values]
            ]
            # Translate reason codes to plain English
            reason_explanations: list[str] = []
            for code in reason_codes:
                if code == "NotAvailableForSubscription":
                    reason_explanations.append(
                        "your subscription does not have quota/access for this SKU"
                    )
                else:
                    reason_explanations.append(code)
            reason_str = (
                "; ".join(reason_explanations) if reason_explanations
                else "restricted by Microsoft"
            )

            alternatives = self._alt_engine.find_alternatives(
                sku_name=sku_name,
                region=region,
                vcpu=sku_info.vcpus or machine.vcpu,
                memory_gb=sku_info.memory_gb or machine.memory_gb,
                vm_family=machine.vm_family or sku_info.family,
                disk_requirements=disk_reqs,
            )
            return AnalysisResult(
                **base,
                status=SkuStatus.BLOCKED,
                reason=(
                    f"CANNOT DEPLOY: '{sku_name}'{specs_str} exists in {display} "
                    f"but is RESTRICTED — {reason_str}. "
                    f"This means the SKU is physically present in the region but "
                    f"Microsoft has blocked it for your subscription. "
                    f"You can request access via an Azure support ticket, or "
                    f"use one of the alternative SKUs listed."
                ),
                alternatives=alternatives,
                alternatives_detail=self._alt_engine.get_alternative_details(alternatives),
            )

        # SKU is available but has zone-level restrictions
        if sku_info.is_zone_limited_in_region(region):
            zone_reasons = [
                r.reason_code
                for r in sku_info.restrictions
                if r.type == "Zone"
                and region.lower() in [v.lower() for v in r.values]
            ]
            zones = sorted(sku_info.zones) if sku_info.zones else []
            zone_list = ", ".join(zones) if zones else "unknown"
            zone_str = ", ".join(zone_reasons) if zone_reasons else "limited zones"

            alternatives = self._alt_engine.find_alternatives(
                sku_name=sku_name,
                region=region,
                vcpu=sku_info.vcpus or machine.vcpu,
                memory_gb=sku_info.memory_gb or machine.memory_gb,
                vm_family=machine.vm_family or sku_info.family,
                disk_requirements=disk_reqs,
            )
            return AnalysisResult(
                **base,
                status=SkuStatus.RISK,
                reason=(
                    f"CAN DEPLOY WITH CAVEATS: '{sku_name}'{specs_str} is available "
                    f"in {display} but NOT in all availability zones "
                    f"(available zones: {zone_list}; restriction: {zone_str}). "
                    f"This is fine for single-zone deployments but could be a "
                    f"problem if you need zone-redundant high availability. "
                    f"Found in {total_regions} region(s) total."
                ),
                alternatives=alternatives,
                alternatives_detail=self._alt_engine.get_alternative_details(alternatives),
            )

        # SKU is available and unrestricted — the good case
        zones = sorted(sku_info.zones) if sku_info.zones else []
        zone_str = ", ".join(zones) if zones else "none listed"
        return AnalysisResult(
            **base,
            status=SkuStatus.OK,
            reason=(
                f"READY TO DEPLOY: '{sku_name}'{specs_str} is confirmed available "
                f"in {display} with no restrictions. "
                f"Availability zones: {zone_str}. "
                f"Found in {total_regions} Azure region(s) total. "
                f"No action needed."
            ),
            alternatives=[],
        )
