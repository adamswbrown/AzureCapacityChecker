"""Capacity validation orchestrator.

Takes Method 1 (catalog check) results, extracts unique SKU+region pairs that
need live capacity testing, deduplicates them, calls the ARM deployment
validator, and enriches the results.

Only SKUs that passed the catalog check (OK or RISK) are validated — there is
no point testing SKUs that are BLOCKED or UNKNOWN in the catalog.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from azure_client.deployment_validator import DeploymentValidator, ValidationResult
from engine.alternatives import AlternativeEngine, DiskRequirements
from models.disk import Disk
from models.result import AnalysisResult, SkuStatus

logger = logging.getLogger(__name__)


class CapacityValidator:
    """Enriches Method 1 results with Method 2 ARM deployment validation."""

    def __init__(
        self,
        deployment_validator: DeploymentValidator,
        alternative_engine: Optional[AlternativeEngine] = None,
        disk_map: Optional[dict[str, list[Disk]]] = None,
    ) -> None:
        self._validator = deployment_validator
        self._alt_engine = alternative_engine
        self._disk_map = disk_map or {}

    def validate_results(
        self,
        results: list[AnalysisResult],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        alt_progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[AnalysisResult]:
        """Run live capacity validation and enrich the analysis results.

        Performs two passes:
        1. Validate primary requested SKUs (OK/RISK from catalog check)
        2. Validate alternative SKUs for any machines that became BLOCKED

        Args:
            results: Method 1 analysis results.
            progress_callback: Called with (done, total) for primary SKU validation.
            alt_progress_callback: Called with (done, total) for alternative validation.

        Returns:
            The same results list with capacity fields populated.
        """
        # --- Pass 1: Validate primary SKUs ---
        pairs_to_validate: set[tuple[str, str]] = set()
        for r in results:
            if r.status in (SkuStatus.OK, SkuStatus.RISK) and r.requested_sku and r.region:
                pairs_to_validate.add((r.requested_sku, r.region))

        if not pairs_to_validate:
            logger.info("No SKU+region pairs to validate (all BLOCKED/UNKNOWN)")
            return results

        logger.info(
            "Pass 1: Validating live capacity for %d unique SKU+region pairs",
            len(pairs_to_validate),
        )

        validation_map = self._validator.validate_skus(
            list(pairs_to_validate),
            progress_callback=progress_callback,
        )

        # Enrich each result from pass 1
        for r in results:
            key = (r.requested_sku, r.region)
            vr = validation_map.get(key)

            if vr is None:
                continue

            if not vr.validated:
                r.capacity_verified = False
                r.capacity_error_code = vr.error_code
                r.capacity_error_message = vr.error_message
                continue

            if vr.capacity_available:
                r.capacity_verified = True
                r.reason = r.reason.replace(
                    "READY TO DEPLOY:", "CAPACITY VERIFIED:"
                ).replace(
                    "CAN DEPLOY WITH CAVEATS:", "CAPACITY VERIFIED (with zone caveats):"
                )
            else:
                r.capacity_verified = False
                r.capacity_error_code = vr.error_code
                r.capacity_error_message = vr.error_message
                self._apply_capacity_failure(r, vr)

        # --- Pass 2: Validate alternative SKUs ---
        self._validate_alternatives(results, validation_map, alt_progress_callback)

        return results

    def _validate_alternatives(
        self,
        results: list[AnalysisResult],
        existing_map: dict[tuple[str, str], ValidationResult],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Validate alternative SKUs for blocked results, enrich alternatives_detail."""
        # Collect unique alt SKU+region pairs that need checking
        alt_pairs: set[tuple[str, str]] = set()
        for r in results:
            if not r.alternatives_detail or not r.region:
                continue
            for alt in r.alternatives_detail:
                pair = (alt["name"], r.region)
                if pair not in existing_map:
                    alt_pairs.add(pair)

        if not alt_pairs:
            # Still mark alternatives from results already in existing_map
            self._enrich_alternatives_from_map(results, existing_map)
            return

        logger.info(
            "Pass 2: Validating %d unique alternative SKU+region pairs",
            len(alt_pairs),
        )

        alt_validation_map = self._validator.validate_skus(
            list(alt_pairs),
            progress_callback=progress_callback,
        )

        # Merge maps
        combined_map = {**existing_map, **alt_validation_map}
        self._enrich_alternatives_from_map(results, combined_map)

    def _enrich_alternatives_from_map(
        self,
        results: list[AnalysisResult],
        validation_map: dict[tuple[str, str], ValidationResult],
    ) -> None:
        """Add capacity status to each alternative and sort verified-first."""
        for r in results:
            if not r.alternatives_detail or not r.region:
                continue
            for alt in r.alternatives_detail:
                vr = validation_map.get((alt["name"], r.region))
                if vr is None:
                    alt["capacity"] = "Not Checked"
                elif vr.capacity_available:
                    alt["capacity"] = "Verified"
                else:
                    alt["capacity"] = "Failed"

            # Sort: Verified first, then Not Checked, then Failed
            order = {"Verified": 0, "Not Checked": 1, "Failed": 2}
            r.alternatives_detail.sort(key=lambda a: order.get(a.get("capacity", "Not Checked"), 1))
            # Re-order the names list to match
            r.alternatives = [a["name"] for a in r.alternatives_detail]

    def _apply_capacity_failure(
        self, result: AnalysisResult, vr: ValidationResult
    ) -> None:
        """Update an AnalysisResult when capacity validation fails.

        Also finds alternative SKUs if an AlternativeEngine is available.
        """
        display = result.display_region
        sku = result.requested_sku

        if vr.error_code in ("SkuNotAvailable", "ZonalAllocationFailed",
                             "AllocationFailed", "OverconstrainedZonalAllocationRequest"):
            result.status = SkuStatus.BLOCKED
            result.reason = (
                f"CAPACITY EXHAUSTED: '{sku}' passed the catalog check (the SKU "
                f"is listed and unrestricted in {display}) but ARM deployment "
                f"validation reports NO PHYSICAL CAPACITY available right now. "
                f"The hardware for this VM size is currently saturated in this "
                f"region. Error: {vr.error_code}. "
                f"Options: try a different region, try a different VM size from "
                f"the same family, or use one of the alternative SKUs listed."
            )
            self._find_alternatives_for_result(result)
        elif vr.error_code == "QuotaExceeded":
            result.status = SkuStatus.BLOCKED
            result.reason = (
                f"QUOTA EXCEEDED: '{sku}' passed the catalog check but your "
                f"subscription's vCPU quota for this VM family in {display} is "
                f"exhausted. You need to request a quota increase via the Azure "
                f"portal (Quotas > Request increase) or use a different VM size."
            )
            self._find_alternatives_for_result(result)
        else:
            # Other ARM validation error — flag as risk, not blocked
            # (could be template-related, not capacity-related)
            if result.status == SkuStatus.OK:
                result.status = SkuStatus.RISK
            result.reason += (
                f" [Live capacity check returned: {vr.error_code} — "
                f"{vr.error_message}]"
            )

    def _find_alternatives_for_result(self, result: AnalysisResult) -> None:
        """Find alternative SKUs for a capacity-blocked result."""
        if self._alt_engine is None:
            return

        machine_disks = self._disk_map.get(result.machine_name, [])
        disk_reqs = DiskRequirements.from_disks(machine_disks) if machine_disks else None

        alternatives = self._alt_engine.find_alternatives(
            sku_name=result.requested_sku,
            region=result.region,
            vcpu=result.vcpu,
            memory_gb=result.memory_gb,
            vm_family=result.vm_family,
            disk_requirements=disk_reqs,
        )
        result.alternatives = alternatives
        result.alternatives_detail = self._alt_engine.get_alternative_details(alternatives)
