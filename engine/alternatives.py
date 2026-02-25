"""Alternative SKU recommendation engine.

When a requested SKU is blocked, restricted, or unavailable, this engine
suggests the best replacement SKUs using a weighted scoring system.

Modeled after the Dr Migrate rightsizer logic:
    1. Same VM family (e.g. D→D, or D→Ds variant)
    2. Same vCPU count — exact match strongly preferred
    3. Memory must not regress (>= original)
    4. Prefer newer generations (v5 > v4 > v3) — always upgrade, not just match
    5. Constrained vCPU variants are valid alternatives (e.g. Standard_D4-2s_v5)
    6. Size tier / cost proximity

Hard constraints (disqualify candidates):
    - Must support at least as many data disks as the machine needs
    - Must support Premium storage if the machine has Premium disks
    - Memory must not drop below what the machine needs
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.config import MAX_ALTERNATIVES
from azure_client.sku_service import SkuInfo, SkuService

logger = logging.getLogger(__name__)

# Weights for the scoring algorithm
WEIGHT_FAMILY: float = 35.0
WEIGHT_VCPU: float = 25.0
WEIGHT_MEMORY: float = 15.0
WEIGHT_GENERATION: float = 15.0
WEIGHT_SIZE_TIER: float = 5.0
WEIGHT_CONSTRAINED: float = 5.0  # Bonus for constrained vCPU variants

# Regex to extract generation number from SKU name (e.g. v5 from Standard_D4s_v5)
_GENERATION_RE = re.compile(r"_v(\d+)$", re.IGNORECASE)

# Regex to extract the numeric size from SKU name (e.g. 4 from Standard_D4s_v5)
_SIZE_RE = re.compile(r"[A-Za-z]+_[A-Za-z]+?(\d+)", re.IGNORECASE)

# Regex to extract the family prefix from SKU name (e.g. D from Standard_D4s_v5)
_FAMILY_PREFIX_RE = re.compile(
    r"Standard_([A-Za-z]+?)(\d+)", re.IGNORECASE
)

# Regex to detect constrained vCPU SKUs (e.g. Standard_D4-2s_v5)
_CONSTRAINED_RE = re.compile(r"Standard_\w+\d+-(\d+)\w*_v\d+", re.IGNORECASE)

# Decompose SKU into base family (removing core number, Standard_, version)
# Mirrors the rightsizer's sku_family extraction logic
_SKU_FAMILY_RE = re.compile(
    r"Standard_([A-Za-z]+)\d+(?:-\d+)?([A-Za-z]*)(?:_v\d+)?$", re.IGNORECASE
)

# Related family groups — families that can serve as alternatives for each other
# Based on Dr Migrate Intel SKU upgrade rules (D→Das, E→Eas, etc.)
_FAMILY_UPGRADE_GROUPS: dict[str, list[str]] = {
    "d": ["d", "ds", "das", "das", "dads", "dlds"],
    "e": ["e", "es", "eas", "eads", "ebs"],
    "f": ["f", "fs", "fas", "fads"],
    "b": ["b", "bs", "bls"],
    "l": ["l", "ls", "las"],
    "m": ["m", "ms", "mds"],
    "n": ["n", "ns", "nas", "nads"],
}


def _extract_generation(sku_name: str) -> Optional[int]:
    """Extract the generation number from a SKU name."""
    match = _GENERATION_RE.search(sku_name)
    if match:
        return int(match.group(1))
    return None


def _extract_size_number(sku_name: str) -> Optional[int]:
    """Extract the numeric size portion of the SKU name."""
    match = _SIZE_RE.search(sku_name)
    if match:
        return int(match.group(1))
    return None


def _extract_family_prefix(sku_name: str) -> Optional[str]:
    """Extract the family letter prefix (e.g. 'd', 'e', 'b', 'ds')."""
    match = _FAMILY_PREFIX_RE.search(sku_name)
    if match:
        return match.group(1).lower()
    return None


def _extract_sku_family(sku_name: str) -> Optional[str]:
    """Extract the full SKU family string (e.g. 'das' from Standard_D4as_v5).

    Mirrors the rightsizer's approach: remove Standard_, core number, and version
    to get the family identifier for upgrade-path matching.
    """
    match = _SKU_FAMILY_RE.search(sku_name)
    if match:
        return (match.group(1) + match.group(2)).lower()
    return None


def _is_constrained_sku(sku_name: str) -> bool:
    """Check if this is a constrained vCPU SKU (e.g. Standard_D4-2s_v5)."""
    return bool(_CONSTRAINED_RE.match(sku_name))


def _families_are_related(family_a: str, family_b: str) -> bool:
    """Check if two SKU families are in the same upgrade group."""
    base_a = family_a[0] if family_a else ""
    base_b = family_b[0] if family_b else ""
    if base_a != base_b:
        return False
    group = _FAMILY_UPGRADE_GROUPS.get(base_a, [])
    return family_a in group and family_b in group


@dataclass
class DiskRequirements:
    """Disk constraints for alternative SKU selection.

    Built from a machine's associated disks to ensure alternatives
    can support the same disk configuration.
    """

    disk_count: int = 0
    requires_premium_storage: bool = False
    total_disk_size_gb: float = 0.0

    @classmethod
    def from_disks(cls, disks: list) -> DiskRequirements:
        """Build requirements from a list of Disk objects."""
        if not disks:
            return cls()
        premium = any(
            d.sku_tier in ("Premium SSD", "Premium SSD v2", "Ultra")
            for d in disks
        )
        total_gb = sum(d.disk_size_gb or 0 for d in disks)
        return cls(
            disk_count=len(disks),
            requires_premium_storage=premium,
            total_disk_size_gb=total_gb,
        )


class AlternativeEngine:
    """Recommends alternative VM SKUs when the requested one is unavailable."""

    def __init__(
        self,
        sku_service: SkuService,
        max_alternatives: int = MAX_ALTERNATIVES,
    ) -> None:
        self._sku_service = sku_service
        self._max_alternatives = max_alternatives

    def find_alternatives(
        self,
        sku_name: str,
        region: str,
        vcpu: Optional[int] = None,
        memory_gb: Optional[float] = None,
        vm_family: Optional[str] = None,
        disk_requirements: Optional[DiskRequirements] = None,
    ) -> list[str]:
        """Find the best alternative SKUs for a given configuration.

        Uses the same logic as the Dr Migrate rightsizer:
        - Same family first, then related families (D→Ds→Das)
        - Same vCPU count strongly preferred
        - Memory must not regress
        - Newer generations preferred (v5 > v4)
        - Constrained vCPU variants included as options

        Args:
            sku_name: The original requested SKU name.
            region: The target region.
            vcpu: Desired vCPU count.
            memory_gb: Desired memory in GB.
            vm_family: Desired VM family.
            disk_requirements: Disk constraints from the machine's associated disks.

        Returns:
            List of alternative SKU names, ordered by relevance (best first).
        """
        candidates = self._sku_service.get_skus_in_region(region)
        if not candidates:
            logger.warning("No SKUs found in region '%s' for alternatives", region)
            return []

        # Filter out the original SKU itself, and SKUs that are restricted
        candidates = [
            c for c in candidates
            if c.name.lower() != sku_name.lower()
            and not c.is_restricted_in_region(region)
        ]

        if not candidates:
            return []

        # Hard filter: must support enough data disks
        if disk_requirements and disk_requirements.disk_count > 0:
            filtered = []
            for c in candidates:
                max_disks_str = c.capabilities.get("MaxDataDiskCount", "")
                if max_disks_str:
                    try:
                        max_disks = int(max_disks_str)
                        if max_disks < disk_requirements.disk_count:
                            continue
                    except ValueError:
                        pass
                # Hard filter: must support Premium storage if required
                if disk_requirements.requires_premium_storage:
                    premium_cap = c.capabilities.get("PremiumIO", "")
                    if premium_cap.lower() == "false":
                        continue
                filtered.append(c)
            candidates = filtered

        if not candidates:
            return []

        # Hard filter: memory must not regress (like the rightsizer checks)
        if memory_gb:
            candidates = [
                c for c in candidates
                if c.memory_gb is None or c.memory_gb >= memory_gb
            ]

        if not candidates:
            return []

        # Extract reference values from the original SKU name
        ref_generation = _extract_generation(sku_name)
        ref_family_prefix = _extract_family_prefix(sku_name)
        ref_sku_family = _extract_sku_family(sku_name)
        ref_size_number = _extract_size_number(sku_name)

        # Use the SKU catalog info for vCPU/memory if we have it
        original_info = self._sku_service.get_sku_info(sku_name)
        if original_info:
            if vcpu is None:
                vcpu = original_info.vcpus
            if memory_gb is None:
                memory_gb = original_info.memory_gb

        scored: list[tuple[float, SkuInfo]] = []
        for candidate in candidates:
            score = self._score(
                candidate=candidate,
                ref_vcpu=vcpu,
                ref_memory_gb=memory_gb,
                ref_family=vm_family,
                ref_family_prefix=ref_family_prefix,
                ref_sku_family=ref_sku_family,
                ref_generation=ref_generation,
                ref_size_number=ref_size_number,
            )
            scored.append((score, candidate))

        # Sort by score descending (higher = better match)
        scored.sort(key=lambda x: x[0], reverse=True)

        result = [s[1].name for s in scored[: self._max_alternatives]]
        logger.debug(
            "Alternatives for %s in %s: %s",
            sku_name,
            region,
            result,
        )
        return result

    def get_alternative_details(
        self, alternatives: list[str]
    ) -> list[dict[str, str]]:
        """Return spec details for a list of alternative SKU names.

        Used by the UI to display alternatives with their specs.
        """
        details = []
        for name in alternatives:
            info = self._sku_service.get_sku_info(name)
            if info:
                vcpu = info.vcpus or "?"
                mem = f"{info.memory_gb:g}" if info.memory_gb else "?"
                max_disks = info.capabilities.get("MaxDataDiskCount", "?")
                details.append({
                    "name": name,
                    "vcpu": str(vcpu),
                    "memory_gb": str(mem),
                    "max_disks": str(max_disks),
                })
            else:
                details.append({"name": name, "vcpu": "?", "memory_gb": "?", "max_disks": "?"})
        return details

    def _score(
        self,
        candidate: SkuInfo,
        ref_vcpu: Optional[int],
        ref_memory_gb: Optional[float],
        ref_family: Optional[str],
        ref_family_prefix: Optional[str],
        ref_sku_family: Optional[str],
        ref_generation: Optional[int],
        ref_size_number: Optional[int],
    ) -> float:
        """Score a candidate SKU against reference criteria.

        Scoring mirrors the Dr Migrate rightsizer logic:
        - Same family exact match is best, related families get partial credit
        - vCPU must match (exact strongly preferred, no downsizing)
        - Memory must not regress (already hard-filtered, but exact match scores higher)
        - Newer generations always preferred over older ones
        - Constrained vCPU variants get a bonus (cost-effective)

        Higher score means better match. Maximum possible score is 100.
        """
        score: float = 0.0

        # 1. Family match (35 points)
        # Uses the rightsizer's decomposition: extract full sku_family (e.g. "das")
        cand_sku_family = _extract_sku_family(candidate.name)
        cand_family_prefix = _extract_family_prefix(candidate.name)

        if ref_sku_family and cand_sku_family:
            if cand_sku_family == ref_sku_family:
                # Exact family match (e.g. das == das)
                score += WEIGHT_FAMILY
            elif _families_are_related(ref_sku_family, cand_sku_family):
                # Related family in the same upgrade group (e.g. d → das)
                score += WEIGHT_FAMILY * 0.75
            elif ref_family_prefix and cand_family_prefix:
                if ref_family_prefix[0] == cand_family_prefix[0]:
                    # Same base letter but different variant (e.g. D vs E is 0, D vs Dlds is partial)
                    score += WEIGHT_FAMILY * 0.4
        elif ref_family and candidate.family:
            if ref_family.lower() in candidate.family.lower():
                score += WEIGHT_FAMILY
            elif candidate.family and ref_family and candidate.family[0].lower() == ref_family[0].lower():
                score += WEIGHT_FAMILY * 0.4

        # 2. vCPU match (25 points)
        # Like the rightsizer: exact match is critical, downsizing penalized heavily
        cand_vcpu = candidate.vcpus
        if ref_vcpu and cand_vcpu:
            if cand_vcpu == ref_vcpu:
                score += WEIGHT_VCPU
            elif cand_vcpu > ref_vcpu:
                # Upsizing is acceptable but not ideal — penalize proportionally
                ratio = ref_vcpu / cand_vcpu
                score += WEIGHT_VCPU * ratio * 0.8
            else:
                # Downsizing — heavy penalty (the rightsizer skips these entirely
                # via `target_cores - Cores > 0` check, but we allow it with low score)
                ratio = cand_vcpu / ref_vcpu
                score += WEIGHT_VCPU * ratio * 0.3

        # 3. Memory match (15 points)
        # Memory already hard-filtered to >= ref, so score closeness
        cand_mem = candidate.memory_gb
        if ref_memory_gb and cand_mem:
            if cand_mem == ref_memory_gb:
                score += WEIGHT_MEMORY
            elif cand_mem > ref_memory_gb:
                # More memory is fine but prefer closest match
                ratio = ref_memory_gb / cand_mem
                score += WEIGHT_MEMORY * ratio
            else:
                # Less memory (shouldn't happen after hard filter, but safety)
                ratio = cand_mem / ref_memory_gb
                score += WEIGHT_MEMORY * ratio * 0.3

        # 4. Generation preference (15 points)
        # Like the rightsizer's upgrade_sku_version: always prefer newest version
        cand_gen = _extract_generation(candidate.name)
        if ref_generation and cand_gen:
            if cand_gen > ref_generation:
                # Newer generation — bonus! The rightsizer always upgrades.
                score += WEIGHT_GENERATION
            elif cand_gen == ref_generation:
                # Same generation — good
                score += WEIGHT_GENERATION * 0.8
            elif cand_gen == ref_generation - 1:
                # One generation older — acceptable
                score += WEIGHT_GENERATION * 0.3
            # Older than that — 0 points
        elif cand_gen and not ref_generation:
            # Original has no version (very old), any versioned SKU is an upgrade
            score += WEIGHT_GENERATION * 0.5

        # 5. Size tier / cost proximity (5 points)
        cand_size = _extract_size_number(candidate.name)
        if ref_size_number and cand_size:
            if cand_size == ref_size_number:
                score += WEIGHT_SIZE_TIER
            else:
                ratio = min(cand_size, ref_size_number) / max(cand_size, ref_size_number)
                score += WEIGHT_SIZE_TIER * ratio

        # 6. Constrained vCPU bonus (5 points)
        # The rightsizer's step 2 finds constrained variants as cost-saving options.
        # These are valid alternatives that keep the same memory/family but fewer active cores.
        if _is_constrained_sku(candidate.name):
            score += WEIGHT_CONSTRAINED

        return score
