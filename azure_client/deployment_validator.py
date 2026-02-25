"""ARM Deployment Validator — tests real-time VM capacity via deployment validation.

Uses the Azure Resource Manager deployment validate endpoint to check whether
a VM SKU can actually be provisioned right now, catching physical capacity
exhaustion and quota limits that the SKU catalog check cannot detect.

This is the "Method 2" check (equivalent to New-AzVM -WhatIf).
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Optional

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
    ARM_DEPLOYMENT_API_VERSION,
    ARM_VALIDATION_MAX_WORKERS,
    ARM_VALIDATION_RATE_LIMIT_PER_MINUTE,
    ARM_VALIDATION_POLL_INTERVAL_SECONDS,
    ARM_VALIDATION_POLL_TIMEOUT_SECONDS,
    ARM_VALIDATION_REQUEST_TIMEOUT_SECONDS,
    AZURE_MAX_RETRIES,
    AZURE_RETRY_WAIT_SECONDS,
    AZURE_RETRY_MAX_WAIT_SECONDS,
)

logger = logging.getLogger(__name__)


class DeploymentValidationError(Exception):
    """Raised when the deployment validation service encounters a fatal error."""


@dataclass
class ValidationResult:
    """Result of an ARM deployment validation for a single SKU+region."""

    sku_name: str
    region: str
    capacity_available: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    validated: bool = True  # False if the API call failed due to transient error


class _RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._timestamps: list[float] = []
        self._lock = Lock()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        while True:
            with self._lock:
                now = time.time()
                # Remove timestamps older than 60 seconds
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
            # Wait before retrying
            time.sleep(0.5)


class DeploymentValidator:
    """Validates VM SKU capacity via ARM deployment validation."""

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        auth_method: AuthMethod = AuthMethod.DEFAULT,
        tenant_id: str = "",
        client_id: str = "",
        client_secret: str = "",
        max_workers: int = ARM_VALIDATION_MAX_WORKERS,
    ) -> None:
        if not subscription_id:
            raise DeploymentValidationError("Subscription ID is required.")
        if not resource_group:
            raise DeploymentValidationError("Resource group name is required.")

        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._auth_method = auth_method
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._max_workers = max_workers
        self._rate_limiter = _RateLimiter(ARM_VALIDATION_RATE_LIMIT_PER_MINUTE)
        self._rg_url = (
            f"https://management.azure.com/subscriptions/{subscription_id}"
            f"/resourcegroups/{resource_group}"
        )
        self._base_url = (
            f"{self._rg_url}/providers/Microsoft.Resources/deployments"
        )

    def ensure_resource_group(self, location: str = "uksouth") -> None:
        """Create the resource group if it does not already exist.

        If the RG already exists (possibly in a different location), it is
        left as-is — the RG location does not affect which regions we can
        validate capacity in.

        Args:
            location: Azure region to create the RG in (only used when the
                      RG does not yet exist).

        Raises:
            DeploymentValidationError: If creation fails.
        """
        token = get_access_token(
            method=self._auth_method,
            tenant_id=self._tenant_id,
            client_id=self._client_id,
            client_secret=self._client_secret,
        )
        url = f"{self._rg_url}?api-version=2024-03-01"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Check if it already exists
        try:
            check = requests.get(url, headers=headers, timeout=15)
        except requests.RequestException as exc:
            raise DeploymentValidationError(
                f"Network error checking resource group: {exc}"
            ) from exc

        if check.status_code == 200:
            existing_location = check.json().get("location", "")
            logger.info(
                "Resource group '%s' already exists in '%s' — reusing",
                self._resource_group,
                existing_location,
            )
            return

        # Does not exist — create it
        try:
            resp = requests.put(
                url, json={"location": location}, headers=headers, timeout=30
            )
        except requests.RequestException as exc:
            raise DeploymentValidationError(
                f"Network error creating resource group: {exc}"
            ) from exc

        if resp.status_code in (200, 201):
            logger.info(
                "Resource group '%s' created in '%s' (HTTP %d)",
                self._resource_group,
                location,
                resp.status_code,
            )
            return

        # Parse error
        try:
            error = resp.json().get("error", {})
            code = error.get("code", "")
            message = error.get("message", resp.text[:300])
        except ValueError:
            code = f"HTTP_{resp.status_code}"
            message = resp.text[:300]

        raise DeploymentValidationError(
            f"Failed to create resource group '{self._resource_group}': "
            f"{code} — {message}"
        )

    def validate_skus(
        self,
        sku_region_pairs: list[tuple[str, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[tuple[str, str], ValidationResult]:
        """Validate multiple SKU+region pairs concurrently.

        Args:
            sku_region_pairs: List of (sku_name, region) tuples to validate.
            progress_callback: Called with (completed_count, total_count) after each.

        Returns:
            Dict mapping (sku_name, region) to ValidationResult.

        Raises:
            DeploymentValidationError: On fatal errors (bad RG, missing permissions).
        """
        if not sku_region_pairs:
            return {}

        token = get_access_token(
            method=self._auth_method,
            tenant_id=self._tenant_id,
            client_id=self._client_id,
            client_secret=self._client_secret,
        )

        total = len(sku_region_pairs)
        results: dict[tuple[str, str], ValidationResult] = {}
        completed = 0
        results_lock = Lock()

        logger.info("Starting ARM deployment validation for %d SKU+region pairs", total)

        def _validate_one(pair: tuple[str, str]) -> tuple[tuple[str, str], ValidationResult]:
            sku_name, region = pair
            self._rate_limiter.acquire()
            result = self._validate_single(sku_name, region, token)
            return pair, result

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(_validate_one, pair): pair
                for pair in sku_region_pairs
            }

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    key, result = future.result()

                    # Check for fatal errors that should abort everything
                    if result.error_code in (
                        "ResourceGroupNotFound",
                        "AuthorizationFailed",
                    ):
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        raise DeploymentValidationError(
                            f"Fatal error during capacity validation: "
                            f"{result.error_code} — {result.error_message}"
                        )

                    with results_lock:
                        results[key] = result
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)

                except DeploymentValidationError:
                    raise
                except Exception as exc:
                    sku_name, region = pair
                    logger.warning(
                        "Validation failed for %s in %s: %s", sku_name, region, exc
                    )
                    with results_lock:
                        results[pair] = ValidationResult(
                            sku_name=sku_name,
                            region=region,
                            capacity_available=False,
                            error_code="ValidationError",
                            error_message=str(exc),
                            validated=False,
                        )
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)

        logger.info(
            "Completed ARM deployment validation: %d/%d pairs validated", completed, total
        )
        return results

    @retry(
        stop=stop_after_attempt(AZURE_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=AZURE_RETRY_WAIT_SECONDS,
            max=AZURE_RETRY_MAX_WAIT_SECONDS,
        ),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _validate_single(
        self, sku_name: str, region: str, token: str
    ) -> ValidationResult:
        """Submit a deployment validation request for a single SKU+region.

        Args:
            sku_name: The VM SKU to validate (e.g. Standard_D2as_v5).
            region: The target Azure region (e.g. uksouth).
            token: Bearer token for authentication.

        Returns:
            A ValidationResult with the capacity check outcome.
        """
        deployment_name = f"cap-check-{uuid.uuid4().hex[:12]}"
        url = (
            f"{self._base_url}/{deployment_name}"
            f"/validate?api-version={ARM_DEPLOYMENT_API_VERSION}"
        )

        payload = self._build_template(sku_name, region)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=ARM_VALIDATION_REQUEST_TIMEOUT_SECONDS,
        )

        # Handle 429 rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "10"))
            logger.warning("Rate limited (429), waiting %d seconds", retry_after)
            time.sleep(retry_after)
            # Retry via tenacity
            raise requests.ConnectionError(f"Rate limited, retry after {retry_after}s")

        # Handle 202 Accepted (async validation)
        if response.status_code == 202:
            location = response.headers.get("Location")
            if location:
                return self._poll_async_validation(location, token, sku_name, region)
            # No location header — treat as unknown
            return ValidationResult(
                sku_name=sku_name,
                region=region,
                capacity_available=False,
                error_code="AsyncNoLocation",
                error_message="202 Accepted but no Location header for polling",
                validated=False,
            )

        return self._parse_response(response, sku_name, region)

    def _poll_async_validation(
        self, location_url: str, token: str, sku_name: str, region: str
    ) -> ValidationResult:
        """Poll an async deployment validation until completion."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        deadline = time.time() + ARM_VALIDATION_POLL_TIMEOUT_SECONDS

        while time.time() < deadline:
            time.sleep(ARM_VALIDATION_POLL_INTERVAL_SECONDS)
            resp = requests.get(
                location_url,
                headers=headers,
                timeout=ARM_VALIDATION_REQUEST_TIMEOUT_SECONDS,
            )
            if resp.status_code == 202:
                continue  # Still processing
            return self._parse_response(resp, sku_name, region)

        return ValidationResult(
            sku_name=sku_name,
            region=region,
            capacity_available=False,
            error_code="PollTimeout",
            error_message=(
                f"Async validation did not complete within "
                f"{ARM_VALIDATION_POLL_TIMEOUT_SECONDS}s"
            ),
            validated=False,
        )

    def _parse_response(
        self, response: requests.Response, sku_name: str, region: str
    ) -> ValidationResult:
        """Parse the ARM validation response into a ValidationResult."""
        # 200 OK — validation succeeded, capacity is available
        if response.status_code == 200:
            return ValidationResult(
                sku_name=sku_name,
                region=region,
                capacity_available=True,
            )

        # 400 Bad Request — check for specific capacity/quota error codes
        if response.status_code == 400:
            try:
                body = response.json()
            except ValueError:
                return ValidationResult(
                    sku_name=sku_name,
                    region=region,
                    capacity_available=False,
                    error_code=f"HTTP_400",
                    error_message=response.text[:500],
                )

            error = body.get("error", {})
            code = error.get("code", "")
            message = error.get("message", "")

            # ARM often nests errors in details
            for detail in error.get("details", []):
                detail_code = detail.get("code", "")
                if detail_code in (
                    "SkuNotAvailable",
                    "QuotaExceeded",
                    "ZonalAllocationFailed",
                    "AllocationFailed",
                    "OverconstrainedZonalAllocationRequest",
                ):
                    code = detail_code
                    message = detail.get("message", message)
                    break

            # Some error codes indicate "not deployable" but are not capacity issues
            # (e.g. InvalidParameter) — still mark capacity_available=False
            return ValidationResult(
                sku_name=sku_name,
                region=region,
                capacity_available=False,
                error_code=code,
                error_message=message[:500],
            )

        # 403 Forbidden — permissions issue
        if response.status_code == 403:
            try:
                body = response.json()
                error = body.get("error", {})
                code = error.get("code", "AuthorizationFailed")
                message = error.get("message", "")
            except ValueError:
                code = "AuthorizationFailed"
                message = response.text[:500]

            return ValidationResult(
                sku_name=sku_name,
                region=region,
                capacity_available=False,
                error_code=code,
                error_message=(
                    f"{message} — Your account needs the "
                    f"Microsoft.Resources/deployments/validate/action permission. "
                    f"The 'Contributor' role includes this."
                ),
            )

        # 404 Not Found — likely bad resource group
        if response.status_code == 404:
            try:
                body = response.json()
                error = body.get("error", {})
                code = error.get("code", "ResourceGroupNotFound")
                message = error.get("message", "")
            except ValueError:
                code = "ResourceGroupNotFound"
                message = response.text[:500]

            return ValidationResult(
                sku_name=sku_name,
                region=region,
                capacity_available=False,
                error_code=code,
                error_message=(
                    f"Resource group '{self._resource_group}' was not found. "
                    f"Please check the name and ensure it exists in your subscription."
                ),
            )

        # Other HTTP errors
        return ValidationResult(
            sku_name=sku_name,
            region=region,
            capacity_available=False,
            error_code=f"HTTP_{response.status_code}",
            error_message=response.text[:500],
            validated=False,
        )

    @staticmethod
    def _build_template(sku_name: str, region: str) -> dict:
        """Build a minimal ARM deployment template for capacity validation.

        The template defines a single VM resource with the target SKU and region.
        The validate endpoint checks feasibility without creating any resources.
        """
        return {
            "properties": {
                "mode": "Incremental",
                "template": {
                    "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
                    "contentVersion": "1.0.0.0",
                    "resources": [
                        {
                            "type": "Microsoft.Compute/virtualMachines",
                            "apiVersion": "2024-07-01",
                            "name": "capacity-check-dummy",
                            "location": region,
                            "properties": {
                                "hardwareProfile": {
                                    "vmSize": sku_name,
                                },
                                "storageProfile": {
                                    "imageReference": {
                                        "publisher": "Canonical",
                                        "offer": "0001-com-ubuntu-server-jammy",
                                        "sku": "22_04-lts-gen2",
                                        "version": "latest",
                                    },
                                },
                                "osProfile": {
                                    "computerName": "capdummy",
                                    "adminUsername": "azureuser",
                                    "adminPassword": "CapCheck-V@lid8!",
                                },
                                "networkProfile": {
                                    "networkInterfaces": [],
                                },
                            },
                        }
                    ],
                },
            },
        }
