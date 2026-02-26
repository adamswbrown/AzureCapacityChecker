"""Azure authentication with multiple credential methods.

Supports:
- Azure CLI login (az login)
- Managed identity
- Service principal (client ID + tenant ID + client secret)
- Device code flow (browser-based interactive login)
- DefaultAzureCredential (automatic chain)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Optional, Union

from azure.identity import (
    ClientSecretCredential,
    DefaultAzureCredential,
    DeviceCodeCredential,
    InteractiveBrowserCredential,
)

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Supported Azure authentication methods."""

    DEFAULT = "Default (CLI / Managed Identity / Env Vars)"
    SERVICE_PRINCIPAL = "Service Principal"
    DEVICE_CODE = "Device Code (Browser Login)"
    INTERACTIVE_BROWSER = "Interactive Browser"

    def __str__(self) -> str:
        return self.value


class AzureAuthError(Exception):
    """Raised when Azure authentication fails."""


CredentialType = Union[
    DefaultAzureCredential,
    ClientSecretCredential,
    DeviceCodeCredential,
    InteractiveBrowserCredential,
]
DeviceCodePromptCallback = Callable[[str, str, int], None]

_credential: Optional[CredentialType] = None
_current_method: Optional[AuthMethod] = None


def reset_credential() -> None:
    """Clear the cached credential so a new one can be created."""
    global _credential, _current_method
    _credential = None
    _current_method = None
    logger.info("Azure credential cache cleared")


def get_credential(
    method: AuthMethod = AuthMethod.DEFAULT,
    tenant_id: str = "",
    client_id: str = "",
    client_secret: str = "",
    device_code_callback: Optional[DeviceCodePromptCallback] = None,
) -> CredentialType:
    """Create or return a cached credential for the given auth method.

    Args:
        method: The authentication method to use.
        tenant_id: Required for Service Principal and Device Code methods.
        client_id: Required for Service Principal method.
        client_secret: Required for Service Principal method.
        device_code_callback: Optional callback for device code flow prompts.

    Returns:
        An authenticated credential object.

    Raises:
        AzureAuthError: If the credential cannot be created.
    """
    global _credential, _current_method

    # Return cached credential if same method
    if _credential is not None and _current_method == method:
        return _credential

    try:
        if method == AuthMethod.SERVICE_PRINCIPAL:
            if not all([tenant_id, client_id, client_secret]):
                raise AzureAuthError(
                    "Service Principal requires Tenant ID, Client ID, and Client Secret."
                )
            _credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
            logger.info("Authenticated via Service Principal (client_id=%s)", client_id)

        elif method == AuthMethod.DEVICE_CODE:
            kwargs: dict = {}
            if tenant_id:
                kwargs["tenant_id"] = tenant_id
            if device_code_callback:
                kwargs["prompt_callback"] = device_code_callback
            _credential = DeviceCodeCredential(**kwargs)
            logger.info("Authenticated via Device Code flow")

        elif method == AuthMethod.INTERACTIVE_BROWSER:
            kwargs = {}
            if tenant_id:
                kwargs["tenant_id"] = tenant_id
            _credential = InteractiveBrowserCredential(**kwargs)
            logger.info("Authenticated via Interactive Browser")

        else:
            _credential = DefaultAzureCredential()
            logger.info("Authenticated via DefaultAzureCredential")

        _current_method = method
        return _credential

    except AzureAuthError:
        raise
    except Exception as exc:
        raise AzureAuthError(f"Failed to initialize Azure credentials: {exc}") from exc


def get_access_token(
    method: AuthMethod = AuthMethod.DEFAULT,
    tenant_id: str = "",
    client_id: str = "",
    client_secret: str = "",
    scope: str = "https://management.azure.com/.default",
    device_code_callback: Optional[DeviceCodePromptCallback] = None,
) -> str:
    """Obtain a bearer token for the Azure Resource Manager API.

    Args:
        method: The authentication method to use.
        tenant_id: For Service Principal / Device Code.
        client_id: For Service Principal.
        client_secret: For Service Principal.
        scope: The OAuth scope to request.
        device_code_callback: Optional callback for device code flow prompts.

    Returns:
        A bearer access token string.

    Raises:
        AzureAuthError: If the token cannot be obtained.
    """
    credential = get_credential(
        method=method,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        device_code_callback=device_code_callback,
    )
    try:
        token = credential.get_token(scope)
        logger.debug("Obtained access token, expires at %s", token.expires_on)
        return token.token
    except Exception as exc:
        raise AzureAuthError(f"Failed to obtain access token: {exc}") from exc


def test_connection(
    subscription_id: str,
    method: AuthMethod = AuthMethod.DEFAULT,
    tenant_id: str = "",
    client_id: str = "",
    client_secret: str = "",
    device_code_callback: Optional[DeviceCodePromptCallback] = None,
) -> bool:
    """Test that authentication works by requesting a token.

    Args:
        subscription_id: Azure subscription ID (validated as non-empty).
        method: Auth method to test.
        tenant_id: For Service Principal / Device Code.
        client_id: For Service Principal.
        client_secret: For Service Principal.
        device_code_callback: Optional callback for device code flow prompts.

    Returns:
        True if a token was successfully obtained.

    Raises:
        AzureAuthError: If authentication fails.
    """
    if not subscription_id:
        raise AzureAuthError("Azure Subscription ID is required.")

    get_access_token(
        method=method,
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        device_code_callback=device_code_callback,
    )
    return True
