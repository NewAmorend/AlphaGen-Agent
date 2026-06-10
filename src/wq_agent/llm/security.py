from __future__ import annotations

import re
from urllib.parse import urlparse


LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}
PLACEHOLDER_SECRET_RE = re.compile(r"^(?:your_|change_?me|placeholder|todo\b)", re.IGNORECASE)


def is_real_secret(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip().strip('"').strip("'")
    return bool(text) and text != "********" and not PLACEHOLDER_SECRET_RE.match(text)


def is_local_url(url: str) -> bool:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower() in LOCAL_HTTP_HOSTS


def validate_api_key(api_key: str, *, env_key: str, provider: str) -> None:
    if is_real_secret(api_key):
        return
    raise ValueError(
        f"{env_key} is required for LLM_PROVIDER={provider.lower()}. "
        "Set a real API key in .env before running LLM generation."
    )


def validate_transport_security(
    base_url: str,
    *,
    env_key: str,
    allow_insecure_http: bool = False,
) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "http":
        return
    host = (parsed.hostname or "").lower()
    if host in LOCAL_HTTP_HOSTS or allow_insecure_http:
        return
    raise ValueError(
        f"{env_key} uses insecure HTTP for a non-local host. "
        "Use HTTPS, a localhost/127.0.0.1 URL, or explicitly allow insecure HTTP "
        "only for a trusted private endpoint."
    )
