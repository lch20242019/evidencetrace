"""SSRF-aware, bounded HTTP fetching for citation sources."""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx


class FetchError(RuntimeError):
    """A source could not be fetched within the safe fetch contract."""


TEMPORARY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
ALLOWED_MIME_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "text/plain"}
)
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.aws.internal",
        "metadata.azure.internal",
        "metadata.google.internal",
        "instance-data.ec2.internal",
    }
)
BLOCKED_IPS = frozenset(
    {
        "169.254.169.254",  # AWS, Azure, GCP, and other cloud metadata
        "100.100.100.200",  # Alibaba Cloud metadata
        "fd00:ec2::254",  # AWS IMDS IPv6
    }
)
SENSITIVE_QUERY_KEYS = frozenset(
    {
        "access_token",
        "api-key",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "key",
        "passwd",
        "password",
        "secret",
        "sig",
        "signature",
        "token",
        "x-amz-credential",
        "x-amz-security-token",
        "x-amz-signature",
        "x-api-key",
        "x-goog-signature",
    }
)

_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f]")
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ReadError,
    httpx.WriteError,
)


@dataclass(frozen=True)
class FetchedSource:
    url: str
    final_url: str
    content: str
    content_type: str
    status_code: int
    retrieved_at: datetime


@dataclass(frozen=True)
class _ValidatedTarget:
    url: str
    hostname: str
    port: int
    explicit_port: bool
    addresses: tuple[str, ...]


def _parse_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse normal and legacy numeric IP forms without doing DNS."""

    unscoped = value.split("%", 1)[0]
    try:
        return ipaddress.ip_address(unscoped)
    except ValueError:
        pass
    if ":" in unscoped:
        return None
    try:
        # inet_aton recognizes decimal, octal, hex, and shortened IPv4
        # spellings which some HTTP stacks still accept (for example 2130706433).
        return ipaddress.IPv4Address(socket.inet_aton(unscoped))
    except OSError:
        return None


def _validate_ip(value: str, *, from_dns: bool = False) -> bool:
    address = _parse_ip(value)
    if address is None:
        if from_dns:
            raise FetchError("DNS resolution returned an invalid address")
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if (
        str(address) in BLOCKED_IPS
        or not address.is_global
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        raise FetchError(
            "URL resolves to a private, loopback, link-local, or metadata address"
        )
    return True


def _validate_dns(hostname: str, port: int) -> tuple[str, ...]:
    try:
        addresses = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except (OSError, UnicodeError):
        raise FetchError("DNS resolution failed") from None
    if not addresses:
        raise FetchError("DNS resolution returned no addresses")
    validated: list[str] = []
    for address in addresses:
        try:
            resolved = address[4][0]
        except (IndexError, TypeError):
            raise FetchError("DNS resolution returned an invalid address") from None
        _validate_ip(resolved, from_dns=True)
        parsed = _parse_ip(resolved)
        if parsed is None:  # Kept explicit so this remains fail-closed.
            raise FetchError("DNS resolution returned an invalid address")
        if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
            parsed = parsed.ipv4_mapped
        validated.append(str(parsed))
    return tuple(dict.fromkeys(validated))


def _validate_target(url: str, *, resolve_dns: bool) -> _ValidatedTarget:
    if not isinstance(url, str) or not url or _CONTROL_OR_SPACE.search(url):
        raise FetchError("URL is empty or contains unsafe characters")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise FetchError("URL is malformed") from None
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise FetchError("only http and https URLs are allowed")
    if parsed.username or parsed.password:
        raise FetchError("URLs with embedded credentials are not allowed")
    if any(
        key.casefold() in SENSITIVE_QUERY_KEYS
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    ):
        # httpx logs request URLs at INFO, so redaction after fetching is too late.
        raise FetchError("URLs with credential query parameters are not allowed")
    if not hostname:
        raise FetchError("URL must contain a hostname")
    if "%" in hostname:
        raise FetchError("scoped IP addresses are not allowed")
    try:
        normalized = hostname.rstrip(".").encode("idna").decode("ascii").casefold()
    except UnicodeError:
        raise FetchError("URL hostname is invalid") from None
    if not normalized:
        raise FetchError("URL must contain a hostname")
    if normalized in BLOCKED_HOSTNAMES or normalized.endswith(".localhost"):
        raise FetchError("localhost and metadata hostnames are blocked")
    is_ip = _validate_ip(normalized)
    target_port = port if port is not None else (443 if scheme == "https" else 80)
    addresses: tuple[str, ...] = ()
    if is_ip:
        parsed_ip = _parse_ip(normalized)
        if parsed_ip is None:  # pragma: no cover - guaranteed by _validate_ip
            raise FetchError("URL contains an invalid address")
        if isinstance(parsed_ip, ipaddress.IPv6Address) and parsed_ip.ipv4_mapped:
            parsed_ip = parsed_ip.ipv4_mapped
        addresses = (str(parsed_ip),)
    elif resolve_dns:
        addresses = _validate_dns(normalized, target_port)
    # Fragments are never sent to a server and can contain accidental secrets.
    clean_url = urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, ""))
    return _ValidatedTarget(
        url=clean_url,
        hostname=normalized,
        port=target_port,
        explicit_port=port is not None,
        addresses=addresses,
    )


def validate_url(url: str, *, resolve_dns: bool = True) -> str:
    """Validate scheme, credentials, hostname, IP ranges, and all DNS targets."""

    return _validate_target(url, resolve_dns=resolve_dns).url


def _redact_url(url: str) -> str:
    """Remove common credential query values from persisted source metadata."""

    parsed = urlsplit(url)
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        safe_value = "[REDACTED]" if key.casefold() in SENSITIVE_QUERY_KEYS else value
        query.append((key, safe_value))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), "")
    )


class SafeFetcher:
    """Fetch text sources with redirect, size, timeout, and retry bounds."""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_bytes: int = 1_000_000,
        max_redirects: int = 5,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
        resolve_dns: bool | None = None,
        user_agent: str = "EvidenceTrace-CI/0.0.0",
    ) -> None:
        if timeout <= 0 or max_bytes <= 0:
            raise ValueError("timeout and maximum response size must be positive")
        if max_redirects < 0 or retries < 0:
            raise ValueError("redirect and retry limits cannot be negative")
        if resolve_dns is False and not isinstance(transport, httpx.MockTransport):
            raise ValueError(
                "DNS validation can only be disabled for a custom transport "
                "backed by httpx.MockTransport"
            )
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.retries = retries
        self.transport = transport
        if resolve_dns is None:
            self.resolve_dns = not isinstance(transport, httpx.MockTransport)
        else:
            self.resolve_dns = resolve_dns
        self._pin_connections = self.resolve_dns and not isinstance(
            transport, httpx.MockTransport
        )
        self.user_agent = user_agent

    def fetch(self, url: str) -> FetchedSource:
        deadline = monotonic() + self.timeout
        current = url
        headers = {
            "Accept": "text/html,application/xhtml+xml,text/plain",
            "User-Agent": self.user_agent,
        }
        with httpx.Client(
            transport=self.transport,
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=False,
            headers=headers,
            trust_env=False,
        ) as client:
            for redirect in range(self.max_redirects + 1):
                response, current = self._request_with_retry(
                    client,
                    current,
                    deadline=deadline,
                )
                try:
                    if response.status_code in REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError("redirect response has no Location header")
                        if redirect >= self.max_redirects:
                            raise FetchError("redirect limit exceeded")
                        try:
                            current = urljoin(current, location)
                        except ValueError:
                            raise FetchError("redirect target is malformed") from None
                        continue
                    if not 200 <= response.status_code < 300:
                        raise FetchError(
                            f"source returned HTTP {response.status_code}"
                        )
                    content_type = (
                        response.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if content_type not in ALLOWED_MIME_TYPES:
                        value = content_type or "missing"
                        raise FetchError(f"unsupported source MIME type: {value}")
                    content = self._read_bounded(response, deadline=deadline)
                    return FetchedSource(
                        url=_redact_url(validate_url(url, resolve_dns=False)),
                        final_url=_redact_url(current),
                        content=content,
                        content_type=content_type,
                        status_code=response.status_code,
                        retrieved_at=datetime.now(UTC),
                    )
                finally:
                    response.close()
        raise FetchError("fetch failed")

    def _request_with_retry(
        self,
        client: httpx.Client,
        url: str,
        *,
        deadline: float,
    ) -> tuple[httpx.Response, str]:
        for attempt in range(self.retries + 1):
            self._remaining_timeout(deadline)
            # Resolve and validate immediately before every attempt. This also
            # prevents a retry or redirect from reusing a previously trusted DNS
            # result after the hostname's answers have changed.
            target = _validate_target(url, resolve_dns=self.resolve_dns)
            remaining = self._remaining_timeout(deadline)
            client.cookies.clear()
            request = self._build_request(client, target, attempt)
            request_timeout = request.extensions.get("timeout")
            if isinstance(request_timeout, dict):
                for phase in ("connect", "read", "write", "pool"):
                    request_timeout[phase] = remaining
            try:
                response = client.send(request, stream=True)
            except _RETRYABLE_EXCEPTIONS:
                if attempt >= self.retries:
                    raise FetchError(
                        "temporary network error exhausted retries"
                    ) from None
                continue
            except httpx.HTTPError:
                raise FetchError("source request failed") from None
            try:
                self._remaining_timeout(deadline)
            except FetchError:
                response.close()
                raise
            if response.status_code in TEMPORARY_STATUSES and attempt < self.retries:
                response.close()
                continue
            return response, target.url
        raise FetchError("temporary HTTP error exhausted retries")

    def _build_request(
        self,
        client: httpx.Client,
        target: _ValidatedTarget,
        attempt: int,
    ) -> httpx.Request:
        request_url = target.url
        if self._pin_connections:
            # Connect to a validated address instead of asking the HTTP stack to
            # resolve the hostname again. Host and SNI retain the source identity.
            address = target.addresses[attempt % len(target.addresses)]
            parsed = urlsplit(target.url)
            ip_host = f"[{address}]" if ":" in address else address
            request_url = urlunsplit(
                (
                    parsed.scheme,
                    f"{ip_host}:{target.port}",
                    parsed.path,
                    parsed.query,
                    "",
                )
            )
        request = client.build_request("GET", request_url)
        if self._pin_connections:
            host = (
                f"[{target.hostname}]" if ":" in target.hostname else target.hostname
            )
            if target.explicit_port:
                host = f"{host}:{target.port}"
            request.headers["host"] = host
            request.headers["connection"] = "close"
            request.extensions["sni_hostname"] = target.hostname
        return request

    def _remaining_timeout(self, deadline: float) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise FetchError("fetch timeout exceeded")
        return min(self.timeout, remaining)

    def _read_bounded(
        self,
        response: httpx.Response,
        *,
        deadline: float,
    ) -> str:
        self._remaining_timeout(deadline)
        declared_length = response.headers.get("content-length")
        if declared_length:
            try:
                parsed_length = int(declared_length)
            except ValueError:
                raise FetchError("source returned an invalid Content-Length") from None
            if parsed_length < 0:
                raise FetchError("source returned an invalid Content-Length")
            if parsed_length > self.max_bytes:
                raise FetchError("source response exceeds maximum size")
        body = bytearray()
        chunk_size = min(65_536, self.max_bytes + 1)
        try:
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                self._remaining_timeout(deadline)
                if len(chunk) > self.max_bytes - len(body):
                    raise FetchError("source response exceeds maximum size")
                body.extend(chunk)
        except httpx.HTTPError:
            raise FetchError("source body could not be read") from None
        try:
            return bytes(body).decode(response.encoding or "utf-8")
        except (LookupError, UnicodeDecodeError):
            raise FetchError(
                "source is not valid text in its declared encoding"
            ) from None


__all__ = ["FetchError", "FetchedSource", "SafeFetcher", "validate_url"]
