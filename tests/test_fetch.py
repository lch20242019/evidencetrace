from __future__ import annotations

import gzip
import socket
from collections.abc import Callable, Iterator

import httpx
import pytest

import evidencetrace.retrieval.fetch as fetch_module
from evidencetrace.retrieval.fetch import FetchError, SafeFetcher, validate_url


Handler = Callable[[httpx.Request], httpx.Response]


class _Chunks(httpx.SyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.chunks)


def _fetcher(
    handler: Handler,
    *,
    timeout: float = 1.0,
    max_bytes: int = 1_000_000,
    max_redirects: int = 5,
    retries: int = 0,
    resolve_dns: bool = False,
) -> SafeFetcher:
    return SafeFetcher(
        timeout=timeout,
        max_bytes=max_bytes,
        max_redirects=max_redirects,
        retries=retries,
        transport=httpx.MockTransport(handler),
        resolve_dns=resolve_dns,
    )


def _ok(request: httpx.Request, *, mime: str = "text/html") -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        headers={"content-type": mime},
        content=b"<p>evidence</p>",
    )


def _address(ip: str, port: int) -> tuple[object, ...]:
    if ":" in ip:
        return (
            socket.AF_INET6,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (ip, port, 0, 0),
        )
    return (
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        (ip, port),
    )


@pytest.mark.parametrize("scheme", ["file", "ftp", "data", "gopher", "ssh"])
def test_only_http_and_https_schemes_are_allowed(scheme: str) -> None:
    with pytest.raises(FetchError, match="only http and https"):
        validate_url(f"{scheme}://example.com/resource", resolve_dns=False)


@pytest.mark.parametrize("scheme", ["http", "https", "HTTP", "HTTPS"])
def test_http_and_https_schemes_are_accepted(scheme: str) -> None:
    result = validate_url(
        f"{scheme}://93.184.216.34/evidence#not-sent",
        resolve_dns=False,
    )

    assert result == f"{scheme.lower()}://93.184.216.34/evidence"


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/data",
        "https://LOCALHOST./data",
        "https://api.localhost/data",
        "https://metadata/data",
        "https://metadata.google.internal/computeMetadata/v1/",
        "https://metadata.azure.internal/data",
        "https://instance-data.ec2.internal/latest/meta-data/",
    ],
)
def test_localhost_and_metadata_hostnames_are_blocked(url: str) -> None:
    with pytest.raises(FetchError, match="localhost and metadata"):
        validate_url(url, resolve_dns=False)


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://127.0.0.1/",
        "http://0.0.0.0/",
        "http://169.254.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.100.100.200/latest/meta-data/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[fd00:ec2::254]/",
        "http://[::ffff:127.0.0.1]/",
        "http://224.0.0.1/",
        "http://2130706433/",
        "http://0177.0.0.1/",
        "http://0x7f000001/",
    ],
)
def test_non_public_and_obfuscated_ip_literals_are_blocked(url: str) -> None:
    with pytest.raises(FetchError, match="private, loopback, link-local"):
        validate_url(url, resolve_dns=False)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com/",
        "https://example.com:99999/",
        "https://[fe80::1%25eth0]/",
        "https://example.com/a path",
        "https:///missing-host",
    ],
)
def test_malformed_or_credential_bearing_urls_are_blocked(url: str) -> None:
    with pytest.raises(FetchError):
        validate_url(url, resolve_dns=False)


def test_dns_uses_the_scheme_port_and_accepts_only_public_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def public_dns(host: str, port: int, **_: object) -> list[tuple[object, ...]]:
        calls.append((host, port))
        return [_address("93.184.216.34", port), _address("2606:2800:220:1::", port)]

    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", public_dns)

    assert validate_url("http://example.com/a") == "http://example.com/a"
    assert validate_url("https://example.com:8443/b") == (
        "https://example.com:8443/b"
    )
    assert calls == [("example.com", 80), ("example.com", 8443)]


def test_production_request_is_pinned_to_the_validated_dns_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def public_dns(host: str, port: int, **_: object) -> list[tuple[object, ...]]:
        del host
        return [_address("93.184.216.34", port)]

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url_host"] = request.url.host
        observed["host_header"] = request.headers["host"]
        observed["sni_hostname"] = request.extensions["sni_hostname"]
        return _ok(request)

    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", public_dns)
    fetcher = _fetcher(handler, resolve_dns=True)
    # MockTransport is deliberately not pinned by default; enable the production
    # branch explicitly while retaining a no-network transport for this test.
    monkeypatch.setattr(fetcher, "_pin_connections", True)

    source = fetcher.fetch("https://docs.example/evidence")

    assert observed == {
        "url_host": "93.184.216.34",
        "host_header": "docs.example",
        "sni_hostname": "docs.example",
    }
    assert source.final_url == "https://docs.example/evidence"


def test_dns_fails_closed_if_any_answer_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mixed_dns(host: str, port: int, **_: object) -> list[tuple[object, ...]]:
        del host
        return [_address("93.184.216.34", port), _address("10.1.2.3", port)]

    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", mixed_dns)

    with pytest.raises(FetchError, match="private, loopback, link-local"):
        validate_url("https://changes.example/source")


def test_dns_resolution_failure_has_a_bounded_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_dns(host: str, port: int, **_: object) -> list[tuple[object, ...]]:
        del host, port
        raise socket.gaierror(socket.EAI_NONAME, "resolver detail")

    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", failed_dns)

    with pytest.raises(FetchError, match=r"^DNS resolution failed$"):
        validate_url("https://unresolved.example/source")


def test_dns_is_revalidated_before_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolutions = 0
    requests = 0

    def changing_dns(
        host: str, port: int, **_: object
    ) -> list[tuple[object, ...]]:
        nonlocal resolutions
        del host
        resolutions += 1
        ip = "93.184.216.34" if resolutions == 1 else "127.0.0.1"
        return [_address(ip, port)]

    def timeout_once(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise httpx.ReadTimeout("temporary", request=request)

    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", changing_dns)
    fetcher = _fetcher(timeout_once, retries=1, resolve_dns=True)

    with pytest.raises(FetchError, match="private, loopback, link-local"):
        fetcher.fetch("https://changing.example/evidence")
    assert resolutions == 2
    assert requests == 1


def test_every_redirect_target_is_validated_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_hosts: list[str] = []
    resolved_hosts: list[str] = []

    def resolver(host: str, port: int, **_: object) -> list[tuple[object, ...]]:
        resolved_hosts.append(host)
        ip = "127.0.0.1" if host == "internal.example" else "93.184.216.34"
        return [_address(ip, port)]

    def redirects(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.host == "start.example":
            return httpx.Response(
                302,
                request=request,
                headers={"location": "https://next.example/step"},
            )
        return httpx.Response(
            307,
            request=request,
            headers={"location": "http://internal.example/admin"},
        )

    monkeypatch.setattr(fetch_module.socket, "getaddrinfo", resolver)
    fetcher = _fetcher(redirects, resolve_dns=True)

    with pytest.raises(FetchError, match="private, loopback, link-local"):
        fetcher.fetch("https://start.example/source")
    assert requested_hosts == ["start.example", "next.example"]
    assert resolved_hosts == ["start.example", "next.example", "internal.example"]


def test_redirect_requires_location_and_obeys_limit() -> None:
    def no_location(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, request=request)

    with pytest.raises(FetchError, match="no Location"):
        _fetcher(no_location).fetch("https://example.com/source")

    def loop(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            request=request,
            headers={"location": "/again"},
        )

    with pytest.raises(FetchError, match="redirect limit exceeded"):
        _fetcher(loop, max_redirects=1).fetch("https://example.com/source")


@pytest.mark.parametrize(
    "mime",
    ["text/html", "text/html; charset=utf-8", "application/xhtml+xml", "text/plain"],
)
def test_allowed_text_mime_types(mime: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(request, mime=mime)

    source = _fetcher(handler).fetch("https://example.com/source")

    assert source.content == "<p>evidence</p>"
    assert source.content_type == mime.split(";", 1)[0]


@pytest.mark.parametrize(
    "mime",
    ["", "application/octet-stream", "application/pdf", "image/svg+xml", "text/htmlx"],
)
def test_missing_or_unsupported_mime_type_is_rejected(mime: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": mime} if mime else {}
        return httpx.Response(200, request=request, headers=headers, content=b"data")

    with pytest.raises(FetchError, match="unsupported source MIME type"):
        _fetcher(handler).fetch("https://example.com/source")


def test_declared_response_size_is_checked_before_reading() -> None:
    stream = _Chunks(b"never consumed")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/plain", "content-length": "100"},
            stream=stream,
        )

    with pytest.raises(FetchError, match="exceeds maximum size"):
        _fetcher(handler, max_bytes=10).fetch("https://example.com/source")


def test_streamed_response_size_is_bounded_without_content_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/plain"},
            stream=_Chunks(b"12345", b"67890"),
        )

    with pytest.raises(FetchError, match="exceeds maximum size"):
        _fetcher(handler, max_bytes=8).fetch("https://example.com/source")


def test_decompressed_response_size_is_bounded() -> None:
    compressed = gzip.compress(b"x" * 1_000)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_Chunks(compressed),
        )

    with pytest.raises(FetchError, match="exceeds maximum size"):
        _fetcher(handler, max_bytes=100).fetch("https://example.com/source")


def test_timeout_is_passed_to_every_httpx_phase() -> None:
    observed: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        timeout = request.extensions["timeout"]
        assert isinstance(timeout, dict)
        observed.update(timeout)
        return _ok(request)

    _fetcher(handler, timeout=0.25).fetch("https://example.com/source")

    assert set(observed) == {"connect", "read", "write", "pool"}
    assert all(0 < value <= 0.25 for value in observed.values())


def test_total_fetch_deadline_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    calls = 0

    def clock() -> float:
        return now[0]

    def slow_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        now[0] = 1.01
        return _ok(request)

    monkeypatch.setattr(fetch_module, "monotonic", clock)

    with pytest.raises(FetchError, match="fetch timeout exceeded"):
        _fetcher(slow_handler, timeout=1.0, retries=3).fetch(
            "https://example.com/source"
        )
    assert calls == 1


@pytest.mark.parametrize("status", sorted(fetch_module.TEMPORARY_STATUSES))
def test_explicit_temporary_statuses_are_retried(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, request=request)
        return _ok(request)

    source = _fetcher(handler, retries=1).fetch("https://example.com/source")

    assert source.status_code == 200
    assert calls == 2


def test_temporary_status_stops_at_retry_limit() -> None:
    calls = 0

    def unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request)

    with pytest.raises(FetchError, match="source returned HTTP 503"):
        _fetcher(unavailable, retries=2).fetch("https://example.com/source")
    assert calls == 3


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ReadError])
def test_explicit_temporary_transport_errors_are_retried(
    error_type: type[httpx.RequestError],
) -> None:
    calls = 0

    def temporary_error(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_type("temporary", request=request)
        return _ok(request)

    source = _fetcher(temporary_error, retries=1).fetch(
        "https://example.com/source"
    )

    assert source.status_code == 200
    assert calls == 2


def test_non_temporary_http_and_transport_errors_are_not_retried() -> None:
    http_calls = 0

    def not_found(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        return httpx.Response(404, request=request)

    with pytest.raises(FetchError, match="source returned HTTP 404"):
        _fetcher(not_found, retries=5).fetch("https://example.com/source")
    assert http_calls == 1

    transport_calls = 0

    def tls_failure(request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        raise httpx.ConnectError("certificate verification failed", request=request)

    with pytest.raises(FetchError, match=r"^source request failed$"):
        _fetcher(tls_failure, retries=5).fetch("https://example.com/source")
    assert transport_calls == 1


def test_sensitive_headers_do_not_reach_requests_or_output() -> None:
    secret = "do-not-persist-this-secret"
    seen_headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers)
        if request.url.path == "/start":
            return httpx.Response(
                302,
                request=request,
                headers={
                    "location": "/final",
                    "set-cookie": f"session={secret}",
                    "authorization": f"Bearer {secret}",
                    "x-api-key": secret,
                },
            )
        assert "cookie" not in request.headers
        return _ok(request)

    source = _fetcher(handler).fetch("https://example.com/start")

    for headers in seen_headers:
        assert "authorization" not in headers
        assert "proxy-authorization" not in headers
        assert "x-api-key" not in headers
        assert "cookie" not in headers
    assert secret not in repr(source)
    assert secret not in source.url
    assert secret not in source.final_url


@pytest.mark.parametrize(
    "query_key",
    ["api_key", "access_token", "Authorization", "X-Amz-Signature"],
)
def test_query_credentials_are_blocked_before_httpx_can_log_them(
    query_key: str,
) -> None:
    secret = "never-enter-httpx"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _ok(request)

    with pytest.raises(FetchError, match="credential query parameters") as captured:
        _fetcher(handler).fetch(
            f"https://example.com/source?{query_key}={secret}"
        )

    assert calls == 0
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)

def test_transport_error_detail_is_not_exposed() -> None:
    secret = "Bearer highly-sensitive-value"
    calls = 0

    def failure(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(secret, request=request)

    with pytest.raises(FetchError) as captured:
        _fetcher(failure, retries=4).fetch("https://example.com/source")

    assert calls == 1
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert captured.value.__suppress_context__ is True


def test_dns_validation_cannot_be_disabled_for_live_transport() -> None:
    with pytest.raises(ValueError, match="custom transport"):
        SafeFetcher(resolve_dns=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0.0},
        {"max_bytes": 0},
        {"max_redirects": -1},
        {"retries": -1},
    ],
)
def test_fetch_limits_are_validated(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        SafeFetcher(transport=httpx.MockTransport(_ok), **kwargs)
