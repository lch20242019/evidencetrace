"""Product pipeline compatibility exports and deterministic HTTP fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from evidencetrace.product import ProductAuditPipeline, ProductPipelineError

# Preserve the original import name while keeping one production orchestration path.
CitationAuditPipeline = ProductAuditPipeline
AuditPipelineError = ProductPipelineError


def fixture_transport(manifest_path: Path) -> httpx.MockTransport:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {item["url"]: item for item in manifest["sources"]}

    def handler(request: httpx.Request) -> httpx.Response:
        item = sources.get(str(request.url))
        if item is None or not item.get("available", False):
            return httpx.Response(
                503,
                request=request,
                headers={"content-type": "text/plain"},
                text="fixture unavailable",
            )
        content = item.get("content", "")
        locator = item.get("locator", "")
        html = (
            f"<html><head><title>{locator}</title></head>"
            f"<body><h1>{locator}</h1><p>{content}</p></body></html>"
        )
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
        )

    return httpx.MockTransport(handler)


__all__ = [
    "AuditPipelineError",
    "CitationAuditPipeline",
    "ProductAuditPipeline",
    "fixture_transport",
]
