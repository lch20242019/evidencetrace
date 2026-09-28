from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_first_screen_states_product_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required = (
        "bounded fact-audit Agent",
        "not a truth detector",
        "evidencetrace demo",
        "--discover",
        "--suggest-patch",
        ".github/workflows/evidencetrace.yml",
        "security-events: write",
        "--sarif",
        "completed_with_known_limitations",
    )
    for phrase in required:
        assert phrase in readme


def test_exactly_three_markdown_input_fixtures_exist() -> None:
    fixtures = sorted(path.name for path in (ROOT / "examples").glob("*.md"))

    assert fixtures == [
        "bad-agent-comparison.md",
        "complex-work-document.md",
        "simple-work-document.md",
    ]


def test_bad_document_has_eight_distinct_controlled_errors() -> None:
    bad_doc = (ROOT / "examples" / "bad-agent-comparison.md").read_text(
        encoding="utf-8"
    )

    for case_number in range(1, 9):
        assert bad_doc.count(f"ET-{case_number:03d}:") == 1


def test_offline_manifest_covers_every_bad_document_url() -> None:
    bad_doc = (ROOT / "examples" / "bad-agent-comparison.md").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (ROOT / "examples" / "evidence" / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    urls = {source["url"] for source in manifest["sources"]}

    expected = {
        "https://fixtures.evidencetrace.invalid/nimbus/benchmark-2026",
        "https://fixtures.evidencetrace.invalid/nimbus/releases/v1.3",
        "https://fixtures.evidencetrace.invalid/nimbus/ownership",
        "https://fixtures.evidencetrace.invalid/nimbus/readiness",
        "https://fixtures.evidencetrace.invalid/nimbus/launch",
        "https://fixtures.evidencetrace.invalid/nimbus/storage",
        "https://fixtures.evidencetrace.invalid/nimbus/retries",
        "https://fixtures.evidencetrace.invalid/nimbus/missing-latency-study",
    }
    assert expected <= urls
    assert all(url in bad_doc for url in expected)


def test_expected_audit_uses_only_contract_labels() -> None:
    audit = (
        ROOT / "examples" / "expected" / "bad-agent-comparison.audit.md"
    ).read_text(encoding="utf-8")

    assert audit.count("## ET-") == 8
    assert "`contradicted`" in audit
    assert "`partially_entailed`" in audit
    assert "`not_in_source`" in audit
    assert "`source_unavailable`" in audit
    assert "truth detector" not in audit.lower()


def test_name_check_does_not_claim_gitHub_availability() -> None:
    name_check = (ROOT / "docs" / "name-availability.md").read_text(encoding="utf-8")

    assert "Exact-name collision" in name_check
    assert "not a trademark opinion" in name_check
    assert "blocked" in name_check
