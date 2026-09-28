"""Build the Phase 4C public-source Chinese diagnostic pack.

This pack is intentionally *not* presented as an independently human-reviewed
holdout.  It snapshots short, citable excerpts from official documentation and
derives labels from literal source spans and explicit diagnostic probes.  The
purpose is to exercise the CJK retriever and the empty-recall
policy on material that was not used by the earlier EvidenceTrace packs.  The
runner therefore reports ``diagnostic_contaminated`` for the resulting metrics.

The script is deterministic: source text, labels, ordering, and hashes are all
fixed in this file.  It also audits the previous eval_sets tree for exact ID,
claim, URL, and source-content-hash reuse before writing the pack.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from evidencetrace.eval.baselines import build_evidence_chunks
from evidencetrace.eval.dataset import compute_case_hash
from evidencetrace.eval.models import EvalCase, SourceFixture
from evidencetrace.models import Relation


PACK = Path(__file__).resolve().parents[1] / "eval_sets" / "phase4c_public_zh_diagnostic"
FIXTURE_RELATIVE = "sources/source_snapshots.jsonl"
DATASET_NAME = "holdout_public_zh.jsonl"
FROZEN_NAME = "holdout_public_zh.frozen_hashes.json"
RETRIEVAL_DATE = "2026-09-02"


SOURCE_TEXTS: dict[str, dict[str, str]] = {
    "phase4c_k8s_objects_zh": {
        "url": "https://kubernetes.io/zh-cn/docs/concepts/overview/working-with-objects/",
        "provenance": (
            "Official Kubernetes zh-cn documentation snapshot, retrieved "
            "2026-09-02; excerpt transcribed from page lines 999-1017 with "
            "inter-sentence display whitespace normalized."
        ),
        "content": (
            "Kubernetes 对象是 Kubernetes 系统中的持久性实体。Kubernetes 使用这些实体表示你的集群状态。"
            "了解 Kubernetes 对象模型以及如何使用这些对象。\n\n"
            "本页说明了在 Kubernetes API 中是如何表示 Kubernetes 对象的，以及如何使用 `.yaml` 格式的文件表示 Kubernetes 对象。\n\n"
            "在 Kubernetes 系统中，Kubernetes 对象是持久化的实体。Kubernetes 使用这些实体去表示整个集群的状态。"
            "具体而言，它们描述了如下信息：哪些容器化应用正在运行（以及在哪些节点上运行）、可以被应用使用的资源、"
            "关于应用运行时行为的策略，比如重启策略、升级策略以及容错策略。\n\n"
            "Kubernetes 对象是一种“意向表达（Record of Intent）”。一旦创建该对象，Kubernetes 系统将不断工作以确保该对象存在。\n\n"
            "通过创建对象，你本质上是在告知 Kubernetes 系统，你想要的集群工作负载状态看起来应是什么样子的，"
            "这就是 Kubernetes 集群所谓的期望状态（Desired State）。\n\n"
            "操作 Kubernetes 对象——无论是创建、修改或者删除——需要使用 Kubernetes API。比如，当使用 `kubectl` 命令行接口（CLI）时，"
            "CLI 会调用必要的 Kubernetes API；也可以在程序中使用客户端库，来直接调用 Kubernetes API。\n\n"
            "几乎每个 Kubernetes 对象包含两个嵌套的对象字段，它们负责管理对象的配置：对象 `spec`（规约）和对象 `status`（状态）。\n\n"
            "`status` 描述了对象的当前状态（Current State），它是由 Kubernetes 系统和组件设置并更新的。\n\n"
            "在任何时刻，Kubernetes 控制平面都一直在积极地管理着对象的实际状态，以使之达成期望状态。\n\n"
            "例如，Kubernetes 中的 Deployment 对象能够表示运行在集群中的应用。当创建 Deployment 时，你可能会设置 Deployment 的 `spec`，"
            "指定该应用要有 3 个副本运行。\n\n"
            "Kubernetes 系统读取 Deployment 的 `spec`，并启动我们所期望的应用的 3 个实例——更新状态以与规约相匹配。\n\n"
            "如果这些实例中有的失败了（一种状态变更），Kubernetes 系统会通过执行修正操作来响应 `spec` 和 `status` 间的不一致——"
            "意味着它会启动一个新的实例来替换。"
        ),
    },
    "phase4c_k8s_manage_zh": {
        "url": "https://kubernetes.io/zh-cn/docs/tasks/manage-kubernetes-objects/",
        "provenance": (
            "Official Kubernetes zh-cn documentation snapshot, retrieved "
            "2026-09-02; excerpt transcribed from page lines 990-1006."
        ),
        "content": (
            "用声明式和命令式范型与 Kubernetes API 交互。\n\n"
            "使用配置文件对 Kubernetes 对象进行声明式管理。\n\n"
            "使用 Kustomize 对 Kubernetes 对象进行声明式管理。\n\n"
            "使用指令式命令管理 Kubernetes 对象。\n\n"
            "使用配置文件对 Kubernetes 对象进行命令式管理。\n\n"
            "使用 kubectl patch 更新 API 对象。\n\n"
            "使用 kubectl patch 更新 Kubernetes API 对象。做一个策略性的合并 patch 或 JSON 合并 patch。"
        ),
    },
    "phase4c_python_tutorial_zh": {
        "url": "https://docs.python.org/zh-cn/3/tutorial/index.html",
        "provenance": (
            "Official Python 3 zh-cn documentation snapshot, retrieved "
            "2026-09-02; excerpt transcribed from page lines 5-16."
        ),
        "content": (
            "本教程被设计为针对新入门 Python 语言的程序员，而不是新入门编程的初学者。\n\n"
            "Python 是一门易于学习、功能强大的编程语言。它提供了高效的高级数据结构，还能简单有效地面向对象编程。\n\n"
            "Python 优雅的语法和动态类型以及解释型语言的本质，使它成为多数平台上写脚本和快速开发应用的理想语言。\n\n"
            "Python 解释器及丰富的标准库针对所有主流系统平台以源代码或二进制形式在 Python 网站 https://www.python.org/ 上免费提供，并可自由分发。\n\n"
            "该网站还包含许多免费的第三方 Python 模块、程序和工具以及附加文档的分享链接。\n\n"
            "Python 解释器易于扩展，使用 C 或 C++（或其他 C 能调用的语言）即可为 Python 扩展新功能和数据类型。\n\n"
            "Python 也可用作定制软件中的扩展程序语言。\n\n"
            "本教程非正式地介绍了 Python 语言和系统的基本概念和特性。请注意它预期你对于编程的总体概念有基本的了解。\n\n"
            "准备好一个 Python 解释器随时上手练习会很有帮助，但所有的示例都是完备自足的，因此可以离线阅读。\n\n"
            "本教程对每一个功能的介绍并不完整，甚至没有涉及全部常用功能，旨在让读者快速感受一下 Python 的特色。"
        ),
    },
    "phase4c_python_venv_zh": {
        "url": "https://docs.python.org/zh-cn/3/library/venv.html",
        "provenance": (
            "Official Python 3 zh-cn documentation snapshot, retrieved "
            "2026-09-02; excerpt transcribed from page lines 9-22, 35-47, and 132-153."
        ),
        "content": (
            "venv 模块支持创建轻量的“虚拟环境”，每个虚拟环境将拥有它们自己独立的安装在其 site 目录中的 Python 软件包集合。\n\n"
            "虚拟环境是在现有的 Python 安装版基础之上创建的，这被称为虚拟环境的“基础”Python，并且默认与基础环境中的软件包隔离开来，"
            "这样只有在虚拟环境中显式安装的软件包才是可用的。\n\n"
            "当在虚拟环境中使用时，常见安装工具如 pip 将把 Python 软件包安装到虚拟环境而无需显式地指明这一点。\n\n"
            "虚拟环境被认为是可丢弃的——它应当能被简单地删除并从头开始重建。\n\n"
            "虚拟环境不被视为是可移动或可复制的——你只能在目标位置重建相同的环境。\n\n"
            "虚拟环境是通过执行 venv 模块来创建的：python -m venv /path/to/new/virtual/environment\n\n"
            "此命令会创建目标目录，并在其中放置一个 pyvenv.cfg 文件；在 Windows 上会创建 Scripts 子目录。\n\n"
            "在 3.5 版本发生变更：现在推荐使用 venv 来创建虚拟环境。\n\n"
            "激活一个虚拟环境的操作不是必需的，因为你完全可以在唤起 Python 时指明特定虚拟环境的 Python 解释器的完整路径。\n\n"
            "PowerShell 激活命令为 PS C:\\> <venv>\\Scripts\\Activate.ps1。\n\n"
            "创建的 pyvenv.cfg 文件还包括 include-system-site-packages 键；如果不使用 --system-site-packages 选项则为 false。\n\n"
            "安装在虚拟环境中的脚本包含指定虚拟环境 Python 解释器的路径，因此虚拟环境在通常情况下都是不可移植的。\n\n"
            "你应当在目标位置重建环境并删除旧环境；否则，安装到该虚拟环境的软件包可能无法正常工作。\n\n"
            "激活脚本位于环境的 Scripts 目录中，安装在虚拟环境中的脚本也可以在不激活环境的情况下运行。"
        ),
    },
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _norm_claim(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _source_fixtures() -> dict[str, SourceFixture]:
    fixtures: dict[str, SourceFixture] = {}
    for source_id, values in SOURCE_TEXTS.items():
        content = values["content"]
        fixtures[source_id] = SourceFixture(
            source_id=source_id,
            url=values["url"],
            content=content,
            provenance=values["provenance"],
            content_hash=_sha256_text(content),
            available=True,
        )
    return fixtures


def _stratum(source: SourceFixture) -> str:
    if not source.available:
        return "unavailable"
    size = len(source.content.encode("utf-8"))
    return "short" if size < 1200 else "medium" if size < 3000 else "long"


def _case(
    *,
    case_id: str,
    source: SourceFixture,
    claim: str,
    relation: Relation,
    evidence: str | None,
    mutation: str,
    split: str,
    expected_route: str | None = None,
) -> EvalCase:
    chunks = build_evidence_chunks(source) if source.available else ()
    if expected_route is None:
        if not source.available:
            expected_route = "deterministic_source_unavailable"
        elif "best" in claim.casefold() or "always" in claim.casefold():
            expected_route = "deterministic_not_checkable"
        else:
            expected_route = "retrieval_judge" if len(chunks) > 1 else "full_context_single_agent"
    value = EvalCase(
        case_id=case_id,
        document_path=None,
        claim_text=claim,
        claim_line=1,
        source_fixture=FIXTURE_RELATIVE,
        source_id=source.source_id,
        source_url=source.url,
        gold_relation=relation,
        gold_evidence_span=evidence,
        claim_type="citation_pair",
        mutation_type=mutation,
        split=split,  # type: ignore[arg-type]
        provenance=(
            "Fresh public-source diagnostic case; source excerpt and label were "
            "frozen by build_phase4c_public_zh_diagnostic.py on 2026-09-02."
        ),
        annotation_status="deterministic_gold",
        annotation_notes=(
            "Deterministic label: substantive labels cite an exact source substring; "
            "not_in_source/not_checkable are author-declared probes. No independent "
            "human review was performed; do not use as a public benchmark claim."
        ),
        source_sha256=source.content_hash,
        source_length_stratum=_stratum(source),
        expected_chunk_count=len(chunks),
        expected_route=expected_route,  # type: ignore[arg-type]
    )
    return value.model_copy(update={"case_hash": compute_case_hash(value)})


def _make_cases(sources: dict[str, SourceFixture]) -> list[EvalCase]:
    # Every claim is intentionally distinct.  A small label-stratified split
    # is fixed here before any prediction or metric is run.  Using labels for
    # stratification is recorded in the manifest; using model output is not.
    specs: list[tuple[str, str, str, Relation, str | None, str]] = [
        # Kubernetes objects
        ("k8sobj", "几乎每个 Kubernetes 对象包含 spec 和 status 两个嵌套字段。", "phase4c_k8s_objects_zh", Relation.ENTAILED, "几乎每个 Kubernetes 对象包含两个嵌套的对象字段，它们负责管理对象的配置：对象 `spec`（规约）和对象 `status`（状态）。", "exact_cjk"),
        ("k8sdep", "Deployment 的 spec 指定应用运行 3 个副本。", "phase4c_k8s_objects_zh", Relation.ENTAILED, "指定该应用要有 3 个副本运行。", "exact_numeric"),
        ("k8spart", "Kubernetes 对象表示集群状态，并且会自动扩容节点。", "phase4c_k8s_objects_zh", Relation.PARTIALLY_ENTAILED, "Kubernetes 使用这些实体表示你的集群状态。", "compound_extra"),
        ("k8snum", "Deployment 的 spec 指定应用运行 5 个副本。", "phase4c_k8s_objects_zh", Relation.CONTRADICTED, "指定该应用要有 3 个副本运行。", "numeric_mutation"),
        ("k8sneg", "操作 Kubernetes 对象不需要 Kubernetes API。", "phase4c_k8s_objects_zh", Relation.CONTRADICTED, "操作 Kubernetes 对象——无论是创建、修改或者删除——需要使用 Kubernetes API。", "negation_mutation"),
        ("k8sabsent", "Kubernetes 对象的 spec 会自动加密 Secret 数据。", "phase4c_k8s_objects_zh", Relation.NOT_IN_SOURCE, None, "unsupported_security_detail"),
        ("k8sraft", "Kubernetes 控制平面使用 Raft 选举 leader。", "phase4c_k8s_objects_zh", Relation.NOT_IN_SOURCE, None, "unsupported_algorithm"),
        ("k8sopinion", "Kubernetes should be the best orchestration system。", "phase4c_k8s_objects_zh", Relation.NOT_CHECKABLE, None, "subjective_comparison"),
        # Kubernetes object management
        ("manageapi", "Kubernetes API 支持声明式和命令式范型。", "phase4c_k8s_manage_zh", Relation.ENTAILED, "用声明式和命令式范型与 Kubernetes API 交互。", "exact_cjk"),
        ("managepatch", "kubectl patch 可以更新 Kubernetes API 对象。", "phase4c_k8s_manage_zh", Relation.ENTAILED, "使用 kubectl patch 更新 Kubernetes API 对象。", "exact_tool"),
        ("managepart", "kubectl patch 更新 API 对象，并且删除 API 对象。", "phase4c_k8s_manage_zh", Relation.PARTIALLY_ENTAILED, "使用 kubectl patch 更新 API 对象。", "compound_extra"),
        ("manageonly", "kubectl patch 只支持 JSON 合并 patch。", "phase4c_k8s_manage_zh", Relation.CONTRADICTED, "使用 kubectl patch 更新 Kubernetes API 对象。做一个策略性的合并 patch 或 JSON 合并 patch。", "qualifier_mutation"),
        ("managexml", "kubectl patch 支持 XML patch。", "phase4c_k8s_manage_zh", Relation.NOT_IN_SOURCE, None, "unsupported_patch_type"),
        ("manageterraform", "Kustomize 会自动把 Kubernetes 对象转换成 Terraform。", "phase4c_k8s_manage_zh", Relation.NOT_IN_SOURCE, None, "unsupported_tool"),
        ("manageversion", "命令式管理只能在 Kubernetes 1.30 使用。", "phase4c_k8s_manage_zh", Relation.NOT_IN_SOURCE, None, "unsupported_version"),
        ("manageopinion", "声明式管理 is always better than imperative management。", "phase4c_k8s_manage_zh", Relation.NOT_CHECKABLE, None, "subjective_comparison"),
        # Python tutorial
        ("pydatas", "Python 提供了高效的高级数据结构。", "phase4c_python_tutorial_zh", Relation.ENTAILED, "它提供了高效的高级数据结构，还能简单有效地面向对象编程。", "exact_cjk"),
        ("pyextend", "Python 可用作定制软件中的扩展程序语言。", "phase4c_python_tutorial_zh", Relation.ENTAILED, "Python 也可用作定制软件中的扩展程序语言。", "exact_cjk"),
        ("pypart", "Python 提供高级数据结构，并且内置数据库服务器。", "phase4c_python_tutorial_zh", Relation.PARTIALLY_ENTAILED, "它提供了高效的高级数据结构，还能简单有效地面向对象编程。", "compound_extra"),
        ("pylinux", "Python 标准库只在 Linux 平台提供。", "phase4c_python_tutorial_zh", Relation.CONTRADICTED, "Python 解释器及丰富的标准库针对所有主流系统平台以源代码或二进制形式在 Python 网站 https://www.python.org/ 上免费提供，并可自由分发。", "platform_qualifier_mutation"),
        ("pynoc", "Python 解释器不能使用 C 扩展。", "phase4c_python_tutorial_zh", Relation.CONTRADICTED, "Python 解释器易于扩展，使用 C 或 C++（或其他 C 能调用的语言）即可为 Python 扩展新功能和数据类型。", "negation_mutation"),
        ("pysql", "Python 标准库默认包含 Kubernetes 客户端。", "phase4c_python_tutorial_zh", Relation.NOT_IN_SOURCE, None, "unsupported_library"),
        ("pycloud", "Python 网站提供云端 GPU 集群。", "phase4c_python_tutorial_zh", Relation.NOT_IN_SOURCE, None, "unsupported_service"),
        ("pyopinion", "Python is the best language for every project。", "phase4c_python_tutorial_zh", Relation.NOT_CHECKABLE, None, "subjective_comparison"),
        # Python venv
        ("venvcreate", "venv 模块支持创建轻量的虚拟环境。", "phase4c_python_venv_zh", Relation.ENTAILED, "venv 模块支持创建轻量的“虚拟环境”，每个虚拟环境将拥有它们自己独立的安装在其 site 目录中的 Python 软件包集合。", "exact_cjk"),
        ("venvisolate", "Python 软件包在虚拟环境中默认与基础环境隔离。", "phase4c_python_venv_zh", Relation.ENTAILED, "虚拟环境是在现有的 Python 安装版基础之上创建的，这被称为虚拟环境的“基础”Python，并且默认与基础环境中的软件包隔离开来，这样只有在虚拟环境中显式安装的软件包才是可用的。", "exact_policy"),
        ("venvpart", "venv 创建环境并且自动把项目代码复制进去。", "phase4c_python_venv_zh", Relation.PARTIALLY_ENTAILED, "虚拟环境是通过执行 venv 模块来创建的：python -m venv /path/to/new/virtual/environment", "compound_extra"),
        ("venvmobile", "虚拟环境是可移动和可复制的。", "phase4c_python_venv_zh", Relation.CONTRADICTED, "虚拟环境不被视为是可移动或可复制的——你只能在目标位置重建相同的环境。", "negation_mutation"),
        ("venvactivate", "激活虚拟环境是运行其中 Python 的必需步骤。", "phase4c_python_venv_zh", Relation.CONTRADICTED, "激活一个虚拟环境的操作不是必需的，因为你完全可以在唤起 Python 时指明特定虚拟环境的 Python 解释器的完整路径。", "necessity_mutation"),
        ("venvgit", "venv 会把环境提交到 Git。", "phase4c_python_venv_zh", Relation.NOT_IN_SOURCE, None, "unsupported_vcs_behavior"),
        ("venvsite", "venv 默认创建 Docker 镜像。", "phase4c_python_venv_zh", Relation.NOT_IN_SOURCE, None, "unsupported_container_behavior"),
        ("venvopinion", "venv is the best way for every Python project。", "phase4c_python_venv_zh", Relation.NOT_CHECKABLE, None, "subjective_comparison"),
    ]
    test_quota = {
        Relation.ENTAILED: 4,
        Relation.PARTIALLY_ENTAILED: 3,
        Relation.CONTRADICTED: 4,
        Relation.NOT_IN_SOURCE: 3,
        Relation.NOT_CHECKABLE: 2,
    }
    assigned_to_test = {relation: 0 for relation in Relation}
    cases: list[EvalCase] = []
    for index, (slug, claim, source_id, relation, evidence, mutation) in enumerate(specs, 1):
        source = sources[source_id]
        case_id = f"phase4c_zh_{slug}_{index:03d}"
        split = "test" if assigned_to_test[relation] < test_quota[relation] else "dev"
        if split == "test":
            assigned_to_test[relation] += 1
        cases.append(
            _case(
                case_id=case_id,
                source=source,
                claim=claim,
                relation=relation,
                evidence=evidence,
                mutation=mutation,
                split=split,
            )
        )
    return cases


def _previous_material(eval_sets: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    old_ids: set[str] = set()
    old_claims: set[str] = set()
    old_urls: set[str] = set()
    old_hashes: set[str] = set()
    for path in eval_sets.rglob("*.jsonl"):
        if PACK in path.parents:
            continue
        try:
            rows = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in rows:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                if isinstance(value.get("case_id"), str):
                    old_ids.add(value["case_id"])
                if isinstance(value.get("claim_text"), str):
                    old_claims.add(_norm_claim(value["claim_text"]))
                if isinstance(value.get("url"), str):
                    old_urls.add(value["url"])
                if isinstance(value.get("content_hash"), str):
                    old_hashes.add(value["content_hash"])
    return old_ids, old_claims, old_urls, old_hashes


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    # Hash the actual bytes on disk.  ``Path.write_text`` applies the host
    # newline convention on Windows, so hashing the pre-write LF string would
    # record a value that cannot verify the artifact byte-for-byte.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    (PACK / "sources").mkdir(parents=True, exist_ok=True)
    sources = _source_fixtures()
    cases = _make_cases(sources)
    old_ids, old_claims, old_urls, old_hashes = _previous_material(PACK.parent)
    new_ids = {case.case_id for case in cases}
    new_claims = {_norm_claim(case.claim_text or "") for case in cases}
    new_urls = {source.url for source in sources.values()}
    new_hashes = {source.content_hash for source in sources.values()}
    audit = {
        "audit_version": "phase4c-leakage-audit-v1",
        "audited_at": RETRIEVAL_DATE,
        "previous_eval_sets_root": "eval_sets",
        "selection_used_labels": True,
        "selection_used_predictions_or_metrics": False,
        "old_case_id_overlap": sorted(new_ids & old_ids),
        "old_claim_overlap": sorted(new_claims & old_claims),
        "old_source_url_overlap": sorted(new_urls & old_urls),
        "old_source_hash_overlap": sorted(new_hashes & old_hashes),
        "new_case_count": len(cases),
        "new_source_count": len(sources),
        "result": "pass" if not any((new_ids & old_ids, new_claims & old_claims, new_urls & old_urls, new_hashes & old_hashes)) else "fail",
        "limitations": [
            "The audit checks exact IDs, normalized claim text, URLs, and source hashes only.",
            "The public source excerpts are deterministically labeled and have not received independent human review.",
            "Semantic paraphrase overlap and external model-training contamination are not ruled out.",
        ],
    }
    if audit["result"] != "pass":
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))

    source_rows = [source.model_dump(mode="json") for source in sources.values()]
    source_path = PACK / FIXTURE_RELATIVE
    source_sha = _write_jsonl(source_path, source_rows)
    dataset_path = PACK / DATASET_NAME
    dataset_sha = _write_jsonl(dataset_path, [case.model_dump(mode="json") for case in cases])
    frozen = {case.case_id: compute_case_hash(case) for case in cases}
    frozen_path = PACK / FROZEN_NAME
    frozen_payload = json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    frozen_path.write_text(frozen_payload, encoding="utf-8")

    manifest = {
        "schema_version": "phase4c-public-zh-diagnostic-v1",
        "pack_id": "phase4c-public-zh-diagnostic",
        "created_on": RETRIEVAL_DATE,
        "benchmark_validity": "diagnostic_contaminated",
        "public_benchmark_eligible": False,
        "annotation_status": "deterministic_gold",
        "annotation_review": "none",
        "source_origin": "official_public_documentation_snapshots",
        "source_count": len(sources),
        "case_count": len(cases),
        "split_counts": {
            "dev": sum(case.split == "dev" for case in cases),
            "test": sum(case.split == "test" for case in cases),
        },
        "relation_distribution": {
            relation.value: sum(case.gold_relation == relation for case in cases)
            for relation in Relation
        },
        "files": {
            DATASET_NAME: dataset_sha,
            FIXTURE_RELATIVE: source_sha,
            FROZEN_NAME: hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        },
        "retrieval_date": RETRIEVAL_DATE,
        "official_source_urls": [source.url for source in sources.values() if source.available],
        "selection_protocol": (
            "Sources and cases were fixed in the builder before evaluation; split assignment "
            "uses fixed per-label quotas for minimum diagnostic coverage and does not inspect "
            "predictions or metrics."
        ),
        "label_protocol": (
            "Entailed/partial/contradicted labels cite literal source substrings; not_in_source "
            "and not_checkable are explicit diagnostic probes. No source_unavailable label was "
            "fabricated because no documented collection failure occurred."
        ),
        "expected_use": "CJK retrieval and empty-recall policy diagnostic only; not a public quality claim.",
        "leakage_audit": audit,
    }
    (PACK / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (PACK / "leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (PACK / "README.md").write_text(
        "# Phase 4C public-source Chinese diagnostic pack\n\n"
        "This pack contains 32 frozen claim/source pairs over four official Chinese documentation snapshots.\n\n"
        "It is **not** an independently human-reviewed holdout: substantive labels are derived from literal source spans, and `not_in_source`/`not_checkable` cases are author-declared probes. The runner must therefore report `diagnostic_contaminated`; the numbers are useful for regression diagnostics, not for a public benchmark or a population-generalization claim.\n\n"
        "Official snapshots: Kubernetes object model, Kubernetes object management, Python tutorial, and Python `venv` documentation. URLs, retrieval date, content hashes, case hashes, and the exact leakage-audit scope are recorded in `manifest.json` and `leakage_audit.json`.\n\n"
        "The pack was frozen before running the deterministic baselines. No model output or metric was used to select cases.\n",
        encoding="utf-8",
    )
    print(json.dumps({"pack": str(PACK), "dataset": str(dataset_path), "cases": len(cases), "sources": len(sources), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
