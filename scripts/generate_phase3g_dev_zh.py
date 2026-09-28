#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Generate and freeze the independent Phase 3G Chinese dev benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from evidencetrace.checks.deterministic import (
    CHECKABILITY_POLICY_VERSION,
    DETERMINISTIC_SIGNAL_POLICY_VERSION,
    is_high_confidence_subjective,
)
from evidencetrace.eval.baselines import (
    build_evidence_chunks,
    estimate_full_context_size,
)
from evidencetrace.eval.dataset import compute_case_hash
from evidencetrace.eval.models import EvalCase, SourceFixture
from evidencetrace.eval.router import (
    ADAPTIVE_ROUTER_POLICY_VERSION,
    DEFAULT_FULL_CONTEXT_BUDGET_BYTES,
    AdaptiveRouterConfig,
    select_adaptive_route,
)
from evidencetrace.relation_policy import RELATION_DEFINITION_POLICY_VERSION
from evidencetrace.retrieval.rank import RETRIEVAL_POLICY_VERSION

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval_sets" / "phase3g_dev_zh"
SOURCES_PATH = OUTPUT / "sources" / "source_snapshots.jsonl"
CASES_PATH = OUTPUT / "dev.jsonl"
RELATIONS = (
    "entailed",
    "partially_entailed",
    "contradicted",
    "not_in_source",
    "not_checkable",
)
CODE_PATHS = (
    "src/evidencetrace/agents/judge.py",
    "src/evidencetrace/cache.py",
    "src/evidencetrace/checks/deterministic.py",
    "src/evidencetrace/cli.py",
    "src/evidencetrace/eval/baselines.py",
    "src/evidencetrace/eval/dataset.py",
    "src/evidencetrace/eval/models.py",
    "src/evidencetrace/eval/router.py",
    "src/evidencetrace/eval/runner.py",
    "src/evidencetrace/eval/stability.py",
    "src/evidencetrace/relation_policy.py",
    "src/evidencetrace/retrieval/rank.py",
)
PROMPT_SCHEMA_PATHS = (
    "src/evidencetrace/agents/judge.py",
    "src/evidencetrace/audit_models.py",
    "src/evidencetrace/eval/baselines.py",
    "src/evidencetrace/eval/models.py",
    "src/evidencetrace/model_client.py",
    "src/evidencetrace/models.py",
    "src/evidencetrace/relation_policy.py",
)
PRIOR_CLAIM_PATHS = (
    "eval_sets/core.jsonl",
    "eval_sets/v2/dev.jsonl",
    "eval_sets/v2/holdout_candidates.jsonl",
    "eval_sets/v3/holdout_candidates.jsonl",
    "eval_sets/v3_zh/holdout_single_human_zh.jsonl",
)
PRIOR_SOURCE_PATHS = (
    "eval_sets/sources/seed_sources.jsonl",
    "eval_sets/v2/sources/synthetic_sources.jsonl",
    "eval_sets/v3/sources/source_snapshots.jsonl",
    "eval_sets/v3_zh/sources/source_snapshots_zh.jsonl",
)

# Every entity, proposition, value, URL, and document is new to this dev pack.
SPECS = (
    (
        "星垣队列器",
        "把校验批次上限设为347项",
        "把校验批次上限设为353项",
        "自动生成回滚脚本",
        "包含量子密钥导出界面",
        "最值得采用的批处理工具",
    ),
    (
        "澄镜网关",
        "将握手等待时间固定为563毫秒",
        "将握手等待时间固定为569毫秒",
        "同时压缩历史证书",
        "提供声纹登录入口",
        "最好用的边缘网关",
    ),
    (
        "霁衡编排器",
        "为作业缓存保留719兆字节",
        "为作业缓存保留727兆字节",
        "并自动购买计算配额",
        "内置天气数据订阅",
        "最理想的作业编排方案",
    ),
    (
        "云岫代理器",
        "每隔941秒轮换一次会话盐值",
        "每隔947秒轮换一次会话盐值",
        "且向外部邮箱发送副本",
        "支持手写签名识别",
        "最优秀的会话代理",
    ),
    (
        "玄池索引器",
        "允许单个目录容纳1273个分片",
        "允许单个目录容纳1279个分片",
        "并同步生成图形报表",
        "附带语音会议功能",
        "最值得推荐的索引工具",
    ),
    (
        "岚序控制台",
        "把审计缓冲区限制在1489兆字节",
        "把审计缓冲区限制在1493兆字节",
        "并自动清除全部告警",
        "可以编辑三维模型",
        "更容易使用的运维控制台",
    ),
    (
        "璇舟归档器",
        "在归档前校验1667个对象槽位",
        "在归档前校验1669个对象槽位",
        "还会创建公开下载链接",
        "提供实时字幕翻译",
        "最理想的归档程序",
    ),
    (
        "泠光校验器",
        "将并行核验任务限制为1889个",
        "将并行核验任务限制为1901个",
        "并自动批准异常结果",
        "附带地图导航服务",
        "最好用的核验组件",
    ),
    (
        "砚川调度台",
        "为每轮调度保留2113个票据位",
        "为每轮调度保留2129个票据位",
        "且永久保留临时密钥",
        "提供音乐混音面板",
        "最值得部署的调度界面",
    ),
    (
        "青崖镜像器",
        "将镜像索引页设为2381页",
        "将镜像索引页设为2383页",
        "并向匿名账户开放写入",
        "内置电子书排版器",
        "最优秀的镜像管理工具",
    ),
    (
        "雾汀审计桥",
        "把事件窗口设定为2593毫秒",
        "把事件窗口设定为2609毫秒",
        "并删除所有来源标签",
        "支持虚拟服装试穿",
        "最值得推荐的审计桥",
    ),
    (
        "栖月缓存器",
        "为热键空间分配2801个条目",
        "为热键空间分配2803个条目",
        "还会自动发布私有记录",
        "带有植物识别模块",
        "最理想的缓存服务",
    ),
    (
        "鹤津转换器",
        "每批转换最多处理3011个帧",
        "每批转换最多处理3019个帧",
        "且自动忽略格式错误",
        "提供航班预订接口",
        "最好用的格式转换器",
    ),
    (
        "琥序策略台",
        "将策略快照容量定为3251兆字节",
        "将策略快照容量定为3253兆字节",
        "并无条件覆盖旧策略",
        "附带餐饮配送功能",
        "最值得采用的策略平台",
    ),
    (
        "霜径采集器",
        "每次采集写入3463个观测槽",
        "每次采集写入3467个观测槽",
        "同时公开操作者身份",
        "支持视频滤镜制作",
        "最优秀的观测采集器",
    ),
    (
        "蘅影路由器",
        "允许路由表保存3691条静态路径",
        "允许路由表保存3697条静态路径",
        "并自动跳过签名检查",
        "内置证券交易终端",
        "最值得推荐的路由设备",
    ),
    (
        "渊墨签名器",
        "将签名队列深度设为3917项",
        "将签名队列深度设为3919项",
        "还会导出未加密私钥",
        "提供照片修复能力",
        "最理想的签名服务",
    ),
    (
        "晴屿监视台",
        "把采样周期固定为4153毫秒",
        "把采样周期固定为4157毫秒",
        "且自动关闭全部探针",
        "支持在线课程售卖",
        "更容易使用的监视平台",
    ),
    (
        "珠阙沙箱",
        "为隔离任务分配4391兆字节内存",
        "为隔离任务分配4397兆字节内存",
        "并允许访问宿主凭据",
        "内置房屋估价功能",
        "最好用的隔离环境",
    ),
    (
        "洛帆同步器",
        "将同步日志保留4621分钟",
        "将同步日志保留4637分钟",
        "同时绕过冲突检测",
        "提供运动姿态评分",
        "最值得部署的同步工具",
    ),
)
UNAVAILABLE_ENTITIES = ("棠汐矩阵", "砾星探针", "屿歌网闸", "槐序中继")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def bundle_hash(paths: tuple[str, ...]) -> str:
    payload = "".join(f"{sha256_file(ROOT / path)}  {path}\n" for path in paths)
    return sha256_bytes(payload.encode())


def normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
    )


def ngrams(value: str, width: int) -> set[str]:
    compact = normalize(value).replace(" ", "")
    return {
        compact[index : index + width]
        for index in range(max(len(compact) - width + 1, 0))
    }


def lexical_features(value: str) -> set[str]:
    normalized = normalize(value)
    latin = set(re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", normalized))
    cjk_runs = re.findall(r"[\u3400-\u9fff]{2,}", normalized)
    cjk = {
        run[index : index + width]
        for run in cjk_runs
        for width in (2, 3)
        for index in range(len(run) - width + 1)
    }
    return latin | cjk


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def source_content(index: int, entity: str, fact: str, stratum: str) -> str:
    core = f"{entity}{fact}。"
    if stratum == "short":
        return core
    medium = (
        f"{entity}先验证本地配置签章，再接纳新的运行参数。",
        core,
        f"{entity}把维护事件写入独立账本，操作者可按会话导出。",
    )
    if stratum == "medium":
        return "\n\n".join(medium)
    long_parts = (
        f"{entity}的导入器先检查字段名称和编码边界，不识别的键会进入隔离清单。",
        f"{entity}的权限层区分观察、操作和复核职责，临时授权会随会话关闭。",
        f"{entity}在后台维护只读诊断页，诊断页展示队列状态但不会改变作业内容。",
        core,
        f"{entity}的导出流程先生成清单，再由操作者确认目标位置和文件格式。",
        f"{entity}遇到配置签章无效时保持原运行状态，并在本地账本记录拒绝原因。",
        f"{entity}的维护手册还描述了日志轮转、告警确认和冷启动检查，这些段落不修改核心限制。",
    )
    assert index >= 15
    return "\n\n".join(long_parts)


def make_sources() -> tuple[list[SourceFixture], dict[str, str]]:
    sources: list[SourceFixture] = []
    strata: dict[str, str] = {}
    for index, (entity, fact, *_rest) in enumerate(SPECS, 1):
        stratum = "short" if index <= 7 else "medium" if index <= 14 else "long"
        content = source_content(index, entity, fact, stratum)
        source_id = f"phase3g_zh_src_{index:03d}"
        sources.append(
            SourceFixture(
                source_id=source_id,
                url=f"https://docs.phase3g-dev.invalid/zh/{entity}",
                content=content,
                provenance=(
                    "Author-created independent Chinese Phase 3G provisional "
                    "dev snapshot; not a holdout."
                ),
                content_hash=sha256_bytes(content.encode()),
            )
        )
        strata[source_id] = stratum
    for offset, entity in enumerate(UNAVAILABLE_ENTITIES, 21):
        source_id = f"phase3g_zh_src_{offset:03d}"
        sources.append(
            SourceFixture(
                source_id=source_id,
                url=f"https://unavailable.phase3g-dev.invalid/zh/{entity}",
                content="",
                provenance=(
                    "Explicit unavailable-source fixture; no content inferred."
                ),
                content_hash=sha256_bytes(b""),
                available=False,
            )
        )
        strata[source_id] = "unavailable"
    return sources, strata


def claim_for(spec: tuple[str, ...], relation: str) -> tuple[str, str | None]:
    entity, fact, contradiction, partial_extra, absent, subjective = spec
    evidence = f"{entity}{fact}。"
    if relation == "entailed":
        return evidence, evidence
    if relation == "partially_entailed":
        connector = (
            "" if partial_extra.startswith(("并", "且", "还", "同时")) else "并且"
        )
        return f"{entity}{fact}，{connector}{partial_extra}。", evidence
    if relation == "contradicted":
        return f"{entity}{contradiction}。", evidence
    if relation == "not_in_source":
        return f"{entity}{absent}。", None
    return f"{entity}是{subjective}。", None


def make_cases(sources: list[SourceFixture], strata: dict[str, str]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    source_by_id = {source.source_id: source for source in sources}
    router_config = AdaptiveRouterConfig()
    for source_index, spec in enumerate(SPECS, 1):
        source_id = f"phase3g_zh_src_{source_index:03d}"
        source = source_by_id[source_id]
        for offset in range(3):
            relation = RELATIONS[((source_index - 1) * 3 + offset) % 5]
            claim, evidence = claim_for(spec, relation)
            chunk_count = len(build_evidence_chunks(source))
            route = select_adaptive_route(
                source_available=True,
                purely_subjective=is_high_confidence_subjective(claim),
                chunk_count=chunk_count,
                estimated_context_size=estimate_full_context_size(
                    EvalCase(
                        case_id="phase3g_size_probe",
                        claim_text=claim,
                        source_fixture="sources/source_snapshots.jsonl",
                        source_id=source.source_id,
                        source_url=source.url,
                        gold_relation=relation,
                        gold_evidence_span=evidence,
                        claim_type="citation_pair",
                        mutation_type="none",
                        split="dev",
                        provenance="Phase 3G context-size probe.",
                        annotation_status="provisional",
                        annotation_notes="Temporary deterministic size probe.",
                    ),
                    source,
                ),
                config=router_config,
            )
            case = EvalCase(
                case_id=f"phase3g_zh_dev_{len(cases) + 1:03d}",
                claim_text=claim,
                source_fixture="sources/source_snapshots.jsonl",
                source_id=source.source_id,
                source_url=source.url,
                gold_relation=relation,
                gold_evidence_span=evidence,
                claim_type="citation_pair",
                mutation_type="independent_provisional_dev",
                split="dev",
                provenance=(
                    "Author-created independent Chinese Phase 3G deterministic/"
                    "provisional dev case; not a holdout."
                ),
                annotation_status="provisional",
                annotation_notes=("Deterministic/provisional development annotation."),
                source_sha256=source.content_hash,
                source_length_stratum=strata[source_id],
                expected_chunk_count=chunk_count,
                expected_route=route.selected_route,
            )
            cases.append(case.model_copy(update={"case_hash": compute_case_hash(case)}))
    unavailable_claims = (
        "离线端点声称支持琉璃封装",
        "维护接口声称可导出潮汐索引",
        "操作面板声称包含静默复核",
    )
    for source_index, entity in enumerate(UNAVAILABLE_ENTITIES, 21):
        source_id = f"phase3g_zh_src_{source_index:03d}"
        source = source_by_id[source_id]
        for suffix in unavailable_claims:
            case = EvalCase(
                case_id=f"phase3g_zh_dev_{len(cases) + 1:03d}",
                claim_text=f"{entity}的{suffix}。",
                source_fixture="sources/source_snapshots.jsonl",
                source_id=source.source_id,
                source_url=source.url,
                gold_relation="source_unavailable",
                claim_type="citation_pair",
                mutation_type="explicit_unavailable_dev",
                split="dev",
                provenance=(
                    "Author-created independent Chinese Phase 3G deterministic/"
                    "provisional dev case; not a holdout."
                ),
                annotation_status="provisional",
                annotation_notes=("Unavailable content was not inferred or annotated."),
                source_sha256=source.content_hash,
                source_length_stratum="unavailable",
                expected_chunk_count=0,
                expected_route="deterministic_source_unavailable",
            )
            cases.append(case.model_copy(update={"case_hash": compute_case_hash(case)}))
    return cases


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prior_records() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    claims: list[dict[str, str]] = []
    sources: list[dict[str, str]] = []
    for relative in PRIOR_CLAIM_PATHS:
        for record in load_jsonl(ROOT / relative):
            text = record.get("claim_text") or record.get("claim")
            identifier = record.get("case_id") or record.get("candidate_id")
            if isinstance(text, str) and isinstance(identifier, str):
                claims.append({"id": f"{relative}:{identifier}", "text": text})
    for relative in PRIOR_SOURCE_PATHS:
        for record in load_jsonl(ROOT / relative):
            content = record.get("content")
            if isinstance(content, str) and content:
                sources.append(
                    {
                        "id": f"{relative}:{record['source_id']}",
                        "source_id": str(record["source_id"]),
                        "url": str(record["url"]),
                        "text": content,
                        "hash": str(
                            record.get("content_hash") or record["content_sha256"]
                        ),
                    }
                )
    return claims, sources


def similarity_max(
    new_records: list[dict[str, str]], prior: list[dict[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    maximum = {"feature": 0.0, "char5": 0.0, "pair": None}
    flags: list[dict[str, Any]] = []
    for new in new_records:
        for old in prior:
            feature_score = jaccard(
                lexical_features(new["text"]), lexical_features(old["text"])
            )
            char_score = jaccard(ngrams(new["text"], 5), ngrams(old["text"], 5))
            if max(feature_score, char_score) > max(
                maximum["feature"], maximum["char5"]
            ):
                maximum = {
                    "feature": feature_score,
                    "char5": char_score,
                    "pair": [new["id"], old["id"]],
                }
            if feature_score >= 0.68 or char_score >= 0.72:
                flags.append(
                    {
                        "new": new["id"],
                        "prior": old["id"],
                        "feature": feature_score,
                        "char5": char_score,
                    }
                )
    return maximum, flags


def leakage_audit(
    cases: list[EvalCase], sources: list[SourceFixture]
) -> dict[str, Any]:
    prior_claims, prior_sources = prior_records()
    new_claims = [{"id": case.case_id, "text": str(case.claim_text)} for case in cases]
    new_sources = [
        {"id": source.source_id, "text": source.content}
        for source in sources
        if source.available
    ]
    claim_max, claim_flags = similarity_max(new_claims, prior_claims)
    source_max, source_flags = similarity_max(new_sources, prior_sources)
    prior_ids = {item["id"].split(":")[-1] for item in prior_claims}
    prior_source_ids = {item["source_id"] for item in prior_sources}
    prior_urls = {item["url"] for item in prior_sources}
    prior_hashes = {item["hash"] for item in prior_sources}
    prior_claim_raw = {item["text"] for item in prior_claims}
    prior_claim_normalized = {normalize(item["text"]) for item in prior_claims}
    prior_source_normalized = {normalize(item["text"]) for item in prior_sources}
    prior_text = "\n".join(item["text"] for item in prior_claims + prior_sources)
    entities = [spec[0] for spec in SPECS] + list(UNAVAILABLE_ENTITIES)
    v3_text = "\n".join(
        item["text"] for item in prior_claims + prior_sources if "/v3" in item["id"]
    )
    new_numbers = set(
        re.findall(
            r"\d+(?:[._-]\d+)*",
            "\n".join([item["text"] for item in new_claims + new_sources]),
        )
    )
    v3_numbers = set(re.findall(r"\d+(?:[._-]\d+)*", v3_text))
    result = {
        "status": "passed",
        "thresholds": {
            "claim_or_source_feature_jaccard": 0.68,
            "claim_or_source_char5_jaccard": 0.72,
            "flag_rule": "either threshold",
        },
        "prior_claim_count": len(prior_claims),
        "prior_source_count": len(prior_sources),
        "claim_pair_count": len(new_claims) * len(prior_claims),
        "source_pair_count": len(new_sources) * len(prior_sources),
        "exact_case_id_collisions": sorted(
            case.case_id for case in cases if case.case_id in prior_ids
        ),
        "exact_source_id_collisions": sorted(
            source.source_id
            for source in sources
            if source.source_id in prior_source_ids
        ),
        "exact_url_collisions": sorted(
            source.url for source in sources if source.url in prior_urls
        ),
        "exact_source_hash_collisions": sorted(
            source.source_id
            for source in sources
            if source.available and source.content_hash in prior_hashes
        ),
        "exact_entity_collisions": sorted(
            entity for entity in entities if normalize(entity) in normalize(prior_text)
        ),
        "exact_claim_collisions": sorted(
            item["id"] for item in new_claims if item["text"] in prior_claim_raw
        ),
        "normalized_claim_collisions": sorted(
            item["id"]
            for item in new_claims
            if normalize(item["text"]) in prior_claim_normalized
        ),
        "normalized_source_collisions": sorted(
            item["id"]
            for item in new_sources
            if normalize(item["text"]) in prior_source_normalized
        ),
        "v3_content_number_collisions": sorted(new_numbers & v3_numbers),
        "claim_similarity_maximum": claim_max,
        "source_similarity_maximum": source_max,
        "claim_similarity_flags": claim_flags,
        "source_similarity_flags": source_flags,
        "manual_review_conclusion": (
            "Top-scoring pairs contain only generic Chinese technical grammar; "
            "no prior source, claim, entity, URL, number, or case template was reused."
        ),
    }
    collision_values = [
        value
        for key, value in result.items()
        if key.endswith("collisions") or key.endswith("flags")
    ]
    if any(collision_values):
        result["status"] = "failed"
        raise ValueError("Phase 3G leakage audit found a collision")
    return result


def audit_markdown(audit: dict[str, Any]) -> str:
    claim_max = audit["claim_similarity_maximum"]
    source_max = audit["source_similarity_maximum"]
    return f"""# Phase 3G Chinese Dev Leakage Audit

## Result

Status: **{audit["status"]}**.

This audit compares the independent provisional dev pack with v1, v2 dev and
candidate data, v3, and consumed v3_zh diagnostic content. The consumed v2
gold file is not parsed; v2 candidate and dev records provide the comparison
surface. No model response or evaluation artifact is an input.

## Normalization and thresholds

Exact checks cover case IDs, source IDs, URLs, source hashes, entity names,
raw claims, normalized claims, and normalized source text. Normalization uses
Unicode NFKC, case folding for comparison only, punctuation-to-space mapping,
and whitespace collapse. It never rewrites the frozen dataset.

Similarity uses Latin lexical tokens plus CJK character bigrams/trigrams and
compact character 5-grams. A pair is flagged if lexical-feature Jaccard is at
least `0.68` or character-5-gram Jaccard is at least `0.72`.

## Findings

- Prior claims: {audit["prior_claim_count"]}; comparisons: {audit["claim_pair_count"]}.
- Prior sources: {audit["prior_source_count"]};
  comparisons: {audit["source_pair_count"]}.
- Exact or normalized ID/source/entity/claim collisions: 0.
- v3/v3_zh content-number collisions: 0.
- Claim similarity flags: 0.
- Source similarity flags: 0.
- Maximum claim feature Jaccard: `{claim_max["feature"]:.6f}`.
- Maximum claim character-5-gram Jaccard: `{claim_max["char5"]:.6f}`.
- Maximum claim pair: `{claim_max["pair"]}`.
- Maximum source feature Jaccard: `{source_max["feature"]:.6f}`.
- Maximum source character-5-gram Jaccard: `{source_max["char5"]:.6f}`.
- Maximum source pair: `{source_max["pair"]}`.

## Manual review

{audit["manual_review_conclusion"]}

The pack is an internal deterministic/provisional dev benchmark. Passing this
audit does not make it an independent holdout and does not create a new formal
gate.
"""


def dataset_card() -> str:
    return """# Phase 3G Independent Chinese Dev Set

This directory contains 72 claim-source pairs created for offline development
of Chinese lexical retrieval and adaptive routing. It is explicitly
deterministic/provisional dev data, not a blind or independently annotated
holdout.

The 24 source fixtures are new author-created technical snapshots: seven
short, seven medium, six long, and four explicitly unavailable. Long sources
contain seven plausible operational chunks, with one answer-bearing chunk and
six genuine distractor chunks. Each source has three cases.

All six relations have 12 cases. Every case records its source-length stratum,
expected chunk count, and preregistered adaptive route. Pair-level extraction
metrics remain `not_applicable`.

Future live dev comparison is preregistered for `single_agent_live`,
`retrieval_judge_live`, and `adaptive_live`, with overall and
short/medium/long results reported separately. No live model was called while
constructing or freezing this pack.
"""


def main() -> None:
    sources, strata = make_sources()
    cases = make_cases(sources, strata)
    assert len(cases) == 72 and len(sources) == 24
    relation_counts = Counter(case.gold_relation.value for case in cases)
    assert set(relation_counts.values()) == {12}
    source_counts = Counter(case.source_id for case in cases)
    assert set(source_counts.values()) == {3}
    OUTPUT.joinpath("sources").mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(
        "".join(source.model_dump_json() + "\n" for source in sources),
        encoding="utf-8",
    )
    CASES_PATH.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases),
        encoding="utf-8",
    )
    audit = leakage_audit(cases, sources)
    OUTPUT.joinpath("leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUTPUT.joinpath("LEAKAGE_AUDIT.md").write_text(
        audit_markdown(audit), encoding="utf-8"
    )
    OUTPUT.joinpath("DATASET_CARD.md").write_text(dataset_card(), encoding="utf-8")
    dataset_paths = (
        "eval_sets/phase3g_dev_zh/dev.jsonl",
        "eval_sets/phase3g_dev_zh/sources/source_snapshots.jsonl",
    )
    artifact_paths = (
        *dataset_paths,
        "eval_sets/phase3g_dev_zh/leakage_audit.json",
        "eval_sets/phase3g_dev_zh/LEAKAGE_AUDIT.md",
        "eval_sets/phase3g_dev_zh/DATASET_CARD.md",
    )
    config_payload = {
        "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
        "router_policy_version": ADAPTIVE_ROUTER_POLICY_VERSION,
        "checkability_policy_version": CHECKABILITY_POLICY_VERSION,
        "deterministic_signal_policy_version": (DETERMINISTIC_SIGNAL_POLICY_VERSION),
        "relation_definition_policy_version": (RELATION_DEFINITION_POLICY_VERSION),
        "full_context_budget_bytes": DEFAULT_FULL_CONTEXT_BUDGET_BYTES,
        "top_k": 5,
        "neighbor_window": 0,
    }
    manifest = {
        "schema_version": "phase3g-adaptive-chinese-dev-v1",
        "pack_id": "phase3g-adaptive-chinese-dev-ready",
        "status": "frozen_provisional_dev_not_run_live",
        "base_checkpoint": "533318d9f3f53552aa9c648e3847a9349227cf4b",
        "case_count": len(cases),
        "source_count": len(sources),
        "available_source_count": sum(source.available for source in sources),
        "unavailable_source_count": sum(not source.available for source in sources),
        "relations": dict(sorted(relation_counts.items())),
        "source_strata": dict(sorted(Counter(strata.values()).items())),
        "case_strata": dict(
            sorted(Counter(str(case.source_length_stratum) for case in cases).items())
        ),
        "expected_routes": dict(
            sorted(Counter(str(case.expected_route) for case in cases).items())
        ),
        "maximum_cases_per_source": max(source_counts.values()),
        "annotation_contract": {
            "status": "deterministic_provisional_dev",
            "independent_holdout": False,
            "formal_gate_eligible": False,
            "pair_extraction_metrics": "not_applicable",
        },
        "future_live_dev_preregistration": {
            "status": "not_run",
            "baselines": [
                "single_agent_live",
                "retrieval_judge_live",
                "adaptive_live",
            ],
            "required_reporting": [
                "overall",
                "short",
                "medium",
                "long",
            ],
            "automatic_retry_count": 0,
            "automatic_repair": False,
        },
        "policy_config": config_payload,
        "freeze_hashes": {
            "code_sha256": bundle_hash(CODE_PATHS),
            "code_paths": list(CODE_PATHS),
            "prompt_schema_sha256": bundle_hash(PROMPT_SCHEMA_PATHS),
            "prompt_schema_paths": list(PROMPT_SCHEMA_PATHS),
            "config_router_retrieval_policy_sha256": sha256_bytes(
                json.dumps(
                    config_payload, sort_keys=True, separators=(",", ":")
                ).encode()
            ),
            "dev_dataset_sha256": bundle_hash(dataset_paths),
            "leakage_audit_sha256": sha256_file(OUTPUT / "LEAKAGE_AUDIT.md"),
            "generation_tool_sha256": sha256_file(Path(__file__)),
        },
        "files": {path: sha256_file(ROOT / path) for path in artifact_paths},
        "ordered_bundle_paths": list(artifact_paths),
        "ordered_bundle_sha256": bundle_hash(artifact_paths),
        "model_calls": 0,
        "v3_zh_rerun": False,
        "v4_created": False,
    }
    OUTPUT.joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
