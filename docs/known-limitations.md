# EvidenceTrace 已知局限

本文是 2026-07-29 的事实快照，用于防止迁移和项目介绍时夸大能力。它不是新的
roadmap；后续任务仍由用户和 Projects 根 v2.0 计划决定。

## 1. 当前状态不能合并成一个“完成”

权威计划同时保留：

```text
self_use_check=operational_partial
latest_run_document_status=partial
interactive_fix=accepted
candidate_patch=unverified
overall_new_self_use_goal=incomplete
Miner P0=improved_but_blocking
Phase 3=completed_with_known_limitations
phase4_eligible=false
```

这些字段描述不同层次和不同 checkpoint。`interactive_fix=accepted` 不能覆盖只读
check 的 partial 运行、历史评测限制或 Miner P0。

## 2. 它不是 truth detector

Verdict 只说明当前 bounded evidence 与 claim 的关系。证据不足、来源冲突、Agent
error、abstain、预算或 transport failure 都必须显式保留为 partial 或
`needs_human`，不能解释成事实真假已经确定。

## 3. 修复范围很窄

交互式 repair 只覆盖有 exact evidence 的数字、百分比、allowlist 日期和 SemVer。
当前不支持：

- 通用事实或自由文本重写；
- 单位换算；
- 自动插入 citation；
- PDF 或目录递归；
- fenced code、URL、front matter、HTML attribute 等禁止位置；
- 无人值守、后台、CI 或 Agent 自动写回。

## 4. 联网路径依赖外部服务

OpenAI-compatible provider 和 Tavily 可能出现 schema、transport、budget 或
availability failure。Search snippet 不是 evidence；候选 URL 必须经 safe fetch
形成 exact bounded span。失败时应安全 partial，而不是增加 retry、token、预算或
循环重跑挑结果。

## 5. 最新历史 check 仍是 partial

Canonical run `20260728T065626Z-40dc78f9` 验证了 citation handoff、真实 Tavily
路径和 not-checkable，但还记录了 Challenger、Judge 和 Scout error。该运行是历史
事实，不能因后续交互式 acceptance 而改写为成功。

## 6. 评测证据有明确边界

- Phase 3 是 `completed_with_known_limitations`。
- Consumed holdouts 不能重新用于正式选择。
- Pair benchmark 没有运行 Claim Miner。
- Provisional full-document smoke 没有证明 Multi-Agent 优于 Single Agent。
- 没有可宣称的生产级高 recall、完整事实覆盖或正式盲测质量优势。

## 7. 发布与运维尚未完成

项目当前不是已验证的 PyPI/public Action 发布物，也没有 Web UI 或生产服务部署。
本轮目标是本地自用 CLI 和可信工程边界，不应把 `uvx`、公开发布或高 star 当成已
完成能力。

## 8. Git 与可恢复性

2026-07-29 快照没有 remote，关键实现位于 dirty worktree。迁移必须保留 `.git`、
修改文件和未跟踪文件，不能只保存 HEAD。任何 reset、clean、checkout 覆盖、
stage、commit 或 push 都需要新的明确授权。

## 9. 可安全表述

可以说：

> EvidenceTrace 是一个 bounded Multi-Agent Markdown/TXT fact-audit CLI，支持
> citation、本地 reference 和受控 discovery，并在交互式 TTY 中对通过全部安全门
> 的 scalar correction 做逐条确认、原子写回和可逆 diff。

不能说：

- 自动修复任意事实；
- production-ready 或完整审计；
- 已证明高 recall；
- Multi-Agent 已证明优于 Single Agent；
- 无需人工即可安全修改文档；
- 历史 partial 或 Phase 3/P0 限制已经消失。

